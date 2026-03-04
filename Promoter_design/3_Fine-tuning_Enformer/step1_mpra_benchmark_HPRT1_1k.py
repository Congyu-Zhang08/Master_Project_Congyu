import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import sys
import time

try:
    from enformer_pytorch import Enformer
except ImportError:
    raise ImportError("pip install enformer-pytorch")

CONFIG = {
    'SEQ_LEN_ENFORMER': 196608,
    'EMBED_DIM': 3072,
    'BATCH_SIZE': 24,
    'NUM_WORKERS': 16,
    'DROPOUT_RATE': 0.0 
}

INFERENCE_CONFIG = {
    "MPRA_FILE_PATH": "mpra_data.txt", 
    "MODEL_PATH": "./checkpoints_final/best_model.pth", 
    "OUTPUT_PREDS_PATH": "./raw_predictions_HPRT1_1k.npy",
    "BACKBONE_FILE": "./final_HPRT1_backbone_1k.txt",
    "TRAIN_PKL": "./processed_data/dataset_train.pkl",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TSS_INDEX": 824,       
    "SEQ_LEN_INPUT": 1024,
    "REPLACE_START_OFFSET": -250, 
    "REPLACE_END_OFFSET": -50
}

class CellSpecificAttentionHead(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.2):
        super().__init__()
        self.attn_net = nn.Sequential(nn.Linear(embed_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.layer1 = nn.Sequential(
            nn.Linear(embed_dim, 1536), nn.LayerNorm(1536), nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1536, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.output_layer = nn.Linear(512, 1)

    def forward(self, embeddings):
        attn_weights = torch.softmax(self.attn_net(embeddings), dim=1)
        pooled = torch.sum(embeddings * attn_weights, dim=1)
        x = self.output_layer(self.layer2(self.layer1(pooled)))
        return x

class EnformerGeneExprModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        self.heads = nn.ModuleList([
            CellSpecificAttentionHead(CONFIG['EMBED_DIM'], CONFIG['DROPOUT_RATE']) 
            for _ in range(3)
        ])

    def forward(self, x):
        full = self.enformer(x, return_only_embeddings=True)
        target = full[:, 442:454, :]
        return torch.cat([head(target) for head in self.heads], dim=1)

def load_fixed_backbone(filepath):
    print(f"Loading fixed 1kb backbone from {filepath}...")
    if not os.path.exists(filepath):
        alt_path = filepath.replace("_with_MinP", "")
        if os.path.exists(alt_path):
            print(f"Warning: Specific file not found, falling back to {alt_path}")
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Backbone file not found: {filepath}")
        
    with open(filepath, "r") as f:
        seq = f.read().strip()
        
    if len(seq) != INFERENCE_CONFIG['SEQ_LEN_INPUT']:
        raise ValueError(f"Backbone length {len(seq)} != Expected {INFERENCE_CONFIG['SEQ_LEN_INPUT']}")
    return seq

def gpu_one_hot_and_pad(batch_seq_indices, device):
    B, L = batch_seq_indices.shape
    TOTAL_LEN = CONFIG['SEQ_LEN_ENFORMER']
    
    embed_weight = torch.tensor([
        [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]
    ], dtype=torch.float32, device=device)
    
    small_one_hot = nn.functional.embedding(batch_seq_indices, embed_weight)
    
    final_input = torch.zeros((B, TOTAL_LEN, 4), dtype=torch.float32, device=device)
    
    start = (TOTAL_LEN - L) // 2
    final_input[:, start : start + L, :] = small_one_hot
    
    return final_input

class MPRA_1KB_Lite_Dataset(Dataset):
    def __init__(self, file_path, backbone_seq):
        try:
            self.df = pd.read_csv(file_path, sep='\t')
        except:
            self.df = pd.read_csv(file_path)
            
        self.df = self.df.dropna(subset=['sequence']).reset_index(drop=True)
        
        self.mapping = {
            'A':0, 'C':1, 'G':2, 'T':3, 'N':4,
            'a':0, 'c':1, 'g':2, 't':3, 'n':4
        }
        
        self.backbone_arr = np.array([self.mapping.get(b, 4) for b in backbone_seq], dtype=np.int8)
        
        tss_idx = INFERENCE_CONFIG['TSS_INDEX']
        self.rep_start = tss_idx + INFERENCE_CONFIG['REPLACE_START_OFFSET']
        self.rep_end = tss_idx + INFERENCE_CONFIG['REPLACE_END_OFFSET']
        self.target_len = self.rep_end - self.rep_start
        
        print(f"Dataset initialized (Lite Mode).")
        print(f"  - Backbone Length: {len(self.backbone_arr)} bp")
        print(f"  - Insert Window: {self.rep_start}-{self.rep_end}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        user_seq = self.df.iloc[idx]['sequence']
        
        current_seq = self.backbone_arr.copy()
        
        u_indices = [self.mapping.get(s, 4) for s in user_seq]
        
        if len(u_indices) > self.target_len:
            u_indices = u_indices[:self.target_len]
        elif len(u_indices) < self.target_len:
            u_indices = u_indices + [4]*(self.target_len - len(u_indices))
            
        current_seq[self.rep_start : self.rep_end] = u_indices
        
        return torch.from_numpy(current_seq).long()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    
    device = INFERENCE_CONFIG['DEVICE']
    print(f"Using Model: {INFERENCE_CONFIG['MODEL_PATH']}")
    print(f"Using Backbone: {INFERENCE_CONFIG['BACKBONE_FILE']}")
    
    backbone_seq = load_fixed_backbone(INFERENCE_CONFIG['BACKBONE_FILE'])
    
    model = EnformerGeneExprModel().to(device)
    if not os.path.exists(INFERENCE_CONFIG['MODEL_PATH']):
        raise FileNotFoundError("Model checkpoint not found.")
    model.load_state_dict(torch.load(INFERENCE_CONFIG['MODEL_PATH'], map_location=device))
    model.eval()
    
    print("Loading training stats...")
    if os.path.exists(INFERENCE_CONFIG['TRAIN_PKL']):
        df_t = pd.read_pickle(INFERENCE_CONFIG['TRAIN_PKL'])
        cols = ['logTPM_K562', 'logTPM_HepG2', 'logTPM_SKNSH']
        t_mean = torch.tensor(df_t[cols].values.mean(axis=0)).float().to(device)
        t_std = torch.tensor(df_t[cols].values.std(axis=0)).float().to(device)
        del df_t
    else:
        print("WARNING: Stats not found, using raw Z-scores.")
        t_mean, t_std = 0.0, 1.0

    dataset = MPRA_1KB_Lite_Dataset(INFERENCE_CONFIG['MPRA_FILE_PATH'], backbone_seq)
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True,
        prefetch_factor=2
    )
    
    all_preds = []
    print(f"Starting Inference on {len(dataset)} sequences...")
    print(f"Configuration: Batch={CONFIG['BATCH_SIZE']}, Workers={CONFIG['NUM_WORKERS']}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch_indices in enumerate(loader):
            batch_indices = batch_indices.to(device, non_blocking=True)
            
            seqs = gpu_one_hot_and_pad(batch_indices, device)
            
            z_fwd = model(seqs)
            z_rc = model(torch.flip(seqs, [1, 2]))
            p_z = (z_fwd + z_rc) / 2
            
            p_raw = p_z * (t_std + 1e-8) + t_mean
            
            all_preds.append(p_raw.cpu().numpy())
            
            if (i+1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = ((i+1) * CONFIG['BATCH_SIZE']) / elapsed
                print(f"Batch {i+1}/{len(loader)} | Speed: {speed:.1f} seqs/sec", end='\r')
            
    final_arr = np.concatenate(all_preds, axis=0)
    np.save(INFERENCE_CONFIG['OUTPUT_PREDS_PATH'], final_arr)
    
    print(f"\nDone. Results saved to {INFERENCE_CONFIG['OUTPUT_PREDS_PATH']}")