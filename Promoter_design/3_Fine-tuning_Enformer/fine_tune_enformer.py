import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random

# Ensure torchmetrics is installed
try:
    from torchmetrics import SpearmanCorrCoef
except ImportError:
    raise ImportError("Please run: pip install torchmetrics")

# ================= CONFIGURATION =================
CONFIG = {
    "DATA_DIR": "./processed_data", 
    "SAVE_DIR": "./checkpoints_final",
    "BATCH_SIZE": 4,              
    "TARGET_BATCH_SIZE": 96,      
    "LEARNING_RATE": 1e-4,        
    "WEIGHT_DECAY": 1e-4,         
    "MAX_EPOCHS": 50,             
    "PATIENCE": 5,                
    "SEQ_LEN_ENFORMER": 196608, 
    "EMBED_DIM": 3072,           
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 4,
    "DROPOUT_RATE": 0.2
}

ACCUM_STEPS = CONFIG['TARGET_BATCH_SIZE'] // CONFIG['BATCH_SIZE']

# ================= 1. DATASET =================
class GeneExpressionDataset(Dataset):
    def __init__(self, pkl_path, split='train', t_mean=None, t_std=None):
        print(f"Loading dataset from {pkl_path}...")
        self.df = pd.read_pickle(pkl_path)
        self.split = split
        self.t_mean = t_mean
        self.t_std = t_std
        self.target_cols = ['logTPM_K562', 'logTPM_HepG2', 'logTPM_SKNSH']
        
        self.one_hot_embed = torch.zeros(256, 4)
        for b, v in [('a',[1,0,0,0]),('c',[0,1,0,0]),('g',[0,0,1,0]),('t',[0,0,0,1]),('n',[0,0,0,0])]:
            self.one_hot_embed[ord(b.lower())] = torch.Tensor(v)
            self.one_hot_embed[ord(b.upper())] = torch.Tensor(v)
        self.total_len = CONFIG['SEQ_LEN_ENFORMER']

    def __len__(self):
        return len(self.df)

    def one_hot_encode(self, sequence):
        # Augmentation only for training
        if self.split == 'train' and random.random() > 0.5:
            trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
            sequence = sequence.translate(trans)[::-1]

        arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        seq_tensor = torch.from_numpy(arr.copy()).long()
        encoded = self.one_hot_embed[seq_tensor] 
        
        current_len = encoded.shape[0]
        if current_len < self.total_len:
            final = torch.zeros(self.total_len, 4)
            start = (self.total_len - current_len) // 2
            final[start : start + current_len] = encoded
            return final
        return encoded

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.one_hot_encode(row['sequence'])
        y = np.array([row[c] for c in self.target_cols], dtype=np.float32)
        
        if self.t_mean is not None and self.t_std is not None:
            y = (y - self.t_mean) / (self.t_std + 1e-8)
            
        return x, torch.from_numpy(y)

# ================= 2. MODEL =================
try:
    from enformer_pytorch import Enformer
except ImportError:
    raise ImportError("pip install enformer-pytorch")

class CellSpecificAttentionHead(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.2):
        super().__init__()
        
        # 1. Attention Mechanism
        # Slightly increased hidden dim for better attention resolution
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, 256), 
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # 2. Deep MLP Regressor (3-Layer Pyramid)
        # Structure: Linear -> LayerNorm -> GELU -> Dropout
        
        # Hidden Layer 1: 3072 -> 1536
        self.layer1 = nn.Sequential(
            nn.Linear(embed_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden Layer 2: 1536 -> 512
        self.layer2 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output Layer: 512 -> 1
        self.output_layer = nn.Linear(512, 1)
        
    def forward(self, embeddings):
        # embeddings shape: [Batch, N_BINS, 3072]
        
        # --- Step 1: Attention Pooling ---
        attn_scores = self.attn_net(embeddings) # [Batch, N_BINS, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted Sum -> [Batch, 3072]
        pooled = torch.sum(embeddings * attn_weights, dim=1)
        
        # --- Step 2: Deep MLP ---
        x = self.layer1(pooled)  # -> 1536
        x = self.layer2(x)       # -> 512
        x = self.output_layer(x) # -> 1
        
        return x

class EnformerGeneExprModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading Enformer (Official Pretrained)...")
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        
        # Full Fine-tuning enabled
        for param in self.enformer.parameters():
            param.requires_grad = True
            
        # Initialize 3-layer MLP heads
        self.heads = nn.ModuleList([
            CellSpecificAttentionHead(CONFIG['EMBED_DIM'], CONFIG['DROPOUT_RATE']) 
            for _ in range(3)
        ])

    def forward(self, x):
        # Freeze BN stats (but allow weight updates)
        for m in self.enformer.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.track_running_stats = False

        # 1. Get full embeddings [Batch, 896, 3072]
        full_embeddings = self.enformer(x, return_only_embeddings=True)
        
        # 2. Smart Crop for 1024bp Insert
        # Enformer bin size = 128bp. 
        # Center 8 bins (444:452) cover exactly 1024bp.
        # We take center 12 bins (442:454) to capture flanking regions/boundary effects.
        # 12 bins * 128bp = 1536bp coverage.
        target_embeddings = full_embeddings[:, 442:454, :]
        
        # 3. Pass to Heads
        outputs = [head(target_embeddings) for head in self.heads]
        return torch.cat(outputs, dim=1)

# ================= 3. LOOPS =================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    
    for i, (seqs, targets) in enumerate(loader):
        seqs, targets = seqs.to(device), targets.to(device)
        
        preds = model(seqs)
        loss = criterion(preds, targets) / ACCUM_STEPS
        loss.backward()
        
        total_loss += loss.item() * ACCUM_STEPS 
        
        if (i + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            
            if (i + 1) % (ACCUM_STEPS * 10) == 0:
                 print(f"  [Batch {i+1}] Train Loss: {loss.item() * ACCUM_STEPS:.4f}")

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    
    # Rigorous Spearman calculation using torchmetrics
    spearman_metric = SpearmanCorrCoef(num_outputs=3).to(device)
    
    try:
        cell_names = loader.dataset.target_cols
    except AttributeError:
        cell_names = ['K562', 'HepG2', 'SKNSH']
    
    print(f"  -> Validating...")
    
    with torch.no_grad():
        for i, (seqs, targets) in enumerate(loader):
            seqs = seqs.to(device)
            targets = targets.to(device)
            
            # TTA (Test Time Augmentation)
            pred_fwd = model(seqs)
            pred_rc = model(torch.flip(seqs, dims=[1, 2]))
            p = (pred_fwd + pred_rc) / 2
            
            spearman_metric.update(p, targets)

    # Compute final metric over the whole validation set
    scores = spearman_metric.compute()
    
    print("\n" + "="*45)
    print(f"{'Cell Line':<15} | {'Spearman':<20}")
    print("-" * 45)
    
    correlations = []
    for i, name in enumerate(cell_names):
        sr = scores[i].item()
        correlations.append(sr)
        
        display_name = name.replace('logTPM_', '')
        print(f"{display_name:<15} | {sr:.4f}")
        
    print("="*45 + "\n")
    
    spearman_metric.reset()
    avg_sr = sum(correlations) / len(correlations)
    return avg_sr

# ================= 4. MAIN =================
if __name__ == "__main__":
    if not os.path.exists(CONFIG['SAVE_DIR']):
        os.makedirs(CONFIG['SAVE_DIR'])
        
    print(f"Device: {CONFIG['DEVICE']} | Batch: {CONFIG['BATCH_SIZE']}")

    train_path = os.path.join(CONFIG['DATA_DIR'], "dataset_train.pkl")
    valid_path = os.path.join(CONFIG['DATA_DIR'], "dataset_valid.pkl")
    
    print("Calculating Statistics...")
    df_t = pd.read_pickle(train_path)
    cols = ['logTPM_K562', 'logTPM_HepG2', 'logTPM_SKNSH']
    m = df_t[cols].values.mean(axis=0).astype(np.float32)
    s = df_t[cols].values.std(axis=0).astype(np.float32)
    del df_t
    print(f"Stats: Mean={m}, Std={s}")

    train_loader = DataLoader(GeneExpressionDataset(train_path, 'train', m, s), 
                              batch_size=CONFIG['BATCH_SIZE'], shuffle=True, 
                              num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    valid_loader = DataLoader(GeneExpressionDataset(valid_path, 'valid', m, s), 
                              batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
                              num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

    model = EnformerGeneExprModel().to(CONFIG['DEVICE'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    criterion = nn.MSELoss()

    best_s = -1.0
    patience_counter = 0
    
    for epoch in range(CONFIG['MAX_EPOCHS']):
        print(f"\n=== Epoch {epoch+1} ===")
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['DEVICE'])
        sr = validate(model, valid_loader, CONFIG['DEVICE'])
        
        print(f"Epoch {epoch+1} Result: Train Loss={t_loss:.4f}, Spearman={sr:.4f}")
        
        if sr > best_s:
            best_s = sr
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], "best_model.pth"))
            print("  >>> SAVE: New Best Model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print("Early stopping.")
                break