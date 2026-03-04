import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys

try:
    from enformer_pytorch import Enformer
    from pybedtools import BedTool
except ImportError:
    raise ImportError("Please install enformer-pytorch and pybedtools")

# ================= CONFIGURATION =================
CONFIG = {
    "GTF_FILE": "./gencode.v49.annotation.gtf",
    "GENOME_FASTA": "./hg38.fa",
    "MODEL_PATH": "./checkpoints_final/best_model.pth", 
    "TRAIN_PKL": "./processed_data/dataset_train.pkl",
    "OUTPUT_FILE": "final_HPRT1_backbone_1k.txt",
    
    "SEQ_LEN_INPUT": 1024,
    "SEQ_LEN_ENFORMER": 196608,
    "TSS_INDEX": 824,  
    "EMBED_DIM": 3072,
    "DROPOUT": 0.0,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "TARGET_GENE": "HPRT1"
}

# MinP  (64bp)
FULL_SEQ = "tagagggtatataatggaagctcgacttccagcttggcaatccggtactgttggtaaagccacc".upper()
MINP_TSS_OFFSET = 32 

# ================= MODEL  =================
class CellSpecificAttentionHead(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.2):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )
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
            CellSpecificAttentionHead(CONFIG['EMBED_DIM'], CONFIG['DROPOUT']) for _ in range(3)
        ])

    def forward(self, x):
        full = self.enformer(x, return_only_embeddings=True)
        target = full[:, 442:454, :]
        return torch.cat([head(target) for head in self.heads], dim=1)

# ================= UTILS =================
def one_hot_encode_and_pad(sequence):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4} 
    arr = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, char in enumerate(sequence):
        idx = mapping.get(char, 4)
        if idx < 4: arr[i, idx] = 1.0
            
    encoded = torch.from_numpy(arr)
    current_len = encoded.shape[0]
    total_len = CONFIG['SEQ_LEN_ENFORMER']
    
    if current_len < total_len:
        final = torch.zeros(total_len, 4)
        start = (total_len - current_len) // 2
        final[start : start + current_len] = encoded
        return final.unsqueeze(0) 
    return encoded.unsqueeze(0)

def fetch_gene_sequence_1k(gtf_path, fasta_path, gene_name):
    print(f"Fetching {gene_name} sequence from genome...")
    chunksize = 50000
    cols = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    reader = pd.read_csv(gtf_path, sep='\t', comment='#', names=cols, chunksize=chunksize)
    
    chrom, strand, tss = None, None, None
    found = False
    
    for chunk in reader:
        genes = chunk[chunk['feature'] == 'gene']
        for _, row in genes.iterrows():
            if f'gene_name "{gene_name}"' in row['attribute']:
                chrom, strand = row['seqname'], row['strand']
                tss = row['start'] if strand == '+' else row['end']
                found = True
                break
        if found: break
    
    if not found:
        raise ValueError(f"Gene {gene_name} not found in GTF")

    print(f"  -> Found {gene_name} at {chrom}:{tss} ({strand})")

    # Total 1024 bp.
    # Upstream needed: 824 bp.
    # Downstream needed: 1024 - 824 = 200 bp.
    
    up_len = CONFIG['TSS_INDEX']      # 824
    down_len = CONFIG['SEQ_LEN_INPUT'] - CONFIG['TSS_INDEX'] # 200

    if strand == '+':
        start = tss - up_len
        end = tss + down_len
    else:
        #start = tss - down_len
        #end = tss + up_len
        start = tss - down_len
        end = tss + up_len

    a = BedTool(f"{chrom} {start} {end} . . {strand}", from_string=True)
    a = a.sequence(fi=fasta_path, s=True)
    
    with open(a.seqfn) as f:
        seq = f.read().splitlines()[1].upper()
        
    if len(seq) != CONFIG['SEQ_LEN_INPUT']:
        print(f"Warning: Fetched length {len(seq)} != Expected {CONFIG['SEQ_LEN_INPUT']}")
        
    return seq

# ================= MAIN =================
def main():
    print(f"Generating 1kb {CONFIG['TARGET_GENE']} Backbone (Standardized with MinP)...")
    
    # 1. Fetch Native Sequence
    native_seq = fetch_gene_sequence_1k(CONFIG['GTF_FILE'], CONFIG['GENOME_FASTA'], CONFIG['TARGET_GENE'])
    seq_list = list(native_seq)
    
    # 2. Implant MinP
    # 824 - 32 = 792
    insert_start = CONFIG['TSS_INDEX'] - MINP_TSS_OFFSET
    insert_end = insert_start + len(FULL_SEQ)
    
    print(f"  - Implanting MinP at index: {insert_start} to {insert_end}")
    print(f"  - Replacing Native Core Promoter with Standardized MinP")
    
    seq_list[insert_start : insert_end] = list(FULL_SEQ)
    final_backbone = "".join(seq_list)
    
    # Save
    with open(CONFIG['OUTPUT_FILE'], "w") as f:
        f.write(final_backbone)
    print(f"  -> Saved to {CONFIG['OUTPUT_FILE']}")

    # 3. Verify Activity
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print("Model checkpoint not found, skipping verification.")
        return

    print("Verifying Basal Activity...")
    
    # Load Stats
    t_mean, t_std = 0.0, 1.0
    if os.path.exists(CONFIG['TRAIN_PKL']):
        df_t = pd.read_pickle(CONFIG['TRAIN_PKL'])
        cols = ['logTPM_K562', 'logTPM_HepG2', 'logTPM_SKNSH']
        t_mean = torch.tensor(df_t[cols].values.mean(axis=0)).to(CONFIG['DEVICE'])
        t_std = torch.tensor(df_t[cols].values.std(axis=0)).to(CONFIG['DEVICE'])
        del df_t

    # Load Model
    model = EnformerGeneExprModel().to(CONFIG['DEVICE'])
    state_dict = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
    model.load_state_dict(state_dict)
    model.eval()

    # Predict
    input_tensor = one_hot_encode_and_pad(final_backbone).to(CONFIG['DEVICE'])
    
    with torch.no_grad():
        pred_z = model(input_tensor)
        pred_rc = model(torch.flip(input_tensor, dims=[1, 2]))
        pred_z = (pred_z + pred_rc) / 2
        
        if isinstance(t_mean, torch.Tensor):
            pred_raw = pred_z * (t_std + 1e-8) + t_mean
            vals = pred_raw.cpu().numpy()[0]
            print("-" * 60)
            print(f"BASELINE ACTIVITY ({CONFIG['TARGET_GENE']} + MinP):")
            print(f"K562 : {vals[0]:.4f}")
            print(f"HepG2: {vals[1]:.4f}")
            print(f"SKNSH: {vals[2]:.4f}")
            print("-" * 60)
        else:
            print(f"Raw Z-scores: {pred_z.cpu().numpy()[0]}")

if __name__ == "__main__":
    main()