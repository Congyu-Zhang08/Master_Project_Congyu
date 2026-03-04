import os
import sys

# ================= CRITICAL CONFIGURATION =================
# These must be set BEFORE importing numpy/pandas
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ==========================================================

import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from Bio import motifs, SeqIO
import MOODS.tools
import MOODS.scan
from adjustText import adjust_text 

# ================= GLOBAL SETTINGS =================

FILE_TEMPLATE = "{cell}_{type}.fasta"
JASPAR_FILE = "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
CELL_LINES = ["SK-N-SH", "HepG2", "K562"]

P_VALUE = 0.0001
PSEUDOCOUNT = 0.0001
BG_LIST = [0.25, 0.25, 0.25, 0.25]

# Marker configuration for the comparative plot
# We use a list to easily loop through all groups for every subplot
MARKER_GROUPS = [
    {
        "name": "SK-N-SH Markers",
        "targets": ["PHOX2B"], # Ensure names match JASPAR exactly (e.g. GATA3, not Gata3)
        "color": "#D7191C" # Red
    },
    {
        "name": "HepG2 Markers",
        "targets": ["HNF4A","FOXC2"],
        "color": "#FDAE61" # Yellow/Orange
    },
    {
        "name": "K562 Markers",
        "targets": ["GATA1::TAL1","TRPS1"],
        "color": "#1A9641" # Green
    }
]

# ================= DATA PROCESSING =================

def load_motifs_in_memory(jaspar_file):
    print(f"[Init] Loading Motifs from {jaspar_file}...")
    with open(jaspar_file) as f:
        bio_motifs = list(motifs.parse(f, "jaspar"))
    
    moods_matrices = []
    moods_thresholds = []
    moods_names = []
    
    bg_arr = np.array(BG_LIST).reshape(4, 1)

    print(f"[Init] Calculating PWMs...")
    for m in bio_motifs:
        try:
            counts = np.array([
                m.counts['A'], m.counts['C'], m.counts['G'], m.counts['T']
            ], dtype=np.float64)
            
            numerator = counts + (PSEUDOCOUNT * bg_arr)
            col_sums = np.sum(numerator, axis=0)
            probs = numerator / col_sums
            pwm = np.log2(probs / bg_arr)
            
            matrix = pwm.tolist()
            threshold = MOODS.tools.threshold_from_p(matrix, BG_LIST, P_VALUE)
            
            moods_matrices.append(matrix)
            moods_thresholds.append(threshold)
            moods_names.append(m.name)
        except Exception:
            continue

    print(f"[Init] Loaded {len(moods_matrices)} matrices.")
    return moods_matrices, moods_thresholds, moods_names

def scan_chunk(seq_chunk, matrices, thresholds):
    local_counts = np.zeros(len(matrices))
    for seq in seq_chunk:
        results = MOODS.scan.scan_dna(seq, matrices, BG_LIST, thresholds, 7)
        for i, matrix_hits in enumerate(results):
            if len(matrix_hits) > 0:
                local_counts[i] += 1
    return local_counts

def get_motif_proportions_parallel(fasta_path, matrices, thresholds):
    if not os.path.exists(fasta_path):
        print(f"[Warn] File missing: {fasta_path}")
        return np.zeros(len(matrices))

    seqs = [str(r.seq).upper() for r in SeqIO.parse(fasta_path, "fasta") if len(r.seq) >= 20]
    n_seqs = len(seqs)
    
    if n_seqs == 0: return np.zeros(len(matrices))

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        num_cores = max(1, os.cpu_count() - 2)

    print(f"[Scan] {fasta_path}: {n_seqs} sequences on {num_cores} cores.")
    
    chunk_size = int(np.ceil(n_seqs / num_cores))
    chunks = [seqs[i:i + chunk_size] for i in range(0, n_seqs, chunk_size)]
    
    total_counts = np.zeros(len(matrices))
    func = partial(scan_chunk, matrices=matrices, thresholds=thresholds)
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(func, chunks)
        for res in results:
            total_counts += res

    return total_counts / n_seqs

def compute_js_distance(p, q):
    if np.sum(p) == 0 or np.sum(q) == 0: return np.nan
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    return jensenshannon(p_norm, q_norm)

# ================= PLOTTING FUNCTIONS =================

def plot_heatmap_fig2b(data):
    print("Generating Figure 2b (Heatmap)...")
    hm_data = []
    y_labels = []
    # Simplified X labels
    x_labels = [f"{c}\n(Test)" for c in CELL_LINES]
    
    for r in CELL_LINES:
        # Row 1: Generated vs Test
        val = [compute_js_distance(data[r]['gen'], data[c]['test']) for c in CELL_LINES]
        hm_data.append(val)
        y_labels.append(f"{r}\n(DNA-Diffusion)")
        
        # Row 2: Train vs Test (Baseline)
        val = [compute_js_distance(data[r]['train'], data[c]['test']) for c in CELL_LINES]
        hm_data.append(val)
        y_labels.append(f"{r}\n(Train)")
        
    df = pd.DataFrame(hm_data, index=y_labels, columns=x_labels)
    
    plt.figure(figsize=(8, 10))
    ax = sns.heatmap(df, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'JS Distance'})
    
    # [FIX] Rotate Y-axis labels to be horizontal
    plt.yticks(rotation=0) 
    
    plt.tight_layout()
    plt.savefig("fig2b_heatmap_final.pdf", transparent=True)
    plt.close()
    print("Saved fig2b_heatmap_final.pdf")

def plot_scatter_fig2c(data, motif_names):
    print("Generating Figure 2c (Comparative Scatter)...")
    
    # Create subplots without shared axes to handle different scales
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Calculate global max for axis limits (optional, for uniform look)
    # Or calculate per-plot limits. Here we use a uniform limit based on max data.
    max_val = 0.0
    for cell in CELL_LINES:
        if cell in data:
            local_max = max(np.max(data[cell]['gen']), np.max(data[cell]['train']))
            if local_max > max_val:
                max_val = local_max
    limit = max_val * 1.1 # 10% padding

    for i, cell in enumerate(CELL_LINES):
        ax = axes[i]
        if cell not in data: continue

        x_vals = data[cell]['gen']
        y_vals = data[cell]['train']
        
        # 1. Background (Gray Dots)
        ax.scatter(x_vals, y_vals, c='#E0E0E0', alpha=0.5, s=25, zorder=1, edgecolors='none')
        
        # 2. Diagonal Line
        ax.plot([0, limit], [0, limit], color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
        
        texts = []
        
        # 3. [FIX] Comparative Logic: Loop through ALL marker groups for EACH plot
        for group in MARKER_GROUPS:
            for target in group['targets']:
                if target in motif_names:
                    idx = motif_names.index(target)
                    xi, yi = x_vals[idx], y_vals[idx]
                    
                    # Plot the colored dot
                    ax.scatter(xi, yi, c=group['color'], s=150, edgecolors='white', 
                               linewidth=1.5, zorder=10)
                    
                    # Add label text object (filter low values to avoid clutter at 0,0)
                    if xi > 0.01 or yi > 0.01:
                        t = ax.text(xi, yi, target, fontsize=12, fontweight='bold', color='black')
                        texts.append(t)
                else:
                    # Optional: Print warning only once per run usually, 
                    # but here it might print multiple times. Useful for debugging.
                    pass 

        # 4. [FIX] Auto-adjust text positions to prevent overlap
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        # Axis Styling
        ax.set_title(cell, fontsize=16, fontweight='bold')
        ax.set_xlabel("Predicted (Gen)", fontsize=14)
        
        # Use the dynamic limit calculated earlier
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        
        if i == 0:
            ax.set_ylabel("Experimental (Train)", fontsize=14)
        
        ax.tick_params(axis='both', which='major', length=6, width=1.5)

    plt.tight_layout()
    plt.savefig("fig2c_scatter_final.pdf", transparent=True)
    plt.close()
    print("Saved fig2c_scatter_final.pdf")

# ================= MAIN EXECUTION =================

def main():
    if not os.path.exists(JASPAR_FILE):
        print(f"Error: JASPAR file {JASPAR_FILE} not found.")
        return

    matrices, thresholds, motif_names = load_motifs_in_memory(JASPAR_FILE)
    
    data = {cell: {} for cell in CELL_LINES}
    
    # 1. Scan Motifs
    for cell in CELL_LINES:
        for dtype in ['train', 'test', 'gen']:
            fname = FILE_TEMPLATE.format(cell=cell, type=dtype)
            data[cell][dtype] = get_motif_proportions_parallel(fname, matrices, thresholds)

    # 2. Plot Figure 2b (Heatmap)
    plot_heatmap_fig2b(data)
    
    # 3. Plot Figure 2c (Scatter)
    plot_scatter_fig2c(data, motif_names)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()