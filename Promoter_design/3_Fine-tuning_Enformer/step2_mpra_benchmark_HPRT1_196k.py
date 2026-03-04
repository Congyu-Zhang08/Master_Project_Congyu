import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os

# ================= CONFIGURATION =================
ANALYSIS_CONFIG = {
    "MPRA_FILE_PATH": "mpra_data.txt", 
    "PREDS_PATH": "raw_predictions_hprt1.npy", 
    "BACKBONE_BASELINE": [3.4630, 3.2010, 2.6145] 
}

# ================= PLOTTING STYLE =================
plt.style.use('default') 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] 
plt.rcParams['pdf.fonttype'] = 42
# Force grid to be off globally
plt.rcParams['axes.grid'] = False

def run_analysis():
    # 1. Load Metadata
    print("Loading MPRA metadata...")
    try:
        df = pd.read_csv(ANALYSIS_CONFIG['MPRA_FILE_PATH'], sep='\t')
    except:
        df = pd.read_csv(ANALYSIS_CONFIG['MPRA_FILE_PATH'])

    df = df.dropna(subset=['sequence']).reset_index(drop=True)
    
    cell_map_norm = {'k562': 'K562', 'hepg2': 'HepG2', 'sknsh': 'SKNSH'}
    if 'target_cell' in df.columns:
        df['target_cell_clean'] = df['target_cell'].astype(str).str.lower().map(cell_map_norm)
    
    # 2. Load Predictions
    print(f"Loading predictions from {ANALYSIS_CONFIG['PREDS_PATH']}...")
    if not os.path.exists(ANALYSIS_CONFIG['PREDS_PATH']):
        raise FileNotFoundError(f"Prediction file not found: {ANALYSIS_CONFIG['PREDS_PATH']}")
        
    preds = np.load(ANALYSIS_CONFIG['PREDS_PATH']) 
    
    if len(df) != len(preds):
        raise ValueError(f"Shape Mismatch: Metadata has {len(df)} rows, Preds has {len(preds)} rows.")
        
    # 3. Calculate Delta
    baseline = np.array(ANALYSIS_CONFIG['BACKBONE_BASELINE'])
    preds_delta = preds - baseline
    
    cells = ['K562', 'HepG2', 'SKNSH']
    for i, cell in enumerate(cells):
        df[f'Pred_{cell}'] = preds[:, i]
        df[f'Delta_{cell}'] = preds_delta[:, i]

    # 4. Correlation Plots
    print("\n=== Generating Correlation Plots ===")
    
    comparisons = [
        ('K562_l2fc', f'Delta_K562', 'K562'),
        ('HepG2_l2fc', f'Delta_HepG2', 'HepG2'),
        ('SKNSH_l2fc', f'Delta_SKNSH', 'SKNSH')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    for i, (exp_col, pred_col, label) in enumerate(comparisons):
        ax = axes[i]
        
        if exp_col not in df.columns:
            if label in df.columns:
                exp_col = label
            else:
                ax.text(0.5, 0.5, f"Missing Col:\n{exp_col}", ha='center')
                continue
            
        valid = df[[exp_col, pred_col]].dropna()
        if len(valid) < 2:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center'); continue

        sr, _ = spearmanr(valid[exp_col], valid[pred_col])
        pr, _ = pearsonr(valid[exp_col], valid[pred_col])
        print(f"{label}: Spearman = {sr:.4f}, Pearson = {pr:.4f}")
        
        ax.scatter(valid[pred_col], valid[exp_col], alpha=0.5, s=20, 
                   color='#2b7bba', edgecolor='none', rasterized=True)
        sns.regplot(data=valid, x=pred_col, y=exp_col, scatter=False, ax=ax, 
                    color='red', line_kws={'alpha':0.5})
        
        ax.set_title(f"{label}\nSpearman: {sr:.3f}")
        
        # LABELS UPDATED
        ax.set_xlabel("Predicted Activity (Log2FC)")
        ax.set_ylabel("Experimental Activity (Log2FC)")
        # GRID REMOVED
    
    plt.savefig('Step2_Correlation_196k.png', dpi=300)
    
    # 5. Specificity Analysis (MinGap)
    if 'MinGap' in df.columns and 'target_cell_clean' in df.columns:
        print("\n=== Generating Specificity Plots ===")
        pred_mingaps = []
        
        for _, row in df.iterrows():
            target = row['target_cell_clean']
            if pd.isna(target) or target not in cells:
                pred_mingaps.append(np.nan)
                continue
            
            target_val = row[f'Delta_{target}']
            off_targets = [c for c in cells if c != target]
            off_vals = [row[f'Delta_{c}'] for c in off_targets]
            
            if off_vals:
                pred_mingaps.append(target_val - max(off_vals))
            else:
                pred_mingaps.append(np.nan)
        
        df['Pred_MinGap'] = pred_mingaps
        
        valid_gap = df[['MinGap', 'Pred_MinGap', 'target_cell_clean']].dropna()
        if len(valid_gap) > 1:
            sr_g, _ = spearmanr(valid_gap['MinGap'], valid_gap['Pred_MinGap'])
            print(f"Overall MinGap Spearman: {sr_g:.4f}")
            
            plt.figure(figsize=(6, 5))
            sns.scatterplot(data=valid_gap, x='Pred_MinGap', y='MinGap', 
                            hue='target_cell_clean', palette='Set1', alpha=0.8)
            plt.title(f"Specificity (MinGap)\nSpearman: {sr_g:.4f}")
            
            # LABELS UPDATED
            plt.xlabel("Predicted Activity (Log2FC)")
            plt.ylabel("Experimental Activity (Log2FC)")
            plt.legend(title='Target Cell')
            # GRID REMOVED
            
            plt.savefig('Step2_Specificity_196k.png', dpi=150)

    out_csv = "final_results_196k_analyzed.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAnalysis complete. Results saved to {out_csv}")

if __name__ == "__main__":
    run_analysis()