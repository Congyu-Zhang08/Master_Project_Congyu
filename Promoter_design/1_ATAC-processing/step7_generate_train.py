import pandas as pd

df_seq = pd.read_csv("sequences_raw.txt", sep="\t", header=None, names=["region", "sequence"])
df_bed = pd.read_csv("all_cells_training.bed", sep="\t", header=None, names=["chr", "start", "end", "TAG"])

if len(df_seq) == len(df_bed):
    final_df = pd.DataFrame()
    final_df['chr'] = df_bed['chr']
    final_df['sequence'] = df_seq['sequence'].str.upper()
    final_df['TAG'] = df_bed['TAG']

    final_df.to_csv("training_data_dna_diffusion.txt", sep="\t", index=False)
    print("Done.")
else:
    print("Error: Row count mismatch.")
