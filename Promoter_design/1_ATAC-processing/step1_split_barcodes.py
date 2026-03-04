import h5py
import random

h5_file = "GSE262189_SK.N.SH_filtered_feature_bc_matrix.h5"

with h5py.File(h5_file, 'r') as f:
    if 'matrix' in f:
        barcodes = f['matrix']['barcodes'][:]
    else:
        barcodes = f['barcodes'][:]

all_barcodes = [x.decode('utf-8') for x in barcodes]
random.shuffle(all_barcodes)

mid_point = len(all_barcodes) // 2
rep1 = all_barcodes[:mid_point]
rep2 = all_barcodes[mid_point:]

with open("sknsh_barcodes_rep1.txt", "w") as f:
    f.write("\n".join(rep1) + "\n")
with open("sknsh_barcodes_rep2.txt", "w") as f:
    f.write("\n".join(rep2) + "\n")