# The Binder Project

## Benchmarking AlphaFold 3 on Cao et al. binder dataset

1. **Download the dataset**
2. **Get sequences** from PDB files and save to FASTA format with `pdb2fasta.py`
3. **Downsampling** run [mmseqs2](https://github.com/soedinglab/MMseqs2) with `mmseqs easy-cluster example/dataset.fasta clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1`, `result_name_rep_seq.fasta` is the downsampled dataset
4. **Generate AlphaFold 3 input JSON files** with `fasta2json.py` (from [Alphafold3_tools](https://github.com/snufoodbiochem/Alphafold3_tools))
5. **Run AlphaFold** for the target protein to get MSA and put MSA in JSON files with `modifyjson.py`
6. **Submit AlphaFold 3 job arrays** on Snellius
7. **Calculate metrics** with... save the metrics to `.csv` files
8. **Merge the AF3 metrics CSV** with CSV containing AF2 metrics of Cao et al. dataset:
    > Bennett, N.R., Coventry, B., Goreshnik, I. et al. Improving de novo protein binder design with deep learning. *Nat Commun* **14**, 2625 (2023). https://doi.org/10.1038/s41467-023-38328-5
9. **ROC plots** for each metric
10. **Classifier training and testing**
11. **Evaluate classifier performance**

---

## Designing protein binder for the target CD117 (KIT)
