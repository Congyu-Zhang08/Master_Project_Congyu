# The Binder Project

## Benchmarking AlphaFold 3 on Cao et al. binder dataset

1. **Download the dataset**
2. **Get sequences** from PDB files and save to FASTA format with `pdb2fasta.py`
3. **Generate AlphaFold 3 input JSON files** with `fasta2json.py` (from [Alphafold3_tools](https://github.com/snufoodbiochem/Alphafold3_tools))
4. **Run AlphaFold** for the target protein to get MSA and put MSA in JSON files with `modifyjson.py`
5. **Submit AlphaFold 3 job arrays** on Snellius
6. **Calculate metrics** with... save the metrics to `.csv` files
7. **Merge the AF3 metrics CSV** with CSV containing AF2 metrics of Cao et al. dataset:
    > Bennett, N.R., Coventry, B., Goreshnik, I. et al. Improving de novo protein binder design with deep learning. *Nat Commun* **14**, 2625 (2023). https://doi.org/10.1038/s41467-023-38328-5
8. **ROC plots** for each metric
9. **Classifier training and testing**
10. **Evaluate classifier performance**

---

## Designing protein binder for the target CD117 (KIT)
