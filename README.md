# The Binder Project

## Benchmarking AlphaFold 3 on Cao et al. binder dataset

1. **Download the dataset**
2. **Get sequences** from PDB files and save to FASTA format with `pdb2fasta.py`
3. **Downsampling** run [mmseqs2](https://github.com/soedinglab/MMseqs2) with `mmseqs easy-cluster example/dataset.fasta clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1`, `result_name_rep_seq.fasta` is the downsampled dataset
4. **Generate AlphaFold 3 input JSON files** with `fasta2json.py` (from [Alphafold3_tools](https://github.com/snufoodbiochem/Alphafold3_tools))
5. **Run AlphaFold** for the target protein to get MSA and put MSA in JSON files with `modifyjson.py`
6. **Submit AlphaFold 3 job arrays** on Snellius
7. **Calculate metrics** with `RMSD_confidence.py`,`score_interface.py` and `calculate_clashes.py`
8. **Merge the AF3 metrics CSV** with CSV containing AF2 metrics of Cao et al. dataset:
    > Bennett, N.R., Coventry, B., Goreshnik, I. et al. Improving de novo protein binder design with deep learning. *Nat Commun* **14**, 2625 (2023). https://doi.org/10.1038/s41467-023-38328-5
9. **Classifier training and testing** `classification.py` trains and validates the models on PDGFR set, test the models on FGFR2 set.
10. **Binder design** In `5_Binder_design` are the scripts of running RFdiffusion-ProteinMPNN pipeline for KIT binder design.
11. **Binder evaluation** `dssp.py` evaluates the secondary sturcutures of the binder designs; `run_plip.py` calculates the protein-protein interaction between the binder and the target for AF3 predicted structures.

---

## Designing protein binder for the target CD117 (KIT)
