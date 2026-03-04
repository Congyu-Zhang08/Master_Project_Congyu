# The Binder Project


1. **Download the dataset** [Binder designs, Cao et al](https://pmc.ncbi.nlm.nih.gov/articles/instance/9117152/bin/41586_2022_4654_MOESM3_ESM.gz); [Binder metrics, Bennett et al](http://files.ipd.uw.edu/pub/improving_dl_binders_2023/supplemental_files/scripts_and_main_pdbs.tar.gz)
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

# The Promoter Project

## Evaluating DNA-Diffusion

1. **Download the dataset** ATAC-seq: ENCFF006OFA (K562), ENCFF906NBO (HepG2) and GSE262189 (SK-N-SH)
2. **Clone DNA-Diffusion repository** [DNA-Diffusion](https://github.com/pinellolab/DNA-Diffusion)
3. **Prepare training data** run scripts in `1_ATAC-processing` to prepare the training data for DNA-Diffusion.
4. **Train DNA-Diffusion** run the train script in DNA-Diffusion repository (default config), after training, generate 10,000 sequences for each cell type.
5. **Evaluate generated sequences** run Blast `blast_TrainGen.sh` and MOODS motif scanning `MOODSandPlots.py` on generated sequence to evaluate the similarity and motif distributions.

## Evaluating Fine-tuned Enformer

1. **Prepare the dataset** RNA-seq: ENCFF556IS6R (K562), ENCFF119KXQ, (HepG2) and ENCFF378TYT (SK-N-SH). Run `build_dataset.py` to generate training dataset \[sequence, expression value\]
2. **Fine-tune Enformer** Run `fine_tune_enformer.py`
3. **Get backbone sequence** For predicting the activity of 200bp synthetic sequences from [Gosai et al](https://pmc.ncbi.nlm.nih.gov/articles/instance/11525185/bin/41586_2024_8070_MOESM14_ESM.txt), we replace the sequences in the -250 to -50 positions of HPRT1 gene (it is a housekeeping gene, representing an open and neutral regulatory background) and subtract the logTPM of the backbone sequence from the predicted logTPM, to represent the promoter-driven activity. Run  `generate_neutral_backbone_1k.py` to get the HPRT1 backbone sequence.
4. Run `step1_mpra_benchmark_HPRT1_1k.py` and `step2_mpra_benchmark_HPRT1_1k.py` to predict the activity of the 200bp synthetic regulatory sequences and plot the correlations of the prediction and true values for each cell type.
