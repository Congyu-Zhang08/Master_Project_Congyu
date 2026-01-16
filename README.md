The Binder Project:
Benchmarking AlphaFold3 on Cao.et al binder dataset
1.download the dataset
2.get sequences from pdb files and save to fasta format with pdb2fasta.py
3.generate alphafold3 input json files with fasta2json.py(from https://github.com/snufoodbiochem/Alphafold3_tools)
4.run alphafold for the target protein to get MSA and put MSA in json files with modifyjson.py
5.submit alphafold3 job arrays on snellius
6.calculate metrics with... save the metrics to csv files
7.merge the af3 metrics csv with csv containing af2 metrics of Cao.et al dataset (Bennett, N.R., Coventry, B., Goreshnik, I. et al. Improving de novo protein binder design with deep learning. Nat Commun 14, 2625 (2023). https://doi.org/10.1038/s41467-023-38328-5)
8.ROC plots for each metric
9.Classifier training and testing
10.evaluate classifier performance

Designing protein binder for the target CD117(KIT) 
