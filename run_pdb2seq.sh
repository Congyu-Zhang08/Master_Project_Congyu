#!/bin/bash
#SBATCH -n 16
#SBATCH -t 8:00:00
#SBATCH --partition=rome

module load 2023
module load SciPy-bundle/2023.07-gfbf-2023a

cp -r $HOME/supplemental_files/design_models_pdb/PDGFR "$TMPDIR"

mkdir "$TMPDIR"/output_dir

python $HOME/pdb2seq.py "$TMPDIR"/PDGFR "$TMPDIR"/output_dir

cp -r "$TMPDIR"/output_dir $HOME
