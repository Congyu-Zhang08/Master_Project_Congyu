#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=gpu_h100
#SBATCH --gpus 1
#SBATCH --time 02:30:00

module load 2024 SciPy-bundle/2024.05-gfbf-2024a 
module load matplotlib/3.9.2-gfbf-2024a
module load 2023 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

rm check.point 2>/dev/null
../mpnn_fr/dl_interface_design.py -pdbdir inputs/7aah_50 -relax_cycles 0 -seqs_per_struct 2 -outpdbdir 7aah_50_out

rm check.point 2>/dev/null
../mpnn_fr/dl_interface_design.py -pdbdir inputs/7aah_100 -relax_cycles 0 -seqs_per_struct 2 -outpdbdir 7aah_100_out
