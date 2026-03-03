#!/bin/bash
#SBATCH -n 16
#SBATCH -t 8:00:00
#SBATCH --partition=rome

module load 2023
module load SciPy-bundle/2023.07-gfbf-2023a


python ./json_modifier.py
