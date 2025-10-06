#!/bin/bash
#
# Slurm submission script for AlphaFold 3 data pipeline only.
#

#SBATCH --job-name=AF3_data         # Name of the job
#SBATCH --nodes=1                   # Use a single node
#SBATCH --time=01:30:00             # Maximum execution time: 1.5 hours
#SBATCH --partition genoa           # Use Genoa CPU partition
#SBATCH --ntasks 1                  # Run a single task
#SBATCH --cpus-per-task 24          # Allocate 24 CPU cores per task. See Note below
#SBATCH --output=AF3_data%j.out     # Standard output log
#SBATCH --error=AF3_data%j.err      # Standard error log

# load environment modules
module load 2024
module load AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0 

# Path of the container that is already hosted in the software stack of Snellius.
AF3_CONTAINER_PATH="/sw/arch/RHEL9/EB_production/2024/software/AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0/bin/alphafold-3.0.1.sif"

# Path of the data. Contains both the (large) .fasta files used for jackhmmer and a simlink to the mmcif files on NVME storage used for MSA deduplication and template matching.
DATA_PATH=/projects/2/managed_datasets/AlphaFold/3.0.0

# Path to the mmcif symlink. For now we point directly to the 'true' location at /scratch-nvme/ml-datasets/AlphaFold/3.0.0/mmcif_files/. 
# Setting it to {DATA_PATH}/mmcif_files should also work due to the symlink
MMCIF_PATH=/scratch-nvme/ml-datasets/AlphaFold/3.0.0/mmcif_files/

PROJECT_SPACE="./"

# Path of the input AF3 json file. Change this to the location of your input file.
INPUT_JSON_PATH=${PROJECT_SPACE}/alphafold3/inputs/rif_proteing_0861_000000005_0001_data.json

# Path where the output files are written to
OUTPUT_PATH=${PROJECT_SPACE}/alphafold3/input_afterdata/
# Path to the model weights. Change to the location of your weights
MODEL_PATH=~/AF3Weights


# AF3 command line arguments
cmd_args="--json_path ${INPUT_JSON_PATH}
--output_dir ${OUTPUT_PATH}
--db_dir ${DATA_PATH}
--pdb_database_path ${MMCIF_PATH}
--run_inference=False" # Do not run inference

# Unset to avoid warnings.
unset LD_PRELOAD
# Run the Alphafold 3 data pipeline.
# -B "$PWD:/workspace" mounts the current directory ($PWD) to /workspace inside the container.
# -B ${DATA_PATH} mounts the data path to the container.
# --pwd sets the working directory inside the container.
apptainer run -B "$PWD:/workspace" \
    -B ${DATA_PATH} \
    --pwd /workspace \
    ${AF3_CONTAINER_PATH}  ${cmd_args}
