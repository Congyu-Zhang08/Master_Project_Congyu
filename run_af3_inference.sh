#!/bin/bash
#
# Slurm submission script for AlphaFold 3 inference only. 
# Run this after doing the AlphaFold 3 data pipeline.
#

#SBATCH --job-name=AF3_inference        # Name of the job
#SBATCH --nodes=1                       # Use a single node
#SBATCH --time=00:10:00                 # Maximum execution time: 10 minutes
#SBATCH --partition gpu_h100            # Use H100 partition for AF3
#SBATCH --gpus 1                        # Use a single GPU
#SBATCH --output=AF3_inference%j.out   # Standard output log
#SBATCH --error=AF3_inference%j.err    # Standard error log

# load environment modules
module load 2024
module load AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0 

# Path of the container that is already hosted in the software stack of Snellius.
AF3_CONTAINER_PATH="/sw/arch/RHEL9/EB_production/2024/software/AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0/bin/alphafold-3.0.1.sif"

PROJECT_SPACE="./"

INPUT_JSON_PATH=${PROJECT_SPACE}/alphafold3/inputs/rif_proteing_0861_000000005_0001_data.json
DATA_JSON_PATH=${PROJECT_SPACE}/alphafold3/input_afterdata/
# Path where the output files are written to
OUTPUT_PATH=${PROJECT_SPACE}/alphafold3/outputs/
# Path to the model weights. Change to the location of your weights
MODEL_PATH=~/AF3Weights


# --- Determine path of pre-processed JSON file ---
# The AF3 data pipeline will use the 'name' field in the input json to create a subdirectory for the output. Retrieve this name to find the output JSON file with the MSAs.
# Change this in case you have it stored in a different location
NAME=$(jq -r '.name' "$INPUT_JSON_PATH")
REAL_INPUT_JSON_PATH=${DATA_JSON_PATH}/${NAME}/${NAME}_data.json


# arguments
cmd_args="--json_path ${REAL_INPUT_JSON_PATH}
--output_dir ${OUTPUT_PATH}
--run_data_pipeline=False
--model_dir ${MODEL_PATH}"


# Unset to avoid warnings.
unset LD_PRELOAD
# Run the Alphafold 3 data pipeline.
# --nv for passing the Nvidia GPU
# -B "$PWD:/workspace" mounts the current directory ($PWD) to /workspace inside the container.
# -B ${DATA_PATH} mounts the data path to the container.
# --pwd sets the working directory inside the container.
apptainer run --nv \
    -B "$PWD:/workspace" \
    --pwd /workspace \
    ${AF3_CONTAINER_PATH}  ${cmd_args}

SEARCH_PATTERN="${OUTPUT_PATH}/${NAME}*"
TARGET_DIR=$(ls -d $SEARCH_PATTERN | head -n 1)
find "$TARGET_DIR" -type f \( -name "TERMS_OF_USE.md" -o -name "ranking_scores.csv" \) -delete
find "$TARGET_DIR" -type d -name "seed*" -prune -exec rm -rf {} +
echo "Cleanup complete"
