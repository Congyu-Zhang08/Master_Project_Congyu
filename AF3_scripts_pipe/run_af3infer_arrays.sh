#!/bin/bash
#
# Slurm submission script for AlphaFold 3 inference only.
# This script is configured as a Job Array to process all pre-processed
# JSON files within a specific input folder (e.g., input_infer_001) using a GPU.
# Run this AFTER the data pipeline has successfully finished.
#

# --- SLURM JOB ARRAY CONFIGURATION ---
# Assuming 1000 tasks, one for each input_infer_XXX folder.
#SBATCH --array=1-1000             # Run a job for each output folder from the data pipeline (Task ID 1 to 1000)
#SBATCH --job-name=AF3_InferArray   # Name of the job
#SBATCH --nodes=1                   # Use a single node
#SBATCH --time=03:00:00             # Maximum execution time: 10 hours (Adjusted time)
#SBATCH --partition gpu_a100        # Use H100 partition for AF3 inference
#SBATCH --gpus=1                    # Use a single GPU per task
#SBATCH --output=AF3_infer_%a.out   # Standard output log (e.g., AF3_infer_1.out)
#SBATCH --error=AF3_infer_%a.err    # Standard error log (e.g., AF3_infer_1.err)

# load environment modules
module load 2024
module load AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0 

# Path of the container.
AF3_CONTAINER_PATH="/sw/arch/RHEL9/EB_production/2024/software/AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0/bin/alphafold-3.0.1.sif"

PROJECT_SPACE="./"

# Base directory where the pre-processed data (from the data pipeline) is located
BASE_DATA_DIR=${PROJECT_SPACE}/alphafold3/inputs_infer/

# Path where the final output files will be written
OUTPUT_PATH=${PROJECT_SPACE}/alphafold3/outputs/

# Path to the model weights. Change this to the location of your weights
MODEL_PATH=~/AF3Weights

# --- ARRAY-SPECIFIC FOLDER SELECTION ---

# Construct the input data folder name using the SLURM Task ID with 3-digit padding.
# Example: If SLURM_ARRAY_TASK_ID=5, INPUT_DATA_FOLDER becomes 'input_infer_005'
INPUT_FOLDER_SUFFIX=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
INPUT_DATA_FOLDER="${BASE_DATA_DIR}/input_infer_${INPUT_FOLDER_SUFFIX}"

# Create the final output directory for this task ID, if it doesn't exist
mkdir -p "$OUTPUT_PATH"

echo "--- Starting AF3 Inference for Task ID: $SLURM_ARRAY_TASK_ID ---"
echo "Reading pre-processed data from folder: $INPUT_DATA_FOLDER"

# Check if the data folder exists
if [ ! -d "$INPUT_DATA_FOLDER" ]; then
    echo "ERROR: Data directory not found: $INPUT_DATA_FOLDER. Exiting task."
    exit 1
fi

# --- LOOP THROUGH ALL PRE-PROCESSED JSON FILES IN THE SELECTED FOLDER ---

# We now use 'find' to recursively locate all *_data.json files within the $INPUT_DATA_FOLDER
find "$INPUT_DATA_FOLDER" -type f -name '*_data.json' | while read REAL_INPUT_JSON_PATH; do
    
    echo "Running Inference on: $REAL_INPUT_JSON_PATH"
    
    # --- AF3 COMMAND LINE ARGUMENTS (Inference Only) ---
    cmd_args="--json_path ${REAL_INPUT_JSON_PATH} \
--output_dir ${OUTPUT_PATH} \
--run_data_pipeline=False \
--model_dir ${MODEL_PATH}"

    # Unset to avoid warnings.
    unset LD_PRELOAD
    
    # --- RUN ALPHAFOLD 3 INFERENCE ---
    
    # --nv flag is CRITICAL for passing the Nvidia GPU to the container
    apptainer run --nv \
        -B "$PWD:/workspace" \
        --pwd /workspace \
        ${AF3_CONTAINER_PATH}  ${cmd_args}
        
    # Check the exit status of the AlphaFold run
    if [ $? -eq 0 ]; then
        echo "Successfully finished inference for $(basename $REAL_INPUT_JSON_PATH)."
    else
        echo "Inference FAILED for $(basename $REAL_INPUT_JSON_PATH). Check error logs."
    fi

    # --- CLEANUP OUTPUT FILES (EXECUTED IMMEDIATELY AFTER INFERENCE) ---
    # This section reduces file count and storage usage immediately after each run.

    # Extract the base name (e.g., 'rif_proteing_0861_000000005_0001') from the full path
    NAME=$(basename "$REAL_INPUT_JSON_PATH" _data.json)
    
    # Search for the output directory created by AF3 (e.g., alphafold3/outputs/rif_proteing_0861_000000005_0001/)
    SEARCH_PATTERN="${OUTPUT_PATH}/${NAME}*"
    TARGET_DIR=$(ls -d $SEARCH_PATTERN 2>/dev/null | head -n 1)
    
    if [ -d "$TARGET_DIR" ]; then
        echo "Cleaning up output directory: $TARGET_DIR"
        # Remove unnecessary files (TERMS_OF_USE.md, ranking_scores.csv)
        find "$TARGET_DIR" -type f \( -name "TERMS_OF_USE.md" -o -name "ranking_scores.csv" \) -delete
        # Remove 'seed*' subdirectories and their contents
        find "$TARGET_DIR" -type d -name "seed*" -prune -exec rm -rf {} +
    fi
    # --- END CLEANUP ---

done

echo "--- Finished all files in Task ID $SLURM_ARRAY_TASK_ID ---"

