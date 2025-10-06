#!/bin/bash
#
# Slurm submission script for AlphaFold 3 data pipeline only.
# Uses a job array to process files distributed across 1000 input folders,
# directing the output for all files in a folder (e.g., input_001) to a single
# corresponding output folder (e.g., input_afterdata/input_infer_001).
#

# --- SLURM JOB ARRAY CONFIGURATION ---
# IMPORTANT: This must match the number of input folders you created (1000)
#SBATCH --array=1-1000              # Run a job for each folder (Task ID from 1 to 1000)
#SBATCH --job-name=AF3_DataArray    # Name of the job
#SBATCH --nodes=1                   # Use a single node
#SBATCH --time=08:00:00             # Maximum execution time: 8 hours (CPU intensive)
#SBATCH --partition=genoa           # Use Genoa CPU partition
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=24          # Allocate 24 CPU cores per task for Jackhmmer
#SBATCH --output=input_infer_%a.out # LOG CHANGE: Output log named after Task ID, e.g., input_infer_1.out
#SBATCH --error=input_infer_%a.err  # LOG CHANGE: Error log named after Task ID, e.g., input_infer_1.err


# load environment modules
module load 2024
module load AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0 

# --- PATHS & CONFIG ---

# Path of the container.
AF3_CONTAINER_PATH="/sw/arch/RHEL9/EB_production/2024/software/AlphaFold/3.0.1-foss-2024a-CUDA-12.6.0/bin/alphafold-3.0.1.sif"

# Path of the data (large .fasta files for jackhmmer, etc.).
DATA_PATH=/projects/2/managed_datasets/AlphaFold/3.0.0

# Path to the mmcif symlink (used for MSA deduplication and template matching).
MMCIF_PATH=/scratch-nvme/ml-datasets/AlphaFold/3.0.0/mmcif_files/

PROJECT_SPACE="./"

# Base directory where the output (processed data) folders will be placed
BASE_OUTPUT_DIR=${PROJECT_SPACE}/alphafold3/inputs_infer/

# --- ARRAY-SPECIFIC FOLDER SELECTION ---

# Construct the input folder name using the SLURM Task ID with 3-digit padding.
# Example: If SLURM_ARRAY_TASK_ID=5, INPUT_FOLDER becomes 'input_005'
INPUT_FOLDER_SUFFIX=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
INPUT_FOLDER="${PROJECT_SPACE}/PDGFR_input/input_${INPUT_FOLDER_SUFFIX}"

# --- NEW OUTPUT DIRECTORY SETUP ---
# Output folder structure: BASE_OUTPUT_DIR/input_infer_001, input_infer_002, etc.
# All processed data for the current array task will go here.
OUTPUT_DIR_NAME="input_infer_${INPUT_FOLDER_SUFFIX}"
OUTPUT_PATH="${BASE_OUTPUT_DIR}/${OUTPUT_DIR_NAME}/"

# Create the single output directory for this task ID
mkdir -p "$OUTPUT_PATH"
echo "--- Output for this task will be aggregated in: $OUTPUT_PATH ---"


echo "--- Starting AF3 Data Pipeline for Task ID: $SLURM_ARRAY_TASK_ID ---"
echo "Processing files from folder: $INPUT_FOLDER"

# Check if the folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "ERROR: Input directory not found: $INPUT_FOLDER"
    exit 1
fi

# --- LOOP THROUGH ALL INPUT JSON FILES IN THE SELECTED FOLDER ---

# Find all JSON input files in the designated array folder
for INPUT_FILE in ${INPUT_FOLDER}/*.json; do
    
    # Check if a file was actually found (in case glob matches nothing)
    if [ ! -f "$INPUT_FILE" ]; then
        echo "No .json files found in $INPUT_FOLDER. Skipping."
        continue
    fi

    # 1. Set the specific input path for the current file
    INPUT_JSON_PATH=$INPUT_FILE
    
    # We no longer need to derive a FILE_NAME_BASE or create file-specific subfolders.
    # AlphaFold 3 handles the output file naming within the provided OUTPUT_PATH.
    
    echo "Processing Input: $INPUT_JSON_PATH"
    
    # --- AF3 COMMAND LINE ARGUMENTS (Data Pipeline Only) ---
    # NOTE: Arguments are flush left to prevent leading spaces being included in the variable value
    cmd_args="--json_path ${INPUT_JSON_PATH}
--output_dir ${OUTPUT_PATH}
--db_dir ${DATA_PATH}
--pdb_database_path ${MMCIF_PATH}
--run_inference=False" # CRITICAL: Set to FALSE to only run data processing

    # Unset to avoid warnings (standard practice for Apptainer/Singularity)
    unset LD_PRELOAD
    
    # --- RUN ALPHAFOLD 3 DATA PIPELINE ---
    
    # Note: No --nv flag needed since this is running on CPU partition (Genoa)
    apptainer run -B "$PWD:/workspace" \
        -B ${DATA_PATH} \
        --pwd /workspace \
        ${AF3_CONTAINER_PATH} ${cmd_args}
        
    # Check the exit status of the AlphaFold run
    if [ $? -eq 0 ]; then
        echo "Successfully finished data pipeline for ${INPUT_FILE}."
    else
        echo "Data pipeline FAILED for ${INPUT_FILE}. Check error logs."
    fi

done

echo "--- Finished all files in folder $INPUT_FOLDER ---"

