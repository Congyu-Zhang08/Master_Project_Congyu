#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --partition=gpu_h100
#SBATCH --gpus 1
#SBATCH --time 06:30:00
#SBATCH --output=RFdiffusion_%A.log
module load 2024 SciPy-bundle/2024.05-gfbf-2024a 
module load matplotlib/3.9.2-gfbf-2024a
module load 2023 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install dgl
# Set environment variables for your files and directories
OUTPUT_DIR="./outputs_D23Target_0.5noise/beta_sheet_targeting"
TARGET_PDB="../helper_scripts/2E9W_D2D3.pdb"
TARGET_SS="../helper_scripts/2E9W_D2D3_ss.pt" # You can generate this using the make_secstruc_adj.py script in the helper_scripts directory
TARGET_ADJ="../helper_scripts/2E9W_D2D3_adj.pt" # You can generate this using the make_secstruc_adj.py script in the helper_scripts directory

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the beta-sheet targeting design
../scripts/run_inference.py \
  inference.output_prefix="$OUTPUT_DIR/beta_strand_test" \
  inference.input_pdb="$TARGET_PDB" \
  inference.num_designs=300 \
  ppi.hotspot_res="[A200,A201,A258,A259,A260,A261]" \
  scaffoldguided.scaffoldguided=True \
  scaffoldguided.target_pdb=True \
  scaffoldguided.target_path="$TARGET_PDB" \
  scaffoldguided.set_binder_beta_sheet_length=6 \
  scaffoldguided.set_target_beta_sheet="[A258-263/0]" \
  scaffoldguided.flexible_beta_sheet=False \
  scaffoldguided.target_ss="$TARGET_SS" \
  scaffoldguided.target_adj="$TARGET_ADJ" \
  +scaffoldguided.binder_length="60-90" \
  denoiser.noise_scale_ca=0.5 \
  denoiser.noise_scale_frame=0.5 \
  logging.save_ss_adj=True \
