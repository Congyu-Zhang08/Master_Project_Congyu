import os
import csv
import warnings
from Bio.PDB import PDBParser, DSSP

# Ignore Biopython PDB format warnings
warnings.filterwarnings("ignore")

def get_ss_counts(pdb_file_path):
    """
    Calculate Alpha Helix and Beta Strand counts for Chain A only.
    Returns: (helix_count, strand_count)
    """
    p = PDBParser()
    
    # Structure loading without error handling
    structure = p.get_structure("id", pdb_file_path)
    model = structure[0]
    
    # Run DSSP directly
    dssp = DSSP(model, pdb_file_path)

    ss_sequence = ""
    
    # Iterate through DSSP keys to filter for Chain A
    for key in dssp.keys():
        chain_id = key[0]
        
        # ONLY process Chain A
        if chain_id != 'A':
            continue
            
        # index 2 is the secondary structure code
        ss_type = dssp[key][2]
        
        # H: Alpha helix, E: Beta strand
        if ss_type not in ['H', 'E']:
            ss_sequence += "-"
        else:
            ss_sequence += ss_type

    # Calculate number of independent fragments for Chain A
    helix_count = 0
    strand_count = 0
    current_type = None
    
    for char in ss_sequence:
        if char == 'H':
            if current_type != 'H':
                helix_count += 1
            current_type = 'H'
        elif char == 'E':
            if current_type != 'E':
                strand_count += 1
            current_type = 'E'
        else:
            current_type = None
            
    return helix_count, strand_count

def batch_process_pdb_folder(input_folder, output_csv):
    """
    Process all PDBs in folder and write secondary structure counts to CSV.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return

    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
    total_files = len(pdb_files)
    
    print(f"Found {total_files} PDB files, starting process (Chain A only)...")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        header = ["Name", "Alpha_Helix_Count_ChainA", "Beta_Strand_Count_ChainA", "Status"]
        writer.writerow(header)

        success_count = 0
        
        for index, filename in enumerate(pdb_files):
            file_path = os.path.join(input_folder, filename)
            
            print(f"[{index + 1}/{total_files}] Analyzing: {filename} ...", end="\r")
            
            # Direct execution without try-except
            h_count, s_count = get_ss_counts(file_path)
            
            writer.writerow([filename, h_count, s_count, "Success"])
            success_count += 1

    print(f"\n\nProcessing complete!")
    print(f"Processed: {success_count}")
    print(f"Results saved to: {output_csv}")

# ==========================================
# Configuration
# ==========================================

INPUT_FOLDER = "./filtered_pdbs"
OUTPUT_CSV = "KITfiltered_secondary_structure_results.csv"

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Tip: Please put your .pdb files in the {INPUT_FOLDER} folder.")
    
    batch_process_pdb_folder(INPUT_FOLDER, OUTPUT_CSV)