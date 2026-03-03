
"""
This script is used to batch calculate the number of clashing atoms in PDB files.
The core logic is based on the calculate_clash_score function from biopython_utils.py.

It scans the input directory, processes all files matching *_model.pdb,
and saves the results (Name, Clashes) to a CSV file.

Example usage:
python calculate_clashes.py --input_dir ./pdbs_scores --output_csv clashes_summary.csv
"""

import os
import glob
import argparse
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

# --- Core Calculation Function (derived from biopython_utils.py) ---
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    """
    Calculates the number of clashing atoms in a PDB file.
    Logic strictly follows the provided biopython_utils.py file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Store detailed info: (chain, res_id, atom_name, coord)

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    if not atoms:
        return 0

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes (not only_ca), only consider clashes between different chains
        # Note: This is the original logic from the utils file, meaning it only calculates inter-chain clashes by default.
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

# --- Main Program ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch calculate clash scores for PDB files and save to CSV.")
    
    parser.add_argument(
        "-i", "--input_dir", 
        default="./pdbs_scores", 
        help="Input directory containing PDB files (default: ./pdbs_scores)"
    )
    parser.add_argument(
        "-o", "--output_csv", 
        default="clashes_summary.csv", 
        help="Output path for the CSV file (default: clashes_summary.csv)"
    )
    parser.add_argument(
        "-t", "--threshold", 
        type=float, 
        default=2.4, 
        help="Clash distance threshold (Angstrom), default 2.4"
    )
    
    args = parser.parse_args()

    # Find files
    search_pattern = os.path.join(args.input_dir, "*_model.pdb")
    pdb_files = glob.glob(search_pattern)

    if not pdb_files:
        print(f"Error: No files matching '*_model.pdb' found in '{args.input_dir}'.", flush=True)
        exit(1)

    print(f"Found {len(pdb_files)} files, starting calculation...", flush=True)

    results = []
    
    for idx, pdb_path in enumerate(pdb_files):
        try:
            # Extract filename as name (remove _model.pdb suffix)
            file_basename = os.path.basename(pdb_path)
            name = file_basename.rsplit('_model.pdb', 1)[0]

            # Calculate clashes
            # Use default only_ca=False (calculate all-atom inter-chain clashes)
            num_clashes = calculate_clash_score(pdb_path, threshold=args.threshold, only_ca=False)
            
            # Print progress
            print(f"[{idx+1}/{len(pdb_files)}] {file_basename}: {num_clashes} clashes", flush=True)

            results.append({
                "name": name,
                "clashes": num_clashes
            })

        except Exception as e:
            print(f"[Warning] Failed to process file {pdb_path}: {e}", flush=True)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        # Ensure the 'name' column is first
        cols = ['name', 'clashes']
        df = df[cols]
        
        df.to_csv(args.output_csv, index=False)
        print(f"\nCalculation complete. Results saved to: {args.output_csv}", flush=True)
    else:
        print("\nNo results generated.", flush=True)