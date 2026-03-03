import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Atom import Atom 
from Bio.PDB.Polypeptide import protein_letters_3to1
from typing import Tuple, List
import os
import glob
import csv
import json

# --- 1. Global Configuration ---

NATIVE_CHAIN_ORDER = ['A', 'B'] 
PRED_CHAIN_ORDER = ['A', 'B']   

PRED_BASE_DIR = "./KIT_05_pdbs/"
NATIVE_BASE_DIR = "../dl_binder_design/examples/KIT_05noise_out/"
CSV_OUTPUT_FILE = "KIT05noise_rmsd_results_with_confidence.csv"

# --- 2. Core Functions (Unchanged) ---

def get_ca_atoms_and_lengths(pdb_file: str, chain_id_list: List[str]) -> Tuple[List[Atom], List[int]]:
    """Extracts CA Atom objects from a PDB file, following the specified chain order."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    all_ca_atoms = []
    chain_lengths = []
    model = structure[0]
    
    for chain_id in chain_id_list:
        chain = model[chain_id]
        
        chain_ca_atoms = [] 
        for atom in chain.get_atoms():
            if atom.get_id() == 'CA':
                chain_ca_atoms.append(atom)
        if not chain_ca_atoms:
            raise ValueError(f"No 'CA' atoms found in chain '{chain_id}' of file '{pdb_file}'.")
        
        chain_lengths.append(len(chain_ca_atoms))
        all_ca_atoms.extend(chain_ca_atoms)
            
    return all_ca_atoms, chain_lengths


def calculate_rmsd(atoms1: List[Atom], atoms2: List[Atom]) -> float:
    """Aligns two lists of Atom objects and calculates the RMSD."""
    if len(atoms1) != len(atoms2):
        raise ValueError(f"Atom list length mismatch: {len(atoms1)} vs {len(atoms2)}")
    if len(atoms1) == 0:
        print("Warning: No atoms provided for RMSD calculation.")
        return 0.0
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    return super_imposer.rms


def get_sequence_from_pdb(pdb_file: str, chain_id_list: List[str]) -> str:
    """Extracts the 1-letter amino acid sequence from specified chains in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("seq_struct", pdb_file)
    model = structure[0]
    full_sequence = []
    
    for chain_id in chain_id_list:
        chain = model[chain_id]
            
        chain_seq = []
        for residue in chain:
            if residue.id[0] == ' ': 
                res_name = residue.get_resname().strip()
                one_letter_code = protein_letters_3to1.get(res_name, 'X')
                chain_seq.append(one_letter_code)
        full_sequence.append("".join(chain_seq))
    
    return "".join(full_sequence)


def calculate_confidence_metrics(
    json_file_path: str, 
    binder_chain_char: str, 
    binder_residue_len: int, 
    total_residue_len: int
) -> dict:
    """
    Loads AF3 'confidences.json' data and calculates pLDDT and PAE metrics.
    Uses ATOM length for pLDDT and RESIDUE length for PAE.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    pae_matrix = np.array(data['pae'])
    plddt_scores = np.array(data['atom_plddts'])
    atom_chain_ids = data['atom_chain_ids']

    # --- pLDDT Calculation (Atom-based) ---
    binder_atom_len = atom_chain_ids.count(binder_chain_char)
    if binder_atom_len == 0:
        raise ValueError(f"Binder chain '{binder_chain_char}' not found in 'atom_chain_ids'.")
    
    plddt_binder_mean = np.mean(plddt_scores[:binder_atom_len])
    
    # --- PAE Calculation (Residue-based) ---
    # Validate that the PAE matrix shape matches the PDB residue count
    if pae_matrix.shape[0] != total_residue_len or pae_matrix.shape[1] != total_residue_len:
        raise ValueError(f"PAE matrix shape {pae_matrix.shape} does not match "
                         f"PDB total residue count ({total_residue_len}).")
                         
    if binder_residue_len == total_residue_len:
        raise ValueError("PDB data shows only binder residues, cannot calculate PAE interaction.")

    # Use the PDB-derived RESIDUE length for slicing the PAE matrix
    # 
    pae_interaction1 = np.mean(pae_matrix[:binder_residue_len, binder_residue_len:])
    pae_interaction2 = np.mean(pae_matrix[binder_residue_len:, :binder_residue_len])
    pae_interaction_total = (pae_interaction1 + pae_interaction2) / 2
    
    return {
        'plddt_binder': plddt_binder_mean,
        'pae_interaction1': pae_interaction1,
        'pae_interaction2': pae_interaction2,
        'pae_interaction_total': pae_interaction_total
    }

# --- 4. Main Execution Script (MODIFIED) ---

def main():
    """Runs the main batch processing workflow."""
    results_data = []
    csv_headers = ['name', 'sequence', 'binder_RMSD', 'complex_RMSD', 
                   'plddt_binder', 'pae_interaction1', 'pae_interaction2', 'pae_interaction_total']
    results_data.append(csv_headers)
    
    pred_base_dir_abs = os.path.abspath(PRED_BASE_DIR)
    native_base_dir_abs = os.path.abspath(NATIVE_BASE_DIR)
    
    search_pattern = os.path.join(pred_base_dir_abs, "*_model.pdb")
    print(f"Searching for files matching: {search_pattern}\n")
    
    for pred_pdb_file in glob.glob(search_pattern):
        
        filename = os.path.basename(pred_pdb_file)
        if len(filename) <= 10 or not filename.endswith("_model.pdb"):
            continue
            
        name = filename[:-len("_model.pdb")]
        print(f"--- Processing: {name} ---")
        
        native_pdb_file = os.path.join(native_base_dir_abs, f"{name}.pdb")
        conf_json_file = os.path.join(pred_base_dir_abs, f"{name}_confidences.json")
        
        if not all(os.path.exists(f) for f in [native_pdb_file, pred_pdb_file, conf_json_file]):
            print(f"  [SKIPPED] One or more required files are missing for name: {name}")
            results_data.append([name, "File Not Found", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue

        # 1. Extract Sequence (Binder only)
        binder_chain_id_list = [NATIVE_CHAIN_ORDER[0]]
        sequence = get_sequence_from_pdb(native_pdb_file, binder_chain_id_list)

        # 2. Get PDB Atom/Residue info
        native_all_atoms, native_lengths = get_ca_atoms_and_lengths(native_pdb_file, NATIVE_CHAIN_ORDER)
        pred_all_atoms, pred_lengths = get_ca_atoms_and_lengths(pred_pdb_file, PRED_CHAIN_ORDER)

        if native_lengths != pred_lengths:
            print(f"  [ERROR] Chain length mismatch (Native: {native_lengths}, Pred: {pred_lengths})")
            results_data.append([name, sequence, "Length Mismatch", "Length Mismatch", 
                                 "N/A", "N/A", "N/A", "N/A"])
            continue
            
        # --- 3. Define RESIDUE lengths ---
        binder_res_len = native_lengths[0]
        total_res_len = sum(native_lengths)

        # 4. Calculate RMSD
        native_binder_atoms = native_all_atoms[:binder_res_len]
        pred_binder_atoms = pred_all_atoms[:binder_res_len]
        native_complex_atoms = native_all_atoms
        pred_complex_atoms = pred_all_atoms
        
        rmsd_binder = calculate_rmsd(native_binder_atoms, pred_binder_atoms)
        rmsd_complex = calculate_rmsd(native_complex_atoms, pred_complex_atoms)

        # 5. Calculate AF3 Confidence Metrics
        binder_char = PRED_CHAIN_ORDER[0]
        
        # Pass the RESIDUE lengths to the function
        conf_metrics = calculate_confidence_metrics(
            conf_json_file, binder_char, binder_res_len, total_res_len
        )
        
        print(f"Success: Binder RMSD: {rmsd_binder:.3f}, Complex RMSD: {rmsd_complex:.3f}")
        print(f"Binder pLDDT: {conf_metrics['plddt_binder']:.3f}, PAE Interaction: {conf_metrics['pae_interaction_total']:.3f}")

        # 6. Store complete results
        results_data.append([
            name,
            sequence,
            f"{rmsd_binder:.3f}",
            f"{rmsd_complex:.3f}",
            f"{conf_metrics['plddt_binder']:.3f}",
            f"{conf_metrics['pae_interaction1']:.3f}",
            f"{conf_metrics['pae_interaction2']:.3f}",
            f"{conf_metrics['pae_interaction_total']:.3f}"
        ])

    # --- 5. Write CSV ---
    
    with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(results_data)
    
    print(f"\n--- Batch processing complete ---")
    print(f"Processed {len(results_data) - 1} total entries.")
    print(f"Results written to: {os.path.abspath(CSV_OUTPUT_FILE)}")
        
if __name__ == "__main__":
    main()
