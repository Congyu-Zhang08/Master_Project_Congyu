"""
This script is used for batch calculation of interface scores for PDB files (with real-time timing).

It scans an input directory (default is ./pdbs_scores) for all files matching *_model.pdb,
calculates the interface score for each file, and summarizes all results into a CSV file.
When running in Slurm, it prints the processing time for each file.

Example usage:
python calculate_scores_batch_timing.py \
    --input_dir ./pdbs_scores \
    --output_csv ./all_scores.csv \
    --binder_chain A \
    --dalphaball_path /path/to/your/bindcraft/functions/DAlphaBall.gcc
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import time  # <-- 1. Import time module
from pprint import pprint

try:
    # --- PyRosetta Core Dependencies ---
    import pyrosetta as pr
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.core.simple_metrics.metrics import TotalEnergyMetric, SasaMetric
    from pyrosetta.rosetta.core.select.residue_selector import LayerSelector

    # --- BioPython Core Dependencies ---
    from Bio.PDB import PDBParser, Selection
    from scipy.spatial import cKDTree
    from Bio.PDB.Selection import unfold_entities

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure PyRosetta, BioPython, NumPy, SciPy, and Pandas are correctly installed.")
    print("pip install pandas")
    print("PyRosetta requires separate installation and a license.")
    exit(1)


####################################################################
# Dependency Functions (from biopython_utils.py)
####################################################################

# hotspot_residues function requires this dictionary
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def hotspot_residues(trajectory_pdb, binder_chain="A", atom_distance_cutoff=4.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    if binder_chain not in structure[0]:
        print(f"Warning: Binder chain '{binder_chain}' not found in {trajectory_pdb}.")
        return {}
        
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    if 'B' not in structure[0]:
        print(f"Warning: Target chain 'B' not found in {trajectory_pdb}.")
        return {}
        
    target_atoms = Selection.unfold_entities(structure[0]['B'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)
    interacting_residues = {}
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

####################################################################
# Core Functions (from pyrosetta_utils.py)
####################################################################

def score_interface(pdb_file, binder_chain="A"):
    pose = pr.pose_from_pdb(pdb_file)
    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{binder_chain}_B") 
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    for pdb_res_num, aa_type in interface_residues_set.items():
        if aa_type in interface_AA:
            interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0.0

    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)
    
    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100
    else:
        interface_hbond_percentage = 0.0
        interface_bunsch_percentage = 0.0

    chain_design = ChainSelector(binder_chain)
    tem = TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    bsasa = SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0.0

    binder_chain_index = -1
    for i in range(1, pose.num_chains() + 1):
        if pose.pdb_info().chain(pose.conformation().chain_begin(i)) == binder_chain:
            binder_chain_index = i
            break
            
    if binder_chain_index == -1:
        surface_hydrophobicity = 0.0
    else:
        binder_pose = pose.split_by_chain(binder_chain_index)
        layer_sel = LayerSelector()
        layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0 
        for i in range(1, len(surface_res) + 1):
            if surface_res[i] == True:
                res = binder_pose.residue(i)
                if res.is_apolar() or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                    exp_apol_count += 1
                total_count += 1
        
        if total_count > 0:
            surface_hydrophobicity = exp_apol_count / total_count
        else:
            surface_hydrophobicity = 0.0

    interface_scores = {
        'binder_score': binder_score,
        'surface_hydrophobicity': surface_hydrophobicity,
        'interface_sc': interface_sc,
        'interface_packstat': interface_packstat,
        'interface_dG': interface_dG,
        'interface_dSASA': interface_dSASA,
        'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
        'interface_fraction': interface_binder_fraction,
        'interface_hydrophobicity': interface_hydrophobicity,
        'interface_nres': interface_nres,
        'interface_interface_hbonds': interface_interface_hbonds,
        'interface_hbond_percentage': interface_hbond_percentage,
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
        'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    interface_scores = {k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

####################################################################
# Script Execution Entry Point
####################################################################

if __name__ == "__main__":
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Batch calculate interface scores for PDB complexes and save to CSV (with real-time timing).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_dir", 
        default="./pdbs_scores",
        help="Input directory containing PDB files (default: ./pdbs_scores)"
    )
    parser.add_argument(
        "-o", "--output_csv", 
        default="interface_scores_summary.csv",
        help="Output CSV file path (default: interface_scores_summary.csv)"
    )
    parser.add_argument(
        "-chain", "--binder_chain", 
        default="A", 
        help="Chain ID to act as 'binder' (default: A).\nScript assumes the other chain is 'B'."
    )
    parser.add_argument(
        "-dalphaball", "--dalphaball_path",
        required=True,
        help="Full path to DAlphaBall.gcc or surf_vol executable.\n(Required for calculating BuriedUnsatHbonds)"
    )
    args = parser.parse_args()


    # --- 2. Check DAlphaBall Path ---
    if not os.path.exists(args.dalphaball_path):
        print(f"Error: DAlphaBall executable not found at: {args.dalphaball_path}")
        print("Please provide the correct path using --dalphaball_path.")
        exit(1)

    # --- 3. Initialize PyRosetta ---
    print("Initializing PyRosetta...")
    init_flags = (
        "-ignore_unrecognized_res "
        "-load_PDB_components false "
        "-holes:dalphaball "
        f"-dalphaball {args.dalphaball_path}"
    )
    pr.init(init_flags) 

    # --- 4. Find PDB Files ---
    search_pattern = os.path.join(args.input_dir, "*_model.pdb")
    pdb_files = glob.glob(search_pattern)

    if not pdb_files:
        print(f"Error: No files matching '*_model.pdb' found in '{args.input_dir}'.")
        exit(1)

    total_files = len(pdb_files)
    print(f"Found {total_files} PDB files. Starting processing...")

    # --- 5. Loop through files ---
    all_results = []
    total_start_time = time.time() 

    for i, pdb_path in enumerate(pdb_files):
        file_basename = os.path.basename(pdb_path)
        print(f"--- Processing file {i+1}/{total_files}: {file_basename} ---")
        
        # <-- 2. Record start time for individual file
        file_start_time = time.time() 
        
        try:
            # Extract 'name'
            name = file_basename.rsplit('_model.pdb', 1)[0]
            
            # Calculate scores
            (
                scores, 
                aa_counts, 
                interface_residues_str
            ) = score_interface(pdb_path, args.binder_chain)
            
            # <-- 3. Record end time and calculate duration
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            print(f"    > Done. Time taken: {duration:.2f} seconds.") 
            
            # --- Prepare data for CSV ---
            row_data = {}
            row_data['name'] = name
            row_data['processing_time_s'] = round(duration, 2) 
            row_data.update(scores)
            row_data['interface_residues_str'] = interface_residues_str
            flat_aa_counts = {f"AA_{aa}": count for aa, count in aa_counts.items()}
            row_data.update(flat_aa_counts)
            
            all_results.append(row_data)

        except Exception as e:
            # <-- 4. Record time even on failure
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            print(f"\n[Warning] Failed to process file {pdb_path}: {e} (Time taken: {duration:.2f} seconds)")
            print("Skipping this file.")

    # --- 6. Save to CSV ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nProcessing complete. Successfully processed {len(all_results)} / {total_files} files.")
    print(f"Total time taken: {total_duration:.2f} seconds.")
    
    if not all_results:
        print("No files were successfully processed.")
        exit(1)
        
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    key_cols = ['name', 'processing_time_s', 'interface_dG', 'interface_dSASA', 'interface_sc', 'interface_packstat', 'interface_nres']
    existing_key_cols = [col for col in key_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_key_cols]
    
    df = df[existing_key_cols + sorted(other_cols)]

    df.to_csv(args.output_csv, index=False)
    
    print(f"Results saved to: {args.output_csv}")