import os
import csv
import subprocess
import xml.etree.ElementTree as ET

# ==========================================
# Configuration
# ==========================================
INPUT_CSV = "KITfiltered_secondary_structure_results.csv" 
OUTPUT_CSV = "final_filtered_pdb_interactions.csv"
PDB_FOLDER = "./filtered_pdbs"

# [IMPORTANT] Set the absolute path to your Singularity image (.sif file)
# Example: "/projects/software/containers/plip_latest.sif"
SINGULARITY_IMAGE_PATH = "./plip_3.0.0.simg" 

# Define the Target Chain (Receptor) and Binder Chain (Ligand)
BINDER_CHAIN = "A"
TARGET_CHAIN = "B"



def run_plip_and_parse(pdb_file_path):
    # 1. Run PLIP
    cmd = [
        SINGULARITY_IMAGE_PATH,
        "-f", pdb_file_path, 
        "--peptide", "A", 
        "--xml", 
        "--pymol",
        "--silent"
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    # 2. Locate XML
    xml_report = "report.xml" 
    if not os.path.exists(xml_report):
        prefix = os.path.basename(pdb_file_path).replace(".pdb", "")
        xml_report = f"{prefix}_report.xml"
        
    # 3. Parse XML
    tree = ET.parse(xml_report)
    root = tree.getroot()
    
    details = {"Hbonds_B_145_150": 0, "SaltBridge_B_90_92": 0}
    
    hbond_count = 0
    saltbridge_found = False
    
    # Iterate over ALL binding sites found (Assuming valid A-B interaction)
    for site in root.findall("bindingsite"):
        
        # --- Check 1: Hydrogen Bonds ---
        hbonds = site.findall("interactions/hydrogen_bonds/hydrogen_bond")
        
        for hb in hbonds:
            res_chain_node = hb.find("reschain")
            res_nr_node = hb.find("resnr")
            
            # Skip if malformed
            if res_chain_node is None or res_nr_node is None:
                continue

            res_chain = res_chain_node.text
            res_nr = int(res_nr_node.text)
            
            # Check Chain B, Residue 145-150
            if res_chain == TARGET_CHAIN:
                if 145 <= res_nr <= 150:
                    hbond_count += 1

        # --- Check 2: Salt Bridges ---
        saltbridges = site.findall("interactions/salt_bridges/salt_bridge")
        
        for sb in saltbridges:
            res_chain_node = sb.find("reschain")
            res_nr_node = sb.find("resnr")

            if res_chain_node is None or res_nr_node is None:
                continue

            res_chain = res_chain_node.text
            res_nr = int(res_nr_node.text)
            
            # Check Chain B, Residue 90 or 92
            if res_chain == TARGET_CHAIN:
                if res_nr == 90 or res_nr == 92:
                    saltbridge_found = True

    # 4. Result
    details["Hbonds_B_145_150"] = hbond_count
    details["SaltBridge_B_90_92"] = 1 if saltbridge_found else 0
    
    passed = (hbond_count > 0) and saltbridge_found

    # 5. Clean up
    if os.path.exists(xml_report):
        os.remove(xml_report)
            
    return passed, details

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    print(f"Reading from {INPUT_CSV}...")
    
    match_count = 0
    
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as f_in, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        
        fieldnames = reader.fieldnames + ["Hbonds_B_145_150", "SaltBridge_B_90_92"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            filename = row.get("Name")
            
            beta_count = int(row.get("Beta_Strand_Count_ChainA", 0))
                
            if beta_count <= 0:
                continue

            pdb_path = os.path.join(PDB_FOLDER, filename)
            
            print(f"Analyzing {filename} ...", end="\r")
            
            passed, details = run_plip_and_parse(pdb_path)
            
            if passed:
                row.update(details)
                writer.writerow(row)
                match_count += 1
                print(f"[MATCH] {filename} passed filters.            ")

    print(f"\nProcessing complete.")
    print(f"Total matching PDBs: {match_count}")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()