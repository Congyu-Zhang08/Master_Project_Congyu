import os
import json
import glob
import sys

# --- CONFIGURATION ---
folder_path = "./PDGFR_input" 
file_pattern = os.path.join(folder_path, "*.json")

REFERENCE_FILE = "example.json" 
# ---------------------

TEMPLATE_DATA = None

def load_template_data_from_reference():
    global TEMPLATE_DATA
    try:
        with open(REFERENCE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(1)
    except Exception:
        sys.exit(1)

    sequences = data.get("sequences", [])
    if len(sequences) < 2:
        sys.exit(1)

    protein_b_data = sequences[1].get("protein", {})
    
    TEMPLATE_DATA = {
        "unpairedMsa_B": protein_b_data.get("unpairedMsa", ""),
        "pairedMsa_B": protein_b_data.get("pairedMsa", ""),
        "templates_B": protein_b_data.get("templates", []) 
    }


def update_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return

    sequences = data.get("sequences", [])
    
    for i, seq_entry in enumerate(sequences):
        protein = seq_entry.get("protein", {})
        
        sequence = protein.get("sequence", "")
        
        # NOTE: 'modifications' field removed here
        
        # Protein A (index 0)
        if i == 0:
            msa_content = f">query\n{sequence}\n"
            protein["unpairedMsa"] = msa_content
            protein["pairedMsa"] = msa_content
            
        # Protein B (index 1) - Insert all three long data items
        elif i == 1:
            protein["unpairedMsa"] = TEMPLATE_DATA.get("unpairedMsa_B", f">query\n{sequence}\n")
            protein["pairedMsa"] = TEMPLATE_DATA.get("pairedMsa_B", f">query\n{sequence}\n")
            
            # Insert 'templates' array inside the protein object
            protein["templates"] = TEMPLATE_DATA.get("templates_B", [])


    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def process_all_files():
    load_template_data_from_reference()
    
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        return

    for file_path in file_list:
        update_json_file(file_path)

if __name__ == "__main__":
    process_all_files()
