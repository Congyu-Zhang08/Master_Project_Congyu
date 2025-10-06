from Bio.PDB import PDBParser, PPBuilder
import os
import sys

def extract_sequences_to_file(directory, output_file):
    """
    Extracts amino acid sequences from PDB files in a given directory
    and writes them to a single output file, including the filename for
    each sequence.
    """
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    with open(output_file, 'w') as out_file:
        for filename in os.listdir(directory):
            if filename.endswith(".pdb"):
                pdb_path = os.path.join(directory, filename)
                try:
                    structure = parser.get_structure(filename, pdb_path)
                    
                    out_file.write(f"--- Sequences from {filename} ---\n")
                    for model in structure:
                        for chain in model:
                            # Build peptides and join their sequences
                            sequence = "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
                            if sequence:  # Only write if a sequence was found
                                out_file.write(f"Filename: {filename}, Chain {chain.id}: {sequence}\n")
                    out_file.write("\n")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    out_file.write(f"--- Error processing {filename}: {e} ---\n\n")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python pdb_sequence_extractor.py <input_directory> <output_directory>")
        sys.exit(1)

    # Get input and output directories from command-line arguments
    pdb_directory = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Define the output file name within the specified output directory
    output_file_name = os.path.join(output_dir, "sequences.txt")

    # Check if the input directory exists
    if not os.path.isdir(pdb_directory):
        print(f"Error: Input directory '{pdb_directory}' not found.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the extraction function
    extract_sequences_to_file(pdb_directory, output_file_name)

    print(f"Sequences have been successfully extracted and saved to {output_file_name}")
