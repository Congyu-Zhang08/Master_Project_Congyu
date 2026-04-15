"""Generate BoltzGen YAML config files from PDBs with sequences.

For each PDB file:
  1. Extract the binder chain sequence.
  2. Emit a YAML that:
       - Loads target + binder chains from the PDB.
       - Marks the binder chain as 'design' (so it gets evaluated).
       - Adds per-position residue_constraints locking each position
         to the ProteinMPNN amino acid. This prevents BoltzGen from
         replacing the sequence during inverse folding.

Usage:
    python generate_yamls.py
"""
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

# ---- EDIT THESE PARAMETERS ----
PDB_DIR = Path("rfdiffusion_outputs")   # Directory with your input PDBs
YAML_DIR = Path("yamls")                # Output directory for YAML + PDB copies
TARGET_CHAIN = "A"                      # Chain ID of c-KIT in your PDBs
BINDER_CHAIN = "B"                      # Chain ID of the designed binder
# -------------------------------


def extract_binder_sequence(pdb_path: Path, binder_chain: str) -> str:
    """Return single-letter sequence of the binder chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    seq = []
    for residue in structure[0][binder_chain]:
        if residue.id[0] == " ":  # Skip hetero / water
            try:
                seq.append(three_to_one(residue.get_resname()))
            except KeyError:
                seq.append("X")
    return "".join(seq)


def generate_yaml(
    pdb_name: str,
    yaml_path: Path,
    sequence: str,
    target_chain: str,
    binder_chain: str,
) -> None:
    """Write a YAML config that locks the binder chain to the given sequence."""
    if "X" in sequence:
        raise ValueError(f"Unknown residue found in binder of {pdb_name}")

    constraints_block = "\n".join(
        f"        - position: {i + 1}\n          allowed: {aa}"
        for i, aa in enumerate(sequence)
    )

    yaml_content = f"""entities:
  - file:
      path: {pdb_name}
      include:
        - chain:
            id: {target_chain}
        - chain:
            id: {binder_chain}
      design:
        - chain:
            id: {binder_chain}
      residue_constraints:
{constraints_block}
"""
    yaml_path.write_text(yaml_content)


def main() -> None:
    YAML_DIR.mkdir(exist_ok=True)
    pdb_paths = sorted(PDB_DIR.glob("*.pdb"))
    print(f"Found {len(pdb_paths)} PDB files in {PDB_DIR}")

    for pdb_path in pdb_paths:
        # BoltzGen resolves 'path:' relative to the YAML file location,
        # so we copy the PDB next to the generated YAML.
        dest_pdb = YAML_DIR / pdb_path.name
        if not dest_pdb.exists():
            dest_pdb.write_bytes(pdb_path.read_bytes())

        seq = extract_binder_sequence(dest_pdb, BINDER_CHAIN)
        yaml_path = YAML_DIR / f"{pdb_path.stem}.yaml"
        generate_yaml(
            pdb_name=pdb_path.name,
            yaml_path=yaml_path,
            sequence=seq,
            target_chain=TARGET_CHAIN,
            binder_chain=BINDER_CHAIN,
        )

    print(f"Generated {len(pdb_paths)} YAML files in {YAML_DIR}")


if __name__ == "__main__":
    main()
