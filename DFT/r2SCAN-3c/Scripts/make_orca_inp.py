#!/usr/bin/env python3
from pathlib import Path

# ---- user settings ----
XYZ_DIR = Path("/Users/kaveen/Desktop/r2SCAN/Prep/xyz")
NEW_PC_DIR = Path("/Users/kaveen/Desktop/r2SCAN/Prep/new_pc")  # First 500 .inp files
OLD_PC_DIR = Path("/Users/kaveen/Desktop/r2SCAN/Prep/old_pc")  # Remaining files
MAX_NEW_PC_FILES = 500  # Maximum files for new_pc folder

# Charge and multiplicity (from your sample: * xyz -2 1)
CHARGE = -2
MULTIPLICITY = 1

HEADER = """! r2SCAN-3c TightSCF SP

%pal
  nprocs 4
end

%maxcore 1000

%scf
  MaxIter 2000
end

%method
  Grid 4
  FinalGrid 5
end

%cpcm
  smd true
  SMDsolvent "water"
end

"""

def xyz_to_orca_block(xyz_path: Path, charge: int, multiplicity: int) -> str:
    """
    Convert an .xyz file to an ORCA * xyz charge mult ... * block.
    Assumes standard XYZ format:
      line 1: number of atoms
      line 2: comment
      line 3+: element x y z
    """
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()

    if len(lines) < 3:
        raise ValueError(f"{xyz_path.name}: too short to be a valid XYZ file.")

    # Try to parse atom count, but don't strictly require it
    # We'll just skip first 2 lines and use the rest as coordinates.
    coord_lines = lines[2:]

    # Filter out blank lines
    coord_lines = [ln.strip() for ln in coord_lines if ln.strip()]

    # Basic validation: each coord line should have at least 4 tokens
    bad = [ln for ln in coord_lines if len(ln.split()) < 4]
    if bad:
        raise ValueError(
            f"{xyz_path.name}: found malformed coordinate lines (need 'El x y z'), e.g.: {bad[0]}"
        )

    block = [f"* xyz {charge} {multiplicity}"]
    block.extend(coord_lines)
    block.append("*")
    return "\n".join(block) + "\n"

def main():
    if not XYZ_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {XYZ_DIR}")

    # Create both output directories
    NEW_PC_DIR.mkdir(parents=True, exist_ok=True)
    OLD_PC_DIR.mkdir(parents=True, exist_ok=True)

    xyz_files = sorted(XYZ_DIR.glob("*.xyz"))
    if not xyz_files:
        print(f"No .xyz files found in: {XYZ_DIR}")
        return

    wrote_new_pc = 0
    wrote_old_pc = 0
    for idx, xyz_path in enumerate(xyz_files):
        try:
            orca_geom = xyz_to_orca_block(xyz_path, CHARGE, MULTIPLICITY)
            
            # Decide which directory to use based on file count
            if idx < MAX_NEW_PC_FILES:
                out_dir = NEW_PC_DIR
                wrote_new_pc += 1
            else:
                out_dir = OLD_PC_DIR
                wrote_old_pc += 1
            
            out_path = out_dir / (xyz_path.stem + ".inp")
            out_path.write_text(HEADER + orca_geom, encoding="utf-8")
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"SKIP {xyz_path.name}: {e}")

    print(f"\nDone. Generated {wrote_new_pc} .inp file(s) in {NEW_PC_DIR}")
    print(f"      Generated {wrote_old_pc} .inp file(s) in {OLD_PC_DIR}")

if __name__ == "__main__":
    main()