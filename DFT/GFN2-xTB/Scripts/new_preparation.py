#!/usr/bin/env python3
from pathlib import Path
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================
# Ligand handling (RDKit): read SDF, add H, keep coords
# ============================================================

def load_ligand_pose_mols_with_h_from_sdf(sdf_path: Path):
    """Read all molecules (poses) in an SDF, add H with coords, keep docking coords."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    out = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            # Fallback only; docking SDF should already have 3D
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        mol_h = Chem.AddHs(mol, addCoords=True)
        out.append((i + 1, mol_h))
    return out


def ligand_formal_charge(mol: Chem.Mol) -> int:
    """Ligand charge strictly from SDF (RDKit formal charges)."""
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def ligand_total_Z(mol: Chem.Mol) -> int:
    """Sum of atomic numbers for ligand (includes added H)."""
    return sum(a.GetAtomicNum() for a in mol.GetAtoms())


def ligand_xyz_lines(mol: Chem.Mol):
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"  {atom.GetSymbol():2s}  {pos.x: .6f}  {pos.y: .6f}  {pos.z: .6f}")
    return lines


# ============================================================
# PDB parsing
# ============================================================

def parse_element_and_charge_field(raw: str):
    """
    PDB cols 77-80 can contain element+charge like 'O1-' or 'N2+'.
    Returns (element, integer_charge).
    """
    s = raw.strip()
    if not s:
        return None, 0

    elem_part = ""
    charge_part = ""
    for ch in s:
        if ch.isalpha() and charge_part == "" and ch not in "+-":
            elem_part += ch
        else:
            charge_part += ch

    charge = 0
    if charge_part:
        sign_char = charge_part[-1]
        num_part = charge_part[:-1]
        if sign_char in "+-":
            sign = 1 if sign_char == "+" else -1
            mag = int(num_part) if num_part.isdigit() else 1
            charge = sign * mag

    elem = elem_part.capitalize() if elem_part else None
    return elem, charge


def parse_pdb_atoms(pdb_path: Path):
    """Parse all ATOM/HETATM lines into a flat list. TER is ignored."""
    atoms = []
    with pdb_path.open() as f:
        for line in f:
            if line.startswith("END"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21:22].strip()
            res_seq_str = line[22:26].strip()

            try:
                res_seq = int(res_seq_str)
            except Exception:
                res_seq = None

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            elem_from_field, charge = parse_element_and_charge_field(line[76:80])
            if elem_from_field is not None:
                element = elem_from_field
            else:
                nm = atom_name.strip()
                element = (nm[1] if (len(nm) >= 2 and nm[0].isdigit()) else nm[0]).upper()

            atoms.append({
                "idx": len(atoms),  # 0-based index in this atoms list
                "element": element,
                "atom_name": atom_name,
                "res_name": res_name,
                "chain": chain,
                "res_seq": res_seq,
                "x": x, "y": y, "z": z,
                "charge": charge,
            })

    if not atoms:
        raise RuntimeError(f"No atoms parsed from PDB: {pdb_path}")
    return atoms


def residue_total_charge_from_atoms(atoms):
    """Sum atomic charges from PDB charge field."""
    return sum(a["charge"] for a in atoms)


def residue_total_Z_from_atoms(atoms):
    """Sum atomic numbers from PDB element symbols."""
    pt = Chem.GetPeriodicTable()
    total = 0
    for a in atoms:
        sym = a["element"]
        try:
            z = pt.GetAtomicNumber(sym)
        except Exception:
            raise RuntimeError(f"Unknown element symbol in PDB: '{sym}'")
        total += z
    return total


def residue_xyz_lines(atoms):
    return [f"  {a['element']:2s}  {a['x']: .6f}  {a['y']: .6f}  {a['z']: .6f}" for a in atoms]


def _dist2(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return dx*dx + dy*dy + dz*dz


def build_residue_and_cap_maps(atoms):
    ace_map = {}
    nme_map = {}
    res_map = {}
    for a in atoms:
        key = (a["chain"], a["res_seq"])
        rn = a["res_name"].upper()
        if rn == "ACE":
            ace_map.setdefault(key, []).append(a["idx"])
        elif rn == "NME":
            nme_map.setdefault(key, []).append(a["idx"])
        else:
            res_map.setdefault(key, []).append(a["idx"])
    return ace_map, nme_map, res_map


def find_cap_connection_constraints(atoms):
    """
    For each real residue at (chain, r):
      ACE expected at (chain, r-1)
      NME expected at (chain, r+1)

    Constrain only the cap atom closest to residue backbone N (ACE side)
    and closest to residue backbone C (NME side).
    """
    ace_map, nme_map, res_map = build_residue_and_cap_maps(atoms)

    constraints = []
    for (chain, r) in sorted(res_map.keys(), key=lambda x: (x[0], x[1] if x[1] is not None else -999999)):
        res_idxs = res_map[(chain, r)]
        if not res_idxs or r is None:
            continue

        res_name = atoms[res_idxs[0]]["res_name"].upper()
        res_tag = f"{res_name}{r}"
        if chain:
            res_tag += f"({chain})"

        n_idx = None
        c_idx = None
        for i in res_idxs:
            if atoms[i]["atom_name"] == "N" and n_idx is None:
                n_idx = i
            if atoms[i]["atom_name"] == "C" and c_idx is None:
                c_idx = i

        ace_key = (chain, r - 1)
        if n_idx is not None and ace_key in ace_map:
            n_atom = atoms[n_idx]
            ace_candidates = ace_map[ace_key]
            best_ace = min(ace_candidates, key=lambda j: _dist2(atoms[j], n_atom))
            constraints.append((best_ace, f"{res_tag} N-term cap (ACE) atom {atoms[best_ace]['atom_name']}"))

        nme_key = (chain, r + 1)
        if c_idx is not None and nme_key in nme_map:
            c_atom = atoms[c_idx]
            nme_candidates = nme_map[nme_key]
            best_nme = min(nme_candidates, key=lambda j: _dist2(atoms[j], c_atom))
            constraints.append((best_nme, f"{res_tag} C-term cap (NME) atom {atoms[best_nme]['atom_name']}"))

    # de-duplicate
    seen = set()
    uniq = []
    for idx, comment in constraints:
        if idx not in seen:
            uniq.append((idx, comment))
            seen.add(idx)
    return uniq


# ============================================================
# Multiplicity
# ============================================================

def infer_multiplicity_from_electron_count(total_Z: int, total_charge: int) -> int:
    """
    total_electrons = total_Z - total_charge
    even -> singlet (1)
    odd  -> doublet (2)
    """
    total_e = total_Z - total_charge
    return 1 if (total_e % 2 == 0) else 2


# ============================================================
# Writers
# ============================================================

def write_xyz_file(lig_xyz, res_xyz, out_path: Path, title: str):
    n_tot = len(lig_xyz) + len(res_xyz)
    lines = [str(n_tot), title]
    lines.extend(l.strip() for l in lig_xyz)
    lines.extend(l.strip() for l in res_xyz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def write_orca_optfreq_input(
    lig_xyz,
    res_xyz,
    total_charge,
    multiplicity,
    out_path: Path,
    constraints_global_indices_with_comments,
):
    lines = []
    lines.append("! GFN2-xTB OPT FREQ VeryTightOPT ALPB(Water)")
    lines.append("")
    lines.append("%pal")
    lines.append("  nprocs 1")
    lines.append("end")
    lines.append("%maxcore 1000")
    lines.append("")
    lines.append("%geom")
    lines.append("  MaxIter 20000")

    if constraints_global_indices_with_comments:
        lines.append("  Constraints")
        for idx, comment in constraints_global_indices_with_comments:
            lines.append(f"    {{ C {idx} C }}   # {comment}")
        lines.append("  end")

    lines.append("end")
    lines.append("")
    lines.append(f"* xyz {total_charge} {multiplicity}")
    lines.extend(lig_xyz)
    lines.extend(res_xyz)
    lines.append("*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    base_dir = Path("/Users/kaveen/Desktop/Preparation").resolve()
    ligands_dir = base_dir / "ligands"
    residue_pdb = base_dir / "Residues" / "All_ACE_NME.pdb"

    # Two output folders only:
    inp_out_dir = base_dir / "orca_inp_outputs"
    xyz_out_dir = base_dir / "orca_xyz_outputs"
    inp_out_dir.mkdir(parents=True, exist_ok=True)
    xyz_out_dir.mkdir(parents=True, exist_ok=True)

    if not ligands_dir.is_dir():
        raise SystemExit(f"ligands directory not found: {ligands_dir}")
    if not residue_pdb.is_file():
        raise SystemExit(f"Residue file not found: {residue_pdb}")

    # Parse residues once
    res_atoms = parse_pdb_atoms(residue_pdb)
    res_charge = residue_total_charge_from_atoms(res_atoms)
    res_Z = residue_total_Z_from_atoms(res_atoms)
    res_xyz = residue_xyz_lines(res_atoms)

    cap_constraints_local = find_cap_connection_constraints(res_atoms)

    sdf_files = sorted(ligands_dir.glob("*.sdf"))
    if not sdf_files:
        raise SystemExit(f"No SDF files found in {ligands_dir}")

    for sdf in sdf_files:
        ligand_name = sdf.stem
        poses = load_ligand_pose_mols_with_h_from_sdf(sdf)
        if not poses:
            print(f"WARNING: no readable molecules in {sdf}")
            continue

        for pose_idx, mol in poses:
            lig_charge = ligand_formal_charge(mol)
            lig_Z = ligand_total_Z(mol)
            lig_xyz = ligand_xyz_lines(mol)
            n_lig = len(lig_xyz)

            # Shift residue constraint indices by ligand atom count (ligand coords first)
            cap_constraints_global = [(n_lig + idx, comment) for (idx, comment) in cap_constraints_local]

            total_charge = lig_charge + res_charge
            total_Z = lig_Z + res_Z
            multiplicity = infer_multiplicity_from_electron_count(total_Z, total_charge)

            label = f"{ligand_name}_pose{pose_idx}__All_ACE_NME__OPTFREQ_ALPB"
            inp_path = inp_out_dir / f"{label}.inp"
            xyz_path = xyz_out_dir / f"{label}.xyz"

            write_orca_optfreq_input(
                lig_xyz=lig_xyz,
                res_xyz=res_xyz,
                total_charge=total_charge,
                multiplicity=multiplicity,
                out_path=inp_path,
                constraints_global_indices_with_comments=cap_constraints_global,
            )

            write_xyz_file(lig_xyz, res_xyz, xyz_path, label)

            print(f"Wrote INP: {inp_path.name}")
            print(f"Wrote XYZ: {xyz_path.name}")
            print(
                f"Charges: ligand={lig_charge:+d}, residues={res_charge:+d}, total={total_charge:+d} | "
                f"Multiplicity={multiplicity} (electrons={total_Z - total_charge})"
            )
            print(f"Cap constraints: {len(cap_constraints_global)}\n")


if __name__ == "__main__":
    main()
