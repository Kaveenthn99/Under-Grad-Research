#!/usr/bin/env python3
"""
Reads:
  1) selected_molecules_for_GFN2_xTB.csv  (column: "Compound")
  2) canonical_db_withBP79.csv           (columns: "DrugBank ID", "SMILES")

For each Compound ID (e.g., DB06909), finds the matching SMILES and computes RDKit LogP.
Outputs:
  /Users/kaveen/Desktop/logp/ID_logp.csv  with columns: ID, logp
"""

import os
import sys
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen

# ---- INPUTS ----
SELECTED_PATH = "/Users/kaveen/Desktop/logp/selected_molecules_for_GFN2_xTB.csv"
CANONICAL_PATH = "/Users/kaveen/Desktop/logp/canonical_db_withBP79.csv"

# ---- OUTPUT ----
OUT_DIR = "/Users/kaveen/Desktop/logp"
OUT_FILE = os.path.join(OUT_DIR, "ID_logp.csv")


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CSVs
    try:
        selected = pd.read_csv(SELECTED_PATH)
    except Exception as e:
        print(f"ERROR reading: {SELECTED_PATH}\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        canonical = pd.read_csv(CANONICAL_PATH)
    except Exception as e:
        print(f"ERROR reading: {CANONICAL_PATH}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Required columns (exact names you gave)
    if "Compound" not in selected.columns:
        raise ValueError('selected_molecules_for_GFN2_xTB.csv must contain column: "Compound"')
    if "DrugBank ID" not in canonical.columns:
        raise ValueError('canonical_db_withBP79.csv must contain column: "DrugBank ID"')
    if "SMILES" not in canonical.columns:
        raise ValueError('canonical_db_withBP79.csv must contain column: "SMILES"')

    # Normalize
    selected["Compound"] = selected["Compound"].map(norm)
    canonical["DrugBank ID"] = canonical["DrugBank ID"].map(norm)
    canonical["SMILES"] = canonical["SMILES"].map(norm)

    # Build DrugBank ID -> SMILES mapping (take first non-empty SMILES if duplicates)
    canonical_nonempty = canonical.copy()
    canonical_nonempty = canonical_nonempty[canonical_nonempty["DrugBank ID"] != ""]

    canonical_nonempty = (
        canonical_nonempty.sort_values(by=["DrugBank ID"])
        .groupby("DrugBank ID", as_index=False)
        .agg({"SMILES": lambda s: next((x for x in s if x), "")})
    )

    id_to_smiles = dict(zip(canonical_nonempty["DrugBank ID"], canonical_nonempty["SMILES"]))

    # Process IDs (unique)
    ids = (
        selected["Compound"]
        .dropna()
        .map(norm)
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    results = []
    missing_smiles = 0
    invalid_smiles = 0

    for dbid in ids:
        smi = id_to_smiles.get(dbid, "")
        if not smi:
            results.append({"ID": dbid, "logp": None})
            missing_smiles += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"ID": dbid, "logp": None})
            invalid_smiles += 1
            continue

        logp = float(Crippen.MolLogP(mol))
        results.append({"ID": dbid, "logp": logp})

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")
    print(f"Total unique Compounds processed: {len(ids)}")
    print(f"Missing SMILES: {missing_smiles}")
    print(f"Invalid SMILES: {invalid_smiles}")


if __name__ == "__main__":
    main()
