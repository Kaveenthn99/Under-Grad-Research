#!/usr/bin/env python3
"""
r2SCAN-3c Descriptor Extraction Script - Nested Folder Version
============================================================
Extracts quantum chemical descriptors from ORCA r2SCAN-3c output files
located in nested subfolders and saves them to a CSV file.

This version searches recursively through subfolders to find .out files.

Descriptors extracted:
1. HOMO-LUMO Gap (eV)
2. Final Single Point Energy (Hartree)
3. HOMO Energy (eV)
4. LUMO Energy (eV)
5. Dispersion Energy D4 Correction (Hartree)
6. Chemical Hardness η (eV)
7. Electrophilicity Index ω (eV)
8. Dipole Moment Magnitude (Debye)
"""

import os
import re
import pandas as pd
import glob
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# Configuration paths
# -----------------------------------------------------------------------------

# Default base directory containing nested ORCA output folders
DEFAULT_BASE_PATH = "/Users/kaveen/Documents/Research/Work/SILICO/DFT/r2SCAN/Results/completed"

# Default output CSV filename (will be created in the current working directory)
DEFAULT_OUTPUT_CSV = "r2scan_descriptors_all.csv"


def find_out_files_recursive(base_path):
    """Find all .out files recursively in subdirectories."""

    out_files = []

    # Use Path.rglob() for recursive search
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return []

    print(f"Searching for .out files in: {base_path}")
    print("This may take a moment for large directory structures...")

    # Find all .out files recursively
    for out_file in base_path.rglob("*.out"):
        out_files.append(str(out_file))

    return sorted(out_files)


def extract_descriptors_from_file(filepath):
    """Extract descriptors from a single ORCA output file."""

    # Get relative path info for better molecule ID extraction
    path_parts = Path(filepath).parts
    filename = os.path.basename(filepath)

    # Try to extract molecule ID from folder structure or filename
    molecule_id = os.path.splitext(filename)[0]

    # If filename has standard format, extract molecule ID
    if filename.startswith("DFT_"):
        parts = filename.split("_")
        if len(parts) > 1:
            molecule_id = parts[1]

    # Alternative: use parent folder name as molecule ID if it's more descriptive
    parent_folder = path_parts[-2] if len(path_parts) > 1 else "unknown"

    # Use parent folder if it looks like a molecule ID (not just numbers)
    if not parent_folder.isdigit() and len(parent_folder) > 2:
        molecule_id = parent_folder

    result = {
        "molecule_id": molecule_id,
        "filename": filename,
        "folder_path": str(Path(filepath).parent),
        "relative_path": str(Path(filepath).relative_to(Path(filepath).parts[0]))
        if len(path_parts) > 1
        else filepath,
        "status": "failed",
    }

    try:
        # Read the file
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Check if calculation completed successfully
        if "FINAL SINGLE POINT ENERGY" not in content:
            return result

        result["status"] = "success"

        # 1. Extract Final Single Point Energy
        energy_match = re.search(
            r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", content
        )
        if energy_match:
            result["final_single_point_energy_hartree"] = float(energy_match.group(1))

        # 2. Extract Dispersion Energy (D4 Correction)
        dispersion_match = re.search(
            r"Dispersion correction\s+([-+]?\d+\.\d+)", content
        )
        if dispersion_match:
            result["dispersion_energy_d4_hartree"] = float(dispersion_match.group(1))

        # 3. Extract Dipole Moment Magnitude
        dipole_match = re.search(r"Magnitude \(Debye\)\s*:\s*([\d.]+)", content)
        if dipole_match:
            result["dipole_moment_magnitude_debye"] = float(dipole_match.group(1))

        # 4. Extract HOMO and LUMO energies from orbital section
        orbital_section = re.search(
            r"ORBITAL ENERGIES.*?\n.*?NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n(.*?)(?=\n\s*\*|$)",
            content,
            re.DOTALL,
        )

        if orbital_section:
            orbital_lines = orbital_section.group(1).strip().split("\n")

            homo_energy_ev = None
            lumo_energy_ev = None

            for line in orbital_lines:
                line = line.strip()
                if not line or line.startswith("*"):
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    try:
                        occupancy = float(parts[1])
                        energy_ev = float(parts[3])

                        # HOMO: last occupied orbital (occupancy = 2.0000)
                        if abs(occupancy - 2.0) < 0.001:
                            homo_energy_ev = energy_ev

                        # LUMO: first unoccupied orbital (occupancy = 0.0000)
                        elif abs(occupancy - 0.0) < 0.001 and lumo_energy_ev is None:
                            lumo_energy_ev = energy_ev
                            break  # Found LUMO, stop searching
                    except (ValueError, IndexError):
                        continue

            # Store HOMO and LUMO energies
            if homo_energy_ev is not None:
                result["homo_energy_ev"] = homo_energy_ev

            if lumo_energy_ev is not None:
                result["lumo_energy_ev"] = lumo_energy_ev

            # Calculate derived descriptors if both HOMO and LUMO are available
            if homo_energy_ev is not None and lumo_energy_ev is not None:
                # 5. HOMO-LUMO Gap
                gap = lumo_energy_ev - homo_energy_ev
                result["homo_lumo_gap_ev"] = gap

                # 6. Chemical Hardness (η) = (LUMO - HOMO) / 2
                result["chemical_hardness_eta_ev"] = gap / 2.0

                # 7. Electronegativity (χ) = -(HOMO + LUMO) / 2
                electronegativity = -(homo_energy_ev + lumo_energy_ev) / 2.0
                result["electronegativity_chi_ev"] = electronegativity

                # 8. Electrophilicity Index (ω) = χ² / (2η)
                if gap > 0:  # Avoid division by zero
                    result["electrophilicity_index_omega_ev"] = (
                        electronegativity**2
                    ) / gap
                else:
                    result["electrophilicity_index_omega_ev"] = float("inf")

        # Quality control: check gCP correction
        gcp_match = re.search(r"gCP correction\s+([-+]?\d+\.\d+)", content)
        if gcp_match:
            gcp_value = float(gcp_match.group(1))
            result["gcp_correction_hartree"] = gcp_value

            # Add quality flag
            if abs(gcp_value) > 0.05:
                result["quality_flag"] = "poor_large_gcp"
            elif abs(gcp_value) > 0.02:
                result["quality_flag"] = "moderate_gcp"
            else:
                result["quality_flag"] = "good"

    except Exception as e:
        result["error_message"] = str(e)

    return result


def process_nested_folders(base_path, output_csv=DEFAULT_OUTPUT_CSV):
    """Process all .out files in nested folders and save to CSV."""

    # Find all .out files recursively
    out_files = find_out_files_recursive(base_path)

    if not out_files:
        print(f"No .out files found in: {base_path}")
        print(
            "Please check the path and ensure ORCA output files with .out extension are present"
        )
        return None

    print(f"Found {len(out_files)} .out files in {base_path}")

    # Show some example paths
    print(f"\nExample file paths found:")
    for i, filepath in enumerate(out_files[:5]):
        rel_path = str(Path(filepath).relative_to(base_path))
        print(f"  {i+1}. {rel_path}")
    if len(out_files) > 5:
        print(f"  ... and {len(out_files) - 5} more files")

    print(f"\nProcessing files...")

    # Process all files
    all_results = []
    success_count = 0
    failed_files = []

    # Process in batches to show progress
    batch_size = 50
    total_batches = (len(out_files) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(out_files))
        batch_files = out_files[start_idx:end_idx]

        print(f"  Batch {batch_num + 1}/{total_batches}: Processing files {start_idx + 1}-{end_idx}")

        for i, filepath in enumerate(batch_files):
            result = extract_descriptors_from_file(filepath)
            all_results.append(result)

            if result["status"] == "success":
                success_count += 1
            else:
                failed_files.append(filepath)

            # Show progress within batch
            if (i + 1) % 10 == 0 or (i + 1) == len(batch_files):
                current_total = start_idx + i + 1
                print(f"    Progress: {current_total}/{len(out_files)} files processed")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        "molecule_id",
        "filename",
        "folder_path",
        "status",
        "homo_lumo_gap_ev",
        "final_single_point_energy_hartree",
        "homo_energy_ev",
        "lumo_energy_ev",
        "dispersion_energy_d4_hartree",
        "chemical_hardness_eta_ev",
        "electrophilicity_index_omega_ev",
        "dipole_moment_magnitude_debye",
        "electronegativity_chi_ev",
        "gcp_correction_hartree",
        "quality_flag",
        "relative_path",
    ]

    # Reorder columns (only include existing ones)
    available_columns = [col for col in column_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in available_columns]
    final_columns = available_columns + remaining_columns

    df = df[final_columns]

    # Save to CSV
    df.to_csv(output_csv, index=False, float_format="%.8f")

    # Print summary
    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Base directory: {base_path}")
    print(f"Total .out files found: {len(out_files)}")
    print(f"Successful extractions: {success_count}")
    print(f"Failed extractions: {len(out_files) - success_count}")
    print(f"Success rate: {success_count/len(out_files)*100:.1f}%")

    # Show some failed files if any
    if failed_files:
        print(f"\nFirst 5 failed files:")
        for i, filepath in enumerate(failed_files[:5]):
            rel_path = str(Path(filepath).relative_to(base_path))
            print(f"  {i+1}. {rel_path}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more failed files")

    # Show successful data statistics
    successful_df = df[df["status"] == "success"]
    if len(successful_df) > 0:
        print(f"\nDescriptor Statistics (successful extractions only):")

        descriptors_to_show = [
            "homo_lumo_gap_ev",
            "chemical_hardness_eta_ev",
            "electrophilicity_index_omega_ev",
            "dipole_moment_magnitude_debye",
            "gcp_correction_hartree",
        ]

        for desc in descriptors_to_show:
            if desc in successful_df.columns:
                values = successful_df[desc].dropna()
                if len(values) > 0:
                    print(
                        f"  {desc:30s}: {values.mean():8.4f} ± {values.std():6.4f} "
                        f"(range: {values.min():8.4f} to {values.max():8.4f})"
                    )

        # Quality assessment
        if "quality_flag" in successful_df.columns:
            print(f"\nQuality Assessment:")
            quality_counts = successful_df["quality_flag"].value_counts()
            for quality, count in quality_counts.items():
                percentage = count / len(successful_df) * 100
                print(f"  {quality:20s}: {count:4d} files ({percentage:5.1f}%)")

        # Check for problematic gCP values
        if "gcp_correction_hartree" in successful_df.columns:
            large_gcp = successful_df[
                abs(successful_df["gcp_correction_hartree"]) > 0.05
            ]
            if len(large_gcp) > 0:
                print(
                    f"\n⚠️  WARNING: {len(large_gcp)} files have large gCP corrections (>0.05 Hartree)"
                )
                print("   These may indicate basis set artifacts or poor geometries")

    print(f"\nResults saved to: {output_csv}")
    print(f"CSV contains {len(df)} rows and {len(df.columns)} columns")

    # Show first few successful results
    if len(successful_df) > 0:
        print(f"\nFirst 3 successful extractions:")
        cols_to_show = [
            "molecule_id",
            "homo_lumo_gap_ev",
            "chemical_hardness_eta_ev",
            "dipole_moment_magnitude_debye",
        ]
        available_cols = [col for col in cols_to_show if col in successful_df.columns]
        if available_cols:
            print(
                successful_df[available_cols]
                .head(3)
                .to_string(index=False, float_format="%.4f")
            )

    return df


def main():
    """Main function."""

    print("r2SCAN-3c Nested Folder Descriptor Extraction Script")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = DEFAULT_BASE_PATH

    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    else:
        output_csv = DEFAULT_OUTPUT_CSV

    print(f"Base directory: {base_path}")
    print(f"Output CSV: {output_csv}")

    # Check if path exists
    if not os.path.exists(base_path):
        print(f"\n❌ Error: Directory does not exist: {base_path}")
        print("Please check the path and try again.")
        return

    print("Directory exists ✓")
    print()

    # Process files
    df = process_nested_folders(base_path, output_csv)

    if df is not None and len(df) > 0:
        success_count = sum(df["status"] == "success")
        print("\n✅ Extraction completed successfully!")
        print(f"✅ Extracted data from {success_count} files")
        print(f"✅ Results saved to: {output_csv}")
        print("\nTo use this data:")
        print("  import pandas as pd")
        print(f"  df = pd.read_csv('{output_csv}')")
        print("  successful_data = df[df['status'] == 'success']")
    else:
        print("\n❌ Extraction failed or no data extracted!")


if __name__ == "__main__":
    main()

