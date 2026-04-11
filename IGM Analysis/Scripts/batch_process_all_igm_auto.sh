#!/bin/bash
# save as: batch_process_all_igm_auto.sh

# Configuration
XYZ_DIR="/home/kaveen/Desktop/IGM/optimised_xyz"
OUTPUT_BASE_DIR="/home/kaveen/Desktop/IGM/outputs"
RESIDUE_ATOM_COUNT=148  # Last 148 atoms are residues

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Counter for progress
total_files=$(ls "$XYZ_DIR"/*.xyz 2>/dev/null | wc -l)
current=0

echo "Found $total_files XYZ files to process"
echo "Residue atoms: Last $RESIDUE_ATOM_COUNT atoms"
echo "Starting IGM analysis..."
echo "================================"

# Loop through all .xyz files
for xyz_file in "$XYZ_DIR"/*.xyz; do
    # Get filename without path and extension
    filename=$(basename "$xyz_file" .xyz)
    
    # Increment counter
    ((current++))
    
    echo "[$current/$total_files] Processing: $filename"
    
    # Read total number of atoms from first line of XYZ file
    total_atoms=$(head -n 1 "$xyz_file")
    
    # Calculate ligand atom range
    ligand_last_atom=$((total_atoms - RESIDUE_ATOM_COUNT))
    
    # Define ranges
    ligand_range="1-${ligand_last_atom}"
    residue_first_atom=$((ligand_last_atom + 1))
    residue_range="${residue_first_atom}-${total_atoms}"
    
    echo "  Total atoms: $total_atoms"
    echo "  Ligand: $ligand_range"
    echo "  Residue: $residue_range"
    
    # Create output directory for this molecule
    output_dir="$OUTPUT_BASE_DIR/$filename"
    mkdir -p "$output_dir"
    
    # Create Multiwfn input commands - FIXED: Added more blank lines and 0 to exit cleanly
    cat > /tmp/multiwfn_igm_temp.txt << EOF
20
10
2
$ligand_range
$residue_range
1

3



0
q
EOF
    
    # Run multiwfn
    cd "$output_dir"
    multiwfn "$xyz_file" < /tmp/multiwfn_igm_temp.txt > "${filename}_igm.log" 2>&1
    
    # Rename output files regardless of multiwfn exit status
    if [ -f "sl2r.cub" ]; then
        mv sl2r.cub "${filename}.sl2r.cub"
        echo "  ✓ Created ${filename}.sl2r.cub"
    else
        echo "  ✗ Missing sl2r.cub"
    fi
    
    if [ -f "dg_inter.cub" ]; then
        mv dg_inter.cub "${filename}.dg_inter.cub"
        echo "  ✓ Created ${filename}.dg_inter.cub"
    else
        echo "  ✗ Missing dg_inter.cub"
    fi
    
    # Copy original XYZ to output folder
    cp "$xyz_file" "$output_dir/${filename}.xyz"
    echo "  ✓ Created ${filename}.xyz"
    
    # Save atom range info to a text file
    cat > "$output_dir/${filename}_atom_info.txt" << INFOEOF
Molecule: $filename
Total atoms: $total_atoms
Ligand atoms: $ligand_range ($ligand_last_atom atoms)
Residue atoms: $residue_range ($RESIDUE_ATOM_COUNT atoms)
INFOEOF
    echo "  ✓ Created ${filename}_atom_info.txt"
    
    # Check if both cube files were created
    if [ -f "${filename}.sl2r.cub" ] && [ -f "${filename}.dg_inter.cub" ]; then
        echo "  ✓ Success! Output in: $output_dir"
    else
        echo "  ⚠ WARNING: Some files missing - check log file"
    fi
    
    echo ""
    
done

# Clean up temp file
rm -f /tmp/multiwfn_igm_temp.txt

echo "================================"
echo "Processing complete!"
echo "Results saved in: $OUTPUT_BASE_DIR"
echo ""
echo "Each molecule folder should contain:"
echo "  - {name}.sl2r.cub          (for coloring)"
echo "  - {name}.dg_inter.cub      (for isosurface)"
echo "  - {name}.xyz               (structure)"
echo "  - {name}_igm.log           (calculation log)"
echo "  - {name}_atom_info.txt     (atom ranges used)"
