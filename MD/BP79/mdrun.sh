#!/usr/bin/env bash
###############################################################################
# MD Pipeline
# - CPU: Minimization (sander)
# - GPU: Heating / Equilibration / Production (pmemd.cuda)
#
# Script location:
# /home/ws2/Research_temp/Simulation/mdrun.sh
#
# Expected structure:
# Simulation/
# ├── mdrun.sh
# ├── positive/
# │   └── amber/
# └── ligand/
#     └── amber/
###############################################################################

set -e

# =============================
# RESOLVE SCRIPT LOCATION
# =============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"

echo "Running MD pipeline from:"
echo "BASE_DIR = ${BASE_DIR}"
echo

# =============================
# GPU SETTINGS (GPU steps only)
# =============================
export CUDA_VISIBLE_DEVICES=0
export AMBER_GPU_MEM=YES

echo "=================================================="
echo " GPU STATUS (verify before run)"
echo "=================================================="
nvidia-smi
echo "=================================================="
echo

# =============================
# SYSTEM LIST
# =============================
SYSTEMS=("ligand_s")

# =============================
# MD ENGINES
# =============================
CPU_ENGINE="pmemd.cuda"
GPU_ENGINE="pmemd.cuda"

# =============================
# MD RUN FUNCTION
# =============================
run_md() {
    local ENGINE="$1"
    local SYSTEM="$2"
    local STEP="$3"
    local MDIN="$4"
    local INPCRD="$5"
    local PREFIX="$6"
    local REF="$7"

    echo "--------------------------------------------------"
    echo ">>> ${SYSTEM} | ${STEP} | ENGINE: ${ENGINE}"
    echo "--------------------------------------------------"

    $ENGINE -O \
        -i "${INPUT_DIR}/${MDIN}" \
        -p "${RUN_DIR}/solvated.parm7" \
        -c "${RUN_DIR}/${INPCRD}" \
        -r "${RUN_DIR}/${PREFIX}.rst7" \
        -o "${RESULTS_DIR}/${PREFIX}.out" \
        -x "${RESULTS_DIR}/${PREFIX}.nc" \
        -inf "${RESULTS_DIR}/${PREFIX}.mdinfo" \
        ${REF:+-ref "${RUN_DIR}/${REF}"}

    echo ">>> ${SYSTEM} | ${STEP} COMPLETED"
    echo
}

# =============================
# MAIN LOOP
# =============================
for SYSTEM in "${SYSTEMS[@]}"; do

    echo "=================================================="
    echo " STARTING MD PIPELINE FOR: ${SYSTEM}"
    echo "=================================================="

    SYSTEM_DIR="${BASE_DIR}/${SYSTEM}"
    INPUT_DIR="${SYSTEM_DIR}/amber"
    RUN_DIR="${SYSTEM_DIR}/run"
    RESULTS_DIR="${SYSTEM_DIR}/results"

    mkdir -p "${RUN_DIR}" "${RESULTS_DIR}"

    echo "Preparing input files..."
    cp "${INPUT_DIR}/solvated.parm7" "${RUN_DIR}/"
    cp "${INPUT_DIR}/solvated.rst7"  "${RUN_DIR}/"
    echo "Input preparation completed."
    echo

    # -----------------------------
    # MD STAGES
    # -----------------------------

    # 1️⃣ Minimization → CPU
    run_md "${CPU_ENGINE}" "${SYSTEM}" "Minimization" \
        "min.mdin" "solvated.rst7" "min" "solvated.rst7"

    # 2️⃣ Heating → GPU
    run_md "${GPU_ENGINE}" "${SYSTEM}" "Heating" \
        "heat.mdin" "min.rst7" "heat" "solvated.rst7"

    # 3️⃣ Equilibration → GPU
    run_md "${GPU_ENGINE}" "${SYSTEM}" "Equilibration" \
        "equil.mdin" "heat.rst7" "equil" "solvated.rst7"

    # 4️⃣ Production → GPU
    run_md "${GPU_ENGINE}" "${SYSTEM}" "Production" \
        "prod.mdin" "equil.rst7" "prod" ""

    echo "=================================================="
    echo " COMPLETED MD PIPELINE FOR: ${SYSTEM} ✅"
    echo " Results directory: ${RESULTS_DIR}"
    echo "=================================================="
    echo
done

echo "🎉 ALL MD SIMULATIONS COMPLETED SUCCESSFULLY 🎉"
