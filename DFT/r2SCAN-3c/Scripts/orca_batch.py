import os
import glob
import shutil
import subprocess
from joblib import Parallel, delayed

# ================= CONFIG =================

BASE_DIR = "/media/ws2/Storage/Research/Kaveen/GFN2"
INPUT_DIR = os.path.join(BASE_DIR, "r_inputs")

OUTPUT_ROOT = os.path.join(BASE_DIR, "GFN2-xTB_all")

# Where completed job OUTPUT FOLDERS will be moved
COMPLETED_OUTPUTS_DIR = os.path.join(OUTPUT_ROOT, "completed_outputs")

# Where input files go
COMPLETED_INPUTS_DIR = os.path.join(BASE_DIR, "completed")

ORCA_EXE = "/home/ws2/Software/ORCA/build/orca"

NJOBS_PARALLEL = 56
THREADS_PER_JOB = 1

# =================================================


def safe_move(src: str, dst_dir: str) -> str:
    """Cut–paste src into dst_dir without overwriting."""
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)

    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        shutil.move(src, dst)
        return dst

    i = 1
    while True:
        candidate = os.path.join(dst_dir, f"{name}_{i}{ext}")
        if not os.path.exists(candidate):
            shutil.move(src, candidate)
            return candidate
        i += 1


def run_one_job(inp_path: str) -> dict:
    inp_name = os.path.basename(inp_path)
    job_name = os.path.splitext(inp_name)[0]

    job_out_dir = os.path.join(OUTPUT_ROOT, job_name)
    os.makedirs(job_out_dir, exist_ok=True)

    local_inp = os.path.join(job_out_dir, inp_name)
    shutil.copy2(inp_path, local_inp)

    out_file = os.path.join(job_out_dir, f"{job_name}.out")

    cmd = [ORCA_EXE, inp_name]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"

    try:
        with open(out_file, "w") as f:
            result = subprocess.run(
                cmd,
                cwd=job_out_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env
            )

        if result.returncode != 0:
            with open(out_file, "a") as f:
                f.write(f"\n\n*** ORCA RETURN CODE: {result.returncode} ***\n")
            return {"job": job_name, "status": "ERROR"}

        # SUCCESS:
        # 1) move input
        safe_move(inp_path, COMPLETED_INPUTS_DIR)

        # 2) move OUTPUT FOLDER
        safe_move(job_out_dir, COMPLETED_OUTPUTS_DIR)

        return {"job": job_name, "status": "DONE"}

    except Exception as e:
        with open(out_file, "a") as f:
            f.write(f"\n\n*** JOB FAILED: {repr(e)} ***\n")
        return {"job": job_name, "status": "ERROR"}


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(COMPLETED_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(COMPLETED_INPUTS_DIR, exist_ok=True)

    inp_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.inp")))
    if not inp_files:
        print("No input files found.")
        return

    print(f"Found {len(inp_files)} input files")
    print(f"Running {NJOBS_PARALLEL} jobs in parallel (1 core/job)")

    results = Parallel(n_jobs=NJOBS_PARALLEL, backend="loky")(
        delayed(run_one_job)(p) for p in inp_files
    )

    done = sum(r["status"] == "DONE" for r in results)
    error = len(results) - done

    print("\n===== SUMMARY =====")
    print(f"DONE    : {done}")
    print(f"ERROR   : {error}")
    print(f"Completed outputs moved to: {COMPLETED_OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
