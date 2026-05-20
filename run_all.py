# run_all.py
# End-to-end runner for PAMAP2 experiments (centralized + FL + audits)

import os
import sys
import subprocess
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("CMD:", " ".join(cmd))
    print("=" * 100)
    p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(f"\n❌ Failed: {' '.join(cmd)}")
    print("✅ Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", default=sys.executable, help="Python executable")
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_central_embed", action="store_true")
    parser.add_argument("--skip_subject_embed", action="store_true")
    parser.add_argument("--skip_run1", action="store_true")
    parser.add_argument("--skip_run2", action="store_true")
    parser.add_argument("--skip_run3", action="store_true")
    parser.add_argument("--skip_run4", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    args = parser.parse_args()

    print("Project root:", ROOT)
    print("Python:", args.py)
    print("Start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ----------------------------------------------------
    # 1) Build X/y/subjects (per_subject normalization)
    # ----------------------------------------------------
    if not args.skip_data:
        run_cmd([args.py, "dataloader.py"])

    # ----------------------------------------------------
    # 2) Centralized MANTIS embeddings -> data/X_mantis.npy
    # ----------------------------------------------------
    if not args.skip_central_embed:
        run_cmd([args.py, "-m", "experiments.extract_mantis_embeddings"])

    # ----------------------------------------------------
    # 3) Per-subject embedding cache -> data/mantis_subject_cache/*.npz
    # ----------------------------------------------------
    if not args.skip_subject_embed:
        run_cmd([args.py, "-m", "experiments.extract_mantis_embeddings_per_subject"])

    # ----------------------------------------------------
    # 4) Audits (leakage, normalization, cache checks)
    # ----------------------------------------------------
    if not args.skip_audit:
        run_cmd([args.py, "-m", "experiments.testScript"])

    # ----------------------------------------------------
    # 5) Run 1: Centralized baselines
    # ----------------------------------------------------
    if not args.skip_run1:
        run_cmd([args.py, "run.py"])

    # ----------------------------------------------------
    # 6) Run 2: Centralized MANTIS heads
    # ----------------------------------------------------
    if not args.skip_run2:
        run_cmd([args.py, "runMantisembeddings.py"])

    # ----------------------------------------------------
    # 7) Run 3: FedAvg baselines
    # ----------------------------------------------------
    if not args.skip_run3:
        run_cmd([args.py, "run_fedavg_baseline.py"])

    # ----------------------------------------------------
    # 8) Run 4: FedAvg MANTIS heads
    # ----------------------------------------------------
    if not args.skip_run4:
        run_cmd([args.py, "run_fedavg_mantis.py"])

    print("\n✅✅ PIPELINE COMPLETE ✅✅")
    print("End:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\nTip:")
    print("  python run_all.py > logs/all.txt 2>&1")

if __name__ == "__main__":
    main()
