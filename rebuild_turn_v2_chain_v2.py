from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})

v2 = ROOT / "models_turn_hybrid/AH1S_TURN_HYBRID_V2_LAST.zip"
dataset = ROOT / "results_turn_hybrid/turn_teacher_dataset_v1.npz"
bc = ROOT / "models_turn_hybrid/AH1S_TURN_BC_WARMSTART.zip"

print("=" * 90)
print("REBUILD TURN V2 CHAIN")
print("This reconstructs dataset + BC warm-start, then V2 RL fine-tuning.")
print("=" * 90)

if not dataset.exists() or not bc.exists():
    print("Dataset/BC missing -> running build_turn_hybrid_v1.py")
    run([sys.executable, "build_turn_hybrid_v1.py"])
else:
    print("Dataset and BC already exist -> skipping V1 build")

if not v2.exists():
    print("V2 missing -> running deneme/fine_tune_turn_rl_v2.py")
    run([sys.executable, "deneme/fine_tune_turn_rl_v2.py"])
else:
    print("V2 already exists -> skipping V2 training")

print("\nFINAL STATUS")
print("DATASET:", "VAR" if dataset.exists() else "YOK")
print("BC     :", "VAR" if bc.exists() else "YOK")
print("V2     :", "VAR" if v2.exists() else "YOK")

if v2.exists() and dataset.exists():
    print("NEXT=REBUILD_V3")
else:
    print("NEXT=CHECK_ERROR")
