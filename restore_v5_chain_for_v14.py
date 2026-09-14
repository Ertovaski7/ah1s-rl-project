from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models_turn_hybrid"
RESULTS = ROOT / "results_turn_hybrid"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DATASET = RESULTS / "turn_teacher_dataset_v1.npz"
BC = MODELS / "AH1S_TURN_BC_WARMSTART.zip"
V2 = MODELS / "AH1S_TURN_HYBRID_V2_LAST.zip"
V3 = MODELS / "AH1S_TURN_HYBRID_V3_REPAIRED.zip"
V5 = MODELS / "AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip"


def run(path: Path):
    print("\n" + "=" * 100)
    print("RUNNING:", path)
    print("=" * 100)
    runpy.run_path(str(path), run_name="__main__")

print("RESTORE V5 CHAIN FOR V14")
print("DATASET:", "VAR" if DATASET.exists() else "YOK")
print("BC     :", "VAR" if BC.exists() else "YOK")
print("V2     :", "VAR" if V2.exists() else "YOK")
print("V3     :", "VAR" if V3.exists() else "YOK")
print("V5     :", "VAR" if V5.exists() else "YOK")

# Recreate the teacher dataset + BC warm-start if either is missing.
if not DATASET.exists() or not BC.exists():
    run(ROOT / "build_turn_hybrid_v1.py")

# V2 depends on BC + dataset.
if not V2.exists():
    run(ROOT / "deneme" / "fine_tune_turn_rl_v2.py")

# V3 depends on V2 + dataset.
if not V3.exists():
    run(ROOT / "deneme" / "repair_turn_hybrid_v3.py")

# V5 depends on V3.
if not V5.exists():
    run(ROOT / "repair_turn_hybrid_v5_collective_only.py")

print("\n" + "=" * 100)
print("FINAL CHECK")
print("DATASET:", "VAR" if DATASET.exists() else "YOK")
print("BC     :", "VAR" if BC.exists() else "YOK")
print("V2     :", "VAR" if V2.exists() else "YOK")
print("V3     :", "VAR" if V3.exists() else "YOK")
print("V5     :", "VAR" if V5.exists() else "YOK")

if V5.exists():
    print("NEXT=RUN_V14")
else:
    print("NEXT=STOP_V5_MISSING")
    raise SystemExit(2)
