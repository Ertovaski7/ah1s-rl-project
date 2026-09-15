from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "models_turn_hybrid/AH1S_TURN_HYBRID_V2_LAST.zip"
DATASET = ROOT / "results_turn_hybrid/turn_teacher_dataset_v1.npz"
SCRIPT = ROOT / "deneme/repair_turn_hybrid_v3.py"

print("=" * 90)
print("REBUILD TURN V3 V2")
print("=" * 90)
print("V2:", "VAR" if V2.exists() else "YOK")
print("DATASET:", "VAR" if DATASET.exists() else "YOK")
print("V3_SCRIPT:", "VAR" if SCRIPT.exists() else "YOK")

missing = []
if not V2.exists():
    missing.append("V2")
if not DATASET.exists():
    missing.append("DATASET")
if not SCRIPT.exists():
    missing.append("V3_SCRIPT")

if missing:
    print("MISSING=" + ",".join(missing))
    if "V2" in missing:
        print("NEXT=REBUILD_V2")
    elif "DATASET" in missing:
        print("NEXT=REBUILD_DATASET")
    else:
        print("NEXT=RESTORE_V3_SCRIPT")
    raise SystemExit(0)

print("STARTING_V3_REBUILD")
runpy.run_path(str(SCRIPT), run_name="__main__")
