from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

V5 = ROOT / "models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip"
V4 = ROOT / "models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt"
V7 = ROOT / "models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt"
V13 = ROOT / "models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt"
V13_SCRIPT = ROOT / "train_turn_full_entry_v13_gated_50_patch.py"


def run(script):
    print("\n" + "=" * 100)
    print("RUNNING:", script)
    print("=" * 100)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def status():
    print("V5 :", "VAR" if V5.exists() else "YOK")
    print("V4 :", "VAR" if V4.exists() else "YOK")
    print("V7 :", "VAR" if V7.exists() else "YOK")
    print("V13:", "VAR" if V13.exists() else "YOK")

print("RESTORE POST-V5 CHAIN FOR V14")
status()

if not V5.exists():
    raise FileNotFoundError("V5 yok. Once restore_v5_chain_for_v14.py calistirilmali.")

if not V4.exists():
    run("train_turn_live_entry_v12_residual_all_targets.py")

if not V7.exists():
    run("train_turn_full_entry_v7_gated_200_patch.py")

if not V13_SCRIPT.exists():
    run("make_v13_from_v7.py")

if not V13.exists():
    run("train_turn_full_entry_v13_gated_50_patch.py")

print("\n" + "=" * 100)
print("FINAL CHECK")
status()
if all(p.exists() for p in (V5, V4, V7, V13)):
    print("NEXT=RUN_V14")
else:
    print("CHAIN_INCOMPLETE")
    raise SystemExit(2)
