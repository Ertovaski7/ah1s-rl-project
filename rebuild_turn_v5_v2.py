from pathlib import Path
import runpy
import urllib.request

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "models_turn_hybrid" / "AH1S_TURN_HYBRID_V3_REPAIRED.zip"
REPAIR = ROOT / "repair_turn_hybrid_v5_collective_only.py"
URL = "https://raw.githubusercontent.com/selincyr/ah1s-rl-project/3b9eb3e9d16b6b63cd59aa346b078918f24c3b7b/repair_turn_hybrid_v5_collective_only.py"

print("=" * 90)
print("TURN V5 REBUILD V2")
print("Bu script V5'i yeniden kurmak icin gerekli eski repair scriptini otomatik getirir.")
print("=" * 90)

if not V3.exists():
    print("V3_YOK:", V3)
    print("NEXT=REBUILD_V3")
    raise SystemExit(2)

print("V3_VAR:", V3)
print("Repair script indiriliyor...")
urllib.request.urlretrieve(URL, REPAIR)
print("REPAIR_READY:", REPAIR)

runpy.run_path(str(REPAIR), run_name="__main__")
