from __future__ import annotations

"""
CURRENT TURN STACK RECOVERY / REBUILD
=====================================

Fresh clone durumunda kayip models_turn_hybrid checkpoint zincirini repository'deki
mevcut egitim/onarim scriptlerinden yeniden kurar.

ONEMLI:
- Var olan checkpoint'i SILMEZ / EZMEZ; sadece eksik adimlari calistirir.
- git reset / clone / rm yapmaz.
- Bu bir "exact binary recovery" degildir. Eski local checkpoint GitHub'a hic
  push edilmediyse ayni byte'lari geri getiremez. Ama mevcut kaynak kod ve sabit
  seed'lerle turn stack'i yeniden uretmek icin resmi recovery zinciridir.
- Her ana adimdan sonra beklenen dosyanin olustugunu kontrol eder.
- Sonunda V22 robustness testini calistirir.

Kullanim:
    python restore_current_turn_stack_v1.py

Uzun surebilir; ozellikle BC/RL/V3/V5/V4/V7/V13/V17/V21 egitimleri CPU'da zaman alir.
Runtime teacher final testte OFF'tur; teacher yalnizca yeniden egitim adimlarinda
supervision/veri uretimi icin kullanilabilir.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT) + os.pathsep + ENV.get("PYTHONPATH", "")

MODELS = ROOT / "models_turn_hybrid"
RESULTS = ROOT / "results_turn_hybrid"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DATASET = RESULTS / "turn_teacher_dataset_v1.npz"
BC = MODELS / "AH1S_TURN_BC_WARMSTART.zip"
V2 = MODELS / "AH1S_TURN_HYBRID_V2_LAST.zip"
V3 = MODELS / "AH1S_TURN_HYBRID_V3_REPAIRED.zip"
V5 = MODELS / "AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip"
V4 = MODELS / "AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt"
V7 = MODELS / "AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt"
V13 = MODELS / "AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt"
V17 = MODELS / "AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt"
V21 = MODELS / "AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH.pt"
V13_SCRIPT = ROOT / "train_turn_full_entry_v13_gated_50_patch.py"


def banner(text: str):
    print("\n" + "=" * 120, flush=True)
    print(text, flush=True)
    print("=" * 120, flush=True)


def run(script: str):
    path = ROOT / script
    if not path.exists():
        raise FileNotFoundError(f"Recovery script bulunamadi: {path}")
    banner(f"RUNNING: {script}")
    subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=ENV, check=True)


def need(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} uretilmedi: {path}")
    print(f"OK  {label}: {path}", flush=True)


def status():
    items = [
        ("DATASET", DATASET), ("BC", BC), ("V2", V2), ("V3", V3),
        ("V5", V5), ("V4", V4), ("V7", V7), ("V13", V13),
        ("V17", V17), ("V21", V21),
    ]
    print("\nCHECKPOINT STATUS")
    for name, p in items:
        print(f"  {name:7s}: {'VAR' if p.exists() else 'YOK'}  {p.relative_to(ROOT)}")


banner("AH-1S TURN STACK RECOVERY V1")
print("Root:", ROOT)
print("Mevcut dosyalar korunacak; sadece eksikler yeniden uretilecek.")
status()

# 1) Teacher dataset + behavior cloning warm-start.
if not DATASET.exists() or not BC.exists():
    run("build_turn_hybrid_v1.py")
need(DATASET, "teacher dataset")
need(BC, "BC warm-start")

# 2) PPO fine-tune V2.
if not V2.exists():
    run("deneme/fine_tune_turn_rl_v2.py")
need(V2, "V2 PPO")

# 3) Post-RL repair V3.
if not V3.exists():
    run("deneme/repair_turn_hybrid_v3.py")
need(V3, "V3 repaired")

# 4) V5 collective-only repair: general turn base.
if not V5.exists():
    run("repair_turn_hybrid_v5_collective_only.py")
need(V5, "V5 base turn PPO")

# 5) Live-entry residual adapter V4 (script name V12, output intentionally V4).
if not V4.exists():
    run("train_turn_live_entry_v12_residual_all_targets.py")
need(V4, "V4 live-entry residual")

# 6) +200 specialist V7.
if not V7.exists():
    run("train_turn_full_entry_v7_gated_200_patch.py")
need(V7, "V7 +200 patch")

# 7) Generate +50 V13 training script from V7 template if source file is absent.
if not V13_SCRIPT.exists():
    run("make_v13_from_v7.py")
need(V13_SCRIPT, "V13 training script")

# 8) +50 V13 specialist.
if not V13.exists():
    run("train_turn_full_entry_v13_gated_50_patch.py")
need(V13, "V13 +50 patch")

# 9) Robust +50 V17. Historical best can remain equivalent to V13; that is OK.
if not V17.exists():
    run("train_turn_full_entry_v17_robust_50_patch.py")
need(V17, "V17 robust +50 patch")

# 10) Strong terminal +50 V21.
if not V21.exists():
    run("train_turn_full_entry_v21_strong_terminal_50_patch.py")
need(V21, "V21 terminal +50 patch")

status()

# 11) Final required baseline validation before arbitrary-angle testing.
banner("FINAL CHECK: V22 RANDOMIZED-ENTRY ROBUSTNESS")
run("test_turn_full_entry_v22_v21_runtime.py")

banner("RECOVERY PIPELINE FINISHED")
print("V22 sonucu PASS ise sonraki komut:")
print("  python test_turn_arbitrary_angles_v1.py 20 -30")
print("Checkpoint'leri tekrar kaybetmemek icin V22 PASS sonrasinda models_turn_hybrid/ klasorunu kalici backup'a alin.")
