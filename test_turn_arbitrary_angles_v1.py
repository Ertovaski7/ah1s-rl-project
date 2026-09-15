from __future__ import annotations

"""
ARBITRARY RELATIVE-TURN SMOKE / ROBUSTNESS TEST
===============================================

Amaç
----
Mevcut, doğrulanmış V22 turn runtime stack'ini değiştirmeden; eğitim/doğrulama
setinin dışında kalan relative-turn komutlarında nasıl genelleştiğini ölçmek.

Örnek kullanım:
    python test_turn_arbitrary_angles_v1.py
    python test_turn_arbitrary_angles_v1.py 20 -30 75 -90 120 150
    python test_turn_arbitrary_angles_v1.py 20 -30 --seeds 7 21 42
    python test_turn_arbitrary_angles_v1.py 20 -30 --robust

Önemli
------
Bu dosya henüz multi-turn / aynı FDM üzerinde ardışık komut testi değildir.
Her target ayrı bir full-entry koşusudur. Önce arbitrary-angle genellemesini
ölçüyoruz; daha sonra aynı canlı FDM üzerinde +20 -> -30 -> ... komut zincirine
geçeceğiz.

Runtime teacher/controller OFF.
Model ağırlıkları değiştirilmez.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22


DEFAULT_TARGETS = [20.0, -30.0, 75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]
ROBUST_SEEDS = [7, 21, 42, 84, 123]


def parse_args():
    p = argparse.ArgumentParser(
        description="V22 stack ile arbitrary relative-turn genelleme testi"
    )
    p.add_argument(
        "targets",
        nargs="*",
        type=float,
        help="Relative turn açıları. Örn: 20 -30 75 -90",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="İsteğe bağlı seed listesi. Örn: --seeds 7 21 42",
    )
    p.add_argument(
        "--robust",
        action="store_true",
        help="5-seed fiziksel randomized-entry robustness testi çalıştır",
    )
    return p.parse_args()


def specialist_name(target_deg: float) -> str:
    target_norm = float(target_deg) / 360.0
    if abs(target_norm - v22.T50) <= v16.GATE_TOL:
        return "V17 + V21 (+50 specialist)"
    if abs(target_norm - v22.T200) <= v16.GATE_TOL:
        return "V7 (+200 specialist)"
    return "V5 + V4 base/generalization"


def require_models():
    required = [
        v16.BASE_MODEL,
        v16.BASE_ADAPTER,
        v16.PATCH_200,
        v22.P50,
        v22.P21,
    ]
    missing = [Path(p) for p in required if not Path(p).exists()]
    if missing:
        print("\nEksik model/checkpoint var:")
        for p in missing:
            print("  YOK:", p)
        raise FileNotFoundError(
            "Arbitrary-angle testi başlamadı. Doğrulanmış checkpoint'leri geri yükleyin."
        )


def load_stack():
    require_models()

    base_model = PPO.load(str(v16.BASE_MODEL))
    base_adapter = v22.load(v16.BASE_ADAPTER, v16.BASE_SCALE)
    patch_200 = v22.load(v16.PATCH_200, v16.PATCH_SCALE)
    patch_50 = v22.load(v22.P50, v16.PATCH_SCALE)

    terminal_50 = v22.TP()
    terminal_50.load_state_dict(torch.load(v22.P21, map_location="cpu"), strict=True)
    terminal_50.eval()
    for p in terminal_50.parameters():
        p.requires_grad = False

    return base_model, base_adapter, patch_200, patch_50, terminal_50


def run_one(stack, target: float, seed: int):
    bm, ba, p200, p50, p21 = stack
    try:
        row = v22.one(bm, ba, p200, p50, p21, float(target), int(seed))
        row["error"] = ""
        return row
    except Exception as exc:
        return {
            "seed": int(seed),
            "target": float(target),
            "extra": -1,
            "ok": False,
            "done": float("nan"),
            "rem": float("nan"),
            "alt": float("nan"),
            "hold": 0.0,
            "safe": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    args = parse_args()
    targets = args.targets if args.targets else DEFAULT_TARGETS

    if args.robust and args.seeds is not None:
        raise ValueError("--robust ve --seeds aynı anda kullanılmasın; birini seçin.")
    seeds = ROBUST_SEEDS if args.robust else (args.seeds or DEFAULT_SEEDS)

    if any(abs(t) < 1.0 for t in targets):
        raise ValueError("|target| en az 1 derece olmalı.")
    if any(abs(t) > 360.0 for t in targets):
        raise ValueError(
            "İlk arbitrary-angle aşamasında güvenli test aralığını [-360, +360] ile sınırlıyoruz."
        )

    print("=" * 124)
    print("ARBITRARY RELATIVE-TURN TEST — CURRENT V22 RUNTIME STACK")
    print("Runtime teacher/controller: OFF")
    print("Bu test model ağırlıklarını değiştirmez.")
    print("Her target ayrı full-entry koşusudur; multi-turn SAME-FDM test bir sonraki aşamadır.")
    print("Targets:", [float(x) for x in targets])
    print("Seeds  :", [int(x) for x in seeds])
    print("=" * 124)

    stack = load_stack()
    rows = []

    for target in targets:
        print("\n" + "-" * 124)
        print(f"TARGET {target:+.1f} deg | runtime path: {specialist_name(target)}")
        print("-" * 124)

        for seed in seeds:
            r = run_one(stack, target, seed)
            rows.append(r)

            if r["error"]:
                print(
                    f"seed={seed:3d} | target={target:+7.1f} | PASS=False | "
                    f"ERROR={r['error']}"
                )
            else:
                print(
                    f"seed={seed:3d} | target={target:+7.1f} | extra={r['extra']:2d} | "
                    f"PASS={str(r['ok']):5s} | done={r['done']:+8.2f} | "
                    f"rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | "
                    f"hold={r['hold']:4.1f} | safety={r['safe']}"
                )

    print("\n" + "=" * 124)
    print("ARBITRARY-ANGLE SUMMARY")
    print("=" * 124)

    all_ok = True
    for target in targets:
        part = [r for r in rows if abs(r["target"] - target) < 1e-9]
        passes = sum(int(r["ok"]) for r in part)
        safety = sum(int(r["safe"]) for r in part)
        valid_rem = [abs(r["rem"]) for r in part if np.isfinite(r["rem"])]
        max_err = max(valid_rem) if valid_rem else float("nan")
        errors = sum(bool(r["error"]) for r in part)

        target_ok = passes == len(part) and safety == 0 and errors == 0
        all_ok = all_ok and target_ok

        print(
            f"target={target:+7.1f} | pass={passes}/{len(part)} | "
            f"safety_fail={safety}/{len(part)} | errors={errors} | "
            f"max_abs_remaining={max_err:.2f} deg | path={specialist_name(target)}"
        )

    total_pass = sum(int(r["ok"]) for r in rows)
    total_safety = sum(int(r["safe"]) for r in rows)
    total_errors = sum(bool(r["error"]) for r in rows)

    print("\n" + "=" * 124)
    print(
        f"TOTAL: PASS={total_pass}/{len(rows)} | "
        f"SAFETY_FAILURES={total_safety}/{len(rows)} | ERRORS={total_errors}"
    )
    print("ARBITRARY-ANGLE SMOKE:", "PASS" if all_ok else "PARTIAL/FAIL")
    print("=" * 124)

    if all_ok:
        print(
            "Sonraki adım: Stage1'i yalnızca bir kez çalıştırıp aynı canlı FDM üzerinde "
            "+20 -> transition -> forward -> -30 -> transition -> forward şeklinde "
            "ardışık komut kabul eden mission controller oluşturmak."
        )
    else:
        print(
            "En az bir unseen açı başarısız. Multi-turn arayüze geçmeden önce hangi açı/işaret "
            "bölgesinin genellemediğini bu çıktıyla belirleyip policy/controller katmanını onaracağız."
        )


if __name__ == "__main__":
    main()
