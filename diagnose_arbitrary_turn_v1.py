from __future__ import annotations

"""
ARBITRARY TURN DIAGNOSTIC V1
============================

Current verified V22 runtime stack'i DEĞİŞTİRMEDEN unseen açıların neden
başarısız olduğunu zaman çizelgesiyle gösterir.

Varsayılan:
    python diagnose_arbitrary_turn_v1.py 20 -30

Runtime teacher/controller OFF. Model ağırlıkları değişmez.
"""

import argparse
import math

import numpy as np
import torch
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float, default=[20.0, -30.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=float, default=1.0)
    return p.parse_args()


def load_stack():
    bm = PPO.load(str(v16.BASE_MODEL))
    ba = v22.load(v16.BASE_ADAPTER, v16.BASE_SCALE)
    p200 = v22.load(v16.PATCH_200, v16.PATCH_SCALE)
    p50 = v22.load(v22.P50, v16.PATCH_SCALE)
    p21 = v22.TP()
    p21.load_state_dict(torch.load(v22.P21, map_location="cpu"), strict=True)
    p21.eval()
    for p in p21.parameters():
        p.requires_grad = False
    return bm, ba, p200, p50, p21


def deg(x):
    return math.degrees(float(x))


def safety_reasons(info):
    reasons = []
    alt = float(info.get("altitude", np.nan))
    roll = abs(deg(info.get("roll", 0.0)))
    pitch = abs(deg(info.get("pitch", 0.0)))
    yaw = abs(deg(info.get("r_rate", 0.0)))
    speed = float(info.get("forward_velocity", np.nan))
    if np.isfinite(alt) and alt < 275.0: reasons.append(f"ALT_LOW({alt:.2f})")
    if np.isfinite(alt) and alt > 325.0: reasons.append(f"ALT_HIGH({alt:.2f})")
    if roll > 18.0: reasons.append(f"ROLL({roll:.2f})")
    if pitch > 15.0: reasons.append(f"PITCH({pitch:.2f})")
    if yaw > 30.0: reasons.append(f"YAWRATE({yaw:.2f})")
    if np.isfinite(speed) and speed < 1.0: reasons.append(f"SPEED({speed:.2f})")
    return reasons


def run_target(stack, target, seed, print_every):
    bm, ba, p200, p50, p21 = stack
    env = v16.RandomizedEntryEnv(target_turn_deg=float(target))
    obs, info = env.reset(seed=seed)
    dt = float(env.CONTROL_DT)
    next_print = 0.0
    prev_rem = float(target)
    crossed = False

    print("\n" + "=" * 150)
    print(f"DIAG target={target:+.1f} deg | seed={seed} | extra_entry_steps={int(info.get('extra_entry_steps', -1))}")
    print("Runtime path: V5 + V4 plus existing V22 hard-gated specialists if target falls in their gate.")
    print("=" * 150)
    print(" t(s) | cumulative | remaining |   alt |    VS | speed |  roll | pitch | yawRate | hold |   a0    a1    a2    a3 | note")
    print("-" * 150)

    max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / dt)
    final_info = info
    try:
        for k in range(max_steps):
            action = v22.act(bm, ba, p200, p50, p21, obs)
            obs, _, term, trunc, final_info = env.step(action)
            obs = np.asarray(obs, np.float32)
            t = (k + 1) * dt
            rem = float(final_info.get("remaining_turn_deg", np.nan))

            note = ""
            if not crossed and np.isfinite(rem) and prev_rem * rem <= 0.0:
                crossed = True
                note = "TARGET_CROSSED"
            if bool(final_info.get("safety_failure", False)):
                rs = safety_reasons(final_info)
                note = "SAFETY:" + (",".join(rs) if rs else "UNKNOWN")
            elif bool(final_info.get("success", False)):
                note = "SUCCESS"
            elif trunc:
                note = "TIMEOUT"

            should_print = t + 1e-9 >= next_print or bool(note) or term or trunc
            if should_print:
                print(
                    f"{t:5.1f} | {float(final_info.get('cumulative_turn_deg', 0.0)):+10.2f} | "
                    f"{rem:+9.2f} | {float(final_info.get('altitude', np.nan)):6.2f} | "
                    f"{float(final_info.get('vertical_speed', np.nan)):+6.2f} | "
                    f"{float(final_info.get('forward_velocity', np.nan)):5.2f} | "
                    f"{deg(final_info.get('roll', 0.0)):+5.1f} | {deg(final_info.get('pitch', 0.0)):+5.1f} | "
                    f"{deg(final_info.get('r_rate', 0.0)):+7.2f} | "
                    f"{float(final_info.get('success_hold_s', 0.0)):4.1f} | "
                    f"{float(action[0]):+5.2f} {float(action[1]):+5.2f} {float(action[2]):+5.2f} {float(action[3]):+5.2f} | {note}"
                )
                while next_print <= t + 1e-9:
                    next_print += print_every

            prev_rem = rem
            if term or trunc:
                break
    finally:
        env.close()

    print("-" * 150)
    print(
        f"FINAL target={target:+.1f} | success={bool(final_info.get('success', False))} | "
        f"done={float(final_info.get('cumulative_turn_deg', 0.0)):+.2f} | "
        f"rem={float(final_info.get('remaining_turn_deg', 999.0)):+.2f} | "
        f"alt={float(final_info.get('altitude', np.nan)):.2f} | "
        f"hold={float(final_info.get('success_hold_s', 0.0)):.1f} | "
        f"safety={bool(final_info.get('safety_failure', False))} | "
        f"reasons={safety_reasons(final_info)}"
    )


def main():
    args = parse_args()
    stack = load_stack()
    for target in args.targets:
        run_target(stack, float(target), int(args.seed), float(args.print_every))


if __name__ == "__main__":
    main()
