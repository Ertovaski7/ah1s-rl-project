from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16

# Evaluate the exact V18 runtime, but inspect +50 for the one failing randomized entry
# and compare it with passing entries. No training and no runtime teacher/controller.
v16.PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt")

SEEDS = [7, 21, 42, 84, 123]
TARGET = 50.0


def deg(x):
    return math.degrees(float(x))


def load_all():
    for p in (v16.BASE_MODEL, v16.BASE_ADAPTER, v16.PATCH_200, v16.PATCH_50):
        v16.require_file(p)
    base_model = PPO.load(str(v16.BASE_MODEL))
    base_adapter = v16.load_adapter(v16.BASE_ADAPTER, v16.BASE_SCALE)
    patch_200 = v16.load_adapter(v16.PATCH_200, v16.PATCH_SCALE)
    patch_50 = v16.load_adapter(v16.PATCH_50, v16.PATCH_SCALE)
    return base_model, base_adapter, patch_200, patch_50


def inspect_seed(base_model, base_adapter, patch_200, patch_50, seed, verbose=False):
    env = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    obs, reset_info = env.reset(seed=seed)
    s0 = env._raw_state()

    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    base, _ = base_model.predict(obs, deterministic=True)
    with torch.no_grad():
        d0 = base_adapter(x).cpu().numpy()[0]
        d50 = patch_50(x).cpu().numpy()[0]
    initial = np.clip(np.asarray(base, np.float32) + d0 + d50, -1.0, 1.0)

    print("\n" + "=" * 118)
    print(
        f"SEED={seed} extra={int(reset_info.get('extra_entry_steps', 0))} | "
        f"alt={float(s0['altitude']):.3f} vs={float(s0['vertical_speed']):+.3f} "
        f"v={float(s0['forward_velocity']):.3f} roll={deg(s0['roll']):+.3f} "
        f"pitch={deg(s0['pitch']):+.3f} yaw_rate={deg(s0['r_rate']):+.3f}"
    )
    print("base action :", np.array2string(np.asarray(base), precision=4, floatmode='fixed'))
    print("V4 residual:", np.array2string(d0, precision=4, floatmode='fixed'))
    print("V17 patch  :", np.array2string(d50, precision=4, floatmode='fixed'))
    print("stacked    :", np.array2string(initial, precision=4, floatmode='fixed'))

    success = False
    info = reset_info
    next_print = 0.0
    max_steps = int(max(90.0, TARGET / 1.10 + 70.0) / env.CONTROL_DT)
    try:
        for step in range(max_steps):
            action, gate = v16.combined_action(base_model, base_adapter, patch_200, patch_50, obs)
            obs, _, terminated, truncated, info = env.step(action)
            t = (step + 1) * env.CONTROL_DT
            if verbose and t + 1e-9 >= next_print:
                st = env._raw_state()
                print(
                    f"t={t:6.1f} done={float(info.get('cumulative_turn_deg',0)):+7.2f} "
                    f"rem={float(info.get('remaining_turn_deg',999)):+7.2f} "
                    f"alt={float(st['altitude']):7.2f} vs={float(st['vertical_speed']):+6.2f} "
                    f"v={float(st['forward_velocity']):6.2f} roll={deg(st['roll']):+6.2f} "
                    f"yaw_rate={deg(st['r_rate']):+6.2f} hold={float(info.get('success_hold_s',0)):.1f}"
                )
                next_print += 2.0
            if terminated or truncated:
                success = bool(info.get("success", False))
                print(
                    f"END seed={seed} pass={success} terminated={terminated} truncated={truncated} | "
                    f"done={float(info.get('cumulative_turn_deg',0)):+.2f} rem={float(info.get('remaining_turn_deg',999)):+.2f} "
                    f"alt={float(info.get('altitude',np.nan)):.2f} hold={float(info.get('success_hold_s',0)):.1f} "
                    f"safety={bool(info.get('safety_failure',False))}"
                )
                break
    finally:
        env.close()


if __name__ == "__main__":
    print("V18 +50 SEED-7 DIAGNOSTIC")
    print("No training. Runtime teacher/controller OFF.")
    base_model, base_adapter, patch_200, patch_50 = load_all()
    for seed in SEEDS:
        inspect_seed(base_model, base_adapter, patch_200, patch_50, seed, verbose=(seed in (7, 21)))
