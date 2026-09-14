from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEEDS = [7, 21, 42, 84, 123]
TARGETS = [-50.0, 50.0, 200.0, 360.0]
TARGET_50_NORM = 50.0 / 360.0
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02
TERM_ENTER_DEG = 6.0

BASE_MODEL = v16.BASE_MODEL
BASE_ADAPTER = v16.BASE_ADAPTER
PATCH_200 = v16.PATCH_200
PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt")
TERM_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V19_TERMINAL_50_PATCH.pt")

AIL_SCALE = 0.10
RUD_SCALE = 0.12

class TerminalPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2), nn.Tanh(),
        )
    def forward(self, obs):
        y = self.net(obs)
        scale = torch.tensor([AIL_SCALE, RUD_SCALE], dtype=torch.float32, device=obs.device)
        return y * scale

def load_adapter(path, scale):
    m = v16.ResidualAdapter(scale)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def is_target(obs, target_norm):
    return abs(float(obs[12]) - target_norm) <= GATE_TOL

def action_stack(base_model, base_adapter, p200, p50, term50, obs, terminal_latched):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d0 = base_adapter(x).cpu().numpy()[0]
        dspec = np.zeros(4, dtype=np.float32)
        if is_target(obs, TARGET_50_NORM):
            dspec = p50(x).cpu().numpy()[0]
        elif is_target(obs, TARGET_200_NORM):
            dspec = p200(x).cpu().numpy()[0]
        a = np.asarray(base, np.float32) + d0 + dspec
        if terminal_latched and is_target(obs, TARGET_50_NORM):
            dt = term50(x).cpu().numpy()[0]
            a[2] += float(dt[0])
            a[3] += float(dt[1])
    return np.clip(a, -1.0, 1.0)

def eval_one(base_model, base_adapter, p200, p50, term50, target, seed):
    env = v16.RandomizedEntryEnv(target_turn_deg=target)
    obs, reset_info = env.reset(seed=seed)
    info = reset_info
    success = False
    terminal_latched = False
    first_latch = None
    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for k in range(max_steps):
            if target == 50.0 and (not terminal_latched):
                rem = 50.0 - env.cumulative_turn_deg
                if abs(rem) <= TERM_ENTER_DEG:
                    terminal_latched = True
                    first_latch = (k * env.CONTROL_DT, rem)
            action = action_stack(base_model, base_adapter, p200, p50, term50, obs, terminal_latched)
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()
    return {
        "seed": seed,
        "target": target,
        "extra": int(reset_info.get("extra_entry_steps", 0)),
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", np.nan)),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
        "latched": terminal_latched,
        "first_latch": first_latch,
    }

def main():
    for p in (BASE_MODEL, BASE_ADAPTER, PATCH_200, PATCH_50, TERM_50):
        if not p.exists():
            raise FileNotFoundError(p)

    print("=" * 120)
    print("V20 RANDOMIZED FULL-ENTRY TEST - LATCHED +50 TERMINAL SPECIALIST")
    print("V19 terminal patch activates when +50 first enters |remaining| <= 6 deg and then stays active until episode end.")
    print("-50/+360 unchanged; +200 keeps V7; +50 keeps V17 plus latched V19. Runtime teacher/controller OFF.")
    print("=" * 120)

    base_model = PPO.load(str(BASE_MODEL))
    base_adapter = load_adapter(BASE_ADAPTER, v16.BASE_SCALE)
    p200 = load_adapter(PATCH_200, v16.PATCH_SCALE)
    p50 = load_adapter(PATCH_50, v16.PATCH_SCALE)
    term50 = TerminalPatch()
    term50.load_state_dict(torch.load(TERM_50, map_location="cpu"), strict=True)
    term50.eval()
    for p in term50.parameters():
        p.requires_grad = False

    rows = []
    for seed in SEEDS:
        for target in TARGETS:
            r = eval_one(base_model, base_adapter, p200, p50, term50, target, seed)
            rows.append(r)
            latch = "-" if r["first_latch"] is None else f"t={r['first_latch'][0]:.1f},rem={r['first_latch'][1]:+.2f}"
            print(f"seed={seed:3d} | target={target:+7.1f} | extra={r['extra']:2d} | pass={str(r['success']):5s} | done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | hold={r['hold']:4.1f} | safety={r['safety']} | latch={latch}")

    print("\n" + "=" * 120)
    print("V20 SUMMARY")
    print("=" * 120)
    total = 0
    safety_total = 0
    for target in TARGETS:
        part = [r for r in rows if r["target"] == target]
        p = sum(int(r["success"]) for r in part)
        s = sum(int(r["safety"]) for r in part)
        total += p
        safety_total += s
        print(f"target={target:+7.1f} | pass={p}/5 | safety_fail={s}/5 | max_abs_err={max(abs(r['rem']) for r in part):.2f}")
    print(f"\nRANDOMIZED-ENTRY TOTAL: PASS={total}/{len(rows)} | SAFETY_FAILURES={safety_total}/{len(rows)}")
    print("V20 ROBUSTNESS:", "PASS" if total == len(rows) and safety_total == 0 else "PARTIAL/FAIL")

if __name__ == "__main__":
    main()
