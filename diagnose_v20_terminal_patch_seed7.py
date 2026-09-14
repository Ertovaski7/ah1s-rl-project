from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 7
TARGET = 50.0
TARGET_50_NORM = 50.0 / 360.0
GATE_TOL = 0.02
TERM_ENTER_DEG = 6.0

BASE_MODEL = v16.BASE_MODEL
BASE_ADAPTER = v16.BASE_ADAPTER
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

def is_target(obs):
    return abs(float(obs[12]) - TARGET_50_NORM) <= GATE_TOL

for p in (BASE_MODEL, BASE_ADAPTER, PATCH_50, TERM_50):
    if not p.exists():
        raise FileNotFoundError(p)

base_model = PPO.load(str(BASE_MODEL))
base_adapter = load_adapter(BASE_ADAPTER, v16.BASE_SCALE)
patch50 = load_adapter(PATCH_50, v16.PATCH_SCALE)
term50 = TerminalPatch()
term50.load_state_dict(torch.load(TERM_50, map_location="cpu"), strict=True)
term50.eval()
for p in term50.parameters():
    p.requires_grad = False

env = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
obs, reset_info = env.reset(seed=SEED)
terminal_latched = False
first_latch = None
info = reset_info

print("="*120)
print("V20 SEED7 TERMINAL PATCH DIAGNOSTIC")
print("No training. Logs V17 action, V19 lateral delta and final action through terminal region.")
print("="*120)
print(f"entry extra={reset_info.get('extra_entry_steps')} alt={reset_info.get('altitude'):.3f} vs={reset_info.get('vertical_speed'):+.3f} v={reset_info.get('forward_velocity'):.3f}")

try:
    for k in range(int(110.0/env.CONTROL_DT)):
        t = k * env.CONTROL_DT
        rem = TARGET - env.cumulative_turn_deg
        if (not terminal_latched) and abs(rem) <= TERM_ENTER_DEG:
            terminal_latched = True
            first_latch = (t, rem)
            print(f"\nLATCH at t={t:.2f}s rem={rem:+.3f}\n")

        base, _ = base_model.predict(obs, deterministic=True)
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            d0 = base_adapter(x).cpu().numpy()[0]
            d50 = patch50(x).cpu().numpy()[0]
            dt = term50(x).cpu().numpy()[0] if terminal_latched and is_target(obs) else np.zeros(2, np.float32)
        a_v17 = np.clip(np.asarray(base, np.float32) + d0 + d50, -1.0, 1.0)
        action = a_v17.copy()
        action[2] += float(dt[0]); action[3] += float(dt[1])
        action = np.clip(action, -1.0, 1.0)

        s = env._raw_state()
        if terminal_latched and (k % 5 == 0 or abs(rem) <= 2.0):
            print(
                f"t={t:5.1f} rem={rem:+6.2f} alt={float(s['altitude']):7.2f} vs={float(s['vertical_speed']):+5.2f} "
                f"roll={math.degrees(float(s['roll'])):+6.2f} yaw_rate={math.degrees(float(s['r_rate'])):+6.2f} "
                f"v17_a2={a_v17[2]:+6.3f} v17_a3={a_v17[3]:+6.3f} "
                f"v19_da2={dt[0]:+7.4f} v19_da3={dt[1]:+7.4f} "
                f"final_a2={action[2]:+6.3f} final_a3={action[3]:+6.3f} hold={env.success_hold_s:3.1f}"
            )

        obs, _, term, trunc, info = env.step(action)
        obs = np.asarray(obs, np.float32)
        if term or trunc:
            print("\nEND", f"pass={bool(info.get('success', False))}", f"done={float(info.get('cumulative_turn_deg',0)):+.2f}", f"rem={float(info.get('remaining_turn_deg',999)):+.2f}", f"hold={float(info.get('success_hold_s',0)):.1f}", f"safety={bool(info.get('safety_failure',False))}")
            break
finally:
    env.close()
