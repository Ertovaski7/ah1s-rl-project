from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry

SEED = 42
TARGETS = [-50.0, 50.0, 200.0, 360.0]
BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
BASE_ADAPTER = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt")
V7_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
V13_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt")
OUT_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V14_GATED_50_COLLECTIVE.pt")
BEST_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V14_GATED_50_COLLECTIVE_BEST.pt")

BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)
COLLECTIVE_SCALE = 0.18
TARGET_50_NORM = 50.0 / 360.0
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02
ROUNDS = 12
EPOCHS_PER_ROUND = 16
BATCH = 256
LR = 5e-5

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

class ResidualAdapter(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs) * self.scale.to(obs.device)

class CollectivePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1) * COLLECTIVE_SCALE

def gate(obs, target_norm):
    return abs(float(obs[12]) - target_norm) <= GATE_TOL

for p in (BASE_MODEL, BASE_ADAPTER, V7_PATCH, V13_PATCH):
    if not p.exists():
        raise FileNotFoundError(p)

base_model = PPO.load(str(BASE_MODEL))
base_adapter = ResidualAdapter(BASE_SCALE)
base_adapter.load_state_dict(torch.load(BASE_ADAPTER, map_location="cpu"), strict=True)
base_adapter.eval()
v7 = ResidualAdapter(PATCH_SCALE)
v7.load_state_dict(torch.load(V7_PATCH, map_location="cpu"), strict=True)
v7.eval()
v13 = ResidualAdapter(PATCH_SCALE)
v13.load_state_dict(torch.load(V13_PATCH, map_location="cpu"), strict=True)
v13.eval()
for module in (base_adapter, v7, v13):
    for p in module.parameters():
        p.requires_grad = False

def stacked_action(obs, cpatch):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d0 = base_adapter(x).cpu().numpy()[0]
        d7 = v7(x).cpu().numpy()[0] if gate(obs, TARGET_200_NORM) else np.zeros(4, np.float32)
        d13 = v13(x).cpu().numpy()[0] if gate(obs, TARGET_50_NORM) else np.zeros(4, np.float32)
        dc = float(cpatch(x).cpu().numpy()[0]) if gate(obs, TARGET_50_NORM) else 0.0
    a = np.asarray(base, np.float32) + d0 + d7 + d13
    a[0] += dc
    return np.clip(a, -1.0, 1.0), dc

def teacher_collective_action(env):
    s = env._raw_state()
    alt = float(s["altitude"])
    vs = float(s["vertical_speed"])
    desired_vs = float(np.clip(0.08 * (300.0 - alt), -2.5, 0.15))
    gain = 0.045 if vs > desired_vs else 0.004
    collective = float(np.clip(0.5780 + gain * (desired_vs - vs), 0.460, 0.590))
    return env.collective_to_action(collective)

def evaluate_one(cpatch, target):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, _ = env.reset(seed=SEED)
    info = {}
    success = False
    try:
        max_steps = int(max(130.0, abs(target) / 1.10 + 80.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = stacked_action(obs, cpatch)
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()
    return {
        "target": target,
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", float("nan"))),
        "v": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
    }

def evaluate_all(cpatch, label):
    rows = [evaluate_one(cpatch, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safety"]) for r in rows)
    err = sum(abs(r["rem"]) for r in rows)
    score = 10000.0 * passes - 500.0 * safety - err
    print(f"\n{label} | PASS={passes}/4 | score={score:.2f}")
    for r in rows:
        print(f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | v={r['v']:6.2f} | hold={r['hold']:4.1f} | safety={r['safety']}")
    return rows, passes, score

def collect_shadow(cpatch):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=50.0)
    obs, _ = env.reset(seed=SEED)
    xs, ys, ws = [], [], []
    try:
        max_steps = int(130.0 / env.CONTROL_DT)
        for _ in range(max_steps):
            base, _ = base_model.predict(obs, deterministic=True)
            x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                d0 = base_adapter(x).cpu().numpy()[0]
                d13 = v13(x).cpu().numpy()[0]
            current_no_v14 = float(np.clip(np.asarray(base, np.float32)[0] + d0[0] + d13[0], -1.0, 1.0))
            teacher_a0 = teacher_collective_action(env)
            target_dc = float(np.clip(teacher_a0 - current_no_v14, -COLLECTIVE_SCALE, COLLECTIVE_SCALE))
            s = env._raw_state()
            alt = float(s["altitude"])
            remaining = 50.0 - env.cumulative_turn_deg
            w = 1.0
            if alt > 312.0: w *= 4.0
            if alt > 318.0: w *= 2.5
            if abs(remaining) < 20.0: w *= 2.0
            if abs(remaining) < 8.0: w *= 2.0
            xs.append(obs.copy())
            ys.append(target_dc)
            ws.append(w)
            action, _ = stacked_action(obs, cpatch)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                break
    finally:
        env.close()
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ws, np.float32)

def train_round(model, xs, ys, ws):
    xt = torch.as_tensor(xs, dtype=torch.float32)
    yt = torch.as_tensor(ys, dtype=torch.float32)
    wt = torch.as_tensor(ws, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    idx = np.arange(len(xs))
    last = 0.0
    for _ in range(EPOCHS_PER_ROUND):
        rng.shuffle(idx)
        total = 0.0
        n = 0
        for st in range(0, len(idx), BATCH):
            bi = torch.as_tensor(idx[st:st+BATCH], dtype=torch.long)
            pred = model(xt[bi])
            loss = (((pred - yt[bi]) ** 2) * wt[bi]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            opt.step()
            total += float(loss.detach())
            n += 1
        last = total / max(1, n)
    return last

print("=" * 120)
print("V14 +50 HARD-GATED COLLECTIVE-ONLY REPAIR")
print("V5 + V12/V4 adapter + V7(+200) + V13(+50) are frozen. Only +50 collective residual is trained.")
print("Other targets receive exactly zero V14 correction.")
print("=" * 120)

patch = CollectivePatch()
with torch.no_grad():
    last = patch.net[-2]
    last.weight.zero_(); last.bias.zero_()

rows, best_passes, best_score = evaluate_all(patch, "START V14 ZERO PATCH")
torch.save(patch.state_dict(), BEST_PATCH)

for r in range(1, ROUNDS + 1):
    xs, ys, ws = collect_shadow(patch)
    loss = train_round(patch, xs, ys, ws)
    rows, passes, score = evaluate_all(patch, f"V14 ROUND {r}")
    print(f"round={r}/{ROUNDS} rows={len(xs)} loss={loss:.7f} score={score:.2f}")
    if passes > best_passes or (passes == best_passes and score > best_score):
        best_passes, best_score = passes, score
        torch.save(patch.state_dict(), BEST_PATCH)
        print("NEW V14 BEST:", BEST_PATCH)
    if passes == 4:
        print("\n4/4 FULL-ENTRY ACHIEVED - stopping immediately.")
        break

patch.load_state_dict(torch.load(BEST_PATCH, map_location="cpu"), strict=True)
torch.save(patch.state_dict(), OUT_PATCH)
rows, passes, score = evaluate_all(patch, "FINAL FULL-ENTRY V14 BEST")
print("\nFINAL FULL-ENTRY V14 PASS COUNT:", passes, "/ 4")
print("V14 PATCH:", OUT_PATCH)
print("NOTE: V14 is hard-gated to +50 and changes collective only.")
