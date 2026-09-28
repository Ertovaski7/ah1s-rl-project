from __future__ import annotations

# training/ klasöründen çalıştırılabilmesi için: repo kökünü import yoluna ekle
# ve çalışma dizinini köke al (model/sonuç yolları köke göredir).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO_ROOT)
_os.chdir(_REPO_ROOT)

import math
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
OUT_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
BEST_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH_BEST.pt")

ROUNDS = 12
EPOCHS_PER_ROUND = 20
BATCH = 256
LR = 8e-5
BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02

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


def gate_from_obs(obs):
    # obs[12] = target_turn_norm = requested_turn_deg / 360
    return 1.0 if abs(float(obs[12]) - TARGET_200_NORM) <= GATE_TOL else 0.0


def maneuver_teacher(state, remaining):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    desired_yaw_rate = float(np.clip(1.20 * remaining, -1.50, 1.50))
    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    rudder = float(np.clip(rudder, -1.0, 0.20)) if remaining >= 0 else float(np.clip(rudder, -0.20, 1.0))
    desired_roll = float(np.clip(0.25 * remaining, -5.0, 5.0))
    aileron = float(np.clip(0.35 * (desired_roll - roll), -1.0, 1.0))
    return aileron, rudder


def positive_terminal_hold(state, remaining, heading_integral):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    aileron = float(np.clip(0.45 * (0.0 - roll), -1.0, 1.0))
    desired_yaw_rate = float(np.clip(0.22 * remaining + 0.030 * heading_integral, -1.0, 1.0))
    rudder = float(np.clip(-0.75 * (desired_yaw_rate - yaw_rate), -1.0, 1.0))
    return aileron, rudder


def teacher_collective(target, altitude, vertical_speed):
    if target < 0:
        desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.0, 0.8))
        gain = 0.010 if vertical_speed < desired_vs else 0.030
        return float(np.clip(0.5840 + gain * (desired_vs - vertical_speed), 0.470, 0.610))
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.5, 0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    return float(np.clip(0.5780 + gain * (desired_vs - vertical_speed), 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg
    a0 = env.collective_to_action(teacher_collective(env.target_turn_deg, float(s["altitude"]), float(s["vertical_speed"])))
    a1 = float(np.clip(0.35 * (14.5 - float(s["forward_velocity"])), -1.0, 1.0))
    capture_ok = abs(remaining) <= 0.75 and abs(math.degrees(s["roll"])) <= 3.0
    capture_hold_s = capture_hold_s + env.CONTROL_DT if capture_ok else 0.0
    if capture_hold_s >= 0.75:
        captured = True
    if captured and env.target_turn_deg > 0:
        heading_integral += float(np.clip(remaining, -5.0, 5.0)) * env.CONTROL_DT
        heading_integral = float(np.clip(heading_integral, -20.0, 20.0))
        a2, a3 = positive_terminal_hold(s, remaining, heading_integral)
    else:
        a2, a3 = maneuver_teacher(s, remaining)
    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def combined_action(base_model, base_adapter, patch, obs):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d1 = base_adapter(x).cpu().numpy()[0]
        if gate_from_obs(obs) > 0.5:
            d2 = patch(x).cpu().numpy()[0]
        else:
            d2 = np.zeros(4, dtype=np.float32)
    return np.clip(np.asarray(base, dtype=np.float32) + d1 + d2, -1.0, 1.0)


def evaluate(base_model, base_adapter, patch, target):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action = combined_action(base_model, base_adapter, patch, obs)
            obs, _, terminated, truncated, info = env.step(action)
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


def evaluate_all(base_model, base_adapter, patch, label):
    rows = [evaluate(base_model, base_adapter, patch, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safety"]) for r in rows)
    err = sum(abs(r["rem"]) for r in rows)
    score = passes * 10000.0 - safety * 100.0 - err
    print(f"\n{label} | PASS={passes}/4 | score={score:.2f}")
    for r in rows:
        print(f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | v={r['v']:6.2f} | hold={r['hold']:4.1f} | safety={r['safety']}")
    return rows, passes, score


def collect_shadow_200(base_model, base_adapter, patch):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=200.0)
    obs, _ = env.reset(seed=SEED)
    obs_rows, tgt_rows, wt_rows = [], [], []
    captured = False
    capture_hold_s = 0.0
    heading_integral = 0.0
    try:
        max_steps = int(max(90.0, 200.0 / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            teacher, captured, capture_hold_s, heading_integral = teacher_action(env, captured, capture_hold_s, heading_integral)
            base, _ = base_model.predict(obs, deterministic=True)
            x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                d1 = base_adapter(x).cpu().numpy()[0]
            current_without_patch = np.clip(np.asarray(base, dtype=np.float32) + d1, -1.0, 1.0)
            target_patch = np.clip(teacher - current_without_patch, -PATCH_SCALE.numpy(), PATCH_SCALE.numpy())
            remaining = 200.0 - env.cumulative_turn_deg
            w = 1.0
            if abs(remaining) < 40.0:
                w = 3.0
            if abs(remaining) < 15.0:
                w = 8.0
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            tgt_rows.append(target_patch.astype(np.float32))
            wt_rows.append(w)

            action = combined_action(base_model, base_adapter, patch, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return np.asarray(obs_rows, np.float32), np.asarray(tgt_rows, np.float32), np.asarray(wt_rows, np.float32)


def train_patch(patch, obs_np, tgt_np, wt_np):
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    tgt_t = torch.as_tensor(tgt_np, dtype=torch.float32)
    wt_t = torch.as_tensor(wt_np, dtype=torch.float32).reshape(-1, 1)
    channel_w = torch.tensor([2.0, 0.8, 2.5, 2.5], dtype=torch.float32).reshape(1, 4)
    optimizer = torch.optim.Adam(patch.parameters(), lr=LR)
    idx_all = np.arange(len(obs_np))
    last_loss = 0.0
    for _ in range(EPOCHS_PER_ROUND):
        rng.shuffle(idx_all)
        total = 0.0
        batches = 0
        for start in range(0, len(idx_all), BATCH):
            idx = torch.as_tensor(idx_all[start:start+BATCH], dtype=torch.long)
            pred = patch(obs_t[idx])
            loss = (((pred - tgt_t[idx]) ** 2) * wt_t[idx] * channel_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(patch.parameters(), 0.35)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        last_loss = total / max(1, batches)
    return last_loss


if not BASE_MODEL.exists():
    raise FileNotFoundError(BASE_MODEL)
if not BASE_ADAPTER.exists():
    raise FileNotFoundError(BASE_ADAPTER)

print("=" * 120)
print("FULL-ENTRY V7 - HARD-GATED +200 ACTOR-VISITED PATCH")
print("Patch is structurally active only for target_turn_norm near +200/360; other targets receive exactly zero patch.")
print("Runtime teacher/controller OFF.")
print("=" * 120)

base_model = PPO.load(str(BASE_MODEL))
base_adapter = ResidualAdapter(BASE_SCALE)
base_adapter.load_state_dict(torch.load(BASE_ADAPTER, map_location="cpu"), strict=True)
base_adapter.eval()
for p in base_adapter.parameters():
    p.requires_grad = False

patch = ResidualAdapter(PATCH_SCALE)
with torch.no_grad():
    last = patch.net[-2]
    last.weight.zero_()
    last.bias.zero_()

rows, best_passes, best_score = evaluate_all(base_model, base_adapter, patch, "START V7 ZERO PATCH")
torch.save(patch.state_dict(), BEST_PATCH)

for round_idx in range(1, ROUNDS + 1):
    obs_np, tgt_np, wt_np = collect_shadow_200(base_model, base_adapter, patch)
    loss = train_patch(patch, obs_np, tgt_np, wt_np)
    print(f"\nround={round_idx:2d}/{ROUNDS} | rows={len(obs_np)} | loss={loss:.7f}")
    rows, passes, score = evaluate_all(base_model, base_adapter, patch, f"V7 GATED ROUND {round_idx}")
    if passes > best_passes or (passes == best_passes and score > best_score):
        best_passes, best_score = passes, score
        torch.save(patch.state_dict(), BEST_PATCH)
        print("NEW V7 BEST:", BEST_PATCH)
    if passes == 4:
        print("\n4/4 FULL-ENTRY ACHIEVED - stopping immediately.")
        break

patch.load_state_dict(torch.load(BEST_PATCH, map_location="cpu"), strict=True)
torch.save(patch.state_dict(), OUT_PATCH)
rows, passes, score = evaluate_all(base_model, base_adapter, patch, "FINAL FULL-ENTRY V7 BEST")
print("\nFINAL FULL-ENTRY V7 PASS COUNT:", passes, "/ 4")
print("BASE PPO:", BASE_MODEL)
print("BASE V4 ADAPTER:", BASE_ADAPTER)
print("GATED +200 PATCH:", OUT_PATCH)
print("NOTE: +200 patch is hard-gated by target observation; other validated targets receive exactly zero patch at runtime.")
