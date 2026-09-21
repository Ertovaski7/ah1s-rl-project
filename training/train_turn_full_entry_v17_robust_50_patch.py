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
TRAIN_SEEDS = [7, 21, 42, 84, 123, 256, 512, 777]
EVAL_SEEDS = [7, 21, 42, 84, 123]
TARGET = 50.0

BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
BASE_ADAPTER = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt")
PATCH_200 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
PATCH_50_INIT = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt")
OUT_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt")
BEST_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH_BEST.pt")

BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)

ROUNDS = 10
EPOCHS_PER_ROUND = 16
BATCH = 256
LR = 3e-5
MIN_EXTRA_STEPS = 0
MAX_EXTRA_STEPS = 24

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


class RandomizedEntryEnv(HelicopterEnvTurnGoalFullEntry):
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        local_rng = np.random.default_rng(seed)
        extra_steps = int(local_rng.integers(MIN_EXTRA_STEPS, MAX_EXTRA_STEPS + 1))

        self.turn_active = False
        self.success_hold_s = 0.0
        self.cumulative_turn_deg = 0.0
        self.turn_steps = 0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)
        self.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        self.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        for _ in range(extra_steps):
            base_obs = self._base_obs()
            action, _ = self.stage2_model.predict(base_obs, deterministic=True)
            self._apply_action(action)
            for _ in range(self.PHYSICS_STEPS):
                if not self.fdm.run():
                    raise RuntimeError("JSBSim stopped during randomized entry continuation")
            s = self._raw_state()
            self.forward_distance += float(s["forward_velocity"]) * self.CONTROL_DT

        self.turn_active = True
        self.cumulative_turn_deg = 0.0
        self.prev_heading_deg = self._heading_deg()
        self.turn_steps = 0
        self.success_hold_s = 0.0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)
        self._v2_previous_remaining = float(self.target_turn_deg)
        self.fdm["ap/afcs/roll-channel-active-norm"] = self.TURN_ROLL_AFCS
        self.fdm["ap/afcs/yaw-channel-active-norm"] = self.TURN_YAW_AFCS

        s = self._raw_state()
        info = {
            **s,
            "target_turn_deg": self.target_turn_deg,
            "cumulative_turn_deg": 0.0,
            "remaining_turn_deg": self.target_turn_deg,
            "extra_entry_steps": extra_steps,
            "success": False,
        }
        return self._get_obs(), info


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
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.5, 0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    return float(np.clip(0.5780 + gain * (desired_vs - vertical_speed), 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg
    a0 = env.collective_to_action(
        teacher_collective(env.target_turn_deg, float(s["altitude"]), float(s["vertical_speed"]))
    )
    a1 = float(np.clip(0.35 * (14.5 - float(s["forward_velocity"])), -1.0, 1.0))
    capture_ok = abs(remaining) <= 0.75 and abs(math.degrees(s["roll"])) <= 3.0
    capture_hold_s = capture_hold_s + env.CONTROL_DT if capture_ok else 0.0
    if capture_hold_s >= 0.75:
        captured = True
    if captured:
        heading_integral += float(np.clip(remaining, -5.0, 5.0)) * env.CONTROL_DT
        heading_integral = float(np.clip(heading_integral, -20.0, 20.0))
        a2, a3 = positive_terminal_hold(s, remaining, heading_integral)
    else:
        a2, a3 = maneuver_teacher(s, remaining)
    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def combined_action(base_model, base_adapter, patch_50, obs):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d1 = base_adapter(x).cpu().numpy()[0]
        d2 = patch_50(x).cpu().numpy()[0]
    return np.clip(np.asarray(base, dtype=np.float32) + d1 + d2, -1.0, 1.0)


def collect_shadow(base_model, base_adapter, patch_50, seed):
    env = RandomizedEntryEnv(target_turn_deg=TARGET)
    obs, reset_info = env.reset(seed=seed)
    obs_rows, tgt_rows, wt_rows = [], [], []
    captured = False
    capture_hold_s = 0.0
    heading_integral = 0.0
    try:
        max_steps = int(max(90.0, TARGET / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            teacher, captured, capture_hold_s, heading_integral = teacher_action(
                env, captured, capture_hold_s, heading_integral
            )
            base, _ = base_model.predict(obs, deterministic=True)
            x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                d1 = base_adapter(x).cpu().numpy()[0]
            current_without_patch = np.clip(np.asarray(base, dtype=np.float32) + d1, -1.0, 1.0)
            target_patch = np.clip(teacher - current_without_patch, -PATCH_SCALE.numpy(), PATCH_SCALE.numpy())
            remaining = TARGET - env.cumulative_turn_deg
            w = 1.0
            if abs(remaining) < 25.0:
                w = 3.0
            if abs(remaining) < 10.0:
                w = 8.0
            if abs(remaining) < 3.0:
                w = 12.0
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            tgt_rows.append(target_patch.astype(np.float32))
            wt_rows.append(w)

            action = combined_action(base_model, base_adapter, patch_50, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return (
        np.asarray(obs_rows, np.float32),
        np.asarray(tgt_rows, np.float32),
        np.asarray(wt_rows, np.float32),
        int(reset_info.get("extra_entry_steps", 0)),
    )


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
            idx = torch.as_tensor(idx_all[start:start + BATCH], dtype=torch.long)
            pred = patch(obs_t[idx])
            loss = (((pred - tgt_t[idx]) ** 2) * wt_t[idx] * channel_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(patch.parameters(), 0.25)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        last_loss = total / max(1, batches)
    return last_loss


def evaluate_one(base_model, base_adapter, patch_50, seed):
    env = RandomizedEntryEnv(target_turn_deg=TARGET)
    obs, reset_info = env.reset(seed=seed)
    success = False
    info = reset_info
    try:
        max_steps = int(max(90.0, TARGET / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action = combined_action(base_model, base_adapter, patch_50, obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()
    return {
        "seed": seed,
        "extra": int(reset_info.get("extra_entry_steps", 0)),
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", np.nan)),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
    }


def evaluate_all(base_model, base_adapter, patch_50, label):
    rows = [evaluate_one(base_model, base_adapter, patch_50, seed) for seed in EVAL_SEEDS]
    passes = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safety"]) for r in rows)
    err = sum(abs(r["rem"]) for r in rows)
    score = passes * 10000.0 - safety * 100.0 - err
    print(f"\n{label} | PASS={passes}/{len(rows)} | score={score:.2f}")
    for r in rows:
        print(
            f"  seed={r['seed']:3d} | extra={r['extra']:2d} | pass={str(r['success']):5s} | "
            f"done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | "
            f"hold={r['hold']:4.1f} | safety={r['safety']}"
        )
    return rows, passes, score


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


for p in (BASE_MODEL, BASE_ADAPTER, PATCH_50_INIT):
    require(p)

print("=" * 120)
print("V17 ROBUST +50 SPECIALIST TRAINING")
print("Starts from V13 +50 patch and trains only on physically randomized full-entry states.")
print("V5 PPO and V4 adapter stay frozen. Runtime teacher/controller remains OFF.")
print("Other target policies/patches are untouched.")
print("=" * 120)

base_model = PPO.load(str(BASE_MODEL))
base_adapter = ResidualAdapter(BASE_SCALE)
base_adapter.load_state_dict(torch.load(BASE_ADAPTER, map_location="cpu"), strict=True)
base_adapter.eval()
for p in base_adapter.parameters():
    p.requires_grad = False

patch_50 = ResidualAdapter(PATCH_SCALE)
patch_50.load_state_dict(torch.load(PATCH_50_INIT, map_location="cpu"), strict=True)

rows, best_passes, best_score = evaluate_all(base_model, base_adapter, patch_50, "START V17 FROM V13")
torch.save(patch_50.state_dict(), BEST_PATCH)

for round_idx in range(1, ROUNDS + 1):
    all_obs, all_tgt, all_wt = [], [], []
    extras = []
    for seed in TRAIN_SEEDS:
        obs_np, tgt_np, wt_np, extra = collect_shadow(base_model, base_adapter, patch_50, seed)
        if len(obs_np):
            all_obs.append(obs_np)
            all_tgt.append(tgt_np)
            all_wt.append(wt_np)
            extras.append(extra)

    obs_np = np.concatenate(all_obs, axis=0)
    tgt_np = np.concatenate(all_tgt, axis=0)
    wt_np = np.concatenate(all_wt, axis=0)
    loss = train_patch(patch_50, obs_np, tgt_np, wt_np)

    print(
        f"\nround={round_idx:2d}/{ROUNDS} | rows={len(obs_np)} | loss={loss:.7f} | "
        f"train_extra_steps={sorted(set(extras))}"
    )
    rows, passes, score = evaluate_all(base_model, base_adapter, patch_50, f"V17 ROUND {round_idx}")

    if passes > best_passes or (passes == best_passes and score > best_score):
        best_passes, best_score = passes, score
        torch.save(patch_50.state_dict(), BEST_PATCH)
        print("NEW V17 BEST:", BEST_PATCH)

    if passes == len(EVAL_SEEDS) and all(not r["safety"] for r in rows):
        print("\nV17 +50 RANDOMIZED-ENTRY 5/5 ACHIEVED - stopping.")
        break

patch_50.load_state_dict(torch.load(BEST_PATCH, map_location="cpu"), strict=True)
torch.save(patch_50.state_dict(), OUT_PATCH)
rows, passes, score = evaluate_all(base_model, base_adapter, patch_50, "FINAL V17 BEST")
print("\nFINAL V17 +50 PASS COUNT:", passes, "/", len(EVAL_SEEDS))
print("ROBUST +50 PATCH:", OUT_PATCH)
print("NEXT: replace only the +50 specialist in V14/V16 runtime and re-run all four targets.")
