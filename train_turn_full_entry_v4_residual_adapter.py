from __future__ import annotations

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
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_PATH = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt"
BEST_ADAPTER_PATH = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER_BEST.pt"

ROLLOUTS_PER_TARGET = 2
BC_EPOCHS = 120
SHADOW_ROUNDS = 8
SHADOW_EPOCHS = 20
BATCH = 256
LR = 2e-4

# Keep residual authority bounded so the frozen PPO remains the dominant policy.
RESIDUAL_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

    def forward(self, obs):
        scale = RESIDUAL_SCALE.to(obs.device)
        return self.net(obs) * scale


def maneuver_teacher(state, remaining):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    desired_yaw_rate = float(np.clip(1.20 * remaining, -1.50, +1.50))
    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    if remaining >= 0.0:
        rudder = float(np.clip(rudder, -1.0, +0.20))
    else:
        rudder = float(np.clip(rudder, -0.20, +1.0))
    desired_roll = float(np.clip(0.25 * remaining, -5.0, +5.0))
    aileron = float(np.clip(0.35 * (desired_roll - roll), -1.0, +1.0))
    return aileron, rudder


def positive_terminal_hold(state, remaining, heading_integral):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    aileron = float(np.clip(0.45 * (0.0 - roll), -1.0, +1.0))
    desired_yaw_rate = float(np.clip(0.22 * remaining + 0.030 * heading_integral, -1.0, +1.0))
    rudder = float(np.clip(-0.75 * (desired_yaw_rate - yaw_rate), -1.0, +1.0))
    return aileron, rudder


def teacher_collective(target_turn_deg, altitude, vertical_speed):
    if target_turn_deg < 0.0:
        desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.00, +0.80))
        gain = 0.010 if vertical_speed < desired_vs else 0.030
        return float(np.clip(0.5840 + gain * (desired_vs - vertical_speed), 0.470, 0.610))
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.50, +0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    return float(np.clip(0.5780 + gain * (desired_vs - vertical_speed), 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg
    a0 = env.collective_to_action(teacher_collective(
        env.target_turn_deg, float(s["altitude"]), float(s["vertical_speed"])
    ))
    a1 = float(np.clip(0.35 * (14.5 - float(s["forward_velocity"])), -1.0, +1.0))

    roll_deg = abs(math.degrees(s["roll"]))
    capture_ok = abs(remaining) <= 0.75 and roll_deg <= 3.0
    capture_hold_s = capture_hold_s + env.CONTROL_DT if capture_ok else 0.0
    if capture_hold_s >= 0.75:
        captured = True

    if captured and env.target_turn_deg > 0.0:
        heading_integral += float(np.clip(remaining, -5.0, +5.0)) * env.CONTROL_DT
        heading_integral = float(np.clip(heading_integral, -20.0, +20.0))
        a2, a3 = positive_terminal_hold(s, remaining, heading_integral)
    else:
        a2, a3 = maneuver_teacher(s, remaining)

    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def residual_action(base_model, adapter, obs):
    base, _ = base_model.predict(obs, deterministic=True)
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        delta = adapter(x).cpu().numpy()[0]
    return np.clip(np.asarray(base, dtype=np.float32) + delta, -1.0, 1.0)


def evaluate(base_model, adapter, target):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action = residual_action(base_model, adapter, obs)
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


def evaluate_all(base_model, adapter, label):
    rows = [evaluate(base_model, adapter, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safety"]) for r in rows)
    err = sum(abs(r["rem"]) for r in rows)
    score = passes * 10000.0 - safety * 100.0 - err
    print(f"\n{label} | PASS={passes}/4 | score={score:.2f}")
    for r in rows:
        print(
            f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | "
            f"done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | "
            f"v={r['v']:6.2f} | hold={r['hold']:4.1f} | safety={r['safety']}"
        )
    return rows, passes, score


def collect_teacher_dataset(base_model):
    obs_rows, delta_rows, weight_rows = [], [], []
    teacher_passes = 0
    for target in TARGETS:
        for rollout in range(ROLLOUTS_PER_TARGET):
            env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
            obs, info = env.reset(seed=SEED + rollout)
            captured = False
            capture_hold_s = 0.0
            heading_integral = 0.0
            success = False
            try:
                max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
                for _ in range(max_steps):
                    teacher, captured, capture_hold_s, heading_integral = teacher_action(
                        env, captured, capture_hold_s, heading_integral
                    )
                    base, _ = base_model.predict(obs, deterministic=True)
                    delta = np.clip(teacher - np.asarray(base, dtype=np.float32), -0.5, 0.5)
                    remaining = env.target_turn_deg - env.cumulative_turn_deg
                    w = 1.0
                    if abs(remaining) < 25.0:
                        w = 3.0
                    if abs(remaining) < 8.0:
                        w = 6.0
                    obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
                    delta_rows.append(delta.astype(np.float32))
                    weight_rows.append(float(w))
                    obs, _, terminated, truncated, info = env.step(teacher)
                    if terminated or truncated:
                        success = bool(info.get("success", False))
                        break
            finally:
                env.close()
            teacher_passes += int(success)
            print(
                f"teacher target={target:+7.1f} rollout={rollout+1} | success={success} | "
                f"done={float(info.get('cumulative_turn_deg',0.0)):+.2f}"
            )
    if teacher_passes != len(TARGETS) * ROLLOUTS_PER_TARGET:
        raise RuntimeError("Teacher is not 8/8 on full-entry distribution; abort residual training")
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(delta_rows, dtype=np.float32),
        np.asarray(weight_rows, dtype=np.float32),
    )


def collect_shadow_dataset(base_model, adapter):
    obs_rows, delta_rows, weight_rows = [], [], []
    for target in TARGETS:
        env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
        obs, _ = env.reset(seed=SEED)
        captured = False
        capture_hold_s = 0.0
        heading_integral = 0.0
        try:
            max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
            for _ in range(max_steps):
                teacher, captured, capture_hold_s, heading_integral = teacher_action(
                    env, captured, capture_hold_s, heading_integral
                )
                base, _ = base_model.predict(obs, deterministic=True)
                delta = np.clip(teacher - np.asarray(base, dtype=np.float32), -0.5, 0.5)
                remaining = env.target_turn_deg - env.cumulative_turn_deg
                w = 1.5
                if abs(remaining) < 25.0:
                    w = 4.0
                if abs(remaining) < 8.0:
                    w = 8.0
                obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
                delta_rows.append(delta.astype(np.float32))
                weight_rows.append(float(w))

                # Actor+adapter visits the state; teacher only labels it.
                action = residual_action(base_model, adapter, obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(delta_rows, dtype=np.float32),
        np.asarray(weight_rows, dtype=np.float32),
    )


def train_adapter(adapter, optimizer, obs_np, delta_np, weight_np, epochs):
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    tgt_t = torch.as_tensor(delta_np, dtype=torch.float32)
    wt_t = torch.as_tensor(weight_np, dtype=torch.float32).reshape(-1, 1)
    idx_all = np.arange(len(obs_np))
    for epoch in range(1, epochs + 1):
        rng.shuffle(idx_all)
        total = 0.0
        batches = 0
        for start in range(0, len(idx_all), BATCH):
            idx = torch.as_tensor(idx_all[start:start+BATCH], dtype=torch.long)
            pred = adapter(obs_t[idx])
            # Slightly prioritize collective and turn channels.
            channel_w = torch.tensor([2.5, 1.0, 2.0, 2.0], dtype=torch.float32).reshape(1, 4)
            loss = (((pred - tgt_t[idx]) ** 2) * wt_t[idx] * channel_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"adapter epoch {epoch:3d}/{epochs} | loss={total/max(1,batches):.7f}")


if not BASE_MODEL.exists():
    raise FileNotFoundError(BASE_MODEL)

print("=" * 120)
print("FULL-ENTRY V4 - FROZEN PPO + LEARNED RESIDUAL ADAPTER")
print("Base PPO is never updated. Teacher labels training data only; runtime teacher/controller OFF.")
print("=" * 120)

base_model = PPO.load(str(BASE_MODEL))
adapter = ResidualAdapter()
optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)

# Initial adapter is near zero only after explicit zero init of final layer.
with torch.no_grad():
    last = adapter.net[-2]
    last.weight.zero_()
    last.bias.zero_()

rows, best_passes, best_score = evaluate_all(base_model, adapter, "START V4 (ZERO RESIDUAL)")
torch.save(adapter.state_dict(), BEST_ADAPTER_PATH)

obs_np, delta_np, weight_np = collect_teacher_dataset(base_model)
print("teacher dataset rows:", len(obs_np))
train_adapter(adapter, optimizer, obs_np, delta_np, weight_np, BC_EPOCHS)
rows, passes, score = evaluate_all(base_model, adapter, "AFTER TEACHER-DATA RESIDUAL BC")
if passes > best_passes or (passes == best_passes and score > best_score):
    best_passes, best_score = passes, score
    torch.save(adapter.state_dict(), BEST_ADAPTER_PATH)

for round_idx in range(1, SHADOW_ROUNDS + 1):
    if best_passes == 4:
        break
    s_obs, s_delta, s_weight = collect_shadow_dataset(base_model, adapter)
    # Mix original qualified teacher trajectories with actor-visited shadow states.
    mix_obs = np.concatenate([obs_np, s_obs], axis=0)
    mix_delta = np.concatenate([delta_np, s_delta], axis=0)
    mix_weight = np.concatenate([0.5 * weight_np, s_weight], axis=0)
    train_adapter(adapter, optimizer, mix_obs, mix_delta, mix_weight, SHADOW_EPOCHS)
    rows, passes, score = evaluate_all(base_model, adapter, f"V4 SHADOW ROUND {round_idx}")
    if passes > best_passes or (passes == best_passes and score > best_score):
        best_passes, best_score = passes, score
        torch.save(adapter.state_dict(), BEST_ADAPTER_PATH)
        print("NEW V4 BEST:", BEST_ADAPTER_PATH)
    if passes == 4:
        print("\n4/4 FULL-ENTRY ACHIEVED - stopping immediately.")
        break

adapter.load_state_dict(torch.load(BEST_ADAPTER_PATH, map_location="cpu"))
torch.save(adapter.state_dict(), ADAPTER_PATH)
rows, passes, score = evaluate_all(base_model, adapter, "FINAL FULL-ENTRY V4 BEST")
print("\nFINAL FULL-ENTRY V4 PASS COUNT:", passes, "/ 4")
print("BASE PPO:", BASE_MODEL)
print("RESIDUAL ADAPTER:", ADAPTER_PATH)
print("NOTE: base PPO weights are frozen; runtime uses PPO + learned residual NN, no teacher/controller.")
