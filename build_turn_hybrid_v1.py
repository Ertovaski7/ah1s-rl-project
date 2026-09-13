from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_turn_goal import HelicopterEnvTurnGoal


SEED = 42
TARGETS = [-50.0, 50.0, 90.0, 200.0, 360.0]
DATA_ROLLOUTS_PER_TARGET = 2
BC_EPOCHS = 260
BC_BATCH_SIZE = 256
BC_LR = 7e-4
RL_TIMESTEPS = 12000

OUT_DIR = Path("models_turn_hybrid")
RESULT_DIR = Path("results_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BC_MODEL_PATH = OUT_DIR / "AH1S_TURN_BC_WARMSTART"
FINAL_MODEL_PATH = OUT_DIR / "AH1S_TURN_HYBRID_FINAL"
DATASET_PATH = RESULT_DIR / "turn_teacher_dataset_v1.npz"

rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def rule(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


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
    desired_yaw_rate = float(np.clip(
        0.22 * remaining + 0.030 * heading_integral,
        -1.0,
        +1.0,
    ))
    rudder = float(np.clip(
        -0.75 * (desired_yaw_rate - yaw_rate),
        -1.0,
        +1.0,
    ))
    return aileron, rudder


def teacher_collective(target_turn_deg, altitude, vertical_speed):
    if target_turn_deg < 0.0:
        ff = 0.5840
        desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.00, +0.80))
        gain = 0.010 if vertical_speed < desired_vs else 0.030
        collective = ff + gain * (desired_vs - vertical_speed)
        return float(np.clip(collective, 0.470, 0.610))

    ff = 0.5780
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.50, +0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    collective = ff + gain * (desired_vs - vertical_speed)
    return float(np.clip(collective, 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg

    collective = teacher_collective(
        env.target_turn_deg,
        float(s["altitude"]),
        float(s["vertical_speed"]),
    )
    a0 = env.collective_to_action(collective)

    speed_error = 14.5 - float(s["forward_velocity"])
    a1 = float(np.clip(0.35 * speed_error, -1.0, +1.0))

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
        # Negative terminal hold deliberately keeps the same controller that
        # passed V19; positive maneuver uses it until capture as well.
        a2, a3 = maneuver_teacher(s, remaining)

    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def collect_teacher_rollout(target):
    env = HelicopterEnvTurnGoal(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)

    obs_rows = []
    action_rows = []
    weight_rows = []

    captured = False
    capture_hold_s = 0.0
    heading_integral = 0.0
    success = False

    try:
        max_steps = int(env.MAX_TURN_TIME_S / env.CONTROL_DT)
        for _ in range(max_steps):
            action, captured, capture_hold_s, heading_integral = teacher_action(
                env, captured, capture_hold_s, heading_integral
            )

            remaining = env.target_turn_deg - env.cumulative_turn_deg
            w = 1.0
            if abs(remaining) < 25.0:
                w = 3.0
            if abs(remaining) < 8.0:
                w = 6.0

            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            action_rows.append(action.copy())
            weight_rows.append(w)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return obs_rows, action_rows, weight_rows, success, info


def evaluate(model, target, detailed=False):
    env = HelicopterEnvTurnGoal(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    next_print = 0.0

    try:
        max_steps = int(env.MAX_TURN_TIME_S / env.CONTROL_DT)
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            t = (step + 1) * env.CONTROL_DT
            if detailed and t >= next_print:
                print(
                    f"t={t:6.1f}s | target={target:+7.1f} | "
                    f"turn={info['cumulative_turn_deg']:+8.2f} | "
                    f"rem={info['remaining_turn_deg']:+7.2f} | "
                    f"alt={info['altitude']:7.2f} | "
                    f"vs={info['vertical_speed']:+6.2f} | "
                    f"v={info['forward_velocity']:6.2f} | "
                    f"hold={info['success_hold_s']:4.1f}"
                )
                next_print += 2.0

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "target": float(target),
        "success": bool(success),
        "completed": float(info.get("cumulative_turn_deg", 0.0)),
        "remaining": float(info.get("remaining_turn_deg", 999.0)),
        "altitude": float(info.get("altitude", float("nan"))),
        "vertical_speed": float(info.get("vertical_speed", float("nan"))),
        "forward_velocity": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety_failure": bool(info.get("safety_failure", False)),
    }


# =====================================================================
# A — TEACHER DATA
# =====================================================================
rule("A — COLLECT GOAL-CONDITIONED TURN TEACHER DATA")

dataset_obs = []
dataset_actions = []
dataset_weights = []

for target in TARGETS:
    for rollout in range(DATA_ROLLOUTS_PER_TARGET):
        obs_rows, action_rows, weight_rows, success, info = collect_teacher_rollout(target)
        dataset_obs.extend(obs_rows)
        dataset_actions.extend(action_rows)
        dataset_weights.extend(weight_rows)
        print(
            f"target={target:+7.1f} | rollout={rollout+1} | "
            f"samples={len(obs_rows):4d} | teacher_success={success} | "
            f"completed={info.get('cumulative_turn_deg', 0.0):+.2f}"
        )


dataset_obs = np.asarray(dataset_obs, dtype=np.float32)
dataset_actions = np.asarray(dataset_actions, dtype=np.float32)
dataset_weights = np.asarray(dataset_weights, dtype=np.float32)
np.savez_compressed(
    DATASET_PATH,
    obs=dataset_obs,
    actions=dataset_actions,
    weights=dataset_weights,
)
print("dataset:", dataset_obs.shape, dataset_actions.shape)
print("saved:", DATASET_PATH)


# =====================================================================
# B — BEHAVIOR CLONING / DISTILLATION
# =====================================================================
rule("B — DISTILL TURN TEACHER INTO GOAL-CONDITIONED PPO ACTOR")

train_env = Monitor(HelicopterEnvTurnGoal(target_choices=TARGETS))
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
)

model = PPO(
    "MlpPolicy",
    train_env,
    policy_kwargs=policy_kwargs,
    learning_rate=2e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=5,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    seed=SEED,
    device="auto",
)

device = model.device
obs_tensor = torch.as_tensor(dataset_obs, dtype=torch.float32, device=device)
target_tensor = torch.as_tensor(dataset_actions, dtype=torch.float32, device=device)
sample_weight_tensor = torch.as_tensor(dataset_weights, dtype=torch.float32, device=device)
action_loss_weight = torch.as_tensor(
    [2.0, 1.5, 2.5, 2.5], dtype=torch.float32, device=device
).reshape(1, 4)

actor_params = [
    p for name, p in model.policy.named_parameters()
    if "mlp_extractor.policy_net" in name or "action_net" in name
]
if not actor_params:
    raise RuntimeError("PPO actor parameters not found")

optimizer = torch.optim.Adam(actor_params, lr=BC_LR)
indices = np.arange(len(dataset_obs))
best_loss = float("inf")

for epoch in range(1, BC_EPOCHS + 1):
    rng.shuffle(indices)
    total = 0.0
    batches = 0

    for start in range(0, len(indices), BC_BATCH_SIZE):
        idx_np = indices[start:start + BC_BATCH_SIZE]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        batch_obs = obs_tensor[idx]
        batch_target = target_tensor[idx]
        batch_weight = sample_weight_tensor[idx].reshape(-1, 1)

        features = model.policy.extract_features(batch_obs)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        pred = model.policy.action_net(latent_pi)

        loss = (((pred - batch_target) ** 2) * action_loss_weight * batch_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
        optimizer.step()

        total += float(loss.detach().cpu())
        batches += 1

    epoch_loss = total / max(1, batches)
    best_loss = min(best_loss, epoch_loss)
    if epoch == 1 or epoch % 20 == 0:
        print(f"BC epoch {epoch:3d}/{BC_EPOCHS} | loss={epoch_loss:.7f} | best={best_loss:.7f}")

model.save(str(BC_MODEL_PATH))
print("BC warm-start saved:", BC_MODEL_PATH)


# =====================================================================
# C — TEACHER-OFF VALIDATION BEFORE RL
# =====================================================================
rule("C — TEACHER-OFF VALIDATION OF DISTILLED ACTOR")

bc_results = []
for target in [-50.0, 50.0, 200.0, 360.0]:
    r = evaluate(model, target, detailed=True)
    bc_results.append(r)
    print("BC RESULT:", r)

with open(RESULT_DIR / "bc_teacher_off_results.json", "w", encoding="utf-8") as f:
    json.dump(bc_results, f, indent=2)


# =====================================================================
# D — REAL PPO REINFORCEMENT LEARNING FINE-TUNING
# =====================================================================
rule("D — PPO REWARD-BASED FINE-TUNING (TEACHER OFF)")
print("Teacher/controller is OFF during .learn(). Reward comes from HelicopterEnvTurnGoal.step().")
model.learn(total_timesteps=RL_TIMESTEPS, reset_num_timesteps=True, progress_bar=True)
model.save(str(FINAL_MODEL_PATH))
print("Final PPO model saved:", FINAL_MODEL_PATH)


# =====================================================================
# E — FINAL TEACHER-OFF VALIDATION
# =====================================================================
rule("E — FINAL TEACHER-OFF TURN VALIDATION")

final_results = []
for target in [-50.0, 50.0, 200.0, 360.0]:
    r = evaluate(model, target, detailed=True)
    final_results.append(r)
    print("FINAL RESULT:", r)

with open(RESULT_DIR / "final_teacher_off_results.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

passes = sum(int(r["success"]) for r in final_results)
print("\nFINAL PASS COUNT:", passes, "/", len(final_results))
print("BC MODEL   :", BC_MODEL_PATH)
print("FINAL MODEL:", FINAL_MODEL_PATH)
print("RESULTS    :", RESULT_DIR)
