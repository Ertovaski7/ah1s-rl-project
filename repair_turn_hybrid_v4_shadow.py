from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
START_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V3_REPAIRED.zip")
BASE_DATASET = Path("results_turn_hybrid/turn_teacher_dataset_v1.npz")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "AH1S_TURN_HYBRID_V4_SHADOW_REPAIRED"

TARGETS = [-50.0, 50.0, 200.0, 360.0]
SHADOW_ROLLOUTS = 4
MAX_REPAIR_EPOCHS = 40
EVAL_EVERY = 1
BATCH = 256
LR = 2e-5

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

if not START_MODEL.exists():
    raise FileNotFoundError(START_MODEL)
if not BASE_DATASET.exists():
    raise FileNotFoundError(BASE_DATASET)


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


def negative_teacher_action(env):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg

    altitude = float(s["altitude"])
    vertical_speed = float(s["vertical_speed"])

    ff = 0.5840
    desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.00, +0.80))
    gain = 0.010 if vertical_speed < desired_vs else 0.030
    collective = ff + gain * (desired_vs - vertical_speed)
    collective = float(np.clip(collective, 0.470, 0.610))
    a0 = env.collective_to_action(collective)

    speed_error = 14.5 - float(s["forward_velocity"])
    a1 = float(np.clip(0.35 * speed_error, -1.0, +1.0))

    a2, a3 = maneuver_teacher(s, remaining)
    return np.array([a0, a1, a2, a3], dtype=np.float32)


def evaluate(model, target):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    min_alt = float("inf")
    try:
        max_steps = int(env.MAX_TURN_TIME_S / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            min_alt = min(min_alt, float(info.get("altitude", 999.0)))
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
        "min_altitude": float(min_alt),
        "forward_velocity": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety_failure": bool(info.get("safety_failure", False)),
    }


def evaluate_all(model, label):
    results = [evaluate(model, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in results)
    safety_count = sum(int(r["safety_failure"]) for r in results)
    error_sum = sum(abs(r["remaining"]) for r in results)
    min_alt_penalty = sum(max(0.0, 285.0 - r["min_altitude"]) for r in results)
    score = passes * 10000.0 - 100.0 * safety_count - error_sum - 5.0 * min_alt_penalty

    print(f"\n{label} | PASS={passes}/4 | score={score:.2f} | safety={safety_count}")
    for r in results:
        print(
            f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | "
            f"done={r['completed']:+8.2f} | rem={r['remaining']:+7.2f} | "
            f"alt={r['altitude']:7.2f} | min_alt={r['min_altitude']:7.2f} | "
            f"v={r['forward_velocity']:6.2f} | hold={r['hold']:4.1f} | "
            f"safety={r['safety_failure']}"
        )
    return results, passes, score


def collect_shadow_labels(model, rollout_id):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=-50.0)
    obs, info = env.reset(seed=SEED + rollout_id)
    rows_obs = []
    rows_act = []
    rows_w = []

    try:
        max_steps = int(env.MAX_TURN_TIME_S / env.CONTROL_DT)
        for _ in range(max_steps):
            # Teacher is shadow-only: label actor-visited state, but DO NOT execute teacher.
            target_action = negative_teacher_action(env)
            s = env._raw_state()
            altitude = float(s["altitude"])
            remaining = env.target_turn_deg - env.cumulative_turn_deg

            # Emphasize the exact off-distribution states that caused V3 failure.
            w = 3.0
            if altitude < 295.0:
                w = 8.0
            if altitude < 288.0:
                w = 14.0
            if abs(remaining) < 10.0:
                w *= 1.5

            rows_obs.append(np.asarray(obs, dtype=np.float32).copy())
            rows_act.append(target_action.copy())
            rows_w.append(float(w))

            actor_action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(actor_action)
            if terminated or truncated:
                break
    finally:
        env.close()

    print(
        f"shadow rollout {rollout_id}: samples={len(rows_obs)} | "
        f"done={float(info.get('cumulative_turn_deg',0.0)):+.2f} | "
        f"alt={float(info.get('altitude',0.0)):.2f} | safety={bool(info.get('safety_failure',False))}"
    )
    return rows_obs, rows_act, rows_w


print("=" * 120)
print("TURN HYBRID V4 - ON-POLICY SHADOW-LABEL REPAIR")
print("Start model:", START_MODEL)
print("Teacher labels actor-visited -50 states; teacher is NEVER executed during rollout/runtime.")
print("=" * 120)

model = PPO.load(str(START_MODEL))
base_results, base_passes, base_score = evaluate_all(model, "START V4")

# Existing teacher dataset provides anchors for already working positive goals.
data = np.load(BASE_DATASET)
base_obs = np.asarray(data["obs"], dtype=np.float32)
base_actions = np.asarray(data["actions"], dtype=np.float32)
base_weights = np.asarray(data["weights"], dtype=np.float32)
base_target_deg = base_obs[:, 12] * 360.0

anchor_mask = base_target_deg > 0.0
anchor_obs = base_obs[anchor_mask]
anchor_actions = base_actions[anchor_mask]
anchor_weights = base_weights[anchor_mask] * 0.20

# Keep 360 especially protected against forgetting.
anchor_target_deg = anchor_obs[:, 12] * 360.0
anchor_weights[anchor_target_deg > 300.0] *= 4.0

shadow_obs = []
shadow_actions = []
shadow_weights = []
for rollout_id in range(1, SHADOW_ROLLOUTS + 1):
    ro, ra, rw = collect_shadow_labels(model, rollout_id)
    shadow_obs.extend(ro)
    shadow_actions.extend(ra)
    shadow_weights.extend(rw)

shadow_obs = np.asarray(shadow_obs, dtype=np.float32)
shadow_actions = np.asarray(shadow_actions, dtype=np.float32)
shadow_weights = np.asarray(shadow_weights, dtype=np.float32)

train_obs = np.concatenate([shadow_obs, anchor_obs], axis=0)
train_actions = np.concatenate([shadow_actions, anchor_actions], axis=0)
train_weights = np.concatenate([shadow_weights, anchor_weights], axis=0)

print("shadow rows:", len(shadow_obs))
print("anchor rows:", len(anchor_obs))
print("train rows :", len(train_obs))

best_passes = base_passes
best_score = base_score
model.save(str(BEST_PATH))

device = model.device
obs_t = torch.as_tensor(train_obs, dtype=torch.float32, device=device)
actions_t = torch.as_tensor(train_actions, dtype=torch.float32, device=device)
weights_t = torch.as_tensor(train_weights, dtype=torch.float32, device=device).reshape(-1, 1)

# Collective gets the largest repair weight: V3 failed on altitude, not turn direction.
loss_w = torch.as_tensor([6.0, 1.5, 2.0, 2.0], dtype=torch.float32, device=device).reshape(1, 4)

actor_params = [
    p for name, p in model.policy.named_parameters()
    if "mlp_extractor.policy_net" in name or "action_net" in name
]
optimizer = torch.optim.Adam(actor_params, lr=LR)
indices = np.arange(len(train_obs))

for epoch in range(1, MAX_REPAIR_EPOCHS + 1):
    rng.shuffle(indices)
    total = 0.0
    batches = 0

    for start in range(0, len(indices), BATCH):
        idx_np = indices[start:start + BATCH]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        bobs = obs_t[idx]
        btgt = actions_t[idx]
        bw = weights_t[idx]

        features = model.policy.extract_features(bobs)
        if isinstance(features, tuple):
            features = features[0]
        latent = model.policy.mlp_extractor.forward_actor(features)
        pred = model.policy.action_net(latent)
        loss = (((pred - btgt) ** 2) * loss_w * bw).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 0.35)
        optimizer.step()

        total += float(loss.detach().cpu())
        batches += 1

    if epoch % EVAL_EVERY == 0:
        avg_loss = total / max(1, batches)
        print(f"\nepoch={epoch:3d} | shadow_repair_loss={avg_loss:.7f}")
        results, passes, score = evaluate_all(model, f"V4 EPOCH {epoch}")

        if passes > best_passes or (passes == best_passes and score > best_score):
            best_passes = passes
            best_score = score
            model.save(str(BEST_PATH))
            print("NEW V4 BEST saved:", BEST_PATH)

        if passes == 4:
            print("\n4/4 ACHIEVED - stopping immediately.")
            break

best = PPO.load(str(BEST_PATH))
results, passes, score = evaluate_all(best, "FINAL V4 BEST")
print("\nFINAL V4 PASS COUNT:", passes, "/ 4")
print("FINAL V4 MODEL:", BEST_PATH)
print("NOTE: teacher/controller remains OFF at runtime. Shadow teacher was used only to label actor-visited training states.")
