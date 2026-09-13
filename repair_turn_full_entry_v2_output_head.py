from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry

SEED = 42
START_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V2_HEAD_REPAIRED"

TARGETS = [-50.0, 50.0, 200.0, 360.0]
ROLLOUTS_PER_TARGET = 2
MAX_EPOCHS = 40
BATCH = 256
LR = 2e-5
EVAL_EVERY = 1

rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        ff = 0.5840
        desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.00, +0.80))
        gain = 0.010 if vertical_speed < desired_vs else 0.030
        return float(np.clip(ff + gain * (desired_vs - vertical_speed), 0.470, 0.610))
    ff = 0.5780
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.50, +0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    return float(np.clip(ff + gain * (desired_vs - vertical_speed), 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg
    collective = teacher_collective(env.target_turn_deg, float(s["altitude"]), float(s["vertical_speed"]))
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
        a2, a3 = maneuver_teacher(s, remaining)

    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def evaluate(model, target):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
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


def evaluate_all(model, label):
    rows = [evaluate(model, t) for t in TARGETS]
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


def collect_teacher_rollout(target, seed):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=seed)
    obs_rows, act_rows, wt_rows = [], [], []
    captured = False
    capture_hold_s = 0.0
    heading_integral = 0.0
    success = False
    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action, captured, capture_hold_s, heading_integral = teacher_action(
                env, captured, capture_hold_s, heading_integral
            )
            remaining = env.target_turn_deg - env.cumulative_turn_deg
            w = 1.0
            if abs(remaining) < 25.0:
                w = 3.0
            if abs(remaining) < 8.0:
                w = 7.0
            # Protect the already-working negative task while focusing repair on positives.
            if target < 0.0:
                w *= 0.45
            else:
                w *= 1.5
            obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
            act_rows.append(action.copy())
            wt_rows.append(float(w))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()
    return obs_rows, act_rows, wt_rows, success, info


if not START_MODEL.exists():
    raise FileNotFoundError(START_MODEL)

print("=" * 120)
print("TURN FULL-ENTRY V2 - OUTPUT-HEAD REPAIR AFTER PPO")
print("Start model:", START_MODEL)
print("Shared PPO policy network: FROZEN")
print("Only action_net output head is updated; runtime teacher/controller: OFF")
print("=" * 120)

model = PPO.load(str(START_MODEL))
base_rows, best_passes, best_score = evaluate_all(model, "START V2")
model.save(str(BEST_PATH))

# First prove that the deterministic teacher itself solves the full-entry distribution.
all_obs, all_act, all_w = [], [], []
teacher_passes = 0
for target in TARGETS:
    for rollout in range(ROLLOUTS_PER_TARGET):
        ro, ra, rw, success, info = collect_teacher_rollout(target, SEED + rollout)
        all_obs.extend(ro)
        all_act.extend(ra)
        all_w.extend(rw)
        teacher_passes += int(success)
        print(
            f"teacher target={target:+7.1f} rollout={rollout+1} | success={success} | "
            f"done={float(info.get('cumulative_turn_deg',0.0)):+.2f} | "
            f"alt={float(info.get('altitude',0.0)):.2f}"
        )

expected_teacher_passes = len(TARGETS) * ROLLOUTS_PER_TARGET
if teacher_passes != expected_teacher_passes:
    raise RuntimeError(
        f"Full-entry teacher is not qualified: {teacher_passes}/{expected_teacher_passes} rollouts passed. "
        "Abort repair rather than distilling a failing teacher."
    )

obs_np = np.asarray(all_obs, dtype=np.float32)
act_np = np.asarray(all_act, dtype=np.float32)
wt_np = np.asarray(all_w, dtype=np.float32)
print("training rows:", len(obs_np))

device = model.device
obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
act_t = torch.as_tensor(act_np, dtype=torch.float32, device=device)
wt_t = torch.as_tensor(wt_np, dtype=torch.float32, device=device).reshape(-1, 1)
loss_w = torch.as_tensor([5.0, 1.5, 3.0, 3.0], dtype=torch.float32, device=device).reshape(1, 4)

# Freeze everything except the final linear action output head.
for p in model.policy.parameters():
    p.requires_grad = False
for p in model.policy.action_net.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(model.policy.action_net.parameters(), lr=LR)
indices = np.arange(len(obs_np))

for epoch in range(1, MAX_EPOCHS + 1):
    rng.shuffle(indices)
    total = 0.0
    batches = 0
    for start in range(0, len(indices), BATCH):
        idx_np = indices[start:start + BATCH]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        bobs = obs_t[idx]
        btgt = act_t[idx]
        bw = wt_t[idx]

        with torch.no_grad():
            features = model.policy.extract_features(bobs)
            if isinstance(features, tuple):
                features = features[0]
            latent = model.policy.mlp_extractor.forward_actor(features)

        pred = model.policy.action_net(latent)
        loss = (((pred - btgt) ** 2) * loss_w * bw).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.action_net.parameters(), 0.25)
        optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1

    if epoch % EVAL_EVERY == 0:
        print(f"\nepoch={epoch:3d} | head_repair_loss={total/max(1,batches):.7f}")
        rows, passes, score = evaluate_all(model, f"V2 EPOCH {epoch}")
        if passes > best_passes or (passes == best_passes and score > best_score):
            best_passes = passes
            best_score = score
            model.save(str(BEST_PATH))
            print("NEW V2 BEST:", BEST_PATH)
        if passes == 4:
            print("\n4/4 FULL-ENTRY ACHIEVED - stopping immediately.")
            break

best = PPO.load(str(BEST_PATH))
rows, passes, score = evaluate_all(best, "FINAL FULL-ENTRY V2 BEST")
print("\nFINAL FULL-ENTRY V2 PASS COUNT:", passes, "/ 4")
print("FINAL MODEL:", BEST_PATH)
print("NOTE: model already contains PPO RL training; this is a post-RL output-head repair. Teacher/controller is OFF at runtime.")
