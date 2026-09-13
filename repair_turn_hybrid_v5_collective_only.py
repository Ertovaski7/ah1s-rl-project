from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
START_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V3_REPAIRED.zip")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY"

TARGETS = [-50.0, 50.0, 200.0, 360.0]
NEG_ROLLOUTS = 4
ANCHOR_ROLLOUTS = 2
MAX_EPOCHS = 60
BATCH = 256
LR = 5e-6

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def negative_teacher_a0(env):
    s = env._raw_state()
    alt = float(s["altitude"])
    vs = float(s["vertical_speed"])
    desired_vs = float(np.clip(0.10 * (300.0 - alt), -2.0, +0.80))
    gain = 0.010 if vs < desired_vs else 0.030
    collective = float(np.clip(0.5840 + gain * (desired_vs - vs), 0.470, 0.610))
    return env.collective_to_action(collective)


def evaluate(model, target):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    min_alt = 999.0
    try:
        for _ in range(int(env.MAX_TURN_TIME_S / env.CONTROL_DT)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            min_alt = min(min_alt, float(info.get("altitude", 999.0)))
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()
    return {
        "target": target,
        "success": success,
        "completed": float(info.get("cumulative_turn_deg", 0.0)),
        "remaining": float(info.get("remaining_turn_deg", 999.0)),
        "altitude": float(info.get("altitude", float("nan"))),
        "min_altitude": min_alt,
        "forward_velocity": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety_failure": bool(info.get("safety_failure", False)),
    }


def evaluate_all(model, label):
    rs = [evaluate(model, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in rs)
    safety = sum(int(r["safety_failure"]) for r in rs)
    err = sum(abs(r["remaining"]) for r in rs)
    alt_pen = sum(max(0.0, 285.0 - r["min_altitude"]) for r in rs)
    score = passes * 10000.0 - 100.0 * safety - err - 5.0 * alt_pen
    print(f"\n{label} | PASS={passes}/4 | score={score:.2f} | safety={safety}")
    for r in rs:
        print(
            f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | "
            f"done={r['completed']:+8.2f} | rem={r['remaining']:+7.2f} | "
            f"alt={r['altitude']:7.2f} | min_alt={r['min_altitude']:7.2f} | "
            f"v={r['forward_velocity']:6.2f} | hold={r['hold']:4.1f} | safety={r['safety_failure']}"
        )
    return passes, score


def collect_negative_shadow(model, seed):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=-50.0)
    obs, info = env.reset(seed=seed)
    rows_obs, rows_a0, rows_w = [], [], []
    try:
        for _ in range(int(env.MAX_TURN_TIME_S / env.CONTROL_DT)):
            s = env._raw_state()
            alt = float(s["altitude"])
            rem = env.target_turn_deg - env.cumulative_turn_deg
            w = 2.0
            if alt < 295.0:
                w = 5.0
            if alt < 288.0:
                w = 8.0
            if abs(rem) < 10.0:
                w *= 1.5
            rows_obs.append(np.asarray(obs, dtype=np.float32).copy())
            rows_a0.append(float(negative_teacher_a0(env)))
            rows_w.append(w)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows_obs, rows_a0, rows_w


def collect_anchor(model, target, seed):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=target)
    obs, info = env.reset(seed=seed)
    rows_obs, rows_a0, rows_w = [], [], []
    try:
        for _ in range(int(env.MAX_TURN_TIME_S / env.CONTROL_DT)):
            action, _ = model.predict(obs, deterministic=True)
            rows_obs.append(np.asarray(obs, dtype=np.float32).copy())
            rows_a0.append(float(np.asarray(action).reshape(-1)[0]))
            rows_w.append(2.0 if target == 360.0 else 1.0)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows_obs, rows_a0, rows_w


print("=" * 120)
print("TURN HYBRID V5 - COLLECTIVE-ONLY SHADOW REPAIR")
print("Only action[0] is trainable; elevator/aileron/rudder policy is frozen.")
print("=" * 120)

model = PPO.load(str(START_MODEL))
base_passes, base_score = evaluate_all(model, "START V5")
model.save(str(BEST_PATH))
best_passes, best_score = base_passes, base_score

obs_rows, a0_rows, w_rows = [], [], []
for i in range(NEG_ROLLOUTS):
    o, a, w = collect_negative_shadow(model, SEED + 10 + i)
    obs_rows += o; a0_rows += a; w_rows += w
for target in [50.0, 200.0, 360.0]:
    for i in range(ANCHOR_ROLLOUTS):
        o, a, w = collect_anchor(model, target, SEED + 100 + i)
        obs_rows += o; a0_rows += a; w_rows += w

obs_arr = np.asarray(obs_rows, dtype=np.float32)
a0_arr = np.asarray(a0_rows, dtype=np.float32)
w_arr = np.asarray(w_rows, dtype=np.float32)
print("training rows:", len(obs_arr))

device = model.device
obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
a0_t = torch.as_tensor(a0_arr, dtype=torch.float32, device=device)
w_t = torch.as_tensor(w_arr, dtype=torch.float32, device=device)

# Freeze the whole policy first.
for p in model.policy.parameters():
    p.requires_grad_(False)
# Only action head tensor is trainable; gradients for rows 1..3 are explicitly zeroed.
model.policy.action_net.weight.requires_grad_(True)
model.policy.action_net.bias.requires_grad_(True)
opt = torch.optim.Adam([model.policy.action_net.weight, model.policy.action_net.bias], lr=LR)
indices = np.arange(len(obs_arr))

for epoch in range(1, MAX_EPOCHS + 1):
    rng.shuffle(indices)
    total = 0.0
    nb = 0
    for start in range(0, len(indices), BATCH):
        idx_np = indices[start:start+BATCH]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        bobs = obs_t[idx]
        btgt = a0_t[idx]
        bw = w_t[idx]

        with torch.no_grad():
            feat = model.policy.extract_features(bobs)
            if isinstance(feat, tuple):
                feat = feat[0]
            latent = model.policy.mlp_extractor.forward_actor(feat)
        pred_all = model.policy.action_net(latent)
        pred_a0 = pred_all[:, 0]
        loss = (((pred_a0 - btgt) ** 2) * bw).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if model.policy.action_net.weight.grad is not None:
                model.policy.action_net.weight.grad[1:, :].zero_()
            if model.policy.action_net.bias.grad is not None:
                model.policy.action_net.bias.grad[1:].zero_()
        torch.nn.utils.clip_grad_norm_([model.policy.action_net.weight, model.policy.action_net.bias], 0.10)
        opt.step()
        total += float(loss.detach().cpu())
        nb += 1

    if epoch == 1 or epoch % 2 == 0:
        print(f"\nepoch={epoch:3d} | collective_loss={total/max(1,nb):.7f}")
        passes, score = evaluate_all(model, f"V5 EPOCH {epoch}")
        if passes > best_passes or (passes == best_passes and score > best_score):
            best_passes, best_score = passes, score
            model.save(str(BEST_PATH))
            print("NEW V5 BEST saved:", BEST_PATH)
        if passes == 4:
            print("\n4/4 ACHIEVED - stopping immediately.")
            break

best = PPO.load(str(BEST_PATH))
passes, score = evaluate_all(best, "FINAL V5 BEST")
print("\nFINAL V5 PASS COUNT:", passes, "/ 4")
print("FINAL V5 MODEL:", BEST_PATH)
print("NOTE: runtime teacher/controller is OFF. Only PPO action[0] collective head was repaired; turn channels were frozen.")
