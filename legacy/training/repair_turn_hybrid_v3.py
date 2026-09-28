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
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
MODEL_PATH = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V2_LAST.zip")
DATASET_PATH = Path("results_turn_hybrid/turn_teacher_dataset_v1.npz")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "AH1S_TURN_HYBRID_V3_REPAIRED"

TARGETS = [-50.0, 50.0, 200.0, 360.0]
MAX_EPOCHS = 80
EVAL_EVERY = 5
BATCH = 256
LR = 5e-5

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH)


def evaluate(model, target):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    try:
        max_steps = int(env.MAX_TURN_TIME_S / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
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
        "forward_velocity": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety_failure": bool(info.get("safety_failure", False)),
    }


def evaluate_all(model, label):
    results = [evaluate(model, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in results)
    error_sum = sum(abs(r["remaining"]) for r in results)
    safety_count = sum(int(r["safety_failure"]) for r in results)
    print(f"\n{label} | PASS={passes}/4 | abs_remaining_sum={error_sum:.2f} | safety={safety_count}")
    for r in results:
        print(
            f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | "
            f"done={r['completed']:+8.2f} | rem={r['remaining']:+7.2f} | "
            f"alt={r['altitude']:7.2f} | v={r['forward_velocity']:6.2f} | "
            f"hold={r['hold']:4.1f} | safety={r['safety_failure']}"
        )
    score = passes * 10000.0 - 50.0 * safety_count - error_sum
    return results, passes, score


print("=" * 120)
print("TURN HYBRID V3 - TARGETED POST-RL REPAIR")
print("Start model:", MODEL_PATH)
print("Goal: preserve +50/+200/+360 while repairing -50 without teacher at runtime")
print("=" * 120)

model = PPO.load(str(MODEL_PATH))

data = np.load(DATASET_PATH)
obs = np.asarray(data["obs"], dtype=np.float32)
actions = np.asarray(data["actions"], dtype=np.float32)
weights = np.asarray(data["weights"], dtype=np.float32)

# Goal feature index 12 stores target_turn/360.
target_deg = obs[:, 12] * 360.0
neg_mask = target_deg < -1.0
pos360_mask = target_deg > 300.0
anchor_mask = ~neg_mask

# Emphasize the failing negative task but retain positive anchors, especially 360.
sample_w = weights.copy()
sample_w[neg_mask] *= 4.0
sample_w[anchor_mask] *= 0.35
sample_w[pos360_mask] *= 3.0

print("dataset rows:", len(obs))
print("negative rows:", int(neg_mask.sum()))
print("360 anchor rows:", int(pos360_mask.sum()))

base_results, base_passes, base_score = evaluate_all(model, "START")
best_passes = base_passes
best_score = base_score
model.save(str(BEST_PATH))

device = model.device
obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
weights_t = torch.as_tensor(sample_w, dtype=torch.float32, device=device).reshape(-1, 1)
loss_w = torch.as_tensor([2.0, 1.5, 2.5, 2.5], dtype=torch.float32, device=device).reshape(1, 4)

actor_params = [
    p for name, p in model.policy.named_parameters()
    if "mlp_extractor.policy_net" in name or "action_net" in name
]
opt = torch.optim.Adam(actor_params, lr=LR)
idx_all = np.arange(len(obs))

for epoch in range(1, MAX_EPOCHS + 1):
    rng.shuffle(idx_all)
    total = 0.0
    nb = 0
    for start in range(0, len(idx_all), BATCH):
        idx_np = idx_all[start:start + BATCH]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        bobs = obs_t[idx]
        btgt = actions_t[idx]
        bw = weights_t[idx]

        feat = model.policy.extract_features(bobs)
        if isinstance(feat, tuple):
            feat = feat[0]
        latent = model.policy.mlp_extractor.forward_actor(feat)
        pred = model.policy.action_net(latent)
        loss = (((pred - btgt) ** 2) * loss_w * bw).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
        opt.step()
        total += float(loss.detach().cpu())
        nb += 1

    if epoch == 1 or epoch % EVAL_EVERY == 0:
        print(f"\nepoch={epoch:3d} | repair_loss={total/max(1,nb):.7f}")
        results, passes, score = evaluate_all(model, f"REPAIR EPOCH {epoch}")
        if passes > best_passes or (passes == best_passes and score > best_score):
            best_passes = passes
            best_score = score
            model.save(str(BEST_PATH))
            print("NEW V3 BEST saved:", BEST_PATH)
        if passes == 4:
            print("\n4/4 ACHIEVED - early stop.")
            break

best = PPO.load(str(BEST_PATH))
results, passes, score = evaluate_all(best, "FINAL V3 BEST")
print("\nFINAL V3 PASS COUNT:", passes, "/ 4")
print("FINAL V3 MODEL:", BEST_PATH)
print("NOTE: runtime teacher/controller remains OFF; this is post-RL repair distillation of the already RL-trained V2 actor.")
