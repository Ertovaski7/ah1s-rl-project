from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2


SEED = 42
TARGETS = [-50.0, 50.0, 200.0, 360.0]
TRAIN_TARGETS = [-50.0, 50.0, 90.0, 200.0, 360.0]

OUT_DIR = Path("models_turn_hybrid")
RESULT_DIR = Path("results_turn_hybrid")
BC_MODEL_PATH = OUT_DIR / "AH1S_TURN_BC_WARMSTART.zip"
DATASET_PATH = RESULT_DIR / "turn_teacher_dataset_v1.npz"
BEST_MODEL_PATH = OUT_DIR / "AH1S_TURN_HYBRID_V2_BEST"
LAST_MODEL_PATH = OUT_DIR / "AH1S_TURN_HYBRID_V2_LAST"
RESULT_PATH = RESULT_DIR / "turn_rl_v2_history.json"

RL_CHUNK_STEPS = 10_000
RL_CHUNKS = 8
RL_LR = 1e-5
EXPLORATION_STD = 0.10
BC_REHEARSAL_EPOCHS = 12
BC_REHEARSAL_LR = 1e-4
BC_BATCH = 256

if not BC_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing BC model: {BC_MODEL_PATH}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Missing teacher dataset: {DATASET_PATH}")

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def evaluate(model, target, detailed=False):
    env = HelicopterEnvTurnGoalV2(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    next_print = 0.0
    success = False
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
                next_print += 5.0
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


def score_results(results):
    passes = sum(int(r["success"]) for r in results)
    safety = sum(int(r["safety_failure"]) for r in results)
    abs_remaining = sum(min(abs(r["remaining"]), 180.0) for r in results)
    hold_sum = sum(r["hold"] for r in results)
    # Lexicographic-like scalar: passing dominates everything else.
    return 10000.0 * passes - 2000.0 * safety - 10.0 * abs_remaining + hold_sum


def validate(model, label, detailed=False):
    results = [evaluate(model, t, detailed=detailed) for t in TARGETS]
    passes = sum(int(r["success"]) for r in results)
    score = score_results(results)
    print(f"\n{label} | PASS={passes}/{len(TARGETS)} | SCORE={score:.2f}")
    for r in results:
        print(
            f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | "
            f"done={r['completed']:+8.2f} | rem={r['remaining']:+7.2f} | "
            f"alt={r['altitude']:7.2f} | v={r['forward_velocity']:6.2f} | "
            f"hold={r['hold']:4.1f} | safety={r['safety_failure']}"
        )
    return results, score


def bc_rehearsal(model, dataset_obs, dataset_actions, dataset_weights):
    device = model.device
    obs_tensor = torch.as_tensor(dataset_obs, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(dataset_actions, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(dataset_weights, dtype=torch.float32, device=device)
    action_weight = torch.as_tensor([2.0, 1.5, 2.5, 2.5], dtype=torch.float32, device=device).reshape(1, 4)

    actor_params = [
        p for name, p in model.policy.named_parameters()
        if "mlp_extractor.policy_net" in name or "action_net" in name
    ]
    opt = torch.optim.Adam(actor_params, lr=BC_REHEARSAL_LR)
    idx_all = np.arange(len(dataset_obs))

    last_loss = None
    for _ in range(BC_REHEARSAL_EPOCHS):
        rng.shuffle(idx_all)
        total = 0.0
        batches = 0
        for start in range(0, len(idx_all), BC_BATCH):
            idx_np = idx_all[start:start + BC_BATCH]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            batch_obs = obs_tensor[idx]
            batch_target = target_tensor[idx]
            batch_weight = weight_tensor[idx].reshape(-1, 1)

            features = model.policy.extract_features(batch_obs)
            if isinstance(features, tuple):
                features = features[0]
            latent = model.policy.mlp_extractor.forward_actor(features)
            pred = model.policy.action_net(latent)
            loss = (((pred - batch_target) ** 2) * action_weight * batch_weight).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 0.5)
            opt.step()
            total += float(loss.detach().cpu())
            batches += 1
        last_loss = total / max(1, batches)
    return float(last_loss if last_loss is not None else 0.0)


print("=" * 120)
print("TURN PPO V2 - CONSERVATIVE FINE-TUNING + BC REHEARSAL")
print("=" * 120)
print("BC model:", BC_MODEL_PATH)
print("RL total max steps:", RL_CHUNK_STEPS * RL_CHUNKS)
print("RL learning rate:", RL_LR)
print("Target exploration std:", EXPLORATION_STD)

train_env = Monitor(HelicopterEnvTurnGoalV2(target_choices=TRAIN_TARGETS))
model = PPO.load(str(BC_MODEL_PATH), env=train_env, device="auto")

# Conservative PPO settings after imitation learning.
model.learning_rate = RL_LR
model.lr_schedule = lambda _: RL_LR
model.n_epochs = 3
model.clip_range = lambda _: 0.08
model.ent_coef = 0.0
model.max_grad_norm = 0.3

# Start close to deterministic teacher behavior instead of std ~1.0.
with torch.no_grad():
    model.policy.log_std.data.fill_(float(np.log(EXPLORATION_STD)))

pack = np.load(DATASET_PATH)
dataset_obs = np.asarray(pack["obs"], dtype=np.float32)
dataset_actions = np.asarray(pack["actions"], dtype=np.float32)
dataset_weights = np.asarray(pack["weights"], dtype=np.float32)

history = []

baseline_results, best_score = validate(model, "BC BASELINE", detailed=False)
model.save(str(BEST_MODEL_PATH))
print("Initial best checkpoint = BC warm-start")

for chunk in range(1, RL_CHUNKS + 1):
    print("\n" + "-" * 120)
    print(f"RL CHUNK {chunk}/{RL_CHUNKS} | +{RL_CHUNK_STEPS} timesteps")
    print("-" * 120)

    model.learn(
        total_timesteps=RL_CHUNK_STEPS,
        reset_num_timesteps=(chunk == 1),
        progress_bar=True,
    )

    rl_results, rl_score = validate(model, f"AFTER PPO CHUNK {chunk}", detailed=False)

    rehearsal_loss = bc_rehearsal(
        model,
        dataset_obs,
        dataset_actions,
        dataset_weights,
    )
    print(f"BC rehearsal loss after chunk {chunk}: {rehearsal_loss:.7f}")

    post_results, post_score = validate(model, f"AFTER REHEARSAL {chunk}", detailed=False)

    record = {
        "chunk": chunk,
        "timesteps_total": chunk * RL_CHUNK_STEPS,
        "ppo_score": rl_score,
        "post_rehearsal_score": post_score,
        "rehearsal_loss": rehearsal_loss,
        "results": post_results,
    }
    history.append(record)

    if post_score > best_score:
        best_score = post_score
        model.save(str(BEST_MODEL_PATH))
        print(f"NEW BEST CHECKPOINT | score={best_score:.2f}")
    else:
        print(f"No improvement over best score={best_score:.2f}; best checkpoint unchanged.")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

model.save(str(LAST_MODEL_PATH))
print("\nTraining finished.")
print("Best model:", BEST_MODEL_PATH)
print("Last model:", LAST_MODEL_PATH)
print("History:", RESULT_PATH)

best_model = PPO.load(str(BEST_MODEL_PATH), device="auto")
final_results, final_score = validate(best_model, "FINAL BEST CHECKPOINT", detailed=True)
passes = sum(int(r["success"]) for r in final_results)
print("\nFINAL BEST PASS COUNT:", passes, "/", len(final_results))
print("FINAL BEST SCORE:", final_score)
