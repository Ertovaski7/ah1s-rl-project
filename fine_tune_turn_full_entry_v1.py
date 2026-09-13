from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry

SEED = 42
START_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V1_BEST"
LAST_PATH = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V1_LAST"

TARGETS = [-50.0, 50.0, 200.0, 360.0]
CHUNK_STEPS = 10000
MAX_CHUNKS = 6
LR = 5e-6

np.random.seed(SEED)
torch.manual_seed(SEED)


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
            f"done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | "
            f"alt={r['alt']:7.2f} | v={r['v']:6.2f} | "
            f"hold={r['hold']:4.1f} | safety={r['safety']}"
        )
    return passes, score


if not START_MODEL.exists():
    raise FileNotFoundError(START_MODEL)

print("=" * 120)
print("TURN FULL-ENTRY V1 - CONSERVATIVE PPO TRANSFER")
print("Start model:", START_MODEL)
print("Runtime teacher/controller: OFF")
print("Learning rate:", LR)
print("=" * 120)

train_env = Monitor(HelicopterEnvTurnGoalFullEntry(target_choices=TARGETS))
model = PPO.load(str(START_MODEL), env=train_env)

# Preserve V5 behavior: tiny LR and low exploration.
model.learning_rate = LR
model.lr_schedule = lambda _: LR
with torch.no_grad():
    model.policy.log_std.fill_(float(np.log(0.05)))
model.ent_coef = 0.0
model.clip_range = lambda _: 0.05

best_passes, best_score = evaluate_all(model, "START FULL-ENTRY")
model.save(str(BEST_PATH))

if best_passes == 4:
    print("\nAlready 4/4 on full-entry reset. No additional RL required.")
else:
    for chunk in range(1, MAX_CHUNKS + 1):
        print("\n" + "-" * 120)
        print(f"PPO FULL-ENTRY CHUNK {chunk}/{MAX_CHUNKS} | +{CHUNK_STEPS} timesteps")
        print("-" * 120)
        model.learn(
            total_timesteps=CHUNK_STEPS,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.save(str(LAST_PATH))

        passes, score = evaluate_all(model, f"AFTER CHUNK {chunk}")
        if passes > best_passes or (passes == best_passes and score > best_score):
            best_passes = passes
            best_score = score
            model.save(str(BEST_PATH))
            print("NEW BEST:", BEST_PATH)

        if passes == 4:
            print("\n4/4 FULL-ENTRY ACHIEVED - stopping.")
            break

best = PPO.load(str(BEST_PATH))
passes, score = evaluate_all(best, "FINAL FULL-ENTRY BEST")
print("\nFINAL FULL-ENTRY PASS COUNT:", passes, "/ 4")
print("FINAL MODEL:", BEST_PATH)
print("NOTE: teacher/controller remains OFF during PPO training and evaluation.")
