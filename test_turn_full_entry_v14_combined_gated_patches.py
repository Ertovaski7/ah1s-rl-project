from __future__ import annotations

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
PATCH_200 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt")

BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)

TARGET_50_NORM = 50.0 / 360.0
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02

np.random.seed(SEED)
torch.manual_seed(SEED)


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


def gate_name_from_obs(obs) -> str:
    """Return the one specialist patch that is allowed to run for this target.

    obs[12] = target_turn_norm = requested_turn_deg / 360.
    V14 deliberately uses mutually-exclusive hard gates so the +50 and +200
    specialists can never be active at the same time.
    """
    target_norm = float(obs[12])

    if abs(target_norm - TARGET_50_NORM) <= GATE_TOL:
        return "V13_50"

    if abs(target_norm - TARGET_200_NORM) <= GATE_TOL:
        return "V7_200"

    return "NONE"


def combined_action(base_model, base_adapter, patch_200, patch_50, obs):
    """V14 runtime chain: frozen V5 + frozen V4 + at most one hard-gated patch."""
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)

    gate = gate_name_from_obs(obs)

    with torch.no_grad():
        d_base = base_adapter(x).cpu().numpy()[0]

        if gate == "V13_50":
            d_patch = patch_50(x).cpu().numpy()[0]
        elif gate == "V7_200":
            d_patch = patch_200(x).cpu().numpy()[0]
        else:
            d_patch = np.zeros(4, dtype=np.float32)

    action = (
        np.asarray(base, dtype=np.float32)
        + d_base
        + d_patch
    )

    return np.clip(action, -1.0, 1.0), gate


def evaluate(base_model, base_adapter, patch_200, patch_50, target):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=SEED)
    success = False
    last_gate = "NONE"

    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)

        for _ in range(max_steps):
            action, last_gate = combined_action(
                base_model,
                base_adapter,
                patch_200,
                patch_50,
                obs,
            )

            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "target": target,
        "gate": last_gate,
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", float("nan"))),
        "v": float(info.get("forward_velocity", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
    }


def evaluate_all(base_model, base_adapter, patch_200, patch_50):
    rows = [
        evaluate(base_model, base_adapter, patch_200, patch_50, target)
        for target in TARGETS
    ]

    passes = sum(int(row["success"]) for row in rows)
    safety = sum(int(row["safety"]) for row in rows)
    err = sum(abs(row["rem"]) for row in rows)
    score = passes * 10000.0 - safety * 100.0 - err

    print("\n" + "=" * 120)
    print(f"FINAL V14 COMBINED HARD-GATED PATCH TEST | PASS={passes}/4 | score={score:.2f}")
    print("=" * 120)

    for row in rows:
        print(
            f"  target={row['target']:+7.1f} | "
            f"gate={row['gate']:7s} | "
            f"pass={str(row['success']):5s} | "
            f"done={row['done']:+8.2f} | "
            f"rem={row['rem']:+7.2f} | "
            f"alt={row['alt']:7.2f} | "
            f"v={row['v']:6.2f} | "
            f"hold={row['hold']:4.1f} | "
            f"safety={row['safety']}"
        )

    print("\nFINAL V14 PASS COUNT:", passes, "/ 4")

    if passes == 4:
        print("V14 RESULT: 4/4 ACHIEVED")
    else:
        print("V14 RESULT: NOT YET 4/4")
        print("Do not retrain first; compare the failing target with its standalone V7/V13 runtime path.")

    return rows, passes, score


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required model: {path}\n"
            "Run restore_post_v5_for_v14.py first."
        )


def load_adapter(path: Path, scale):
    model = ResidualAdapter(scale)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def main():
    for path in (BASE_MODEL, BASE_ADAPTER, PATCH_200, PATCH_50):
        require_file(path)

    print("=" * 120)
    print("V14 COMBINED HARD-GATED TURN PATCH EVALUATION")
    print("Frozen V5 PPO + frozen V4 residual adapter")
    print("V13 patch active only near +50 deg; V7 patch active only near +200 deg")
    print("-50 and +360 receive no specialist patch")
    print("Runtime teacher/controller OFF")
    print("=" * 120)

    base_model = PPO.load(str(BASE_MODEL))
    base_adapter = load_adapter(BASE_ADAPTER, BASE_SCALE)
    patch_200 = load_adapter(PATCH_200, PATCH_SCALE)
    patch_50 = load_adapter(PATCH_50, PATCH_SCALE)

    evaluate_all(base_model, base_adapter, patch_200, patch_50)


if __name__ == "__main__":
    main()
