from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry


# Canonical targets already validated by V14.
CANONICAL_TARGETS = [-50.0, 50.0, 200.0, 360.0]

# Extra targets are exploratory only. The specialist patches are intentionally
# hard-gated, so most of these run only through frozen V5 + frozen V4.
EXPLORATORY_TARGETS = [-25.0, 100.0, 150.0, 270.0]

# Multiple reset seeds check repeatability / sensitivity to any seeded reset
# variation in the environment. If the underlying reset is fully deterministic,
# identical rows across seeds are still a useful reproducibility result.
SEEDS = [7, 21, 42, 84, 123]

BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
BASE_ADAPTER = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt")
PATCH_200 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt")
OUT_CSV = Path("v15_robustness_results.csv")

BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)
TARGET_50_NORM = 50.0 / 360.0
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02


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
    target_norm = float(obs[12])
    if abs(target_norm - TARGET_50_NORM) <= GATE_TOL:
        return "V13_50"
    if abs(target_norm - TARGET_200_NORM) <= GATE_TOL:
        return "V7_200"
    return "NONE"


def combined_action(base_model, base_adapter, patch_200, patch_50, obs):
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

    action = np.asarray(base, dtype=np.float32) + d_base + d_patch
    return np.clip(action, -1.0, 1.0), gate


def evaluate_once(base_model, base_adapter, patch_200, patch_50, target, seed, suite):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=target)
    obs, info = env.reset(seed=seed)
    success = False
    last_gate = "NONE"
    steps = 0

    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for steps in range(1, max_steps + 1):
            action, last_gate = combined_action(
                base_model, base_adapter, patch_200, patch_50, obs
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "suite": suite,
        "seed": seed,
        "target": target,
        "gate": last_gate,
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", float("nan"))),
        "v": float(info.get("forward_velocity", float("nan"))),
        "vs": float(info.get("vertical_speed", float("nan"))),
        "roll_deg": float(np.degrees(info.get("roll", float("nan")))),
        "pitch_deg": float(np.degrees(info.get("pitch", float("nan")))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
        "steps": steps,
    }


def print_row(r):
    print(
        f"  {r['suite']:11s} | seed={r['seed']:3d} | "
        f"target={r['target']:+7.1f} | gate={r['gate']:7s} | "
        f"pass={str(r['success']):5s} | done={r['done']:+8.2f} | "
        f"rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | "
        f"v={r['v']:6.2f} | vs={r['vs']:+6.2f} | "
        f"hold={r['hold']:4.1f} | safety={r['safety']}"
    )


def summarize(rows, suite, targets):
    subset = [r for r in rows if r["suite"] == suite]
    print("\n" + "=" * 120)
    print(f"{suite.upper()} SUMMARY")
    print("=" * 120)

    for target in targets:
        rs = [r for r in subset if r["target"] == target]
        passes = sum(int(r["success"]) for r in rs)
        safety = sum(int(r["safety"]) for r in rs)
        mean_abs_err = float(np.mean([abs(r["rem"]) for r in rs]))
        max_abs_err = float(np.max([abs(r["rem"]) for r in rs]))
        mean_alt = float(np.mean([r["alt"] for r in rs]))
        print(
            f"target={target:+7.1f} | pass={passes}/{len(rs)} | "
            f"safety_fail={safety}/{len(rs)} | "
            f"mean_abs_err={mean_abs_err:6.2f} | max_abs_err={max_abs_err:6.2f} | "
            f"mean_final_alt={mean_alt:7.2f}"
        )

    passes = sum(int(r["success"]) for r in subset)
    safety = sum(int(r["safety"]) for r in subset)
    print(f"\n{suite.upper()} TOTAL: PASS={passes}/{len(subset)} | SAFETY_FAILURES={safety}/{len(subset)}")
    return passes, len(subset), safety


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required model: {path}\nRun restore_post_v5_for_v14.py first."
        )


def load_adapter(path: Path, scale):
    model = ResidualAdapter(scale)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def write_csv(rows):
    fieldnames = [
        "suite", "seed", "target", "gate", "success", "done", "rem",
        "alt", "v", "vs", "roll_deg", "pitch_deg", "hold", "safety", "steps"
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV SAVED:", OUT_CSV)


def main():
    for path in (BASE_MODEL, BASE_ADAPTER, PATCH_200, PATCH_50):
        require_file(path)

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 120)
    print("V15 ROBUSTNESS / GENERALIZATION EVALUATION")
    print("Frozen V5 PPO + frozen V4 adapter + V13(+50) / V7(+200) mutually-exclusive hard gates")
    print("Canonical suite checks repeatability on the four validated targets.")
    print("Exploratory suite probes unseen target angles; failures there do NOT invalidate the V14 4/4 result.")
    print("Runtime teacher/controller OFF")
    print("=" * 120)

    base_model = PPO.load(str(BASE_MODEL))
    base_adapter = load_adapter(BASE_ADAPTER, BASE_SCALE)
    patch_200 = load_adapter(PATCH_200, PATCH_SCALE)
    patch_50 = load_adapter(PATCH_50, PATCH_SCALE)

    rows = []

    print("\nCANONICAL REPEATABILITY RUNS")
    for seed in SEEDS:
        for target in CANONICAL_TARGETS:
            row = evaluate_once(
                base_model, base_adapter, patch_200, patch_50,
                target=target, seed=seed, suite="canonical"
            )
            rows.append(row)
            print_row(row)

    print("\nEXPLORATORY UNSEEN-TARGET RUNS")
    for seed in SEEDS:
        for target in EXPLORATORY_TARGETS:
            row = evaluate_once(
                base_model, base_adapter, patch_200, patch_50,
                target=target, seed=seed, suite="exploratory"
            )
            rows.append(row)
            print_row(row)

    canonical_passes, canonical_total, canonical_safety = summarize(
        rows, "canonical", CANONICAL_TARGETS
    )
    exploratory_passes, exploratory_total, exploratory_safety = summarize(
        rows, "exploratory", EXPLORATORY_TARGETS
    )

    write_csv(rows)

    print("\n" + "=" * 120)
    print("V15 FINAL")
    print("=" * 120)
    print(f"Canonical:   {canonical_passes}/{canonical_total} pass | safety failures={canonical_safety}")
    print(f"Exploratory: {exploratory_passes}/{exploratory_total} pass | safety failures={exploratory_safety}")

    if canonical_passes == canonical_total and canonical_safety == 0:
        print("CANONICAL ROBUSTNESS: PASS")
    else:
        print("CANONICAL ROBUSTNESS: NEEDS REVIEW")

    print("Exploratory results are diagnostic only; inspect which unseen angles generalize before changing training.")


if __name__ == "__main__":
    main()
