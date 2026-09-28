from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry


SEEDS = [7, 21, 42, 84, 123]
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

# Keep the perturbation physically generated: after FullEntry reaches its normal
# handoff envelope, continue the frozen straight-flight Stage-2 policy for a
# small seed-dependent number of control steps, then begin the turn.
MIN_EXTRA_STEPS = 0
MAX_EXTRA_STEPS = 24


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


class RandomizedEntryEnv(HelicopterEnvTurnGoalFullEntry):
    """Full-entry environment with a small physically generated handoff jitter.

    No state variable is teleported.  We simply keep flying the existing frozen
    Stage-2 straight-flight policy for a seed-dependent number of extra control
    steps before enabling the turn.  This produces genuinely different entry
    altitude/vertical-speed/forward-speed/attitude states while staying on the
    same JSBSim trajectory family.
    """

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        rng = np.random.default_rng(seed)
        extra_steps = int(rng.integers(MIN_EXTRA_STEPS, MAX_EXTRA_STEPS + 1))

        # Temporarily restore straight-flight Stage-2 semantics.
        self.turn_active = False
        self.success_hold_s = 0.0
        self.cumulative_turn_deg = 0.0
        self.turn_steps = 0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)
        self.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        self.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        for _ in range(extra_steps):
            base_obs = self._base_obs()
            action, _ = self.stage2_model.predict(base_obs, deterministic=True)
            self._apply_action(action)

            jsbsim_ok = True
            for _ in range(self.PHYSICS_STEPS):
                if not self.fdm.run():
                    jsbsim_ok = False
                    break
            if not jsbsim_ok:
                raise RuntimeError("JSBSim stopped while applying randomized entry continuation")

            s = self._raw_state()
            self.forward_distance += float(s["forward_velocity"]) * self.CONTROL_DT

        # Re-arm the exact turn semantics after the physical continuation.
        self.turn_active = True
        self.cumulative_turn_deg = 0.0
        self.prev_heading_deg = self._heading_deg()
        self.turn_steps = 0
        self.success_hold_s = 0.0
        self.previous_turn_action = np.zeros(4, dtype=np.float32)
        self._v2_previous_remaining = float(self.target_turn_deg)
        self.fdm["ap/afcs/roll-channel-active-norm"] = self.TURN_ROLL_AFCS
        self.fdm["ap/afcs/yaw-channel-active-norm"] = self.TURN_YAW_AFCS

        s = self._raw_state()
        info = {
            **s,
            "target_turn_deg": self.target_turn_deg,
            "cumulative_turn_deg": 0.0,
            "remaining_turn_deg": self.target_turn_deg,
            "full_entry": True,
            "randomized_entry": True,
            "extra_entry_steps": extra_steps,
            "success": False,
        }
        return self._get_obs(), info


def gate_name_from_obs(obs):
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


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required model: {path}\nRun training/restore_current_turn_stack_v1.py first.")


def load_adapter(path: Path, scale):
    model = ResidualAdapter(scale)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_one(base_model, base_adapter, patch_200, patch_50, target, seed):
    env = RandomizedEntryEnv(target_turn_deg=target)
    obs, reset_info = env.reset(seed=seed)
    success = False
    info = reset_info
    last_gate = gate_name_from_obs(obs)

    entry = {
        "entry_alt": float(reset_info.get("altitude", np.nan)),
        "entry_vs": float(reset_info.get("vertical_speed", np.nan)),
        "entry_v": float(reset_info.get("forward_velocity", np.nan)),
        "entry_roll": float(reset_info.get("roll", np.nan)),
        "entry_pitch": float(reset_info.get("pitch", np.nan)),
        "extra_steps": int(reset_info.get("extra_entry_steps", 0)),
    }

    try:
        max_steps = int(max(90.0, abs(target) / 1.10 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action, last_gate = combined_action(base_model, base_adapter, patch_200, patch_50, obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "seed": seed,
        "target": target,
        "gate": last_gate,
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", np.nan)),
        "v": float(info.get("forward_velocity", np.nan)),
        "vs": float(info.get("vertical_speed", np.nan)),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
        **entry,
    }


def main():
    for path in (BASE_MODEL, BASE_ADAPTER, PATCH_200, PATCH_50):
        require_file(path)

    print("=" * 120)
    print("V16 RANDOMIZED FULL-ENTRY ROBUSTNESS EVALUATION")
    print("Same frozen V14 controller; no retraining.")
    print("Entry perturbation is created by extra real Stage-2 flight steps, not by teleporting JSBSim state.")
    print(f"extra_entry_steps sampled in [{MIN_EXTRA_STEPS}, {MAX_EXTRA_STEPS}]")
    print("Runtime teacher/controller OFF")
    print("=" * 120)

    base_model = PPO.load(str(BASE_MODEL))
    base_adapter = load_adapter(BASE_ADAPTER, BASE_SCALE)
    patch_200 = load_adapter(PATCH_200, PATCH_SCALE)
    patch_50 = load_adapter(PATCH_50, PATCH_SCALE)

    rows = []
    for seed in SEEDS:
        for target in TARGETS:
            row = evaluate_one(base_model, base_adapter, patch_200, patch_50, target, seed)
            rows.append(row)
            print(
                f"seed={seed:3d} | target={target:+7.1f} | gate={row['gate']:7s} | "
                f"entry_steps={row['extra_steps']:2d} | entry_alt={row['entry_alt']:7.2f} | "
                f"entry_vs={row['entry_vs']:+6.2f} | entry_v={row['entry_v']:6.2f} | "
                f"pass={str(row['success']):5s} | done={row['done']:+8.2f} | rem={row['rem']:+7.2f} | "
                f"final_alt={row['alt']:7.2f} | hold={row['hold']:4.1f} | safety={row['safety']}"
            )

    print("\n" + "=" * 120)
    print("V16 SUMMARY")
    print("=" * 120)

    total_pass = 0
    total_safety = 0
    for target in TARGETS:
        part = [r for r in rows if r["target"] == target]
        p = sum(int(r["success"]) for r in part)
        s = sum(int(r["safety"]) for r in part)
        total_pass += p
        total_safety += s
        mean_err = float(np.mean([abs(r["rem"]) for r in part]))
        max_err = float(np.max([abs(r["rem"]) for r in part]))
        entry_alt_min = float(np.min([r["entry_alt"] for r in part]))
        entry_alt_max = float(np.max([r["entry_alt"] for r in part]))
        entry_v_min = float(np.min([r["entry_v"] for r in part]))
        entry_v_max = float(np.max([r["entry_v"] for r in part]))
        print(
            f"target={target:+7.1f} | pass={p}/{len(part)} | safety_fail={s}/{len(part)} | "
            f"mean_abs_err={mean_err:6.2f} | max_abs_err={max_err:6.2f} | "
            f"entry_alt=[{entry_alt_min:.2f},{entry_alt_max:.2f}] | entry_v=[{entry_v_min:.2f},{entry_v_max:.2f}]"
        )

    print(f"\nRANDOMIZED-ENTRY TOTAL: PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={total_safety}/{len(rows)}")
    if total_pass == len(rows) and total_safety == 0:
        print("V16 ROBUSTNESS: PASS")
    else:
        print("V16 ROBUSTNESS: PARTIAL/FAIL")
        print("This does not erase V14 4/4; it identifies sensitivity to realistic handoff variation.")


if __name__ == "__main__":
    main()
