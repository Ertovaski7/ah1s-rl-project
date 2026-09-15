from __future__ import annotations

"""
ARBITRARY RELATIVE TURN V2 — DYNAMIC CAPTURE HANDOFF
=====================================================

This is a non-destructive runtime experiment on top of the verified V22 stack.
No model weights are changed.

Idea:
- V22 turn policy performs the maneuver.
- For unseen/generalized targets, once the aircraft enters a dynamic braking
  window based on remaining heading error and current yaw rate, hand control
  back to the already-trained Stage-2 straight-flight policy / AFCS.
- The same JSBSim FDM continues; there is no reset or state teleport.
- HelicopterEnvTurnGoal.step() continues to accumulate heading and apply the
  original success/safety criteria during capture.

Known V22 specialist targets (+50 and +200 gate regions) keep their existing
V22 path so this test does not disturb the validated specialist behavior.
"""

import argparse
import math

import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22

DEFAULT_TARGETS = [20.0, -30.0]
DEFAULT_SEEDS = [42]
ROBUST_SEEDS = [7, 21, 42, 84, 123]

# Dynamic capture: lead angle ~= |yaw_rate| * LEAD_TIME + margin.
LEAD_TIME_S = 1.00
LEAD_MARGIN_DEG = 0.60
MIN_CAPTURE_DEG = 2.00
MAX_CAPTURE_DEG = 6.00


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--robust", action="store_true")
    return p.parse_args()


def in_existing_specialist_gate(target: float) -> bool:
    tn = float(target) / 360.0
    return (
        abs(tn - v22.T50) <= v16.GATE_TOL
        or abs(tn - v22.T200) <= v16.GATE_TOL
    )


def raw_yaw_rate_deg(env) -> float:
    s = env._raw_state()
    return math.degrees(float(s["r_rate"]))


def capture_threshold_deg(yaw_rate_deg: float) -> float:
    return float(np.clip(
        abs(float(yaw_rate_deg)) * LEAD_TIME_S + LEAD_MARGIN_DEG,
        MIN_CAPTURE_DEG,
        MAX_CAPTURE_DEG,
    ))


def enter_capture(env):
    # Same FDM. Only switch the control semantics back to the already learned
    # straight-flight Stage-2 policy and restore stronger AFCS stabilization.
    env.turn_active = False
    env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0


def run_one(stack, target: float, seed: int):
    bm, ba, p200, p50, p21 = stack
    env = v16.RandomizedEntryEnv(target_turn_deg=float(target))
    obs, reset_info = env.reset(seed=int(seed))
    info = reset_info
    capture = False
    capture_at = None
    capture_yaw = None
    capture_lead = None
    capture_remaining = None
    specialist = in_existing_specialist_gate(target)
    success = False

    max_steps = int(max(100.0, abs(target) / 1.0 + 80.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
            remaining = float(target) - float(env.cumulative_turn_deg)
            yaw_rate = raw_yaw_rate_deg(env)

            if not specialist and not capture:
                threshold = capture_threshold_deg(yaw_rate)
                moving_toward = (remaining * yaw_rate) > 0.0
                meaningful_progress = abs(float(env.cumulative_turn_deg)) >= min(5.0, 0.25 * abs(float(target)))

                if moving_toward and meaningful_progress and abs(remaining) <= threshold:
                    capture = True
                    capture_at = float(env.turn_steps * env.CONTROL_DT)
                    capture_yaw = float(yaw_rate)
                    capture_lead = float(threshold)
                    capture_remaining = float(remaining)
                    enter_capture(env)

            if capture:
                base_obs = env._base_obs()
                action, _ = env.stage2_model.predict(base_obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                # Keep strong roll/yaw stabilization armed during terminal capture.
                env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
                env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
            else:
                action = v22.act(bm, ba, p200, p50, p21, obs)

            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)

            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        env.close()

    return {
        "seed": int(seed),
        "target": float(target),
        "extra": int(reset_info.get("extra_entry_steps", 0)),
        "success": bool(success),
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", np.nan)),
        "vs": float(info.get("vertical_speed", np.nan)),
        "v": float(info.get("forward_velocity", np.nan)),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safe": bool(info.get("safety_failure", False)),
        "capture": bool(capture),
        "capture_at": capture_at,
        "capture_remaining": capture_remaining,
        "capture_yaw": capture_yaw,
        "capture_lead": capture_lead,
        "specialist": specialist,
    }


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    if args.robust and args.seeds is not None:
        raise ValueError("Use either --robust or --seeds, not both")
    seeds = ROBUST_SEEDS if args.robust else (args.seeds or DEFAULT_SEEDS)

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    stack = arb.load_stack()
    rows = []

    print("=" * 132)
    print("ARBITRARY TURN V2 — DYNAMIC CAPTURE HANDOFF")
    print("V22 turn policy -> dynamic terminal handoff -> Stage2 PPO/AFCS; SAME FDM; no retraining")
    print(f"lead_time={LEAD_TIME_S:.2f}s margin={LEAD_MARGIN_DEG:.2f}deg capture=[{MIN_CAPTURE_DEG:.1f},{MAX_CAPTURE_DEG:.1f}]deg")
    print("targets:", targets, "seeds:", seeds)
    print("=" * 132)

    for target in targets:
        for seed in seeds:
            r = run_one(stack, target, seed)
            rows.append(r)
            cap = "specialist-V22" if r["specialist"] else (
                f"cap@{r['capture_at']:.1f}s rem={r['capture_remaining']:+.2f} yaw={r['capture_yaw']:+.2f} lead={r['capture_lead']:.2f}"
                if r["capture"] else "NO_CAPTURE"
            )
            print(
                f"seed={seed:3d} target={target:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} done={r['done']:+8.2f} rem={r['rem']:+7.2f} "
                f"alt={r['alt']:7.2f} vs={r['vs']:+5.2f} v={r['v']:5.2f} hold={r['hold']:3.1f} "
                f"safety={r['safe']} | {cap}"
            )

    print("\n" + "=" * 132)
    total_pass = sum(int(r["success"]) for r in rows)
    total_safe = sum(int(r["safe"]) for r in rows)
    print(f"TOTAL PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={total_safe}/{len(rows)}")
    for target in targets:
        q = [r for r in rows if r["target"] == float(target)]
        p = sum(int(r["success"]) for r in q)
        sf = sum(int(r["safe"]) for r in q)
        print(f"target={target:+7.1f}: pass={p}/{len(q)} safety_fail={sf}/{len(q)} max_abs_rem={max(abs(r['rem']) for r in q):.2f}")
    print("ARBITRARY CAPTURE V2:", "PASS" if total_pass == len(rows) and total_safe == 0 else "PARTIAL/FAIL")
    print("=" * 132)


if __name__ == "__main__":
    main()
