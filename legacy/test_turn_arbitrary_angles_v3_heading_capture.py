from __future__ import annotations

"""
ARBITRARY RELATIVE TURN V3 — ABSOLUTE-HEADING AFCS CAPTURE
==========================================================

Non-destructive runtime experiment on top of the verified V22 stack.
No model weights are changed.

Why V2 failed
-------------
V2 re-enabled the Stage-2/AFCS stabilization without changing the AFCS yaw
reference.  The AH-1S FDM is initialized with psi-trim = pi (the original
straight-flight heading), so full yaw AFCS naturally tried to steer the
helicopter back toward the pre-turn heading.

V3 fixes exactly that handoff:
- V22 PPO performs the arbitrary turn until the dynamic capture window.
- At capture, compute the desired absolute final heading from
      current_heading + remaining_turn
  and write that value into ap/afcs/psi-trim-rad.
- Re-enable strong roll/yaw AFCS on that NEW heading.
- During the short terminal stabilization window, use a small bounded
  altitude/speed trim law for collective/elevator instead of the Stage-2 PPO,
  because V2 showed the Stage-2 policy can over-correct altitude when entered
  from a turning state.
- Same JSBSim FDM continues; no reset or teleport.

Existing V22 specialist regions (+50 and +200) keep their validated V22 path.
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

LEAD_TIME_S = 0.90
LEAD_MARGIN_DEG = 0.50
MIN_CAPTURE_DEG = 2.0
MAX_CAPTURE_DEG = 5.0

TARGET_ALT_FT = 300.0
TARGET_SPEED_FPS = 14.5

# Conservative forward-flight terminal trim.
COLLECTIVE_BASE = 0.596
ALT_KP = 0.0009
VS_KD = 0.0045
COLLECTIVE_MIN = 0.590
COLLECTIVE_MAX = 0.608

ELEVATOR_BASE = -0.145
SPEED_KP = 0.0020
ELEVATOR_MIN = -0.165
ELEVATOR_MAX = -0.125


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


def yaw_rate_deg(env) -> float:
    return math.degrees(float(env._raw_state()["r_rate"]))


def capture_threshold(yaw_deg_s: float) -> float:
    return float(np.clip(
        abs(float(yaw_deg_s)) * LEAD_TIME_S + LEAD_MARGIN_DEG,
        MIN_CAPTURE_DEG,
        MAX_CAPTURE_DEG,
    ))


def wrap360(x: float) -> float:
    return float(float(x) % 360.0)


def enter_heading_capture(env, remaining: float):
    current_heading = float(env._heading_deg())
    target_heading = wrap360(current_heading + float(remaining))

    # Switch out of the turn-specific physical action mapping, but keep the
    # exact same live JSBSim FDM/state.
    env.turn_active = False

    # Critical V3 fix: AFCS yaw reference is the NEW desired final heading,
    # not the original pre-turn psi trim.
    env.fdm["ap/afcs/psi-trim-rad"] = math.radians(target_heading)
    env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
    env.fdm["ap/afcs/pitch-channel-active-norm"] = 0.5

    return current_heading, target_heading


def terminal_action(env) -> np.ndarray:
    s = env._raw_state()

    collective = (
        COLLECTIVE_BASE
        + ALT_KP * (TARGET_ALT_FT - float(s["altitude"]))
        - VS_KD * float(s["vertical_speed"])
    )
    collective = float(np.clip(collective, COLLECTIVE_MIN, COLLECTIVE_MAX))

    # Stage-2 physical mapping is collective = 0.620 + 0.030*a0.
    a0 = float(np.clip((collective - 0.620) / 0.030, -1.0, 1.0))

    elevator = (
        ELEVATOR_BASE
        + SPEED_KP * (TARGET_SPEED_FPS - float(s["forward_velocity"]))
    )
    elevator = float(np.clip(elevator, ELEVATOR_MIN, ELEVATOR_MAX))

    # Stage-2 physical mapping is elevator = -0.145 + 0.035*a1.
    a1 = float(np.clip((elevator + 0.145) / 0.035, -1.0, 1.0))

    # Let strong AFCS handle roll/yaw around the aircraft's established trims.
    return np.asarray([a0, a1, 0.0, 0.0], dtype=np.float32)


def run_one(stack, target: float, seed: int):
    bm, ba, p200, p50, p21 = stack
    env = v16.RandomizedEntryEnv(target_turn_deg=float(target))
    obs, reset_info = env.reset(seed=int(seed))
    info = reset_info

    specialist = in_existing_specialist_gate(target)
    capture = False
    capture_at = None
    capture_remaining = None
    capture_yaw = None
    capture_lead = None
    capture_heading = None
    target_heading = None
    success = False

    max_steps = int(max(100.0, abs(target) + 80.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
            remaining = float(target) - float(env.cumulative_turn_deg)
            yr = yaw_rate_deg(env)

            if not specialist and not capture:
                lead = capture_threshold(yr)
                moving_toward = (remaining * yr) > 0.0
                progress_needed = min(5.0, 0.25 * abs(float(target)))
                meaningful_progress = abs(float(env.cumulative_turn_deg)) >= progress_needed

                if moving_toward and meaningful_progress and abs(remaining) <= lead:
                    capture = True
                    capture_at = float(env.turn_steps * env.CONTROL_DT)
                    capture_remaining = float(remaining)
                    capture_yaw = float(yr)
                    capture_lead = float(lead)
                    capture_heading, target_heading = enter_heading_capture(env, remaining)

            if capture:
                # Reassert the intended new yaw reference each control step.
                env.fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
                env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
                env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
                action = terminal_action(env)
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
        "capture_heading": capture_heading,
        "target_heading": target_heading,
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

    print("=" * 144)
    print("ARBITRARY TURN V3 — ABSOLUTE-HEADING AFCS CAPTURE")
    print("V22 PPO -> dynamic capture -> NEW psi-trim heading + bounded terminal stabilization; SAME FDM")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 144)

    for target in targets:
        for seed in seeds:
            r = run_one(stack, target, seed)
            rows.append(r)

            if r["specialist"]:
                cap = "specialist-V22"
            elif r["capture"]:
                cap = (
                    f"cap@{r['capture_at']:.1f}s rem={r['capture_remaining']:+.2f} "
                    f"yaw={r['capture_yaw']:+.2f} lead={r['capture_lead']:.2f} "
                    f"hdg={r['capture_heading']:.1f}->{r['target_heading']:.1f}"
                )
            else:
                cap = "NO_CAPTURE"

            print(
                f"seed={r['seed']:3d} target={r['target']:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} done={r['done']:+8.2f} rem={r['rem']:+7.2f} "
                f"alt={r['alt']:7.2f} vs={r['vs']:+5.2f} v={r['v']:5.2f} "
                f"hold={r['hold']:3.1f} safety={r['safe']} | {cap}"
            )

    print("\n" + "=" * 144)
    total_pass = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safe"]) for r in rows)
    print(f"TOTAL PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={safety}/{len(rows)}")
    for target in targets:
        q = [r for r in rows if r["target"] == float(target)]
        print(
            f"target={target:+7.1f}: pass={sum(int(r['success']) for r in q)}/{len(q)} "
            f"safety_fail={sum(int(r['safe']) for r in q)}/{len(q)} "
            f"max_abs_rem={max(abs(r['rem']) for r in q):.2f}"
        )
    print("ARBITRARY CAPTURE V3:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 144)


if __name__ == "__main__":
    main()
