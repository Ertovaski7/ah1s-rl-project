from __future__ import annotations

"""
ARBITRARY TURN V6 — LONG-POSITIVE ALTITUDE GUARD
=================================================

Purpose
-------
V5 proved small arbitrary turns (+20/-30) and -90 can work, but longer positive
unseen targets (+75/+120/+150) can climb through the 325 ft safety ceiling
before or during terminal capture.

V6 is a runtime-only hybrid safety/stabilization experiment:
- Keep the verified V22 PPO stack for the turn action.
- For NON-specialist positive targets >= 60 deg only, apply a soft altitude /
  vertical-speed guard to PPO action[0] (collective).  The guard can only LOWER
  collective relative to the PPO request while the aircraft is climbing/high;
  a1/a2/a3 remain PPO outputs.
- Use a momentum-aware earlier terminal capture for those same long positive
  targets so high yaw-rate does not carry the aircraft far through the target.
- Keep V5's direct-FDM terminal capture after handoff.
- Same live JSBSim FDM, no reset/teleport/retraining, runtime teacher OFF.

This file does not modify any model weights.
"""

import argparse
import math
import types

import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v5_direct_capture as v5
import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22

DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]
ROBUST_SEEDS = [7, 21, 42, 84, 123]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--robust", action="store_true")
    return p.parse_args()


def long_positive_mode(target: float) -> bool:
    return float(target) >= 60.0 and not v5.in_existing_specialist_gate(float(target))


def long_positive_capture_threshold(yaw_deg_s: float) -> float:
    # Earlier capture when turn momentum is high.  At ~6 deg/s this gives
    # roughly 10 deg of lead instead of V5's 5 deg ceiling.
    return float(np.clip(abs(float(yaw_deg_s)) * 1.55 + 0.75, 3.0, 12.0))


def guarded_turn_action(env, raw_action: np.ndarray, target: float):
    action = np.asarray(raw_action, dtype=np.float32).copy()
    guard_active = False
    requested_collective = env.action_to_collective(float(action[0]))
    guarded_collective = requested_collective

    if long_positive_mode(target):
        s = env._raw_state()
        alt = float(s["altitude"])
        vs = float(s["vertical_speed"])

        # Engage before the hard 325-ft envelope.  This is intentionally a
        # one-sided limiter: it does not invent extra lift; it only suppresses
        # excessive positive collective while climbing/high.
        if alt >= 302.0 or vs >= 0.80:
            feedback_cap = (
                0.558
                + 0.0010 * (300.0 - alt)
                - 0.0070 * max(vs, 0.0)
            )
            feedback_cap = float(np.clip(feedback_cap, 0.500, 0.565))
            guarded_collective = min(requested_collective, feedback_cap)
            action[0] = env.collective_to_action(guarded_collective)
            guard_active = guarded_collective < requested_collective - 1e-6

    return action, guard_active, requested_collective, guarded_collective


def run_one(stack, target: float, seed: int):
    bm, ba, p200, p50, p21 = stack
    env = v16.RandomizedEntryEnv(target_turn_deg=float(target))
    obs, reset_info = env.reset(seed=int(seed))
    info = reset_info

    specialist = v5.in_existing_specialist_gate(target)
    capture = False
    target_heading = None
    capture_at = None
    capture_remaining = None
    capture_yaw = None
    capture_lead = None
    capture_heading = None
    success = False

    guard_steps = 0
    min_guarded_collective = float("inf")
    max_requested_collective = -float("inf")

    max_steps = int(max(120.0, abs(float(target)) + 100.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
            remaining = float(target) - float(env.cumulative_turn_deg)
            yr = v5.v3.yaw_rate_deg(env)

            if not specialist and not capture:
                if long_positive_mode(target):
                    lead = long_positive_capture_threshold(yr)
                else:
                    lead = v5.v3.capture_threshold(yr)

                moving_toward = (remaining * yr) > 0.0
                progress_needed = min(5.0, 0.25 * abs(float(target)))
                meaningful_progress = abs(float(env.cumulative_turn_deg)) >= progress_needed

                if moving_toward and meaningful_progress and abs(remaining) <= lead:
                    capture = True
                    capture_at = float(env.turn_steps * env.CONTROL_DT)
                    capture_remaining = float(remaining)
                    capture_yaw = float(yr)
                    capture_lead = float(lead)
                    capture_heading, target_heading = v5.v3.enter_heading_capture(env, remaining)
                    env._apply_action = types.MethodType(v5.direct_capture_apply, env)

            if capture:
                env.fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
                env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
                env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
                action = v5.terminal_physical_action(env)
            else:
                base_action = v22.act(bm, ba, p200, p50, p21, obs)
                action, active, req_c, guard_c = guarded_turn_action(env, base_action, target)
                if active:
                    guard_steps += 1
                    min_guarded_collective = min(min_guarded_collective, guard_c)
                    max_requested_collective = max(max_requested_collective, req_c)

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
        "guard_steps": int(guard_steps),
        "guard_cmin": float(min_guarded_collective) if guard_steps else np.nan,
        "guard_reqmax": float(max_requested_collective) if guard_steps else np.nan,
        "cap_cmin": float(getattr(env, "_v5_collective_min", np.nan)),
        "cap_cmax": float(getattr(env, "_v5_collective_max", np.nan)),
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

    print("=" * 168)
    print("ARBITRARY TURN V6 — LONG-POSITIVE ALTITUDE GUARD + MOMENTUM-AWARE CAPTURE")
    print("V22 PPO + one-sided live collective guard -> V5 direct terminal capture | SAME FDM | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 168)

    for target in targets:
        for seed in seeds:
            r = run_one(stack, float(target), int(seed))
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

            guard = (
                f"guard_steps={r['guard_steps']} cmin={r['guard_cmin']:.4f} reqmax={r['guard_reqmax']:.4f}"
                if r["guard_steps"] else "guard=OFF"
            )

            print(
                f"seed={r['seed']:3d} target={r['target']:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} done={r['done']:+8.2f} rem={r['rem']:+7.2f} "
                f"alt={r['alt']:7.2f} vs={r['vs']:+5.2f} v={r['v']:5.2f} "
                f"hold={r['hold']:3.1f} safety={r['safe']} | {cap} | {guard}"
            )

    print("\n" + "=" * 168)
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
    print("ARBITRARY V6:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 168)


if __name__ == "__main__":
    main()
