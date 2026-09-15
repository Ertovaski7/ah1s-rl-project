from __future__ import annotations

"""
ARBITRARY RELATIVE TURN V4 — ADAPTIVE TERMINAL CAPTURE
=======================================================

Non-destructive runtime test on top of V3.

V3 proved the new absolute-heading AFCS capture works: -30 passed, while +20
reached the correct heading but climbed through the 325 ft safety ceiling.

V4 keeps V3's heading-reference fix and replaces only the terminal vertical /
longitudinal trim with the already-proven post-turn transition idea from the
full-mission V23 path:
- reduce collective when sideslip/lateral velocity is high,
- use altitude + vertical-speed feedback once lateral motion settles,
- slew collective smoothly,
- keep elevator fixed at the proven transition trim (-0.145), rather than
  pitching up to chase speed during a turning-state handoff.

Same live JSBSim FDM. No reset. No state teleport. No model retraining.
Runtime teacher OFF.
"""

import argparse
import numpy as np

import test_turn_arbitrary_angles_v3_heading_capture as v3

DEFAULT_TARGETS = [20.0, -30.0]
DEFAULT_SEEDS = [42]
ROBUST_SEEDS = [7, 21, 42, 84, 123]

TARGET_ALT_FT = 300.0

# Proven transition trim from the earlier successful full-mission path.
TRANSITION_ELEVATOR = -0.145


def adaptive_collective_target(env) -> float:
    s = env._raw_state()
    alt = float(s["altitude"])
    vs = float(s["vertical_speed"])
    lat_abs = abs(float(s["lateral_velocity"]))

    # High sideslip / lateral motion: stay conservative to avoid the climb
    # seen in V3 +20. This is the same shape used by the V23 transition.
    if lat_abs >= 7.0:
        feedback_target = 0.580 + 0.0003 * (TARGET_ALT_FT - alt) - 0.0040 * vs

        if lat_abs >= 18.0:
            collective_cap = 0.574
        elif lat_abs >= 14.0:
            collective_cap = 0.576
        elif lat_abs >= 10.0:
            collective_cap = 0.580
        else:
            collective_cap = 0.585

        return float(np.clip(feedback_target, 0.574, collective_cap))

    # Once lateral motion is small, recover altitude/vertical speed smoothly.
    return float(np.clip(
        0.590 + 0.0008 * (TARGET_ALT_FT - alt) - 0.0060 * vs,
        0.580,
        0.610,
    ))


def terminal_action_v4(env) -> np.ndarray:
    target_collective = adaptive_collective_target(env)

    if not hasattr(env, "_v4_transition_collective"):
        # Start from the conservative trim that previously worked in the
        # full-mission +50 transition, then slew toward feedback target.
        env._v4_transition_collective = 0.574

    step = float(np.clip(
        target_collective - float(env._v4_transition_collective),
        -0.0005,
        +0.0005,
    ))
    env._v4_transition_collective = float(np.clip(
        float(env._v4_transition_collective) + step,
        0.574,
        0.610,
    ))

    # Stage-2 physical mapping:
    #   collective = 0.620 + 0.030*a0
    #   elevator   = -0.145 + 0.035*a1
    a0 = float(np.clip((env._v4_transition_collective - 0.620) / 0.030, -1.0, 1.0))
    a1 = float(np.clip((TRANSITION_ELEVATOR + 0.145) / 0.035, -1.0, 1.0))

    # Strong AFCS handles roll/yaw about the NEW psi trim established by V3.
    return np.asarray([a0, a1, 0.0, 0.0], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--robust", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    if args.robust and args.seeds is not None:
        raise ValueError("Use either --robust or --seeds, not both")
    seeds = ROBUST_SEEDS if args.robust else (args.seeds or DEFAULT_SEEDS)

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    # Patch only V3's terminal action. All heading-capture / V22 behavior stays
    # exactly in V3; specialists (+50/+200) remain untouched.
    v3.terminal_action = terminal_action_v4

    stack = v3.arb.load_stack()
    rows = []

    print("=" * 150)
    print("ARBITRARY TURN V4 — ADAPTIVE SIDESLIP-LIMITED HEADING CAPTURE")
    print("V22 PPO -> V3 new-heading AFCS capture -> V23-style adaptive terminal trim; SAME FDM; teacher OFF")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 150)

    for target in targets:
        for seed in seeds:
            r = v3.run_one(stack, float(target), int(seed))
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

    print("\n" + "=" * 150)
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
    print("ARBITRARY CAPTURE V4:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 150)


if __name__ == "__main__":
    main()
