from __future__ import annotations

"""
ARBITRARY TURN V10 — V23-STYLE SAME-FDM RECOVERY
=================================================

V8 showed the fixed-trim recovery safely damps lateral/yaw motion, but its
low-lateral vertical law (inherited from V5 capture) settles near ~291 ft.
V9 proved that handing this post-turn state directly to Stage2 PPO is unsafe:
it rapidly climbs through the 325 ft ceiling.

V10 therefore keeps the safe V8 recovery structure and changes only the
low-lateral vertical feedback to the already-proven V23 post-turn formula:

    collective = 0.590 + 0.0008*(300-alt) - 0.0060*VS

When lateral motion is still large, the same conservative sideslip-limited
caps used by V23/V5 remain active.  Same live JSBSim FDM, no reset, no state
teleport, no retraining, runtime teacher OFF.
"""

import argparse
import math

import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_arbitrary_angles_v8_recovery_supervisor as v8

DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]

TARGET_ALT_FT = 300.0
RECOVERY_MAX_S = 60.0
RECOVERY_HOLD_S = 3.0
REC_ALT_MIN = 295.0
REC_ALT_MAX = 305.0
REC_MAX_ABS_VS = 0.60
REC_MAX_ABS_ROLL_DEG = 4.0
REC_MAX_ABS_YAW_DEG_S = 2.0
REC_MAX_ABS_LAT_V = 3.0
REC_MIN_FORWARD_V = 8.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def v23_recovery_collective_target(env) -> float:
    s = env._raw_state()
    alt = float(s["altitude"])
    vs = float(s["vertical_speed"])
    lat_abs = abs(float(s["lateral_velocity"]))

    # Exact high-lateral shape already used in the successful V23 transition.
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

    # Important V10 change: this is the proven V23 low-sideslip vertical
    # recovery law, not V5's lower 0.586-biased capture law.
    return float(np.clip(
        0.590 + 0.0008 * (TARGET_ALT_FT - alt) - 0.0060 * vs,
        0.580,
        0.610,
    ))


def recover_between_primitives_v10(fdm, fdm_id: int):
    env, _ = v7.make_env_on_fdm(fdm, 1.0)
    heading_ref = v7.heading_deg(fdm)
    elapsed = 0.0
    stable_s = 0.0
    safety = False
    success = False

    try:
        collective = float(np.clip(float(fdm["fcs/collective-cmd-norm"]), 0.560, 0.610))
    except Exception:
        collective = 0.585

    try:
        env.turn_active = False
        fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        max_steps = int(RECOVERY_MAX_S / env.CONTROL_DT)
        for _ in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("Shared FDM identity changed during V10 recovery")

            c_target = v23_recovery_collective_target(env)
            collective += float(np.clip(c_target - collective, -0.00075, +0.00075))
            collective = float(np.clip(collective, 0.560, 0.610))

            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(heading_ref))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
            fdm["fcs/collective-cmd-norm"] = collective
            fdm["fcs/elevator-cmd-norm"] = -0.145
            fdm["fcs/aileron-cmd-norm"] = 0.19095
            fdm["fcs/rudder-cmd-norm"] = 0.390

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            elapsed += env.CONTROL_DT
            s = env._raw_state()
            roll_deg = abs(math.degrees(float(s["roll"])))
            pitch_deg = abs(math.degrees(float(s["pitch"])))
            yaw_deg_s = abs(math.degrees(float(s["r_rate"])))

            safety = bool(
                (not js_ok)
                or float(s["altitude"]) < 275.0
                or float(s["altitude"]) > 325.0
                or roll_deg > 18.0
                or pitch_deg > 15.0
                or yaw_deg_s > 30.0
                or float(s["forward_velocity"]) < 1.0
            )
            if safety:
                break

            stable = bool(
                REC_ALT_MIN <= float(s["altitude"]) <= REC_ALT_MAX
                and abs(float(s["vertical_speed"])) <= REC_MAX_ABS_VS
                and roll_deg <= REC_MAX_ABS_ROLL_DEG
                and yaw_deg_s <= REC_MAX_ABS_YAW_DEG_S
                and abs(float(s["lateral_velocity"])) <= REC_MAX_ABS_LAT_V
                and float(s["forward_velocity"]) >= REC_MIN_FORWARD_V
            )
            stable_s = stable_s + env.CONTROL_DT if stable else 0.0
            if stable_s >= RECOVERY_HOLD_S:
                success = True
                break

        s = env._raw_state()
        return {
            "success": bool(success),
            "safe": bool(safety),
            "elapsed": float(elapsed),
            "alt": float(s["altitude"]),
            "vs": float(s["vertical_speed"]),
            "v": float(s["forward_velocity"]),
            "lat": float(s["lateral_velocity"]),
            "roll": float(math.degrees(float(s["roll"]))),
            "yaw": float(math.degrees(float(s["r_rate"]))),
            "heading": float(v7.heading_deg(fdm)),
            "collective": float(collective),
        }
    finally:
        v7.release_wrapper(env)


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    seeds = args.seeds or DEFAULT_SEEDS

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    # V8.run_command resolves its recovery function from the v8 module global.
    # Replace only that function; planner and RL/AFCS primitives stay unchanged.
    v8.recover_between_primitives = recover_between_primitives_v10

    stack = arb.load_stack()
    rows = []

    print("=" * 168)
    print("ARBITRARY TURN V10 — V23-STYLE SAME-FDM RECOVERY")
    print("primitive -> V23 closed-loop recovery -> next primitive | SAME FDM | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 168)

    for target in targets:
        for seed in seeds:
            plan = v7.plan_command(float(target))
            print("\n" + "-" * 160)
            print(f"USER COMMAND {target:+.1f} deg | plan={plan}")
            print("-" * 160)

            r = v8.run_command(stack, float(target), int(seed))
            rows.append(r)
            print(
                f"seed={seed:3d} target={target:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} | "
                f"done={r['done']:+8.2f} rem={r['rem']:+7.2f} | "
                f"heading={r['start_hdg']:.2f}->{r['end_hdg']:.2f} | "
                f"sim_dt={r['sim_dt']:.2f}s | fdm={r['fdm_id']}"
            )

    print("\n" + "=" * 168)
    total_pass = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safe"]) for r in rows)
    print(f"TOTAL PASS={total_pass}/{len(rows)} | SAFETY_FAILURES={safety}/{len(rows)}")
    for target in targets:
        q = [r for r in rows if r["target"] == float(target)]
        print(
            f"target={target:+7.1f} | pass={sum(int(r['success']) for r in q)}/{len(q)} | "
            f"safety_fail={sum(int(r['safe']) for r in q)}/{len(q)} | "
            f"max_abs_remaining={max(abs(r['rem']) for r in q):.2f}"
        )
    print("ARBITRARY V10:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 168)


if __name__ == "__main__":
    main()
