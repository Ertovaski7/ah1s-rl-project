from __future__ import annotations

"""
ARBITRARY TURN V8 — SUPERVISORY PRIMITIVES + SAME-FDM RECOVERY
===============================================================

V7 showed that decomposition itself is useful, but back-to-back validated
primitives can still accumulate vertical/lateral energy.  The second/third
primitive then starts from a state that is outside the entry distribution in
which the primitive was validated.

V8 keeps the exact V7 planner and primitive implementations, but inserts a
short closed-loop recovery phase BETWEEN successful primitives:

    primitive -> strong AFCS on current heading -> live altitude/VS recovery
              -> next primitive

The recovery:
- uses the SAME live JSBSim FDM (no reset / no teleport),
- keeps runtime teacher OFF,
- does not modify any PPO weights,
- reuses V5's live altitude + vertical-speed collective feedback,
- holds the newly established heading with strong roll/yaw AFCS,
- waits for a tighter entry envelope before launching the next primitive.

This tests whether the V7 failures were caused by state accumulation between
otherwise valid primitives rather than by the primitive policies themselves.
"""

import argparse
import math

import numpy as np

import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v5_direct_capture as v5
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16


DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]

RECOVERY_MAX_S = 45.0
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


def recover_between_primitives(fdm, fdm_id: int):
    """Tighten the live state before the next primitive, without resetting FDM."""
    env, _ = v7.make_env_on_fdm(fdm, 1.0)
    heading_ref = v7.heading_deg(fdm)
    elapsed = 0.0
    stable_s = 0.0
    safety = False
    success = False

    # Start from the currently commanded physical collective so the handoff is
    # smooth, then slew toward V5's live feedback target.
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
                raise RuntimeError("Shared FDM identity changed during primitive recovery")

            c_target = v5.adaptive_collective_target(env)
            collective += float(np.clip(c_target - collective, -0.00075, +0.00075))
            collective = float(np.clip(collective, 0.560, 0.610))

            # Hold the heading established by the previous primitive.  These are
            # the same proven forward/transition trims used in V5/V7 capture.
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


def run_command(stack, target: float, seed: int):
    plan = v7.plan_command(float(target))
    bootstrap_target = float(plan[0][1]) if plan else float(target)

    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=bootstrap_target)
    _, reset_info = bootstrap.reset(seed=int(seed))
    fdm = bootstrap.fdm
    fdm_id = id(fdm)
    start_hdg = v7.heading_deg(fdm)
    start_sim = float(fdm["simulation/sim-time-sec"])

    rows = []
    recoveries = []
    ok = True

    try:
        for idx, (kind, angle) in enumerate(plan, start=1):
            if kind == "rl":
                r = v7.run_rl_primitive(stack, fdm, float(angle), fdm_id)
            else:
                r = v7.fine_afcs_tail(fdm, float(angle), fdm_id)
            rows.append(r)

            print(
                f"    primitive {idx}/{len(plan)} {kind.upper():4s} {angle:+6.1f} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} "
                f"done={r['done']:+7.2f} rem={r['rem']:+6.2f} alt={r['alt']:7.2f} | {r['capture']}"
            )

            if not r["success"] or r["safe"]:
                ok = False
                break

            # Tight same-FDM recovery only when another primitive remains.
            if idx < len(plan):
                rec = recover_between_primitives(fdm, fdm_id)
                recoveries.append(rec)
                print(
                    f"      RECOVERY | PASS={str(rec['success']):5s} safety={rec['safe']} "
                    f"t={rec['elapsed']:5.1f}s alt={rec['alt']:7.2f} vs={rec['vs']:+5.2f} "
                    f"v={rec['v']:5.2f} lat={rec['lat']:+5.2f} roll={rec['roll']:+5.2f} "
                    f"yaw={rec['yaw']:+5.2f} c={rec['collective']:.4f}"
                )
                if not rec["success"] or rec["safe"]:
                    ok = False
                    break

        end_hdg = v7.heading_deg(fdm)
        end_sim = float(fdm["simulation/sim-time-sec"])
        total_done = float(sum(r["done"] for r in rows))
        final_rem = float(target - total_done)

        command_ok = bool(ok and abs(final_rem) <= 2.0 and id(fdm) == fdm_id)
        return {
            "seed": int(seed),
            "target": float(target),
            "plan": plan,
            "success": command_ok,
            "safe": any(bool(r["safe"]) for r in rows) or any(bool(x["safe"]) for x in recoveries),
            "done": total_done,
            "rem": final_rem,
            "start_hdg": start_hdg,
            "end_hdg": end_hdg,
            "sim_dt": end_sim - start_sim,
            "fdm_id": fdm_id,
            "extra": int(reset_info.get("extra_entry_steps", 0)),
            "recoveries": recoveries,
        }
    finally:
        try:
            bootstrap.close()
        except Exception:
            pass


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    seeds = args.seeds or DEFAULT_SEEDS

    if any(abs(t) < 1.0 or abs(t) > 360.0 for t in targets):
        raise ValueError("Targets must satisfy 1 <= |target| <= 360 deg")

    stack = arb.load_stack()
    rows = []

    print("=" * 168)
    print("ARBITRARY TURN V8 — SUPERVISORY PRIMITIVES + SAME-FDM RECOVERY")
    print("primitive -> closed-loop recovery -> next primitive | SAME FDM | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 168)

    for target in targets:
        for seed in seeds:
            plan = v7.plan_command(float(target))
            print("\n" + "-" * 160)
            print(f"USER COMMAND {target:+.1f} deg | plan={plan}")
            print("-" * 160)

            r = run_command(stack, float(target), int(seed))
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
    print("ARBITRARY V8 RECOVERY SUPERVISOR:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 168)


if __name__ == "__main__":
    main()
