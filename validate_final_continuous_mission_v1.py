from __future__ import annotations

"""
FINAL CONTINUOUS MISSION VALIDATOR V1
=====================================

Purpose
-------
Validate the final controller architecture before exposing live user input:

    Stage1 takeoff -> stable ~300 ft
    -> Stage2 forward flight
    -> arbitrary relative turn command
    -> V11 bumpless post-command recovery
    -> Stage2 forward continuation
    -> next arbitrary command
    -> ...

All commands below execute on ONE continuous JSBSim FDM object.  There is no
reset, state teleport, prerecorded telemetry, teacher controller, or model
retraining between commands.

The turn command itself uses the V11 supervisory architecture:
validated PPO/AFCS primitives + bumpless same-FDM recovery.  A final recovery
is also applied after each complete user command so Stage2 never receives the
high-lateral specialist endpoint state that caused the V9 handoff failure.
"""

import argparse

import numpy as np

import validate_live_multiturn_same_fdm_v1 as live
import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_arbitrary_angles_v11_bumpless_supervisor as v11


DEFAULT_TARGETS = [20.0, -30.0, 75.0, -90.0, 120.0, 150.0]
DEFAULT_FORWARD_SECONDS = 3.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--forward-seconds", type=float, default=DEFAULT_FORWARD_SECONDS)
    return p.parse_args()


def execute_command_same_fdm(stack, fdm, fdm_id: int, target: float):
    """Execute one arbitrary relative command on the already-running FDM."""
    if id(fdm) != fdm_id:
        raise RuntimeError("Shared FDM identity changed before command")

    plan = v7.plan_command(float(target))
    start_hdg = live.heading_deg(fdm)
    start_sim = live.telemetry(fdm)["sim_t"]
    total_done = 0.0
    ok = True
    safety = False
    executed = []
    recoveries = []

    print(f"  plan={plan}")
    live.print_tel("COMMAND START", fdm)

    for idx, (kind, planned_angle) in enumerate(plan, start=1):
        angle = float(planned_angle)

        # For a planned AFCS tail, use the true live residual accumulated so far.
        if kind == "afcs" and idx == len(plan):
            live_rem = float(target - total_done)
            if abs(live_rem) >= 0.5:
                angle = live_rem

        if kind == "rl":
            r = v7.run_rl_primitive(stack, fdm, angle, fdm_id)
        else:
            r = v7.fine_afcs_tail(fdm, angle, fdm_id)

        executed.append(r)
        total_done += float(r["done"])
        safety = safety or bool(r["safe"])

        print(
            f"    primitive {idx}/{len(plan)} {kind.upper():4s} {angle:+7.2f} | "
            f"PASS={str(r['success']):5s} safety={r['safe']} "
            f"done={r['done']:+7.2f} rem={r['rem']:+6.2f} alt={r['alt']:7.2f} | {r['capture']}"
        )

        if not r["success"] or r["safe"]:
            ok = False
            break

        # Preserve V11's proven recovery between consecutive RL primitives.
        if idx < len(plan) and plan[idx][0] == "rl":
            rec = v11.recover_bumpless(fdm, fdm_id)
            recoveries.append(rec)
            total_done += float(rec["heading_delta"])
            safety = safety or bool(rec["safe"])
            print(
                f"      INTER-PRIMITIVE RECOVERY | PASS={str(rec['success']):5s} safety={rec['safe']} "
                f"t={rec['elapsed']:5.1f}s alt={rec['alt']:7.2f} vs={rec['vs']:+5.2f} "
                f"v={rec['v']:5.2f} lat={rec['lat']:+5.2f} roll={rec['roll']:+5.2f} "
                f"yaw={rec['yaw']:+5.2f} dH={rec['heading_delta']:+.2f}"
            )
            if not rec["success"] or rec["safe"]:
                ok = False
                break

    # Critical final handoff: return to the validated stable envelope BEFORE
    # giving the aircraft back to Stage2.  This avoids the V9 immediate-Stage2
    # failure from the high-lateral +50 specialist endpoint.
    final_recovery = None
    if ok:
        final_recovery = v11.recover_bumpless(fdm, fdm_id)
        recoveries.append(final_recovery)
        total_done += float(final_recovery["heading_delta"])
        safety = safety or bool(final_recovery["safe"])
        print(
            f"      FINAL BUMPLESS RECOVERY   | PASS={str(final_recovery['success']):5s} "
            f"safety={final_recovery['safe']} t={final_recovery['elapsed']:5.1f}s "
            f"alt={final_recovery['alt']:7.2f} vs={final_recovery['vs']:+5.2f} "
            f"v={final_recovery['v']:5.2f} lat={final_recovery['lat']:+5.2f} "
            f"roll={final_recovery['roll']:+5.2f} yaw={final_recovery['yaw']:+5.2f} "
            f"dH={final_recovery['heading_delta']:+.2f}"
        )
        if not final_recovery["success"] or final_recovery["safe"]:
            ok = False

    # Remove the small heading error accumulated by primitives + recoveries.
    correction = None
    final_rem = float(target - total_done)
    if ok and 0.5 <= abs(final_rem) <= 10.0:
        correction = v7.fine_afcs_tail(fdm, final_rem, fdm_id)
        total_done += float(correction["done"])
        safety = safety or bool(correction["safe"])
        print(
            f"      FINAL AFCS CORRECTION {final_rem:+7.2f} | "
            f"PASS={str(correction['success']):5s} safety={correction['safe']} "
            f"done={correction['done']:+7.2f} rem={correction['rem']:+6.2f} alt={correction['alt']:7.2f}"
        )
        if not correction["success"] or correction["safe"]:
            ok = False

    final_rem = float(target - total_done)
    end_hdg = live.heading_deg(fdm)
    end_sim = live.telemetry(fdm)["sim_t"]

    if id(fdm) != fdm_id:
        raise RuntimeError("Shared FDM identity changed inside command")

    live.print_tel("COMMAND END", fdm)
    command_ok = bool(ok and not safety and abs(final_rem) <= 2.0)
    print(
        f"  COMMAND RESULT target={target:+.1f} | PASS={command_ok} safety={safety} | "
        f"done={total_done:+.2f} rem={final_rem:+.2f} | "
        f"heading={start_hdg:.2f}->{end_hdg:.2f} | sim_dt={end_sim-start_sim:.2f}s"
    )

    return {
        "target": float(target),
        "success": bool(command_ok),
        "safety": bool(safety),
        "done": float(total_done),
        "remaining": float(final_rem),
        "start_heading": float(start_hdg),
        "end_heading": float(end_hdg),
        "sim_dt": float(end_sim - start_sim),
    }


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS

    if any(abs(float(t)) < 1.0 or abs(float(t)) > 360.0 for t in targets):
        raise ValueError("Each command must satisfy 1 <= |target| <= 360 deg")
    if args.forward_seconds < 0.0:
        raise ValueError("--forward-seconds must be >= 0")

    print("=" * 176)
    print("FINAL CONTINUOUS MISSION VALIDATOR V1")
    print("Stage1 -> Stage2 -> arbitrary command -> bumpless recovery -> forward -> repeat")
    print("ONE live FDM | teacher OFF | no reset/teleport/retraining")
    print("Commands:", [float(x) for x in targets])
    print("=" * 176)

    stack = arb.load_stack()
    env1 = env2 = None
    fdm = None
    results = []

    try:
        env1, env2, fdm, fdm_id = live.build_initial_live_mission()
        initial_sim = live.telemetry(fdm)["sim_t"]

        for i, target in enumerate(targets):
            print("\n" + "-" * 176)
            print(f"USER COMMAND {i+1}/{len(targets)}: {float(target):+.1f} deg relative")
            print("-" * 176)

            r = execute_command_same_fdm(stack, fdm, fdm_id, float(target))
            results.append(r)
            if not r["success"] or r["safety"]:
                break

            if i < len(targets) - 1 and args.forward_seconds > 0.0:
                print(f"\n  Stage2 forward continuation: {args.forward_seconds:.1f}s")
                live.fly_stage2_between_turns(env2, fdm, args.forward_seconds, fdm_id)

        final_sim = live.telemetry(fdm)["sim_t"]
        all_pass = len(results) == len(targets) and all(
            r["success"] and not r["safety"] for r in results
        )

        print("\n" + "=" * 176)
        print("FINAL CONTINUOUS MISSION SUMMARY")
        print("=" * 176)
        for i, r in enumerate(results, 1):
            print(
                f"#{i} target={r['target']:+7.1f} | PASS={str(r['success']):5s} | "
                f"safety={r['safety']} | rem={r['remaining']:+6.2f} | "
                f"heading={r['start_heading']:7.2f}->{r['end_heading']:7.2f}"
            )
        print(f"Shared FDM id remained: {fdm_id}")
        print(f"Continuous simulated time advanced: {final_sim - initial_sim:.2f}s")
        print("FINAL CONTINUOUS MISSION:", "PASS" if all_pass else "PARTIAL/FAIL")
        print("=" * 176)
        return 0 if all_pass else 1

    finally:
        if env2 is not None:
            env2.fdm = None
            try:
                env2.close()
            except Exception:
                pass
        if env1 is not None:
            try:
                env1.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
