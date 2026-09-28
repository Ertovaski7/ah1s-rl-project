from __future__ import annotations

"""
ARBITRARY TURN V11 — BUMPLESS SUPERVISOR
=========================================

Integrates the recovery law validated by
`diagnose_post50_handoff_v5_descent_triggered_vertical.py` into the V7
supervisory primitive planner.

Architecture for one user command:
    validated RL primitive
      -> SAME-FDM bumpless E/A/R slew while holding the real endpoint collective
      -> live altitude/vertical-speed collective recovery
      -> next RL primitive
      -> optional small AFCS residual correction

Important:
- The exact same live JSBSim FDM is preserved throughout one command.
- No reset / teleport between primitives.
- Runtime teacher is OFF.
- PPO/model weights are unchanged.
- Recovery is inserted only when another RL primitive follows.  A final AFCS
  tail is allowed to operate directly after the preceding successful primitive.
"""

import argparse
import math

import numpy as np

from helicopter_env_turn_goal import wrap_deg
import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v7_supervisory_primitives as v7
import test_turn_full_entry_v16_randomized_entry_robustness as v16


DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]

# Exact validated recovery constants from post50 diagnostic V5.
RECOVERY_MAX_S = 40.0
TARGET_ALT = 300.0
TARGET_ELEVATOR = -0.1450
TARGET_AILERON = 0.19095
TARGET_RUDDER = 0.3900
SLEW_AILERON_PER_STEP = 0.0030
SLEW_RUDDER_PER_STEP = 0.0015
SLEW_ELEVATOR_PER_STEP = 0.0006
ALT_TO_VS = 0.12
DESIRED_VS_LIMIT = 1.0
VS_ERROR_GAIN = 0.0025
ALT_DIRECT_GAIN = 0.00025
STEP_UP_MAX = 0.0010
STEP_DOWN_MAX = 0.0008
COLLECTIVE_MIN = 0.515
COLLECTIVE_MAX = 0.600


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def slew(x: float, target: float, step: float) -> float:
    return float(x + np.clip(target - x, -step, +step))


def fdm_float(fdm, key: str, default=float("nan")) -> float:
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def recover_bumpless(fdm, fdm_id: int, progress_callback=None):
    """Validated SAME-FDM primitive-to-primitive recovery."""
    env, _ = v7.make_env_on_fdm(fdm, 1.0)
    env.turn_active = False

    heading_start = v7.heading_deg(fdm)
    heading_ref = float(heading_start)

    collective = fdm_float(fdm, "fcs/collective-cmd-norm")
    elevator = fdm_float(fdm, "fcs/elevator-cmd-norm")
    aileron = fdm_float(fdm, "fcs/aileron-cmd-norm")
    rudder = fdm_float(fdm, "fcs/rudder-cmd-norm")
    endpoint_collective = float(collective)

    dt = env.CONTROL_DT
    max_steps = int(RECOVERY_MAX_S / dt)
    stable_s = 0.0
    vertical_started = False
    safety = False
    elapsed = 0.0
    max_alt = -1e9
    min_alt = +1e9
    min_abs_lat = 1e9
    max_abs_vs = 0.0
    max_collective = float(collective)

    try:
        fdm["ap/afcs/psi-trim-rad"] = math.radians(heading_ref)
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        for i in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("Shared FDM identity changed during bumpless recovery")

            # Phase A: do not create a collective bump.  Keep exactly the real
            # primitive endpoint collective while E/A/R slew to recovery trims.
            elevator = slew(elevator, TARGET_ELEVATOR, SLEW_ELEVATOR_PER_STEP)
            aileron = slew(aileron, TARGET_AILERON, SLEW_AILERON_PER_STEP)
            rudder = slew(rudder, TARGET_RUDDER, SLEW_RUDDER_PER_STEP)

            slew_done = bool(
                abs(elevator - TARGET_ELEVATOR) < 1e-6
                and abs(aileron - TARGET_AILERON) < 1e-6
                and abs(rudder - TARGET_RUDDER) < 1e-6
            )

            s_pre = env._raw_state()
            if not slew_done:
                collective = endpoint_collective
            else:
                vertical_started = True
                alt = float(s_pre["altitude"])
                vs = float(s_pre["vertical_speed"])
                desired_vs = float(np.clip(
                    ALT_TO_VS * (TARGET_ALT - alt),
                    -DESIRED_VS_LIMIT,
                    +DESIRED_VS_LIMIT,
                ))
                vs_error = desired_vs - vs
                raw_dc = VS_ERROR_GAIN * vs_error + ALT_DIRECT_GAIN * (TARGET_ALT - alt)
                dc = float(np.clip(raw_dc, -STEP_DOWN_MAX, +STEP_UP_MAX))
                collective = float(np.clip(
                    collective + dc,
                    COLLECTIVE_MIN,
                    COLLECTIVE_MAX,
                ))

            fdm["ap/afcs/psi-trim-rad"] = math.radians(heading_ref)
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
            fdm["fcs/collective-cmd-norm"] = collective
            fdm["fcs/elevator-cmd-norm"] = elevator
            fdm["fcs/aileron-cmd-norm"] = aileron
            fdm["fcs/rudder-cmd-norm"] = rudder

            js_ok = True
            for _ in range(env.PHYSICS_STEPS):
                if not fdm.run():
                    js_ok = False
                    break

            elapsed = (i + 1) * dt
            s = env._raw_state()
            alt = float(s["altitude"])
            vs = float(s["vertical_speed"])
            v = float(s["forward_velocity"])
            lat = float(s["lateral_velocity"])
            roll = math.degrees(float(s["roll"]))
            pitch = math.degrees(float(s["pitch"]))
            yaw = math.degrees(float(s["r_rate"]))

            if progress_callback is not None and i % 5 == 0:
                progress_callback("TURN / RECOVERY")

            max_alt = max(max_alt, alt)
            min_alt = min(min_alt, alt)
            min_abs_lat = min(min_abs_lat, abs(lat))
            max_abs_vs = max(max_abs_vs, abs(vs))
            max_collective = max(max_collective, collective)

            safety = bool(
                (not js_ok)
                or alt < 275.0
                or alt > 325.0
                or abs(roll) > 18.0
                or abs(pitch) > 15.0
                or abs(yaw) > 30.0
                or v < 1.0
            )
            if safety:
                break

            stable = bool(
                vertical_started
                and 295.0 <= alt <= 305.0
                and abs(vs) <= 0.75
                and abs(roll) <= 5.0
                and abs(yaw) <= 3.0
                and abs(lat) <= 8.0
                and v >= 8.0
            )
            stable_s = stable_s + dt if stable else 0.0
            if stable_s >= 2.0:
                break

        s = env._raw_state()
        heading_end = v7.heading_deg(fdm)
        return {
            "success": bool(stable_s >= 2.0),
            "safe": bool(safety),
            "elapsed": float(elapsed),
            "alt": float(s["altitude"]),
            "vs": float(s["vertical_speed"]),
            "v": float(s["forward_velocity"]),
            "lat": float(s["lateral_velocity"]),
            "roll": math.degrees(float(s["roll"])),
            "yaw": math.degrees(float(s["r_rate"])),
            "collective": float(collective),
            "endpoint_collective": float(endpoint_collective),
            "heading_start": float(heading_start),
            "heading_end": float(heading_end),
            "heading_delta": float(wrap_deg(heading_end - heading_start)),
            "max_alt": float(max_alt),
            "min_alt": float(min_alt),
            "max_abs_vs": float(max_abs_vs),
            "min_abs_lat": float(min_abs_lat),
            "max_collective": float(max_collective),
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
    start_sim = fdm_float(fdm, "simulation/sim-time-sec", 0.0)

    executed = []
    recoveries = []
    total_done = 0.0
    ok = True

    try:
        for idx, (kind, planned_angle) in enumerate(plan, start=1):
            angle = float(planned_angle)

            # If V7 already planned a final AFCS tail, use the ACTUAL live
            # residual so heading drift accumulated during recoveries is removed.
            if kind == "afcs" and idx == len(plan):
                live_rem = float(target - total_done)
                if abs(live_rem) >= 0.5:
                    angle = live_rem

            if kind == "rl":
                r = v7.run_rl_primitive(stack, fdm, angle, fdm_id)
            else:
                r = v7.fine_afcs_tail(fdm, angle, fdm_id)

            executed.append((kind, angle, r))
            total_done += float(r["done"])
            print(
                f"    primitive {idx}/{len(plan)} {kind.upper():4s} {angle:+7.2f} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} "
                f"done={r['done']:+7.2f} rem={r['rem']:+6.2f} alt={r['alt']:7.2f} | {r['capture']}"
            )

            if not r["success"] or r["safe"]:
                ok = False
                break

            # Recovery is needed only when another RL primitive follows.  A
            # final AFCS tail can start directly from the successful turn state.
            if idx < len(plan) and plan[idx][0] == "rl":
                rec = recover_bumpless(fdm, fdm_id)
                recoveries.append(rec)
                total_done += float(rec["heading_delta"])
                print(
                    f"      RECOVERY | PASS={str(rec['success']):5s} safety={rec['safe']} "
                    f"t={rec['elapsed']:5.1f}s alt={rec['alt']:7.2f} vs={rec['vs']:+5.2f} "
                    f"v={rec['v']:5.2f} lat={rec['lat']:+5.2f} roll={rec['roll']:+5.2f} "
                    f"yaw={rec['yaw']:+5.2f} c={rec['collective']:.4f} "
                    f"dH={rec['heading_delta']:+.2f}deg"
                )
                if not rec["success"] or rec["safe"]:
                    ok = False
                    break

        # Compensate a small accumulated residual (including recovery heading
        # drift) on the same live FDM.  This is intentionally bounded.
        final_rem = float(target - total_done)
        correction = None
        if ok and 0.5 <= abs(final_rem) <= 10.0:
            correction = v7.fine_afcs_tail(fdm, final_rem, fdm_id)
            total_done += float(correction["done"])
            print(
                f"    FINAL AFCS CORRECTION {final_rem:+7.2f} | "
                f"PASS={str(correction['success']):5s} safety={correction['safe']} "
                f"done={correction['done']:+7.2f} rem={correction['rem']:+6.2f} alt={correction['alt']:7.2f}"
            )
            if not correction["success"] or correction["safe"]:
                ok = False

        final_rem = float(target - total_done)
        end_hdg = v7.heading_deg(fdm)
        end_sim = fdm_float(fdm, "simulation/sim-time-sec", 0.0)

        return {
            "seed": int(seed),
            "target": float(target),
            "plan": plan,
            "success": bool(ok and abs(final_rem) <= 2.0 and id(fdm) == fdm_id),
            "safe": any(bool(r[2]["safe"]) for r in executed)
                    or any(bool(x["safe"]) for x in recoveries)
                    or bool(correction and correction["safe"]),
            "done": float(total_done),
            "rem": float(final_rem),
            "start_hdg": float(start_hdg),
            "end_hdg": float(end_hdg),
            "sim_dt": float(end_sim - start_sim),
            "fdm_id": int(fdm_id),
            "extra": int(reset_info.get("extra_entry_steps", 0)),
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

    print("=" * 176)
    print("ARBITRARY TURN V11 — BUMPLESS SAME-FDM SUPERVISOR")
    print("validated primitive -> bumpless recovery -> next primitive | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 176)

    for target in targets:
        for seed in seeds:
            plan = v7.plan_command(float(target))
            print("\n" + "-" * 168)
            print(f"USER COMMAND {target:+.1f} deg | plan={plan}")
            print("-" * 168)
            r = run_command(stack, float(target), int(seed))
            rows.append(r)
            print(
                f"seed={seed:3d} target={target:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} | "
                f"done={r['done']:+8.2f} rem={r['rem']:+7.2f} | "
                f"heading={r['start_hdg']:.2f}->{r['end_hdg']:.2f} | "
                f"sim_dt={r['sim_dt']:.2f}s | fdm={r['fdm_id']}"
            )

    print("\n" + "=" * 176)
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
    print("ARBITRARY V11 BUMPLESS SUPERVISOR:", "PASS" if total_pass == len(rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 176)


if __name__ == "__main__":
    main()
