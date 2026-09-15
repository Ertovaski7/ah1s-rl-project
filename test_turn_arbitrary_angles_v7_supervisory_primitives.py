from __future__ import annotations

"""
ARBITRARY TURN V7 — SUPERVISORY PRIMITIVE PLANNER
==================================================

Why this exists
---------------
V6 showed that trying to force one continuously-generalized PPO trajectory for
unseen long positive commands is not reliable:
- +75/+120 still accumulated too much vertical energy before/through capture.
- +150 became safe only by applying the collective guard for a long time, but
  the same guard then suppressed the maneuver enough that the target was never
  reached.

V7 therefore changes the architecture instead of adding another blind gain
patch.  A supervisory planner decomposes a requested relative heading change
into already-validated turn primitives while preserving the exact same live
JSBSim FDM throughout the requested command.

Positive command examples:
    +75  -> +50 specialist -> +20 arbitrary-V5 -> +5 AFCS fine capture
    +120 -> +50 specialist -> +50 specialist -> +20 arbitrary-V5
    +150 -> +50 specialist -> +50 specialist -> +50 specialist

Negative commands up to 100 deg currently use the already-validated arbitrary
V5 path directly (e.g. -90).  The fine residual (<= 10 deg) is completed by a
bounded AFCS heading capture with the same live altitude/vertical-speed
feedback used in V5.  No FDM reset, no state teleport, no retraining, runtime
teacher OFF.

This is a hybrid supervisory controller: PPO turn primitives + AFCS terminal
capture.  It does NOT claim that one PPO policy smoothly generalizes to every
angle.  The user command remains arbitrary; decomposition is internal.
"""

import argparse
import math
import types

import numpy as np

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry
import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v5_direct_capture as v5
import test_turn_full_entry_v16_randomized_entry_robustness as v16
import test_turn_full_entry_v22_v21_runtime as v22
from helicopter_env_turn_goal import wrap_deg

DEFAULT_TARGETS = [75.0, -90.0, 120.0, 150.0]
DEFAULT_SEEDS = [42]
POLICY_FORWARD_COORD = 403.424
FINE_TAIL_MAX_DEG = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def heading_deg(fdm) -> float:
    for key in ("attitude/heading-true-rad", "attitude/psi-rad"):
        try:
            return float(math.degrees(float(fdm[key])) % 360.0)
        except Exception:
            pass
    return float("nan")


def plan_command(target: float):
    """Return [('rl', angle), ... , ('afcs', residual)] for one user command."""
    t = float(target)
    if abs(t) < 1.0 or abs(t) > 360.0:
        raise ValueError("Target must satisfy 1 <= |target| <= 360 deg")

    # Current negative V5 evidence includes -30 robust and -90 smoke PASS.
    # Keep those commands on the direct V5 path rather than inventing a new
    # decomposition before needed.
    if t < 0.0 and abs(t) <= 100.0:
        return [("rl", t)]

    parts = []
    rem = abs(t)
    sign = 1.0 if t >= 0.0 else -1.0

    # Positive long turns intentionally use the strongly validated +50
    # specialist.  For large negative commands we conservatively use -30
    # primitives because -30 has 5/5 V5 robustness evidence.
    major = 50.0 if sign > 0.0 else 30.0
    while rem >= major - 1e-9:
        parts.append(("rl", sign * major))
        rem -= major

    # +20 / -30 are the robust arbitrary primitives.  For positive remainder,
    # consume +20 chunks before the very small AFCS tail.
    if sign > 0.0:
        while rem >= 20.0 - 1e-9:
            parts.append(("rl", 20.0))
            rem -= 20.0

    # For negative >100, after -30 chunks rem is automatically <30.  If it is
    # larger than the fine-tail range, let V5 handle that final arbitrary piece.
    if rem > FINE_TAIL_MAX_DEG + 1e-9:
        parts.append(("rl", sign * rem))
        rem = 0.0

    if rem >= 0.5:
        parts.append(("afcs", sign * rem))

    return parts


def make_env_on_fdm(fdm, target: float):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=float(target))
    env.fdm = fdm
    env.phase = 1
    env.turn_active = True
    env.target_turn_deg = float(target)
    env.cumulative_turn_deg = 0.0
    env.prev_heading_deg = env._heading_deg()
    env.turn_steps = 0
    env.success_hold_s = 0.0
    env.previous_turn_action = np.zeros((4,), dtype=np.float32)
    env.forward_distance = POLICY_FORWARD_COORD
    env._v2_previous_remaining = float(target)
    env.fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    env.fdm["ap/afcs/roll-channel-active-norm"] = env.TURN_ROLL_AFCS
    env.fdm["ap/afcs/yaw-channel-active-norm"] = env.TURN_YAW_AFCS
    return env, np.asarray(env._get_obs(), dtype=np.float32)


def release_wrapper(env):
    # Never let a temporary wrapper close the shared live FDM.
    try:
        env.fdm = None
        env.close()
    except Exception:
        pass


def run_rl_primitive(stack, fdm, target: float, fdm_id: int):
    bm, ba, p200, p50, p21 = stack
    env, obs = make_env_on_fdm(fdm, target)
    specialist = v5.in_existing_specialist_gate(float(target))

    capture = False
    target_heading = None
    capture_at = None
    capture_remaining = None
    info = {
        "success": False,
        "safety_failure": False,
        "cumulative_turn_deg": 0.0,
        "remaining_turn_deg": float(target),
    }
    success = False

    max_steps = int(max(100.0, abs(float(target)) + 80.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("Shared FDM identity changed inside RL primitive")

            remaining = float(target) - float(env.cumulative_turn_deg)
            yr = v5.v3.yaw_rate_deg(env)

            if not specialist and not capture:
                lead = v5.v3.capture_threshold(yr)
                moving_toward = (remaining * yr) > 0.0
                progress_needed = min(5.0, 0.25 * abs(float(target)))
                meaningful_progress = abs(float(env.cumulative_turn_deg)) >= progress_needed

                if moving_toward and meaningful_progress and abs(remaining) <= lead:
                    capture = True
                    capture_at = float(env.turn_steps * env.CONTROL_DT)
                    capture_remaining = float(remaining)
                    _, target_heading = v5.v3.enter_heading_capture(env, remaining)
                    env._apply_action = types.MethodType(v5.direct_capture_apply, env)

            if capture:
                env.fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
                env.fdm["ap/afcs/roll-channel-active-norm"] = 1.0
                env.fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
                action = v5.terminal_physical_action(env)
            else:
                action = v22.act(bm, ba, p200, p50, p21, obs)

            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break

        if success:
            # Preserve the newly established heading as the straight-flight
            # AFCS reference for the next primitive.
            if target_heading is None:
                target_heading = heading_deg(fdm)
            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        return {
            "kind": "rl",
            "target": float(target),
            "success": bool(success),
            "safe": bool(info.get("safety_failure", False)),
            "done": float(info.get("cumulative_turn_deg", 0.0)),
            "rem": float(info.get("remaining_turn_deg", 999.0)),
            "alt": float(info.get("altitude", np.nan)),
            "capture": "specialist-V22" if specialist else (
                f"V5@{capture_at:.1f}s rem={capture_remaining:+.2f}" if capture else "NONE"
            ),
        }
    finally:
        release_wrapper(env)


def fine_afcs_tail(fdm, delta_deg: float, fdm_id: int):
    """Small residual heading capture on the same FDM, with live V5 vertical trim."""
    env, _ = make_env_on_fdm(fdm, float(delta_deg))
    start_hdg = heading_deg(fdm)
    target_hdg = (start_hdg + float(delta_deg)) % 360.0
    stable_s = 0.0
    elapsed = 0.0
    collective = float(np.clip(float(fdm["fcs/collective-cmd-norm"]), 0.570, 0.605))
    success = False
    safety = False

    try:
        env.turn_active = False
        fdm["ap/afcs/psi-trim-rad"] = math.radians(target_hdg)
        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
        fdm["ap/afcs/roll-channel-active-norm"] = 1.0
        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        max_steps = int(35.0 / env.CONTROL_DT)
        for _ in range(max_steps):
            if id(fdm) != fdm_id:
                raise RuntimeError("Shared FDM identity changed inside AFCS tail")

            # Reuse the proven V5 live altitude / vertical-speed feedback.
            c_target = v5.adaptive_collective_target(env)
            collective += float(np.clip(c_target - collective, -0.00075, +0.00075))
            collective = float(np.clip(collective, 0.560, 0.610))

            fdm["ap/afcs/psi-trim-rad"] = math.radians(target_hdg)
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
            hdg_err = wrap_deg(target_hdg - heading_deg(fdm))
            roll_deg = abs(math.degrees(float(s["roll"])))
            yaw_deg = abs(math.degrees(float(s["r_rate"])))

            safety = bool(
                (not js_ok)
                or s["altitude"] < 275.0
                or s["altitude"] > 325.0
                or roll_deg > 18.0
                or abs(math.degrees(float(s["pitch"]))) > 15.0
                or yaw_deg > 30.0
                or s["forward_velocity"] < 1.0
            )
            if safety:
                break

            stable = bool(
                abs(hdg_err) <= 1.5
                and roll_deg <= 4.0
                and yaw_deg <= 6.0
                and 285.0 <= s["altitude"] <= 315.0
                and abs(float(s["vertical_speed"])) <= 1.5
                and float(s["forward_velocity"]) >= 8.0
            )
            stable_s = stable_s + env.CONTROL_DT if stable else 0.0
            if stable_s >= 3.0:
                success = True
                break

        s = env._raw_state()
        actual_delta = wrap_deg(heading_deg(fdm) - start_hdg)
        return {
            "kind": "afcs",
            "target": float(delta_deg),
            "success": bool(success),
            "safe": bool(safety),
            "done": float(actual_delta),
            "rem": float(delta_deg - actual_delta),
            "alt": float(s["altitude"]),
            "capture": f"psi={start_hdg:.1f}->{target_hdg:.1f}",
        }
    finally:
        release_wrapper(env)


def run_command(stack, target: float, seed: int):
    plan = plan_command(float(target))

    # Build one randomized full-entry live state.  The target passed here is
    # irrelevant after reset; it is immediately replaced by the first planned
    # primitive without resetting the FDM.
    bootstrap_target = float(plan[0][1]) if plan else float(target)
    bootstrap = v16.RandomizedEntryEnv(target_turn_deg=bootstrap_target)
    _, reset_info = bootstrap.reset(seed=int(seed))
    fdm = bootstrap.fdm
    fdm_id = id(fdm)
    start_hdg = heading_deg(fdm)
    start_sim = float(fdm["simulation/sim-time-sec"])

    rows = []
    ok = True
    try:
        for idx, (kind, angle) in enumerate(plan, start=1):
            if kind == "rl":
                r = run_rl_primitive(stack, fdm, float(angle), fdm_id)
            else:
                r = fine_afcs_tail(fdm, float(angle), fdm_id)
            rows.append(r)
            print(
                f"    primitive {idx}/{len(plan)} {kind.upper():4s} {angle:+6.1f} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} "
                f"done={r['done']:+7.2f} rem={r['rem']:+6.2f} alt={r['alt']:7.2f} | {r['capture']}"
            )
            if not r["success"] or r["safe"]:
                ok = False
                break

        end_hdg = heading_deg(fdm)
        end_sim = float(fdm["simulation/sim-time-sec"])
        total_done = float(sum(r["done"] for r in rows))
        final_rem = float(target - total_done)

        # Final command-level success uses the executed unwrapped primitive sum,
        # not compass subtraction, so this remains meaningful near wraparound.
        command_ok = bool(ok and abs(final_rem) <= 2.0 and id(fdm) == fdm_id)
        return {
            "seed": int(seed),
            "target": float(target),
            "plan": plan,
            "success": command_ok,
            "safe": any(bool(r["safe"]) for r in rows),
            "done": total_done,
            "rem": final_rem,
            "start_hdg": start_hdg,
            "end_hdg": end_hdg,
            "sim_dt": end_sim - start_sim,
            "fdm_id": fdm_id,
            "extra": int(reset_info.get("extra_entry_steps", 0)),
        }
    finally:
        # bootstrap owns the one live FDM for this command and is closed only
        # after every primitive has finished.
        try:
            bootstrap.close()
        except Exception:
            pass


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    seeds = args.seeds or DEFAULT_SEEDS
    stack = arb.load_stack()

    print("=" * 160)
    print("ARBITRARY TURN V7 — SUPERVISORY PPO/AFCS PRIMITIVE PLANNER")
    print("Arbitrary user command -> validated internal primitives | SAME FDM within each command | teacher OFF | no retraining")
    print(f"targets={targets} seeds={seeds}")
    print("=" * 160)

    all_rows = []
    for target in targets:
        print("\n" + "-" * 160)
        print(f"USER COMMAND {target:+.1f} deg | plan={plan_command(float(target))}")
        print("-" * 160)
        for seed in seeds:
            r = run_command(stack, float(target), int(seed))
            all_rows.append(r)
            print(
                f"seed={seed:3d} target={target:+7.1f} extra={r['extra']:2d} | "
                f"PASS={str(r['success']):5s} safety={r['safe']} | "
                f"done={r['done']:+8.2f} rem={r['rem']:+7.2f} | "
                f"heading={r['start_hdg']:.2f}->{r['end_hdg']:.2f} | sim_dt={r['sim_dt']:.2f}s | fdm={r['fdm_id']}"
            )

    print("\n" + "=" * 160)
    total = sum(int(r["success"]) for r in all_rows)
    safety = sum(int(r["safe"]) for r in all_rows)
    print(f"TOTAL PASS={total}/{len(all_rows)} | SAFETY_FAILURES={safety}/{len(all_rows)}")
    for target in targets:
        q = [r for r in all_rows if r["target"] == float(target)]
        print(
            f"target={target:+7.1f} | pass={sum(int(r['success']) for r in q)}/{len(q)} | "
            f"safety_fail={sum(int(r['safe']) for r in q)}/{len(q)} | "
            f"max_abs_remaining={max(abs(r['rem']) for r in q):.2f}"
        )
    print("ARBITRARY V7 SUPERVISOR:", "PASS" if total == len(all_rows) and safety == 0 else "PARTIAL/FAIL")
    print("=" * 160)


if __name__ == "__main__":
    main()
