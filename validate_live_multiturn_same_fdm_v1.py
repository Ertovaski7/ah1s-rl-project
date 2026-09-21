from __future__ import annotations

"""
LIVE MULTI-TURN SAME-FDM VALIDATOR V1
=====================================

Purpose
-------
Validate the next architectural step after arbitrary-turn V5:

    Stage1 once
    -> Stage2 forward once
    -> relative turn command #1
    -> capture / stabilization
    -> Stage2 forward on the NEW heading
    -> relative turn command #2
    -> capture / stabilization
    -> ...

All phases share the exact same live JSBSim FDM object.  There is no reset,
state teleport or prerecorded telemetry between commands.  Runtime teacher is
OFF.  Turn control uses the current verified V22 stack plus the arbitrary-turn
V5 direct-FDM terminal capture for non-specialist angles.

This file is deliberately a deterministic validator first.  Once this same-FDM
chain passes, the exact state machine can be exposed through a live UI where
commands arrive while the simulation is running.
"""

import argparse
import math
import types

import numpy as np

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry
import test_turn_arbitrary_angles_v1 as arb
import test_turn_arbitrary_angles_v5_direct_capture as v5
import test_turn_full_entry_v22_v21_runtime as v22


SEED = 42
DEFAULT_TARGETS = [20.0, -30.0]
ENTRY_FORWARD_FT = 160.0
INTER_TURN_FORWARD_S = 5.0
POLICY_FORWARD_COORD = 403.424


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument(
        "--forward-seconds",
        type=float,
        default=INTER_TURN_FORWARD_S,
        help="Stage2 forward-flight time between consecutive turn commands",
    )
    return p.parse_args()


def load_locked_defs():
    # Kilitli Stage 1 / Stage 2 modelleri ve handoff yardımcıları.
    # (Eskiden deneme/diagnose_stage4_entry_margin_v3.py'nin ilk kısmı exec
    # ile çalıştırılıyordu; aynı tanımlar artık locked_stage1_stage2.py'de.)
    import locked_stage1_stage2

    return vars(locked_stage1_stage2)


ns = load_locked_defs()
Env1 = ns["HelicopterEnvStage1Distill"]
Env2 = ns["HelicopterEnvStage2RefineMapped"]
stage1_model = ns["stage1_model"]
stage2_model = ns["stage2_model"]
get_fdm = ns["get_fdm"]
env_control_dt = ns["env_control_dt"]
info_float = ns["info_float"]
AILERON_SCALE = ns["AILERON_SCALE"]
RUDDER_SCALE = ns["RUDDER_SCALE"]
HANDOFF_STABLE_TIME = ns["HANDOFF_STABLE_TIME"]
STAGE1_MAX_TIME = ns["STAGE1_MAX_TIME"]


def fdm_float(fdm, key: str, default=float("nan")) -> float:
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def heading_deg(fdm) -> float:
    for key in ("attitude/heading-true-rad", "attitude/psi-rad"):
        value = fdm_float(fdm, key)
        if np.isfinite(value):
            return float(math.degrees(value) % 360.0)
    return float("nan")


def telemetry(fdm):
    return {
        "sim_t": fdm_float(fdm, "simulation/sim-time-sec", 0.0),
        "alt": fdm_float(fdm, "position/h-agl-ft"),
        "vs": fdm_float(fdm, "velocities/h-dot-fps"),
        "v": fdm_float(fdm, "velocities/u-aero-fps"),
        "lat_v": fdm_float(fdm, "velocities/v-aero-fps"),
        "roll": math.degrees(fdm_float(fdm, "attitude/roll-rad", 0.0)),
        "pitch": math.degrees(fdm_float(fdm, "attitude/pitch-rad", 0.0)),
        "yaw_rate": math.degrees(fdm_float(fdm, "velocities/r-rad_sec", 0.0)),
        "heading": heading_deg(fdm),
    }


def print_tel(label: str, fdm):
    s = telemetry(fdm)
    print(
        f"{label:24s} | sim={s['sim_t']:8.2f}s | hdg={s['heading']:7.2f}deg | "
        f"alt={s['alt']:7.2f} | vs={s['vs']:+6.2f} | v={s['v']:6.2f} | "
        f"lat={s['lat_v']:+6.2f} | roll={s['roll']:+6.2f} | "
        f"pitch={s['pitch']:+6.2f} | yawRate={s['yaw_rate']:+6.2f}"
    )


def physical_safety_ok(fdm) -> bool:
    s = telemetry(fdm)
    return bool(
        275.0 <= s["alt"] <= 325.0
        and abs(s["roll"]) <= 18.0
        and abs(s["pitch"]) <= 15.0
        and abs(s["yaw_rate"]) <= 30.0
        and s["v"] >= 1.0
    )


def build_initial_live_mission(progress_callback=None):
    print("\n[1/2] Stage1: takeoff -> stable 300 ft hover")
    env1 = Env1(teacher_model_path=None, training_mode=False)
    obs1, _ = env1.reset(seed=SEED)
    fdm = get_fdm(env1)
    original_fdm_id = id(fdm)
    dt1 = env_control_dt(env1)
    stable_time = 0.0

    for step_idx in range(int(STAGE1_MAX_TIME / dt1)):
        action, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info = env1.step(action)
        if progress_callback is not None and step_idx % 5 == 0:
            progress_callback("STAGE 1 — TAKEOFF / HOVER", fdm)

        alt = info_float(info, "altitude")
        vs = info_float(info, "vertical_speed")
        vn = info_float(info, "vn", 0.0)
        ve = info_float(info, "ve", 0.0)
        drift = info_float(info, "drift", 999.0)

        stable = (
            295.0 <= alt <= 305.0
            and abs(vs) <= 0.5
            and float(np.hypot(vn, ve)) <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt1 if stable else 0.0

        if stable_time >= HANDOFF_STABLE_TIME:
            break
        if truncated or (terminated and not bool(info.get("success", False))):
            raise RuntimeError("Stage1 failed before stable handoff")

    if stable_time < HANDOFF_STABLE_TIME:
        raise RuntimeError("Stage1 stable hover handoff was not reached")

    print_tel("STAGE1 HANDOFF", fdm)

    print("\n[2/2] Stage2: same FDM -> forward-flight turn entry")
    env2 = Env2(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
    env2.reset(seed=SEED)
    env2.fdm = fdm
    env2.phase = 1
    env2.forward_distance = 0.0
    env2.previous_action = np.zeros((4,), dtype=np.float32)
    env2.steps = 0

    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    dt2 = env_control_dt(env2)

    for step_idx in range(int(70.0 / dt2)):
        action, _ = stage2_model.predict(obs2, deterministic=True)
        obs2, _, terminated, truncated, _ = env2.step(action)
        obs2 = np.asarray(obs2, dtype=np.float32)
        if progress_callback is not None and step_idx % 5 == 0:
            progress_callback("STAGE 2 — FORWARD FLIGHT", fdm)

        if env2.forward_distance >= ENTRY_FORWARD_FT:
            break
        if terminated or truncated:
            raise RuntimeError("Stage2 failed before initial turn entry")

    if env2.forward_distance < ENTRY_FORWARD_FT:
        raise RuntimeError("Stage2 did not reach initial turn entry")
    if id(fdm) != original_fdm_id or env2.fdm is not fdm:
        raise RuntimeError("FDM identity changed during Stage1 -> Stage2 handoff")

    print_tel("STAGE2 TURN ENTRY", fdm)
    print(f"Shared FDM id: {original_fdm_id}")
    return env1, env2, fdm, original_fdm_id


def fly_stage2_between_turns(env2, fdm, seconds: float, fdm_id: int):
    if id(fdm) != fdm_id or env2.fdm is not fdm:
        raise RuntimeError("Shared FDM identity changed before Stage2 continuation")

    # Reset only Stage2 bookkeeping, never JSBSim state.
    env2.phase = 1
    env2.forward_distance = 0.0
    env2.previous_action = np.zeros((4,), dtype=np.float32)
    env2.steps = 0

    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    obs = np.asarray(env2._get_obs(), dtype=np.float32)
    dt = env_control_dt(env2)
    steps = max(1, int(float(seconds) / dt))

    start_time = telemetry(fdm)["sim_t"]
    for _ in range(steps):
        action, _ = stage2_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env2.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        if not physical_safety_ok(fdm):
            raise RuntimeError("Safety envelope violated during inter-turn Stage2 flight")
        if terminated or truncated:
            # The short continuation should not finish Stage2.  This is a
            # validator, so treat unexpected termination as a failure.
            raise RuntimeError("Stage2 terminated during inter-turn continuation")

    if id(fdm) != fdm_id:
        raise RuntimeError("FDM identity changed during inter-turn Stage2 flight")

    end_time = telemetry(fdm)["sim_t"]
    print_tel("FORWARD CONTINUATION", fdm)
    print(f"  simulated forward interval: {end_time - start_time:.2f}s")


def make_turn_env(fdm, target: float):
    env = HelicopterEnvTurnGoalFullEntry(target_turn_deg=float(target))

    # Attach the ALREADY RUNNING live FDM.  reset() is intentionally not called.
    env.fdm = fdm
    env.phase = 1
    env.turn_active = True
    env.target_turn_deg = float(target)
    env.cumulative_turn_deg = 0.0
    env.prev_heading_deg = env._heading_deg()
    env.turn_steps = 0
    env.success_hold_s = 0.0
    env.previous_turn_action = np.zeros((4,), dtype=np.float32)

    # Match the turn policy's trained observation coordinate without moving the
    # helicopter.  This is bookkeeping only; FDM position/state is untouched.
    env.forward_distance = POLICY_FORWARD_COORD
    env._v2_previous_remaining = float(target)

    env.fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    env.fdm["ap/afcs/roll-channel-active-norm"] = env.TURN_ROLL_AFCS
    env.fdm["ap/afcs/yaw-channel-active-norm"] = env.TURN_YAW_AFCS

    return env, np.asarray(env._get_obs(), dtype=np.float32)


def run_turn_same_fdm(stack, fdm, target: float, fdm_id: int):
    if id(fdm) != fdm_id:
        raise RuntimeError("FDM identity changed before turn")

    bm, ba, p200, p50, p21 = stack
    env, obs = make_turn_env(fdm, target)

    specialist = v5.in_existing_specialist_gate(float(target))
    capture = False
    target_heading = None
    capture_at = None
    capture_remaining = None
    success = False
    info = {
        "remaining_turn_deg": float(target),
        "cumulative_turn_deg": 0.0,
        "success": False,
        "safety_failure": False,
    }

    print_tel(f"TURN {target:+.1f} START", fdm)
    start_hdg = heading_deg(fdm)
    start_time = telemetry(fdm)["sim_t"]

    max_steps = int(max(100.0, abs(float(target)) + 80.0) / env.CONTROL_DT)

    try:
        for _ in range(max_steps):
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

            if id(fdm) != fdm_id:
                raise RuntimeError("FDM identity changed inside turn")

            if terminated or truncated:
                success = bool(info.get("success", False))
                break

        # Re-arm straight-flight AFCS on the newly established heading.
        if success:
            if target_heading is None:
                target_heading = heading_deg(fdm)
            fdm["ap/afcs/psi-trim-rad"] = math.radians(float(target_heading))
            fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
            fdm["ap/afcs/roll-channel-active-norm"] = 1.0
            fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

        end_hdg = heading_deg(fdm)
        end_time = telemetry(fdm)["sim_t"]
        print_tel(f"TURN {target:+.1f} END", fdm)
        print(
            f"  result: PASS={success} safety={bool(info.get('safety_failure', False))} | "
            f"done={float(info.get('cumulative_turn_deg', 0.0)):+.2f}deg | "
            f"remaining={float(info.get('remaining_turn_deg', 999.0)):+.2f}deg | "
            f"heading={start_hdg:.2f}->{end_hdg:.2f}deg | "
            f"sim_dt={end_time - start_time:.2f}s | "
            f"capture={'specialist-V22' if specialist else (f'@{capture_at:.1f}s rem={capture_remaining:+.2f}' if capture else 'NONE')}"
        )

        return {
            "target": float(target),
            "success": bool(success),
            "safety": bool(info.get("safety_failure", False)),
            "done": float(info.get("cumulative_turn_deg", 0.0)),
            "remaining": float(info.get("remaining_turn_deg", 999.0)),
            "start_heading": float(start_hdg),
            "end_heading": float(end_hdg),
        }
    finally:
        # Never let this temporary turn wrapper own/close the shared FDM.
        env.fdm = None
        try:
            env.close()
        except Exception:
            pass


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS
    if any(abs(float(t)) < 1.0 or abs(float(t)) > 360.0 for t in targets):
        raise ValueError("Each command must satisfy 1 <= |target| <= 360 deg")
    if args.forward_seconds < 0.0:
        raise ValueError("--forward-seconds must be >= 0")

    print("=" * 160)
    print("LIVE MULTI-TURN SAME-FDM VALIDATOR V1")
    print("Stage1 once -> Stage2 -> turn -> forward -> turn -> ... | SAME live FDM | runtime teacher OFF")
    print("Commands:", [float(x) for x in targets])
    print("=" * 160)

    stack = arb.load_stack()
    env1 = env2 = None
    fdm = None
    results = []

    try:
        env1, env2, fdm, fdm_id = build_initial_live_mission()
        initial_sim_time = telemetry(fdm)["sim_t"]

        for i, target in enumerate(targets):
            print("\n" + "-" * 160)
            print(f"COMMAND {i + 1}/{len(targets)}: relative turn {target:+.1f} deg")
            print("-" * 160)

            r = run_turn_same_fdm(stack, fdm, float(target), fdm_id)
            results.append(r)
            if not r["success"] or r["safety"]:
                break

            if i < len(targets) - 1 and args.forward_seconds > 0.0:
                print(f"\nStage2 forward continuation for {args.forward_seconds:.1f}s on new heading...")
                fly_stage2_between_turns(env2, fdm, args.forward_seconds, fdm_id)

        final_sim_time = telemetry(fdm)["sim_t"]
        all_pass = len(results) == len(targets) and all(
            r["success"] and not r["safety"] for r in results
        )

        print("\n" + "=" * 160)
        print("SAME-FDM MULTI-TURN SUMMARY")
        print("=" * 160)
        for i, r in enumerate(results, 1):
            print(
                f"#{i} target={r['target']:+7.1f} | PASS={str(r['success']):5s} | "
                f"safety={r['safety']} | rem={r['remaining']:+6.2f} | "
                f"heading={r['start_heading']:7.2f}->{r['end_heading']:7.2f}"
            )
        print(f"Shared FDM id remained: {fdm_id}")
        print(f"Continuous simulated time advanced: {final_sim_time - initial_sim_time:.2f}s")
        print("SAME-FDM MULTI-TURN:", "PASS" if all_pass else "PARTIAL/FAIL")
        print("=" * 160)

        return 0 if all_pass else 1

    finally:
        # env1 owns the real shared FDM.  Detach env2 before final close so the
        # shared simulator is closed only once.
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
