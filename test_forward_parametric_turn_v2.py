from pathlib import Path
import csv
import json
import math

import numpy as np

# =====================================================================
# AH-1S / JSBSim
# PARAMETRIC FORWARD-FLIGHT TURN TEACHER V2
#
# Input examples:
#   +50  -> positive-direction 50 deg turn
#   +200 -> positive-direction 200 deg turn (NOT shortened to -160)
#   -50  -> negative-direction 50 deg turn
#   +360 -> one full positive revolution
#
# Stage-2 policy remains the base controller.
# Turn teacher adds residuals only to action[2] (aileron) and
# action[3] (rudder). The requested turn is tracked cumulatively so
# commands larger than 180 deg and full revolutions remain meaningful.
# =====================================================================

AUTH_SOURCE = Path("diagnose_forward_turn_authority_v1.py")
RESULT_DIR = Path("results_forward_parametric_turn_v2")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TURN_DEG = float(
    input("Kac derece donsun? (orn: 50, 200, 360, -50): ")
)
if abs(TARGET_TURN_DEG) < 1e-6:
    raise ValueError("Donus acisi 0 olamaz.")

# ---------------------------------------------------------------------
# Load the already-validated Stage-1/Stage-2 forward-entry helpers
# without re-running the authority experiment at module import.
# ---------------------------------------------------------------------
if not AUTH_SOURCE.exists():
    raise FileNotFoundError(f"Missing required source: {AUTH_SOURCE}")

source_text = AUTH_SOURCE.read_text(encoding="utf-8")
RUN_MARKER = (
    'print("=" * 120)\n'
    'print("FORWARD-FLIGHT TURN AUTHORITY IDENTIFICATION V1")'
)
if RUN_MARKER not in source_text:
    raise RuntimeError("Could not locate run marker in authority source.")

ns = {
    "__name__": "parametric_turn_v2_base",
    "__file__": str(AUTH_SOURCE),
}
exec(compile(source_text.split(RUN_MARKER, 1)[0], str(AUTH_SOURCE), "exec"), ns)

# Start the turn during established Stage-2 forward flight.
ns["TURN_ENTRY_FORWARD_FT"] = 160.0

build_forward_entry = ns["build_forward_entry"]
close_case = ns["close_case"]
snapshot = ns["snapshot"]
raw_policy_cycle = ns["raw_policy_cycle"]
stage2_model = ns["stage2_model"]


def wrap_deg(x):
    """Wrap ONLY incremental/display angles to [-180, 180)."""
    return float((float(x) + 180.0) % 360.0 - 180.0)


# =====================================================================
# TURN TEACHER PARAMETERS
# Positive turn authority was identified previously as approximately
# aileron +0.30 / rudder -0.60. Negative direction is mirrored here
# and must still be validated separately in JSBSim.
# =====================================================================
MAX_AILERON_TURN = 0.30
MAX_RUDDER_TURN = 0.60
MAX_AILERON_BRAKE = 0.20
MAX_RUDDER_BRAKE = 0.35

HEADING_KP_RUDDER = 0.12
YAW_RATE_KD_RUDDER = 0.12
HEADING_KP_AILERON = 0.06
ROLL_LEVEL_KP_AILERON = 0.02

TARGET_HEADING_TOL_DEG = 1.0
TARGET_YAW_RATE_TOL_DEG_S = 0.50
TARGET_MAX_ABS_ROLL_DEG = 5.0
TARGET_HOLD_SECONDS = 3.0
POST_TURN_FORWARD_SECONDS = 10.0
MAX_TEST_TIME_S = 180.0

SAFE_ALT_MIN_FT = 285.0
SAFE_ALT_MAX_FT = 315.0
SAFE_MAX_ABS_ROLL_DEG = 15.0
SAFE_MAX_ABS_PITCH_DEG = 12.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 25.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0


def parametric_turn_teacher(state, remaining_turn_deg):
    """Return lateral/yaw residuals for the remaining UNWRAPPED turn."""
    yaw_rate_deg_s = math.degrees(state["yaw_rate_rad_s"])
    roll_deg = math.degrees(state["roll_rad"])

    # Project sign convention:
    # positive heading demand -> positive aileron, negative rudder.
    delta_a3 = (
        -HEADING_KP_RUDDER * remaining_turn_deg
        + YAW_RATE_KD_RUDDER * yaw_rate_deg_s
    )
    delta_a2 = (
        HEADING_KP_AILERON * remaining_turn_deg
        - ROLL_LEVEL_KP_AILERON * roll_deg
    )

    if remaining_turn_deg >= 0.0:
        delta_a2 = float(np.clip(delta_a2, -MAX_AILERON_BRAKE, +MAX_AILERON_TURN))
        delta_a3 = float(np.clip(delta_a3, -MAX_RUDDER_TURN, +MAX_RUDDER_BRAKE))
    else:
        delta_a2 = float(np.clip(delta_a2, -MAX_AILERON_TURN, +MAX_AILERON_BRAKE))
        delta_a3 = float(np.clip(delta_a3, -MAX_RUDDER_BRAKE, +MAX_RUDDER_TURN))

    return delta_a2, delta_a3


def safety_reason(state):
    roll_deg = math.degrees(state["roll_rad"])
    pitch_deg = math.degrees(state["pitch_rad"])
    yaw_rate_deg_s = math.degrees(state["yaw_rate_rad_s"])

    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"
    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"
    if abs(roll_deg) > SAFE_MAX_ABS_ROLL_DEG:
        return "roll_limit"
    if abs(pitch_deg) > SAFE_MAX_ABS_PITCH_DEG:
        return "pitch_limit"
    if abs(yaw_rate_deg_s) > SAFE_MAX_ABS_YAW_RATE_DEG_S:
        return "yaw_rate_limit"
    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS:
        return "forward_speed_low"
    return ""


print("=" * 120)
print("PARAMETRIC FORWARD-FLIGHT TURN V2")
print(f"REQUESTED TURN = {TARGET_TURN_DEG:+.2f} deg")
print("=" * 120)

start = build_forward_entry()

try:
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0 = start["lat0"]
    lon0 = start["lon0"]
    mission_heading = start["mission_heading"]
    dt = start["dt"]

    initial = snapshot(fdm, lat0, lon0, mission_heading)
    initial_heading_deg = float(initial["heading_error_deg"])

    target_unwrapped_deg = initial_heading_deg + TARGET_TURN_DEG
    target_wrapped_deg = wrap_deg(target_unwrapped_deg)
    new_target_heading_rad = mission_heading + math.radians(target_wrapped_deg)

    # IMPORTANT: Stage-2 keeps its original heading reference while the
    # turn teacher owns the maneuver. The new heading is handed to Stage-2
    # only after the turn has actually completed.
    previous_wrapped_heading = initial_heading_deg
    cumulative_turn_deg = 0.0

    trace = []
    target_hold_s = 0.0
    post_turn_s = 0.0
    turn_complete = False
    success = False
    termination = "time_limit"

    max_steps = int(MAX_TEST_TIME_S / dt)

    for step in range(max_steps):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        remaining_turn_deg = TARGET_TURN_DEG - cumulative_turn_deg

        obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
        base_action, _ = stage2_model.predict(obs2, deterministic=True)
        base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
        action = base_action.copy()

        if not turn_complete:
            delta_a2, delta_a3 = parametric_turn_teacher(before, remaining_turn_deg)
            action[2] = float(np.clip(base_action[2] + delta_a2, -1.0, +1.0))
            action[3] = float(np.clip(base_action[3] + delta_a3, -1.0, +1.0))
        else:
            delta_a2 = 0.0
            delta_a3 = 0.0

        # Apply this step first, then update cumulative heading from the
        # resulting state. This avoids a one-control-step tracking lag.
        state, used_action = raw_policy_cycle(
            env2, fdm, action, lat0, lon0, mission_heading
        )

        current_wrapped_heading = float(state["heading_error_deg"])
        incremental_heading_change = wrap_deg(
            current_wrapped_heading - previous_wrapped_heading
        )
        cumulative_turn_deg += incremental_heading_change
        previous_wrapped_heading = current_wrapped_heading
        remaining_turn_deg = TARGET_TURN_DEG - cumulative_turn_deg

        yaw_rate_deg_s = math.degrees(state["yaw_rate_rad_s"])
        roll_deg = math.degrees(state["roll_rad"])
        pitch_deg = math.degrees(state["pitch_rad"])

        angle_ok = abs(remaining_turn_deg) <= TARGET_HEADING_TOL_DEG
        yaw_ok = abs(yaw_rate_deg_s) <= TARGET_YAW_RATE_TOL_DEG_S
        roll_ok = abs(roll_deg) <= TARGET_MAX_ABS_ROLL_DEG

        if not turn_complete:
            if angle_ok and yaw_ok and roll_ok:
                target_hold_s += dt
            else:
                target_hold_s = 0.0

            if target_hold_s >= TARGET_HOLD_SECONDS:
                turn_complete = True
                if hasattr(env2, "target_heading"):
                    env2.target_heading = float(new_target_heading_rad)
                print(
                    f"TURN COMPLETE | requested={TARGET_TURN_DEG:+.2f} | "
                    f"completed={cumulative_turn_deg:+.2f}"
                )
                print("Teacher OFF -> Stage-2 continues on the new heading.")

        if turn_complete:
            post_turn_s += dt
            if post_turn_s >= POST_TURN_FORWARD_SECONDS:
                success = True
                termination = "turn_and_new_heading_forward_pass"

        trace.append({
            "time_s": float((step + 1) * dt),
            "requested_turn_deg": float(TARGET_TURN_DEG),
            "cumulative_turn_deg": float(cumulative_turn_deg),
            "remaining_turn_deg": float(remaining_turn_deg),
            "heading_error_deg": float(state["heading_error_deg"]),
            "target_wrapped_heading_deg": float(target_wrapped_deg),
            "altitude_ft": float(state["altitude_ft"]),
            "vertical_speed_fps": float(state["vertical_speed_fps"]),
            "forward_speed_fps": float(state["forward_speed_fps"]),
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_rate_deg_s": float(yaw_rate_deg_s),
            "teacher_delta_a2": float(delta_a2),
            "teacher_delta_a3": float(delta_a3),
            "used_collective": float(used_action[0]),
            "used_elevator": float(used_action[1]),
            "used_aileron": float(used_action[2]),
            "used_rudder": float(used_action[3]),
            "turn_complete": bool(turn_complete),
        })

        reason = safety_reason(state)
        if reason:
            termination = "safety:" + reason
            print("SAFETY STOP:", reason)
            break

        if success:
            break

        if step % max(1, int(1.0 / dt)) == 0:
            print(
                f"t={(step + 1) * dt:6.1f}s | "
                f"TURN={cumulative_turn_deg:+7.2f}/{TARGET_TURN_DEG:+7.2f} | "
                f"REM={remaining_turn_deg:+7.2f} | "
                f"HDG={state['heading_error_deg']:+7.2f} | "
                f"ROLL={roll_deg:+6.2f} | YR={yaw_rate_deg_s:+6.2f} | "
                f"ALT={state['altitude_ft']:7.2f} | "
                f"V={state['forward_speed_fps']:6.2f}"
            )

    if trace:
        with (RESULT_DIR / "parametric_turn_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)

    final_state = snapshot(fdm, lat0, lon0, mission_heading)
    summary = {
        "success": bool(success),
        "termination": str(termination),
        "requested_turn_deg": float(TARGET_TURN_DEG),
        "cumulative_turn_deg": float(cumulative_turn_deg),
        "turn_error_deg": float(TARGET_TURN_DEG - cumulative_turn_deg),
        "target_wrapped_heading_deg": float(target_wrapped_deg),
        "final_heading_error_deg": float(final_state["heading_error_deg"]),
        "final_forward_speed_fps": float(final_state["forward_speed_fps"]),
        "final_altitude_ft": float(final_state["altitude_ft"]),
        "teacher_off_after_turn": bool(turn_complete),
        "post_turn_forward_seconds": float(post_turn_s),
    }

    with (RESULT_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 120)
    print("PARAMETRIC TURN PASS:", success)
    print(f"REQUESTED: {TARGET_TURN_DEG:+.3f} deg")
    print(f"COMPLETED: {cumulative_turn_deg:+.3f} deg")
    print(f"ERROR: {TARGET_TURN_DEG - cumulative_turn_deg:+.3f} deg")
    print("TERMINATION:", termination)
    print("=" * 120)

finally:
    close_case(start)
