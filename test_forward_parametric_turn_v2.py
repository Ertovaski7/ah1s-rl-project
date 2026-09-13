from pathlib import Path
import csv, json, math
import numpy as np

AUTH_SOURCE = Path("diagnose_forward_turn_authority_v1.py")
RESULT_DIR = Path("results_forward_parametric_turn_v3")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TURN_DEG = float(input("Kaç derece dönsün? (örn: 50, 200, 360, -50): "))
if abs(TARGET_TURN_DEG) < 1e-6:
    raise ValueError("Dönüş açısı 0 olamaz.")

source_text = AUTH_SOURCE.read_text(encoding="utf-8")
RUN_MARKER = 'print("=" * 120)\nprint("FORWARD-FLIGHT TURN AUTHORITY IDENTIFICATION V1")'
prefix = source_text.split(RUN_MARKER, 1)[0]
prefix = prefix.replace(
    'BASE_SOURCE = Path("diagnose_stage4_entry_margin_v3.py")',
    'BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")',
)
ns = {"__name__": "parametric_turn_v3_base", "__file__": str(AUTH_SOURCE)}
exec(compile(prefix, str(AUTH_SOURCE), "exec"), ns)
ns["TURN_ENTRY_FORWARD_FT"] = 160.0

build_forward_entry = ns["build_forward_entry"]
close_case = ns["close_case"]
snapshot = ns["snapshot"]
raw_policy_cycle = ns["raw_policy_cycle"]
stage2_model = ns["stage2_model"]

def wrap_deg(x):
    return float((float(x) + 180.0) % 360.0 - 180.0)

TURN_AILERON_SCALE = 0.300
TURN_RUDDER_SCALE = 0.500
MAX_BANK_DEG = 5.0
MAX_YAW_RATE_DEG_S = 1.50
TARGET_HEADING_TOL_DEG = 1.0
TARGET_YAW_RATE_TOL_DEG_S = 0.50
TARGET_MAX_ABS_ROLL_DEG = 5.0
TARGET_HOLD_SECONDS = 2.0
POST_TURN_FORWARD_SECONDS = 10.0
MAX_TEST_TIME_S = max(45.0, abs(TARGET_TURN_DEG) / 1.20 + 40.0)

SAFE_ALT_MIN_FT = 285.0
SAFE_ALT_MAX_FT = 315.0
SAFE_MAX_ABS_ROLL_DEG = 15.0
SAFE_MAX_ABS_PITCH_DEG = 12.0
SAFE_MAX_ABS_YAW_RATE_DEG_S = 25.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0

def turn_teacher(state, remaining):
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])
    desired_yaw_rate = float(np.clip(1.20 * remaining, -1.50, +1.50))
    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    if remaining >= 0.0:
        rudder = float(np.clip(rudder, -1.0, +0.20))
    else:
        rudder = float(np.clip(rudder, -0.20, +1.0))
    desired_roll = float(np.clip(0.25 * remaining, -MAX_BANK_DEG, +MAX_BANK_DEG))
    aileron = float(np.clip(0.35 * (desired_roll - roll), -1.0, +1.0))
    return aileron, rudder, desired_roll, desired_yaw_rate

def safety_reason(state):
    roll = math.degrees(state["roll_rad"])
    pitch = math.degrees(state["pitch_rad"])
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    if state["altitude_ft"] < SAFE_ALT_MIN_FT: return "altitude_low"
    if state["altitude_ft"] > SAFE_ALT_MAX_FT: return "altitude_high"
    if abs(roll) > SAFE_MAX_ABS_ROLL_DEG: return "roll_limit"
    if abs(pitch) > SAFE_MAX_ABS_PITCH_DEG: return "pitch_limit"
    if abs(yaw_rate) > SAFE_MAX_ABS_YAW_RATE_DEG_S: return "yaw_rate_limit"
    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS: return "forward_speed_low"
    return ""

print("=" * 120)
print("PARAMETRIC FORWARD-FLIGHT TURN V3")
print(f"REQUESTED TURN = {TARGET_TURN_DEG:+.2f} deg")
print(f"MAX TEST TIME = {MAX_TEST_TIME_S:.1f} s")
print("=" * 120)

start = build_forward_entry()
try:
    env2 = start["env2"]
    fdm = start["fdm"]
    lat0, lon0 = start["lat0"], start["lon0"]
    mission_heading, dt = start["mission_heading"], start["dt"]

    env2.mapped_rudder_scale = TURN_RUDDER_SCALE
    env2.mapped_aileron_scale = TURN_AILERON_SCALE

    original_apply_action = env2._apply_action
    def turn_apply_action(action):
        controls = original_apply_action(action)
        alt = float(fdm["position/h-agl-ft"])
        vs = float(fdm["velocities/h-dot-fps"])
        collective = float(np.clip(0.587 + 0.0010*(300.0-alt) - 0.0060*vs, 0.575, 0.600))
        fdm["fcs/collective-cmd-norm"] = collective
        return controls
    env2._apply_action = turn_apply_action

    initial = snapshot(fdm, lat0, lon0, mission_heading)
    initial_hdg = float(initial["heading_error_deg"])
    target_unwrapped = initial_hdg + TARGET_TURN_DEG
    target_wrapped = wrap_deg(target_unwrapped)
    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading + math.radians(target_wrapped))

    print(f"ENTRY HEADING ERROR = {initial_hdg:+.3f} deg")
    print(f"TARGET UNWRAPPED    = {target_unwrapped:+.3f} deg")
    print(f"TARGET WRAPPED      = {target_wrapped:+.3f} deg")

    prev_hdg = initial_hdg
    cumulative = 0.0
    hold_s = 0.0
    post_s = 0.0
    complete = False
    success = False
    termination = "time_limit"
    trace = []

    for step in range(int(MAX_TEST_TIME_S / dt)):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        remaining = TARGET_TURN_DEG - cumulative
        obs = np.asarray(env2._get_obs(), dtype=np.float32)
        base, _ = stage2_model.predict(obs, deterministic=True)
        base = np.asarray(base, dtype=np.float32).reshape(-1)
        action = base.copy()

        if not complete:
            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)
            alt_corr = float(np.clip(0.050*(300.0-before["altitude_ft"]) - 0.120*before["vertical_speed_fps"], -0.40, +0.40))
            action[0] = float(np.clip(base[0] + alt_corr, -1.0, +1.0))
            action[2] = a2
            action[3] = a3
        else:
            a2 = a3 = desired_roll = desired_yaw_rate = 0.0

        state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)
        current_hdg = float(state["heading_error_deg"])
        cumulative += wrap_deg(current_hdg - prev_hdg)
        prev_hdg = current_hdg
        remaining = TARGET_TURN_DEG - cumulative

        yaw_rate = math.degrees(state["yaw_rate_rad_s"])
        roll = math.degrees(state["roll_rad"])
        pitch = math.degrees(state["pitch_rad"])

        if not complete:
            if abs(remaining) <= TARGET_HEADING_TOL_DEG and abs(yaw_rate) <= TARGET_YAW_RATE_TOL_DEG_S and abs(roll) <= TARGET_MAX_ABS_ROLL_DEG:
                hold_s += dt
            else:
                hold_s = 0.0
            if hold_s >= TARGET_HOLD_SECONDS:
                complete = True
                print(f"TURN COMPLETE | requested={TARGET_TURN_DEG:+.2f} | completed={cumulative:+.2f}")

        if complete:
            post_s += dt
            if post_s >= POST_TURN_FORWARD_SECONDS:
                success = True
                termination = "turn_and_new_heading_forward_pass"

        trace.append({
            "time_s": float((step+1)*dt),
            "requested_turn_deg": float(TARGET_TURN_DEG),
            "cumulative_turn_deg": float(cumulative),
            "remaining_turn_deg": float(remaining),
            "heading_error_deg": float(state["heading_error_deg"]),
            "altitude_ft": float(state["altitude_ft"]),
            "vertical_speed_fps": float(state["vertical_speed_fps"]),
            "forward_speed_fps": float(state["forward_speed_fps"]),
            "roll_deg": float(roll),
            "pitch_deg": float(pitch),
            "yaw_rate_deg_s": float(yaw_rate),
            "desired_roll_deg": float(desired_roll),
            "desired_yaw_rate_deg_s": float(desired_yaw_rate),
            "teacher_a2": float(a2),
            "teacher_a3": float(a3),
            "used_a2": float(used[2]),
            "used_a3": float(used[3]),
            "turn_complete": bool(complete),
        })

        reason = safety_reason(state)
        if reason:
            termination = "safety:" + reason
            print("SAFETY STOP:", reason)
            break
        if success:
            break
        if step % max(1, int(1.0/dt)) == 0:
            print(f"t={(step+1)*dt:6.1f}s | TURN={cumulative:+7.2f}/{TARGET_TURN_DEG:+7.2f} | REM={remaining:+7.2f} | HDG={state['heading_error_deg']:+7.2f} | ROLL={roll:+6.2f} | YR={yaw_rate:+6.2f} | ALT={state['altitude_ft']:7.2f} | V={state['forward_speed_fps']:6.2f}")

    if trace:
        with (RESULT_DIR/"parametric_turn_trace.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader(); w.writerows(trace)

    final = snapshot(fdm, lat0, lon0, mission_heading)
    summary = {
        "success": bool(success),
        "termination": termination,
        "requested_turn_deg": float(TARGET_TURN_DEG),
        "cumulative_turn_deg": float(cumulative),
        "turn_error_deg": float(TARGET_TURN_DEG-cumulative),
        "final_heading_error_deg": float(final["heading_error_deg"]),
        "final_altitude_ft": float(final["altitude_ft"]),
        "final_forward_speed_fps": float(final["forward_speed_fps"]),
    }
    with (RESULT_DIR/"final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("="*120)
    print("PARAMETRIC TURN PASS:", success)
    print(f"REQUESTED: {TARGET_TURN_DEG:+.3f} deg")
    print(f"COMPLETED: {cumulative:+.3f} deg")
    print(f"ERROR: {TARGET_TURN_DEG-cumulative:+.3f} deg")
    print("TERMINATION:", termination)
    print("="*120)
finally:
    close_case(start)
