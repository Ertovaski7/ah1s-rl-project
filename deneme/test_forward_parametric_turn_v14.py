from pathlib import Path
import csv, json, math
import numpy as np

AUTH_SOURCE = Path("diagnose_forward_turn_authority_v1.py")
RESULT_DIR = Path("results_forward_parametric_turn_v14")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TURN_DEG = float(input("Kaç derece dönsün? (örn: 50, 200, 360, -50): "))
if abs(TARGET_TURN_DEG) < 1e-6:
    raise ValueError("Dönüş açısı 0 olamaz.")

# Load only the reusable authority/entry helpers. No V8/V11/V12/V13 wrapper chain.
source_text = AUTH_SOURCE.read_text(encoding="utf-8")
RUN_MARKER = 'print("=" * 120)\nprint("FORWARD-FLIGHT TURN AUTHORITY IDENTIFICATION V1")'
prefix = source_text.split(RUN_MARKER, 1)[0]
prefix = prefix.replace(
    'BASE_SOURCE = Path("diagnose_stage4_entry_margin_v3.py")',
    'BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")',
)
ns = {"__name__": "parametric_turn_v14_base", "__file__": str(AUTH_SOURCE)}
exec(compile(prefix, str(AUTH_SOURCE), "exec"), ns)
ns["TURN_ENTRY_FORWARD_FT"] = 160.0

build_forward_entry = ns["build_forward_entry"]
close_case = ns["close_case"]
snapshot = ns["snapshot"]
raw_policy_cycle = ns["raw_policy_cycle"]
stage2_model = ns["stage2_model"]


def wrap_deg(x):
    return float((float(x) + 180.0) % 360.0 - 180.0)


# ---------------- TURN ----------------
TURN_AILERON_SCALE = 0.300
TURN_RUDDER_SCALE = 0.500
MAX_BANK_DEG = 5.0
MAX_YAW_RATE_CMD_DEG_S = 1.50
TARGET_HEADING_TOL_DEG = 0.75
TARGET_CAPTURE_HOLD_S = 0.75
POST_TURN_VERIFY_S = 10.0
MAX_TEST_TIME_S = max(55.0, abs(TARGET_TURN_DEG) / 1.20 + 50.0)

# Reduced AFCS during teacher turn; smooth recovery after capture.
TURN_ROLL_AFCS = 0.15
TURN_YAW_AFCS = 0.15
ROLL_AFCS_RAMP_PER_S = 0.45
YAW_AFCS_RAMP_PER_S = 0.14
VERIFY_MIN_YAW_AFCS = 0.90

# ---------------- ALTITUDE ----------------
COLLECTIVE_FF = 0.5780
ALT_TO_VS_GAIN = 0.08
MAX_UPWARD_VS_CMD = 0.15
MAX_DOWNWARD_VS_CMD = 2.50
UPWARD_VS_GAIN = 0.045
DOWNWARD_VS_GAIN = 0.004
PHYS_COLLECTIVE_MIN = 0.460
PHYS_COLLECTIVE_MAX = 0.590

# ---------------- FORWARD SPEED ----------------
TARGET_FORWARD_SPEED_FPS = 14.5
FORWARD_SPEED_KP = 0.35
MAX_FORWARD_ACTION = 1.0

# ---------------- SAFETY ----------------
SAFE_ALT_MIN_FT = 280.0
SAFE_ALT_MAX_FT = 320.0
SAFE_MAX_ABS_ROLL_DEG = 15.0
SAFE_MAX_ABS_PITCH_DEG = 12.0
SAFE_TURN_YAW_RATE_DEG_S = 25.0
SAFE_RECOVERY_YAW_RATE_DEG_S = 45.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0


def turn_teacher(state, remaining):
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])

    desired_yaw_rate = float(np.clip(
        1.20 * remaining,
        -MAX_YAW_RATE_CMD_DEG_S,
        +MAX_YAW_RATE_CMD_DEG_S,
    ))

    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    if remaining >= 0.0:
        rudder = float(np.clip(rudder, -1.0, +0.20))
    else:
        rudder = float(np.clip(rudder, -0.20, +1.0))

    desired_roll = float(np.clip(
        0.25 * remaining,
        -MAX_BANK_DEG,
        +MAX_BANK_DEG,
    ))
    aileron = float(np.clip(
        0.35 * (desired_roll - roll),
        -1.0,
        +1.0,
    ))
    return aileron, rudder, desired_roll, desired_yaw_rate


def safety_reason(state, recovery=False):
    roll = math.degrees(state["roll_rad"])
    pitch = math.degrees(state["pitch_rad"])
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    yaw_limit = SAFE_RECOVERY_YAW_RATE_DEG_S if recovery else SAFE_TURN_YAW_RATE_DEG_S

    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"
    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"
    if abs(roll) > SAFE_MAX_ABS_ROLL_DEG:
        return "roll_limit"
    if abs(pitch) > SAFE_MAX_ABS_PITCH_DEG:
        return "pitch_limit"
    if abs(yaw_rate) > yaw_limit:
        return "yaw_rate_limit"
    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS:
        return "forward_speed_low"
    return ""


print("=" * 120)
print("PARAMETRIC FORWARD-FLIGHT TURN V14 - STANDALONE")
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

    # Preserve parent mapping and override only physical collective.
    original_apply_action = env2._apply_action
    last_alt = {"desired_vs": 0.0, "gain": 0.0, "collective": COLLECTIVE_FF}

    def turn_apply_action(action):
        controls = original_apply_action(action)
        alt = float(fdm["position/h-agl-ft"])
        vs = float(fdm["velocities/h-dot-fps"])

        desired_vs = float(np.clip(
            ALT_TO_VS_GAIN * (300.0 - alt),
            -MAX_DOWNWARD_VS_CMD,
            +MAX_UPWARD_VS_CMD,
        ))
        vs_error = desired_vs - vs
        gain = UPWARD_VS_GAIN if vs > desired_vs else DOWNWARD_VS_GAIN
        collective = float(np.clip(
            COLLECTIVE_FF + gain * vs_error,
            PHYS_COLLECTIVE_MIN,
            PHYS_COLLECTIVE_MAX,
        ))

        fdm["fcs/collective-cmd-norm"] = collective
        last_alt.update(desired_vs=desired_vs, gain=gain, collective=collective)
        return controls

    env2._apply_action = turn_apply_action

    initial = snapshot(fdm, lat0, lon0, mission_heading)
    initial_hdg = float(initial["heading_error_deg"])
    target_unwrapped = initial_hdg + TARGET_TURN_DEG
    target_wrapped = wrap_deg(target_unwrapped)
    final_target_heading = float(mission_heading + math.radians(target_wrapped))

    # Important: Stage-2 keeps the ORIGINAL heading until commanded turn capture.
    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading)

    fdm["ap/afcs/roll-channel-active-norm"] = TURN_ROLL_AFCS
    fdm["ap/afcs/yaw-channel-active-norm"] = TURN_YAW_AFCS

    print(f"ENTRY HEADING ERROR = {initial_hdg:+.3f} deg")
    print(f"TARGET UNWRAPPED    = {target_unwrapped:+.3f} deg")
    print(f"TARGET WRAPPED      = {target_wrapped:+.3f} deg")

    prev_hdg = initial_hdg
    cumulative = 0.0
    capture_hold_s = 0.0
    verify_s = 0.0
    captured = False
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

        # Longitudinal loop remains active before and after turn capture.
        speed_error = TARGET_FORWARD_SPEED_FPS - before["forward_speed_fps"]
        action[1] = float(np.clip(
            FORWARD_SPEED_KP * speed_error,
            -MAX_FORWARD_ACTION,
            +MAX_FORWARD_ACTION,
        ))

        if not captured:
            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)
            action[2] = a2
            action[3] = a3
        else:
            # Teacher completely releases lateral/yaw channels after capture.
            a2 = a3 = desired_roll = desired_yaw_rate = 0.0
            action[2] = 0.0
            action[3] = 0.0

            # Smoothly restore AFCS authority; yaw intentionally slower.
            roll_afcs = float(fdm["ap/afcs/roll-channel-active-norm"])
            yaw_afcs = float(fdm["ap/afcs/yaw-channel-active-norm"])
            fdm["ap/afcs/roll-channel-active-norm"] = min(
                1.0, roll_afcs + ROLL_AFCS_RAMP_PER_S * dt
            )
            fdm["ap/afcs/yaw-channel-active-norm"] = min(
                1.0, yaw_afcs + YAW_AFCS_RAMP_PER_S * dt
            )

        state, used = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)

        current_hdg = float(state["heading_error_deg"])
        cumulative += wrap_deg(current_hdg - prev_hdg)
        prev_hdg = current_hdg
        remaining = TARGET_TURN_DEG - cumulative

        yaw_rate = math.degrees(state["yaw_rate_rad_s"])
        roll = math.degrees(state["roll_rad"])
        pitch = math.degrees(state["pitch_rad"])

        if not captured:
            capture_ok = bool(
                abs(remaining) <= TARGET_HEADING_TOL_DEG
                and abs(roll) <= 3.0
            )
            capture_hold_s = capture_hold_s + dt if capture_ok else 0.0

            if capture_hold_s >= TARGET_CAPTURE_HOLD_S:
                captured = True
                if hasattr(env2, "target_heading"):
                    env2.target_heading = final_target_heading
                # Keep AFCS at reduced authority here. Ramp starts next step.
                fdm["ap/afcs/roll-channel-active-norm"] = TURN_ROLL_AFCS
                fdm["ap/afcs/yaw-channel-active-norm"] = TURN_YAW_AFCS
                print(
                    f"TURN CAPTURED | requested={TARGET_TURN_DEG:+.2f} | "
                    f"captured={cumulative:+.2f} | hdg={current_hdg:+.2f}"
                )
        else:
            yaw_afcs_now = float(fdm["ap/afcs/yaw-channel-active-norm"])
            verify_ok = bool(
                yaw_afcs_now >= VERIFY_MIN_YAW_AFCS
                and abs(remaining) <= 2.0
                and abs(roll) <= 5.0
                and 285.0 <= state["altitude_ft"] <= 315.0
                and abs(state["vertical_speed_fps"]) <= 1.5
                and state["forward_speed_fps"] >= 8.0
                and abs(yaw_rate) <= 8.0
            )
            verify_s = verify_s + dt if verify_ok else 0.0
            if verify_s >= POST_TURN_VERIFY_S:
                success = True
                termination = "turn_capture_smooth_afcs_and_new_heading_verified"

        trace.append({
            "time_s": float((step + 1) * dt),
            "requested_turn_deg": float(TARGET_TURN_DEG),
            "cumulative_turn_deg": float(cumulative),
            "remaining_turn_deg": float(remaining),
            "heading_error_deg": float(state["heading_error_deg"]),
            "altitude_ft": float(state["altitude_ft"]),
            "vertical_speed_fps": float(state["vertical_speed_fps"]),
            "desired_vs_fps": float(last_alt["desired_vs"]),
            "physical_collective": float(last_alt["collective"]),
            "forward_speed_fps": float(state["forward_speed_fps"]),
            "roll_deg": float(roll),
            "pitch_deg": float(pitch),
            "yaw_rate_deg_s": float(yaw_rate),
            "longitudinal_action1": float(action[1]),
            "physical_elevator": float(fdm["fcs/elevator-cmd-norm"]),
            "teacher_a2": float(a2),
            "teacher_a3": float(a3),
            "roll_afcs": float(fdm["ap/afcs/roll-channel-active-norm"]),
            "yaw_afcs": float(fdm["ap/afcs/yaw-channel-active-norm"]),
            "captured": bool(captured),
            "verify_s": float(verify_s),
        })

        reason = safety_reason(state, recovery=captured)
        if reason:
            termination = "safety:" + reason
            print("SAFETY STOP:", reason)
            break
        if success:
            break

        if step % max(1, int(1.0 / dt)) == 0:
            print(
                f"t={(step + 1) * dt:6.1f}s | TURN={cumulative:+7.2f}/{TARGET_TURN_DEG:+7.2f} | "
                f"REM={remaining:+7.2f} | HDG={state['heading_error_deg']:+7.2f} | "
                f"ROLL={roll:+6.2f} | YR={yaw_rate:+6.2f} | ALT={state['altitude_ft']:7.2f} | "
                f"VS={state['vertical_speed_fps']:+6.2f} | COLL={last_alt['collective']:.4f} | "
                f"V={state['forward_speed_fps']:6.2f} | A1={action[1]:+5.2f} | "
                f"ELEV={float(fdm['fcs/elevator-cmd-norm']):+.4f} | "
                f"AFCS_R={float(fdm['ap/afcs/roll-channel-active-norm']):.2f} | "
                f"AFCS_Y={float(fdm['ap/afcs/yaw-channel-active-norm']):.2f} | "
                f"CAP={int(captured)} | VERIFY={verify_s:4.1f}s"
            )

    if trace:
        with (RESULT_DIR / "parametric_turn_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)

    final = snapshot(fdm, lat0, lon0, mission_heading)
    summary = {
        "success": bool(success),
        "termination": termination,
        "requested_turn_deg": float(TARGET_TURN_DEG),
        "cumulative_turn_deg": float(cumulative),
        "turn_error_deg": float(TARGET_TURN_DEG - cumulative),
        "captured": bool(captured),
        "verify_s": float(verify_s),
        "final_heading_error_deg": float(final["heading_error_deg"]),
        "final_altitude_ft": float(final["altitude_ft"]),
        "final_vertical_speed_fps": float(final["vertical_speed_fps"]),
        "final_forward_speed_fps": float(final["forward_speed_fps"]),
        "final_roll_afcs": float(fdm["ap/afcs/roll-channel-active-norm"]),
        "final_yaw_afcs": float(fdm["ap/afcs/yaw-channel-active-norm"]),
    }
    with (RESULT_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 120)
    print("PARAMETRIC TURN PASS:", success)
    print(f"REQUESTED: {TARGET_TURN_DEG:+.3f} deg")
    print(f"COMPLETED: {cumulative:+.3f} deg")
    print(f"ERROR: {TARGET_TURN_DEG - cumulative:+.3f} deg")
    print("CAPTURED:", captured)
    print(f"VERIFIED: {verify_s:.2f} s")
    print("TERMINATION:", termination)
    print("=" * 120)
finally:
    close_case(start)
