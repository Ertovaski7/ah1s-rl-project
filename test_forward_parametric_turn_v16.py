from pathlib import Path
import math
import numpy as np

AUTH_SOURCE = Path("diagnose_forward_turn_authority_v1.py")
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
ns = {"__name__": "parametric_turn_v16_base", "__file__": str(AUTH_SOURCE)}
exec(compile(prefix, str(AUTH_SOURCE), "exec"), ns)
ns["TURN_ENTRY_FORWARD_FT"] = 160.0

build_forward_entry = ns["build_forward_entry"]
close_case = ns["close_case"]
snapshot = ns["snapshot"]
raw_policy_cycle = ns["raw_policy_cycle"]
stage2_model = ns["stage2_model"]


def wrap_deg(x):
    return float((float(x) + 180.0) % 360.0 - 180.0)


# ---------------- MANEUVER ----------------
TURN_AILERON_SCALE = 0.300
TURN_RUDDER_SCALE = 0.500
TURN_ROLL_AFCS = 0.15
TURN_YAW_AFCS = 0.15
MAX_BANK_DEG = 5.0
MAX_YAW_RATE_CMD = 1.50
CAPTURE_TOL_DEG = 0.75
CAPTURE_HOLD_S = 0.75
VERIFY_S = 10.0
MAX_TEST_TIME_S = max(70.0, abs(TARGET_TURN_DEG) / 1.10 + 60.0)

# ---------------- TERMINAL HOLD ----------------
# Level the aircraft first; use heading PI on rudder to remove steady-state bias.
HOLD_ROLL_KP = 0.45
HOLD_AILERON_MAX = 1.0
HOLD_HEADING_KP = 0.22
HOLD_HEADING_KI = 0.030
HOLD_YAW_RATE_KD = 0.75
HOLD_RUDDER_MAX = 1.0
HOLD_INTEGRAL_LIMIT = 20.0

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

# ---------------- SAFETY ----------------
SAFE_ALT_MIN_FT = 280.0
SAFE_ALT_MAX_FT = 320.0
SAFE_MAX_ROLL_DEG = 15.0
SAFE_MAX_PITCH_DEG = 12.0
SAFE_MAX_YAW_RATE_DEG_S = 25.0
SAFE_MIN_FORWARD_SPEED_FPS = 2.0


def maneuver_teacher(state, remaining):
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])

    desired_yaw_rate = float(np.clip(
        1.20 * remaining,
        -MAX_YAW_RATE_CMD,
        +MAX_YAW_RATE_CMD,
    ))
    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    if remaining >= 0.0:
        rudder = float(np.clip(rudder, -1.0, +0.20))
    else:
        rudder = float(np.clip(rudder, -0.20, +1.0))

    desired_roll = float(np.clip(0.25 * remaining, -MAX_BANK_DEG, +MAX_BANK_DEG))
    aileron = float(np.clip(0.35 * (desired_roll - roll), -1.0, +1.0))
    return aileron, rudder, desired_roll, desired_yaw_rate


def terminal_hold_teacher(state, remaining, heading_integral):
    """Bias-free terminal hold: zero bank + PI heading/yaw control."""
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])

    # Level wings. Persistent bank was the V15 steady-state bias source.
    desired_roll = 0.0
    aileron = float(np.clip(
        HOLD_ROLL_KP * (desired_roll - roll),
        -HOLD_AILERON_MAX,
        +HOLD_AILERON_MAX,
    ))

    # Desired yaw rate includes a small integral term so constant trim/coupling
    # cannot leave a 2-3 degree terminal heading offset.
    desired_yaw_rate = float(np.clip(
        HOLD_HEADING_KP * remaining + HOLD_HEADING_KI * heading_integral,
        -1.0,
        +1.0,
    ))
    rudder = float(np.clip(
        -HOLD_YAW_RATE_KD * (desired_yaw_rate - yaw_rate),
        -HOLD_RUDDER_MAX,
        +HOLD_RUDDER_MAX,
    ))
    return aileron, rudder, desired_roll, desired_yaw_rate


def safety_reason(state):
    roll = math.degrees(state["roll_rad"])
    pitch = math.degrees(state["pitch_rad"])
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    if state["altitude_ft"] < SAFE_ALT_MIN_FT:
        return "altitude_low"
    if state["altitude_ft"] > SAFE_ALT_MAX_FT:
        return "altitude_high"
    if abs(roll) > SAFE_MAX_ROLL_DEG:
        return "roll_limit"
    if abs(pitch) > SAFE_MAX_PITCH_DEG:
        return "pitch_limit"
    if abs(yaw_rate) > SAFE_MAX_YAW_RATE_DEG_S:
        return "yaw_rate_limit"
    if state["forward_speed_fps"] < SAFE_MIN_FORWARD_SPEED_FPS:
        return "forward_speed_low"
    return ""


print("=" * 120)
print("PARAMETRIC FORWARD-FLIGHT TURN V16 - PI TERMINAL HOLD")
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

    if hasattr(env2, "target_heading"):
        env2.target_heading = float(mission_heading)

    fdm["ap/afcs/roll-channel-active-norm"] = TURN_ROLL_AFCS
    fdm["ap/afcs/yaw-channel-active-norm"] = TURN_YAW_AFCS

    original_apply_action = env2._apply_action
    last_collective = [COLLECTIVE_FF]

    def controlled_apply_action(action):
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
        last_collective[0] = collective
        return controls

    env2._apply_action = controlled_apply_action

    initial = snapshot(fdm, lat0, lon0, mission_heading)
    initial_hdg = float(initial["heading_error_deg"])
    target_unwrapped = initial_hdg + TARGET_TURN_DEG
    target_wrapped = wrap_deg(target_unwrapped)

    print(f"ENTRY HEADING ERROR = {initial_hdg:+.3f} deg")
    print(f"TARGET UNWRAPPED    = {target_unwrapped:+.3f} deg")
    print(f"TARGET WRAPPED      = {target_wrapped:+.3f} deg")

    prev_hdg = initial_hdg
    cumulative = 0.0
    capture_hold_s = 0.0
    verify_s = 0.0
    heading_integral = 0.0
    captured = False
    success = False
    termination = "time_limit"

    for step in range(int(MAX_TEST_TIME_S / dt)):
        before = snapshot(fdm, lat0, lon0, mission_heading)
        remaining = TARGET_TURN_DEG - cumulative

        obs = np.asarray(env2._get_obs(), dtype=np.float32)
        base, _ = stage2_model.predict(obs, deterministic=True)
        action = np.asarray(base, dtype=np.float32).reshape(-1).copy()

        speed_error = TARGET_FORWARD_SPEED_FPS - before["forward_speed_fps"]
        action[1] = float(np.clip(FORWARD_SPEED_KP * speed_error, -1.0, +1.0))

        if not captured:
            a2, a3, desired_roll, desired_yaw_rate = maneuver_teacher(before, remaining)
        else:
            # Integrate only a bounded terminal heading error (anti-windup).
            heading_integral += float(np.clip(remaining, -5.0, +5.0)) * dt
            heading_integral = float(np.clip(
                heading_integral,
                -HOLD_INTEGRAL_LIMIT,
                +HOLD_INTEGRAL_LIMIT,
            ))
            a2, a3, desired_roll, desired_yaw_rate = terminal_hold_teacher(
                before, remaining, heading_integral
            )

        action[2] = a2
        action[3] = a3

        state, _ = raw_policy_cycle(env2, fdm, action, lat0, lon0, mission_heading)

        current_hdg = float(state["heading_error_deg"])
        cumulative += wrap_deg(current_hdg - prev_hdg)
        prev_hdg = current_hdg
        remaining = TARGET_TURN_DEG - cumulative

        yaw_rate = math.degrees(state["yaw_rate_rad_s"])
        roll = math.degrees(state["roll_rad"])

        if not captured:
            capture_ok = abs(remaining) <= CAPTURE_TOL_DEG and abs(roll) <= 3.0
            capture_hold_s = capture_hold_s + dt if capture_ok else 0.0
            if capture_hold_s >= CAPTURE_HOLD_S:
                captured = True
                verify_s = 0.0
                heading_integral = 0.0
                print(
                    f"TURN CAPTURED | requested={TARGET_TURN_DEG:+.2f} | "
                    f"captured={cumulative:+.2f} | hdg={current_hdg:+.2f}"
                )
        else:
            verify_ok = bool(
                abs(remaining) <= 1.50
                and abs(roll) <= 4.0
                and abs(yaw_rate) <= 6.0
                and 285.0 <= state["altitude_ft"] <= 315.0
                and abs(state["vertical_speed_fps"]) <= 1.5
                and state["forward_speed_fps"] >= 8.0
            )
            verify_s = verify_s + dt if verify_ok else 0.0
            if verify_s >= VERIFY_S:
                success = True
                termination = "turn_and_pi_terminal_heading_hold_verified"

        reason = safety_reason(state)
        if reason:
            termination = "safety:" + reason
            print("SAFETY STOP:", reason)
            break
        if success:
            break

        if step % max(1, int(1.0 / dt)) == 0:
            mode = "HOLD" if captured else "TURN"
            print(
                f"t={(step+1)*dt:6.1f}s | MODE={mode:4s} | "
                f"TURN={cumulative:+7.2f}/{TARGET_TURN_DEG:+7.2f} | REM={remaining:+6.2f} | "
                f"HDG={state['heading_error_deg']:+7.2f} | ROLL={roll:+6.2f} | YR={yaw_rate:+6.2f} | "
                f"ALT={state['altitude_ft']:7.2f} | VS={state['vertical_speed_fps']:+6.2f} | "
                f"COLL={last_collective[0]:.4f} | V={state['forward_speed_fps']:6.2f} | "
                f"A1={action[1]:+5.2f} | A2={action[2]:+5.2f} | A3={action[3]:+5.2f} | "
                f"I={heading_integral:+6.2f} | VERIFY={verify_s:4.1f}s"
            )

    final = snapshot(fdm, lat0, lon0, mission_heading)
    print("=" * 120)
    print("PARAMETRIC TURN PASS:", success)
    print(f"REQUESTED: {TARGET_TURN_DEG:+.3f} deg")
    print(f"COMPLETED: {cumulative:+.3f} deg")
    print(f"ERROR: {TARGET_TURN_DEG - cumulative:+.3f} deg")
    print("CAPTURED:", captured)
    print(f"VERIFIED: {verify_s:.2f} s")
    print(f"FINAL ALT: {final['altitude_ft']:.2f} ft")
    print(f"FINAL V: {final['forward_speed_fps']:.2f} ft/s")
    print("TERMINATION:", termination)
    print("=" * 120)
finally:
    close_case(start)
