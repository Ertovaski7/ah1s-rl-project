from pathlib import Path

# V18 keeps V17's verified maneuver and asymmetric altitude controller.
# Only terminal hold is changed. For negative turns, use the same nested
# yaw-rate / bank-control sign convention that successfully performs the turn.

source = Path("test_forward_parametric_turn_v17.py").read_text(encoding="utf-8")

old = '''def terminal_hold_teacher(state, remaining, heading_integral):
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])

    desired_roll = 0.0
    aileron = float(np.clip(
        HOLD_ROLL_KP * (desired_roll - roll),
        -HOLD_AILERON_MAX,
        +HOLD_AILERON_MAX,
    ))

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
'''

new = '''def terminal_hold_teacher(state, remaining, heading_integral):
    yaw_rate = math.degrees(state["yaw_rate_rad_s"])
    roll = math.degrees(state["roll_rad"])

    if TARGET_TURN_DEG < 0.0:
        # Negative direction: preserve the same control signs that were proven
        # during the maneuver. Keep a small bank command from heading error and
        # close the inner loop on yaw rate. Do not use the V16 PI rudder law,
        # which saturated while driving the aircraft away from -50 deg.
        desired_yaw_rate = float(np.clip(0.35 * remaining, -0.80, +0.80))
        rudder = -1.60 * (desired_yaw_rate - yaw_rate)
        if remaining <= 0.0:
            rudder = float(np.clip(rudder, -0.20, +1.0))
        else:
            rudder = float(np.clip(rudder, -1.0, +0.20))

        desired_roll = float(np.clip(0.18 * remaining, -2.0, +2.0))
        aileron = float(np.clip(
            0.35 * (desired_roll - roll),
            -0.75,
            +0.75,
        ))
        return aileron, rudder, desired_roll, desired_yaw_rate

    # Positive direction keeps the V16 PI hold that already passed +200/+360.
    desired_roll = 0.0
    aileron = float(np.clip(
        HOLD_ROLL_KP * (desired_roll - roll),
        -HOLD_AILERON_MAX,
        +HOLD_AILERON_MAX,
    ))

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
'''

if old not in source:
    raise RuntimeError("V18 terminal-hold patch target not found in V17")
source = source.replace(old, new, 1)
source = source.replace(
    "PARAMETRIC FORWARD-FLIGHT TURN V17 - ASYMMETRIC ALTITUDE + PI HOLD",
    "PARAMETRIC FORWARD-FLIGHT TURN V18 - DIRECTION-CONSISTENT HOLD",
    1,
)
source = source.replace(
    'ns = {"__name__": "parametric_turn_v17_base", "__file__": str(AUTH_SOURCE)}',
    'ns = {"__name__": "parametric_turn_v18_base", "__file__": str(AUTH_SOURCE)}',
    1,
)

exec(compile(source, "test_forward_parametric_turn_v18.py", "exec"), globals(), globals())
