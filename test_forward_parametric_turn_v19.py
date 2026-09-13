from pathlib import Path

# V19 keeps V17's verified negative-turn altitude compensation.
# Key change: for negative turns, do NOT switch controller after capture.
# The exact maneuver_teacher already drove the aircraft safely to -50 deg;
# keep using that same signed remaining -> yaw-rate/bank controller during hold.

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
    # Negative turns: controller continuity is more important than changing
    # to a different terminal law. The maneuver controller itself already
    # reached -50 deg while maintaining altitude/speed in V17.
    if TARGET_TURN_DEG < 0.0:
        return maneuver_teacher(state, remaining)

    # Positive direction keeps the V16 PI terminal hold that already passed
    # +50, +200 and +360 degree tests.
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

if old not in source:
    raise RuntimeError("V19 terminal-hold patch target not found in V17")
source = source.replace(old, new, 1)
source = source.replace(
    "PARAMETRIC FORWARD-FLIGHT TURN V17 - ASYMMETRIC ALTITUDE + PI HOLD",
    "PARAMETRIC FORWARD-FLIGHT TURN V19 - NEGATIVE CONTROLLER CONTINUITY",
    1,
)
source = source.replace(
    'ns = {"__name__": "parametric_turn_v17_base", "__file__": str(AUTH_SOURCE)}',
    'ns = {"__name__": "parametric_turn_v19_base", "__file__": str(AUTH_SOURCE)}',
    1,
)

exec(compile(source, "test_forward_parametric_turn_v19.py", "exec"), globals(), globals())
