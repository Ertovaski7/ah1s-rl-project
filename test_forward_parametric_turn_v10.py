from pathlib import Path

# V10 keeps V9 vertical-control tuning and adds a dedicated forward-speed
# controller on action[1] (longitudinal cyclic / elevator). This isolates the
# new coupling: collective handles altitude, elevator holds forward speed.
source = Path("test_forward_parametric_turn_v8.py").read_text(encoding="utf-8")

replacements = {
    'results_forward_parametric_turn_v8': 'results_forward_parametric_turn_v10',
    'parametric_turn_v8_base': 'parametric_turn_v10_base',
    'PARAMETRIC FORWARD-FLIGHT TURN V8': 'PARAMETRIC FORWARD-FLIGHT TURN V10',
    'COLLECTIVE_FF = 0.5820': 'COLLECTIVE_FF = 0.5780',
    'MAX_DOWNWARD_VS_CMD = 2.00': 'MAX_DOWNWARD_VS_CMD = 2.50',
    'UPWARD_VS_GAIN = 0.020': 'UPWARD_VS_GAIN = 0.032',
    'DOWNWARD_VS_GAIN = 0.004': 'DOWNWARD_VS_GAIN = 0.003',
    'PHYS_COLLECTIVE_MIN = 0.500': 'PHYS_COLLECTIVE_MIN = 0.460',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"V10 patch target not found: {old}")
    source = source.replace(old, new, 1)

# Insert forward-speed hold constants immediately before safety limits.
anchor = 'SAFE_ALT_MIN_FT = 280.0\n'
insert = '''# V10 longitudinal hold. Stage-2 mapping is approximately:\n# action1=-1 -> elevator=-0.180 (~0 ft/s)\n# action1= 0 -> elevator=-0.145 (~11-12 ft/s region)\n# action1=+1 -> elevator=-0.110 (~22 ft/s)\nTARGET_FORWARD_SPEED_FPS = 11.5\nFORWARD_SPEED_KP = 0.12\nMAX_FORWARD_ACTION = 1.0\n\n'''
if anchor not in source:
    raise RuntimeError("V10 safety anchor not found")
source = source.replace(anchor, insert + anchor, 1)

# Replace the turn-action block so action[1] directly holds forward speed.
old_block = '''        if not complete:\n            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)\n            action[2] = a2\n            action[3] = a3\n        else:\n            a2 = a3 = desired_roll = desired_yaw_rate = 0.0\n'''
new_block = '''        if not complete:\n            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)\n\n            # Dedicated longitudinal speed hold. At the nominal 11.5 ft/s\n            # target action1 is near zero -> physical elevator near -0.145.\n            # If speed decays, action1 becomes positive (less negative\n            # elevator) and restores forward flight. If speed is too high,\n            # action1 becomes negative and trims it back.\n            speed_error = TARGET_FORWARD_SPEED_FPS - before["forward_speed_fps"]\n            action[1] = float(np.clip(\n                FORWARD_SPEED_KP * speed_error,\n                -MAX_FORWARD_ACTION,\n                +MAX_FORWARD_ACTION,\n            ))\n\n            action[2] = a2\n            action[3] = a3\n        else:\n            speed_error = TARGET_FORWARD_SPEED_FPS - before["forward_speed_fps"]\n            action[1] = float(np.clip(\n                FORWARD_SPEED_KP * speed_error,\n                -MAX_FORWARD_ACTION,\n                +MAX_FORWARD_ACTION,\n            ))\n            a2 = a3 = desired_roll = desired_yaw_rate = 0.0\n'''
if old_block not in source:
    raise RuntimeError("V10 action block not found")
source = source.replace(old_block, new_block, 1)

# Add longitudinal diagnostics to trace.
old_trace = '            "teacher_a3": float(a3),\n            "turn_complete": bool(complete),\n'
new_trace = '            "teacher_a3": float(a3),\n            "speed_error_fps": float(speed_error),\n            "longitudinal_action1": float(action[1]),\n            "physical_elevator": float(fdm["fcs/elevator-cmd-norm"]),\n            "turn_complete": bool(complete),\n'
if old_trace not in source:
    raise RuntimeError("V10 trace anchor not found")
source = source.replace(old_trace, new_trace, 1)

# Extend console diagnostics.
old_print = '                f"K={last_alt[\'gain\']:.3f} | COLL={last_alt[\'collective\']:.4f} | "\n                f"V={state[\'forward_speed_fps\']:6.2f}"\n'
new_print = '                f"K={last_alt[\'gain\']:.3f} | COLL={last_alt[\'collective\']:.4f} | "\n                f"V={state[\'forward_speed_fps\']:6.2f} | A1={action[1]:+5.2f} | "\n                f"ELEV={float(fdm[\'fcs/elevator-cmd-norm\']):+.4f}"\n'
if old_print not in source:
    raise RuntimeError("V10 print anchor not found")
source = source.replace(old_print, new_print, 1)

exec(compile(source, "test_forward_parametric_turn_v10.py", "exec"), globals(), globals())
