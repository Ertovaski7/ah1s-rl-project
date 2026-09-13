from pathlib import Path

# V11 fixes the controller architecture rather than only changing gains.
# It reuses the V8 mission harness, but during the commanded turn:
#   1) roll/yaw AFCS authority is reduced so it does not fight the turn teacher,
#   2) Stage-2 keeps the ORIGINAL heading reference until the turn is complete,
#   3) elevator holds forward speed near the Stage-2 calibrated ~15 ft/s region,
#   4) collective rejects positive vertical speed earlier.
source = Path("test_forward_parametric_turn_v8.py").read_text(encoding="utf-8")

replacements = {
    'results_forward_parametric_turn_v8': 'results_forward_parametric_turn_v11',
    'parametric_turn_v8_base': 'parametric_turn_v11_base',
    'PARAMETRIC FORWARD-FLIGHT TURN V8': 'PARAMETRIC FORWARD-FLIGHT TURN V11',
    'COLLECTIVE_FF = 0.5820': 'COLLECTIVE_FF = 0.5780',
    'MAX_UPWARD_VS_CMD = 0.10': 'MAX_UPWARD_VS_CMD = 0.15',
    'MAX_DOWNWARD_VS_CMD = 2.00': 'MAX_DOWNWARD_VS_CMD = 2.50',
    'UPWARD_VS_GAIN = 0.020': 'UPWARD_VS_GAIN = 0.045',
    'DOWNWARD_VS_GAIN = 0.004': 'DOWNWARD_VS_GAIN = 0.004',
    'PHYS_COLLECTIVE_MIN = 0.500': 'PHYS_COLLECTIVE_MIN = 0.460',
    'SAFE_ALT_MIN_FT = 280.0': 'SAFE_ALT_MIN_FT = 280.0',
    'SAFE_ALT_MAX_FT = 320.0': 'SAFE_ALT_MAX_FT = 320.0',
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"V11 patch target not found: {old}")
    source = source.replace(old, new, 1)

# Longitudinal speed hold. Stage-2's own refinement reward is centered at
# ~15 ft/s, so use 14.5 ft/s during the turn instead of the too-low 11.5.
anchor = 'SAFE_ALT_MIN_FT = 280.0\n'
insert = '''TARGET_FORWARD_SPEED_FPS = 14.5\nFORWARD_SPEED_KP = 0.35\nMAX_FORWARD_ACTION = 1.0\nTURN_ROLL_AFCS = 0.15\nTURN_YAW_AFCS = 0.15\n\n'''
if anchor not in source:
    raise RuntimeError("V11 safety anchor not found")
source = source.replace(anchor, insert + anchor, 1)

# During the turn, do NOT hand Stage-2 the final wrapped heading yet.
# The teacher tracks the unwrapped requested angle. The final reference is
# installed only after the commanded turn has actually completed.
old_heading = '''    if hasattr(env2, "target_heading"):\n        env2.target_heading = float(mission_heading + math.radians(target_wrapped))\n'''
new_heading = '''    final_target_heading = float(mission_heading + math.radians(target_wrapped))\n    if hasattr(env2, "target_heading"):\n        env2.target_heading = float(mission_heading)\n\n    # Stage-2 normally keeps roll/yaw AFCS fully active. That fights the\n    # turn teacher. Reduce those channels only while the turn is active.\n    fdm["ap/afcs/roll-channel-active-norm"] = TURN_ROLL_AFCS\n    fdm["ap/afcs/yaw-channel-active-norm"] = TURN_YAW_AFCS\n'''
if old_heading not in source:
    raise RuntimeError("V11 heading block not found")
source = source.replace(old_heading, new_heading, 1)

old_action = '''        if not complete:\n            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)\n            action[2] = a2\n            action[3] = a3\n        else:\n            a2 = a3 = desired_roll = desired_yaw_rate = 0.0\n'''
new_action = '''        # Dedicated forward-speed loop. Less-negative elevator means more\n        # forward speed in the Stage-2 calibration.\n        speed_error = TARGET_FORWARD_SPEED_FPS - before["forward_speed_fps"]\n        action[1] = float(np.clip(\n            FORWARD_SPEED_KP * speed_error,\n            -MAX_FORWARD_ACTION,\n            +MAX_FORWARD_ACTION,\n        ))\n\n        if not complete:\n            a2, a3, desired_roll, desired_yaw_rate = turn_teacher(before, remaining)\n            action[2] = a2\n            action[3] = a3\n        else:\n            a2 = a3 = desired_roll = desired_yaw_rate = 0.0\n'''
if old_action not in source:
    raise RuntimeError("V11 action block not found")
source = source.replace(old_action, new_action, 1)

# When the turn is complete, restore normal stabilization and install the
# new heading reference exactly once.
old_complete = '''            if hold_s >= TARGET_HOLD_SECONDS:\n                complete = True\n                print(f"TURN COMPLETE | requested={TARGET_TURN_DEG:+.2f} | completed={cumulative:+.2f}")\n'''
new_complete = '''            if hold_s >= TARGET_HOLD_SECONDS:\n                complete = True\n                if hasattr(env2, "target_heading"):\n                    env2.target_heading = final_target_heading\n                fdm["ap/afcs/roll-channel-active-norm"] = 1.0\n                fdm["ap/afcs/yaw-channel-active-norm"] = 1.0\n                print(f"TURN COMPLETE | requested={TARGET_TURN_DEG:+.2f} | completed={cumulative:+.2f}")\n'''
if old_complete not in source:
    raise RuntimeError("V11 completion block not found")
source = source.replace(old_complete, new_complete, 1)

# Add the longitudinal diagnostics to CSV and console.
old_trace = '            "teacher_a3": float(a3),\n            "turn_complete": bool(complete),\n'
new_trace = '            "teacher_a3": float(a3),\n            "speed_error_fps": float(speed_error),\n            "longitudinal_action1": float(action[1]),\n            "physical_elevator": float(fdm["fcs/elevator-cmd-norm"]),\n            "roll_afcs": float(fdm["ap/afcs/roll-channel-active-norm"]),\n            "yaw_afcs": float(fdm["ap/afcs/yaw-channel-active-norm"]),\n            "turn_complete": bool(complete),\n'
if old_trace not in source:
    raise RuntimeError("V11 trace anchor not found")
source = source.replace(old_trace, new_trace, 1)

old_print = '                f"K={last_alt[\'gain\']:.3f} | COLL={last_alt[\'collective\']:.4f} | "\n                f"V={state[\'forward_speed_fps\']:6.2f}"\n'
new_print = '                f"K={last_alt[\'gain\']:.3f} | COLL={last_alt[\'collective\']:.4f} | "\n                f"V={state[\'forward_speed_fps\']:6.2f} | A1={action[1]:+5.2f} | "\n                f"ELEV={float(fdm[\'fcs/elevator-cmd-norm\']):+.4f} | "\n                f"AFCS_R={float(fdm[\'ap/afcs/roll-channel-active-norm\']):.2f} | "\n                f"AFCS_Y={float(fdm[\'ap/afcs/yaw-channel-active-norm\']):.2f}"\n'
if old_print not in source:
    raise RuntimeError("V11 print anchor not found")
source = source.replace(old_print, new_print, 1)

exec(compile(source, "test_forward_parametric_turn_v11.py", "exec"), globals(), globals())
