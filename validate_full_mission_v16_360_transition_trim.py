from pathlib import Path

V15 = Path("validate_full_mission_v15_360_turnstack_stabilization.py")
if not V15.exists():
    raise FileNotFoundError(V15)

text = V15.read_text(encoding="utf-8")

# Keep V15's proven wrapper structure, but replace only the failed
# turn-stack stabilization action with a neutral transition trim.
old_apply = '''        obs_hold = np.asarray(turn_env._get_obs(), dtype=np.float32)\\n        action_hold, _, _ = turn_action(turn_model, base_adapter, patch, obs_hold)\\n        turn_env._apply_action(action_hold)\\n'''
new_apply = '''        fdm["ap/afcs/psi-trim-rad"] = math.radians(post_target_heading)\\n        fdm["ap/afcs/pitch-channel-active-norm"] = 0.5\\n        fdm["ap/afcs/roll-channel-active-norm"] = 1.0\\n        fdm["ap/afcs/yaw-channel-active-norm"] = 1.0\\n        fdm["fcs/collective-cmd-norm"] = 0.540\\n        fdm["fcs/elevator-cmd-norm"] = -0.145\\n        fdm["fcs/aileron-cmd-norm"] = 0.19095\\n        fdm["fcs/rudder-cmd-norm"] = 0.390\\n'''
if old_apply not in text:
    raise RuntimeError("Could not locate V15 turn-stack action block")
text = text.replace(old_apply, new_apply, 1)

# During transition trim the turn is already captured, so do not keep
# integrating the old turn-goal coordinate or require it for settle success.
old_update = '''        current_hdg = heading_deg(fdm)\\n        turn_env.cumulative_turn_deg += wrap_deg(current_hdg - turn_env.prev_heading_deg)\\n        turn_env.prev_heading_deg = current_hdg\\n        turn_env.forward_distance += mh["u"] * turn_env.CONTROL_DT\\n\\n        remaining_hold = requested_turn - turn_env.cumulative_turn_deg\\n        heading_error_hold = wrap_deg(post_target_heading - current_hdg)\\n'''
new_update = '''        current_hdg = heading_deg(fdm)\\n        turn_env.forward_distance += mh["u"] * turn_env.CONTROL_DT\\n        remaining_hold = requested_turn - turn_env.cumulative_turn_deg\\n        heading_error_hold = wrap_deg(post_target_heading - current_hdg)\\n'''
if old_update not in text:
    raise RuntimeError("Could not locate V15 turn-coordinate update block")
text = text.replace(old_update, new_update, 1)

old_criterion = '''            abs(remaining_hold) <= 1.5\\n            and abs(heading_error_hold) <= 1.5\\n'''
new_criterion = '''            abs(heading_error_hold) <= 1.5\\n'''
if old_criterion not in text:
    raise RuntimeError("Could not locate V15 stabilization criterion")
text = text.replace(old_criterion, new_criterion, 1)

text = text.replace(
    'POST-TURN TURN-STACK STABILIZATION',
    'POST-TURN TRANSITION TRIM',
)
text = text.replace(
    'V15 ACTIVE - 360 TURN-STACK STABILIZATION -> SMOOTH STAGE2 HANDOFF',
    'V16 ACTIVE - 360 TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF',
)
text = text.replace(
    'Same FDM; learned policies only; runtime teacher/controller OFF; thresholds unchanged.',
    'Same FDM; built-in AFCS transition trim; runtime teacher OFF; thresholds unchanged.',
)

# Execute the modified V15 wrapper in memory. It will patch V11 once and run.
exec(compile(text, str(V15), "exec"), {"__name__": "__main__", "__file__": str(V15)})
