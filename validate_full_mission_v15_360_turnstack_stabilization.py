from pathlib import Path

SOURCE = Path("validate_full_mission_v11_final_gated.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# Proven V12 smooth Stage2 handoff.
# -----------------------------------------------------------------------------
old_a = '''post_target_heading = heading_deg(fdm)\nenv2.fdm = fdm\n'''
new_a = '''post_target_heading = heading_deg(fdm)\n\nhandoff_collective = float(fdm["fcs/collective-cmd-norm"])\nhandoff_elevator = float(fdm["fcs/elevator-cmd-norm"])\nhandoff_action = np.zeros(4, dtype=np.float32)\nhandoff_action[0] = float(np.clip((handoff_collective - 0.620) / 0.030, -1.0, 1.0))\nhandoff_action[1] = float(np.clip((handoff_elevator + 0.145) / 0.035, -1.0, 1.0))\nhandoff_action[2] = 0.0\nhandoff_action[3] = 0.0\nPOST_BLEND_COLLECTIVE_S = 9.0\nPOST_BLEND_ELEVATOR_S = 4.0\nPOST_BLEND_LATERAL_S = 1.0\n\nenv2.fdm = fdm\n'''
if old_a not in text:
    raise RuntimeError("Could not locate post-turn handoff start")
text = text.replace(old_a, new_a, 1)

old_b = '''env2.previous_action = np.zeros(4, dtype=np.float32)\n'''
new_b = '''env2.previous_action = handoff_action.copy()\n'''
if old_b not in text:
    raise RuntimeError("Could not locate previous_action reset")
text = text.replace(old_b, new_b, 1)

old_action = '''    action_post, _ = stage2_model.predict(obs_post, deterministic=True)\n    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n'''
new_action = '''    policy_action, _ = stage2_model.predict(obs_post, deterministic=True)\n    policy_action = np.asarray(policy_action, dtype=np.float32)\n\n    alpha_c = float(np.clip(post_elapsed / POST_BLEND_COLLECTIVE_S, 0.0, 1.0))\n    alpha_e = float(np.clip(post_elapsed / POST_BLEND_ELEVATOR_S, 0.0, 1.0))\n    alpha_l = float(np.clip(post_elapsed / POST_BLEND_LATERAL_S, 0.0, 1.0))\n    alpha_c = alpha_c * alpha_c * (3.0 - 2.0 * alpha_c)\n    alpha_e = alpha_e * alpha_e * (3.0 - 2.0 * alpha_e)\n    alpha_l = alpha_l * alpha_l * (3.0 - 2.0 * alpha_l)\n\n    action_post = policy_action.copy()\n    action_post[0] = (1.0 - alpha_c) * handoff_action[0] + alpha_c * policy_action[0]\n    action_post[1] = (1.0 - alpha_e) * handoff_action[1] + alpha_e * policy_action[1]\n    action_post[2] = (1.0 - alpha_l) * handoff_action[2] + alpha_l * policy_action[2]\n    action_post[3] = (1.0 - alpha_l) * handoff_action[3] + alpha_l * policy_action[3]\n    action_post = np.clip(action_post, -1.0, 1.0).astype(np.float32)\n    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n'''
if old_action not in text:
    raise RuntimeError("Could not locate Stage2 post-turn action block")
text = text.replace(old_action, new_action, 1)

old_print = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id}"\n'''
new_print = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id} | "\n    f"blend_c/e/lat={POST_BLEND_COLLECTIVE_S:.1f}/{POST_BLEND_ELEVATOR_S:.1f}/{POST_BLEND_LATERAL_S:.1f}s | "\n    f"handoff_action={np.array2string(handoff_action, precision=3)}"\n'''
if old_print not in text:
    raise RuntimeError("Could not locate post-turn handoff print")
text = text.replace(old_print, new_print, 1)

# -----------------------------------------------------------------------------
# 360-degree only: stabilize with the learned TURN STACK first.
# This keeps the broader turn collective authority [0.460, 0.620].
# Teacher/controller remain OFF.
# -----------------------------------------------------------------------------
needle = '''post_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))\nfor _ in range(post_steps):\n'''

replacement = '''if abs(requested_turn) >= 300.0:\n    settle_elapsed = 0.0\n    settle_hold = 0.0\n    settle_safety = False\n\n    # Keep the captured turn goal fixed.  Continue the learned turn stack\n    # manually on the SAME FDM until lateral/vertical transients decay.\n    turn_env.fdm = fdm\n    turn_env.turn_active = True\n    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5\n    fdm["ap/afcs/roll-channel-active-norm"] = turn_env.TURN_ROLL_AFCS\n    fdm["ap/afcs/yaw-channel-active-norm"] = turn_env.TURN_YAW_AFCS\n\n    while settle_elapsed < 15.0:\n        obs_hold = np.asarray(turn_env._get_obs(), dtype=np.float32)\n        action_hold, _, _ = turn_action(turn_model, base_adapter, patch, obs_hold)\n        turn_env._apply_action(action_hold)\n\n        jsbsim_ok = True\n        for _ in range(turn_env.PHYSICS_STEPS):\n            if not fdm.run():\n                jsbsim_ok = False\n                break\n\n        mh = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        turn_env.cumulative_turn_deg += wrap_deg(current_hdg - turn_env.prev_heading_deg)\n        turn_env.prev_heading_deg = current_hdg\n        turn_env.forward_distance += mh["u"] * turn_env.CONTROL_DT\n\n        remaining_hold = requested_turn - turn_env.cumulative_turn_deg\n        heading_error_hold = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n\n        settle_ok = bool(\n            abs(remaining_hold) <= 1.5\n            and abs(heading_error_hold) <= 1.5\n            and abs(mh["vs"]) <= 1.0\n            and abs(mh["v_lat"]) <= 5.0\n            and abs(mh["roll"]) <= 3.0\n            and abs(mh["r"]) <= 3.0\n            and 285.0 <= mh["alt"] <= 315.0\n            and mh["u"] >= 8.0\n        )\n        settle_hold = settle_hold + turn_env.CONTROL_DT if settle_ok else 0.0\n\n        settle_safety = settle_safety or (not jsbsim_ok) or (not hard_safe(mh))\n        if settle_safety or settle_hold >= 1.0:\n            break\n\n    print(\n        "POST-TURN TURN-STACK STABILIZATION | "\n        f"duration={settle_elapsed:.2f}s | hold={settle_hold:.2f}s | "\n        f"alt={mh['alt']:.2f}ft | vs={mh['vs']:+.2f}fps | "\n        f"u={mh['u']:.2f}fps | v_lat={mh['v_lat']:+.2f}fps | "\n        f"roll={mh['roll']:+.2f}deg | yaw_rate={mh['r']:+.2f}deg/s | "\n        f"remaining={remaining_hold:+.2f}deg | heading_error={heading_error_hold:+.2f}deg | "\n        f"safety_failure={settle_safety}"\n    )\n\n    post_safety_failure = post_safety_failure or settle_safety\n\n    # Rebase Stage2 handoff from the ACTUAL controls at the end of stabilization.\n    handoff_collective = float(fdm["fcs/collective-cmd-norm"])\n    handoff_elevator = float(fdm["fcs/elevator-cmd-norm"])\n    handoff_action[0] = float(np.clip((handoff_collective - 0.620) / 0.030, -1.0, 1.0))\n    handoff_action[1] = float(np.clip((handoff_elevator + 0.145) / 0.035, -1.0, 1.0))\n    env2.previous_action = handoff_action.copy()\n\n    fdm["ap/afcs/psi-trim-rad"] = math.radians(post_target_heading)\n    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5\n    fdm["ap/afcs/roll-channel-active-norm"] = 1.0\n    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0\n    obs_post = np.asarray(env2._get_obs(), dtype=np.float32)\n\n    # Strict qualification starts only after stabilization.\n    post_elapsed = 0.0\n    post_stable_time = 0.0\n    post_min_alt = float("inf")\n    post_max_alt = -float("inf")\n    post_min_speed = float("inf")\n    post_max_abs_heading_error = 0.0\n    post_max_abs_vs = 0.0\n    post_max_abs_roll = 0.0\n    post_max_abs_yaw_rate = 0.0\n\npost_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))\nfor _ in range(post_steps):\n'''

if needle not in text:
    raise RuntimeError("Could not locate post-turn qualification loop")
text = text.replace(needle, replacement, 1)

print("=" * 120)
print("V15 ACTIVE - 360 TURN-STACK STABILIZATION -> SMOOTH STAGE2 HANDOFF")
print("Same FDM; learned policies only; runtime teacher/controller OFF; thresholds unchanged.")
print("=" * 120)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
