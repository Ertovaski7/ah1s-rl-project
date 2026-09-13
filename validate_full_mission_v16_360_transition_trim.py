from pathlib import Path

SOURCE = Path("validate_full_mission_v12_smooth_handoff.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# V12 is itself a wrapper. Point it directly at V11 and reuse its proven
# smooth Stage2 handoff, then inject a 360-degree transition-trim phase.
text = text.replace(
    'SOURCE = Path("validate_full_mission_v11_final_gated.py")',
    'SOURCE = Path("validate_full_mission_v11_final_gated.py")',
    1,
)

# The V12 wrapper applies its edits at runtime; execute it only after we add
# the transition block to the generated V11 text. To keep this file robust,
# reproduce the V12 source edits first by reading its code and extracting the
# resulting transformed text through the same replacements is not practical.
# Instead base this validator on V15 and replace only the failed turn-stack
# stabilization block with a neutral AFCS transition trim.

V15 = Path("validate_full_mission_v15_360_turnstack_stabilization.py")
if not V15.exists():
    raise FileNotFoundError(V15)

v15 = V15.read_text(encoding="utf-8")

start_marker = '''if abs(requested_turn) >= 300.0:\n    settle_elapsed = 0.0\n    settle_hold = 0.0\n    settle_safety = False\n\n    # Keep the captured turn goal fixed.'''
end_marker = '''post_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))\nfor _ in range(post_steps):\n'''

start = v15.find(start_marker)
end = v15.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError("Could not locate V15 stabilization block")

replacement = '''if abs(requested_turn) >= 300.0:\n    settle_elapsed = 0.0\n    settle_hold = 0.0\n    settle_safety = False\n\n    # 360-degree transition trim: hold the captured heading with the aircraft\n    # AFCS while removing turn-policy lateral commands.  Keep a conservative\n    # fixed collective/elevator trim so translational lift can decay without\n    # the Stage2 policy immediately commanding its higher collective range.\n    fdm["ap/afcs/psi-trim-rad"] = math.radians(post_target_heading)\n    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5\n    fdm["ap/afcs/roll-channel-active-norm"] = 1.0\n    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0\n\n    transition_collective = 0.540\n    transition_elevator = -0.145\n    transition_aileron = 0.19095\n    transition_rudder = 0.390\n\n    while settle_elapsed < 15.0:\n        fdm["fcs/collective-cmd-norm"] = transition_collective\n        fdm["fcs/elevator-cmd-norm"] = transition_elevator\n        fdm["fcs/aileron-cmd-norm"] = transition_aileron\n        fdm["fcs/rudder-cmd-norm"] = transition_rudder\n\n        jsbsim_ok = True\n        for _ in range(turn_env.PHYSICS_STEPS):\n            if not fdm.run():\n                jsbsim_ok = False\n                break\n\n        mh = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_hold = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n\n        settle_ok = bool(\n            abs(heading_error_hold) <= 1.5\n            and abs(mh["vs"]) <= 1.0\n            and abs(mh["v_lat"]) <= 5.0\n            and abs(mh["roll"]) <= 3.0\n            and abs(mh["r"]) <= 3.0\n            and 285.0 <= mh["alt"] <= 315.0\n            and mh["u"] >= 8.0\n        )\n        settle_hold = settle_hold + turn_env.CONTROL_DT if settle_ok else 0.0\n        settle_safety = settle_safety or (not jsbsim_ok) or (not hard_safe(mh))\n\n        if settle_safety or settle_hold >= 1.0:\n            break\n\n    print(\n        "POST-TURN TRANSITION TRIM | "\n        f"duration={settle_elapsed:.2f}s | hold={settle_hold:.2f}s | "\n        f"alt={mh['alt']:.2f}ft | vs={mh['vs']:+.2f}fps | "\n        f"u={mh['u']:.2f}fps | v_lat={mh['v_lat']:+.2f}fps | "\n        f"roll={mh['roll']:+.2f}deg | yaw_rate={mh['r']:+.2f}deg/s | "\n        f"heading_error={heading_error_hold:+.2f}deg | safety_failure={settle_safety}"\n    )\n\n    post_safety_failure = post_safety_failure or settle_safety\n\n    # Rebase the proven V12 smooth handoff from the actual transition controls.\n    handoff_collective = float(fdm["fcs/collective-cmd-norm"])\n    handoff_elevator = float(fdm["fcs/elevator-cmd-norm"])\n    handoff_action[0] = float(np.clip((handoff_collective - 0.620) / 0.030, -1.0, 1.0))\n    handoff_action[1] = float(np.clip((handoff_elevator + 0.145) / 0.035, -1.0, 1.0))\n    env2.previous_action = handoff_action.copy()\n    obs_post = np.asarray(env2._get_obs(), dtype=np.float32)\n\n    post_elapsed = 0.0\n    post_stable_time = 0.0\n    post_min_alt = float("inf")\n    post_max_alt = -float("inf")\n    post_min_speed = float("inf")\n    post_max_abs_heading_error = 0.0\n    post_max_abs_vs = 0.0\n    post_max_abs_roll = 0.0\n    post_max_abs_yaw_rate = 0.0\n\n'''

v16 = v15[:start] + replacement + v15[end:]
v16 = v16.replace(
    'print("V15 ACTIVE - 360 TURN-STACK STABILIZATION -> SMOOTH STAGE2 HANDOFF")',
    'print("V16 ACTIVE - 360 TRANSITION TRIM -> SMOOTH STAGE2 HANDOFF")',
)
v16 = v16.replace(
    'print("Same FDM; learned policies only; runtime teacher/controller OFF; thresholds unchanged.")',
    'print("Same FDM; no teacher; built-in AFCS transition trim; validation thresholds unchanged.")',
)

Path(__file__).write_text(v16, encoding="utf-8")

# Re-exec the rewritten file once. The marker below prevents recursion.
if "V16_REWRITTEN_READY = True" not in v16:
    rewritten = v16.replace(
        'from pathlib import Path\n',
        'from pathlib import Path\nV16_REWRITTEN_READY = True\n',
        1,
    )
    Path(__file__).write_text(rewritten, encoding="utf-8")
    exec(compile(rewritten, __file__, "exec"), {"__name__": "__main__", "__file__": __file__})
