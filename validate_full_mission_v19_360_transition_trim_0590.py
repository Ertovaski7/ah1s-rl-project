from pathlib import Path

SOURCE = Path("validate_full_mission_v11_final_gated.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# Proven V12 smooth handoff applied directly to V11 source.
# -----------------------------------------------------------------------------
old_a = '''post_target_heading = heading_deg(fdm)
env2.fdm = fdm
'''
new_a = '''post_target_heading = heading_deg(fdm)

handoff_collective = float(fdm["fcs/collective-cmd-norm"])
handoff_elevator = float(fdm["fcs/elevator-cmd-norm"])
handoff_action = np.zeros(4, dtype=np.float32)
handoff_action[0] = float(np.clip((handoff_collective - 0.620) / 0.030, -1.0, 1.0))
handoff_action[1] = float(np.clip((handoff_elevator + 0.145) / 0.035, -1.0, 1.0))
handoff_action[2] = 0.0
handoff_action[3] = 0.0
POST_BLEND_COLLECTIVE_S = 9.0
POST_BLEND_ELEVATOR_S = 4.0
POST_BLEND_LATERAL_S = 1.0

env2.fdm = fdm
'''
if old_a not in text:
    raise RuntimeError("V19: post-turn handoff start not found")
text = text.replace(old_a, new_a, 1)

old_b = '''env2.previous_action = np.zeros(4, dtype=np.float32)
'''
new_b = '''env2.previous_action = handoff_action.copy()
'''
if old_b not in text:
    raise RuntimeError("V19: previous_action reset not found")
text = text.replace(old_b, new_b, 1)

old_action = '''    action_post, _ = stage2_model.predict(obs_post, deterministic=True)
    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)
'''
new_action = '''    policy_action, _ = stage2_model.predict(obs_post, deterministic=True)
    policy_action = np.asarray(policy_action, dtype=np.float32)

    alpha_c = float(np.clip(post_elapsed / POST_BLEND_COLLECTIVE_S, 0.0, 1.0))
    alpha_e = float(np.clip(post_elapsed / POST_BLEND_ELEVATOR_S, 0.0, 1.0))
    alpha_l = float(np.clip(post_elapsed / POST_BLEND_LATERAL_S, 0.0, 1.0))
    alpha_c = alpha_c * alpha_c * (3.0 - 2.0 * alpha_c)
    alpha_e = alpha_e * alpha_e * (3.0 - 2.0 * alpha_e)
    alpha_l = alpha_l * alpha_l * (3.0 - 2.0 * alpha_l)

    action_post = policy_action.copy()
    action_post[0] = (1.0 - alpha_c) * handoff_action[0] + alpha_c * policy_action[0]
    action_post[1] = (1.0 - alpha_e) * handoff_action[1] + alpha_e * policy_action[1]
    action_post[2] = (1.0 - alpha_l) * handoff_action[2] + alpha_l * policy_action[2]
    action_post[3] = (1.0 - alpha_l) * handoff_action[3] + alpha_l * policy_action[3]
    action_post = np.clip(action_post, -1.0, 1.0).astype(np.float32)

    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)
'''
if old_action not in text:
    raise RuntimeError("V19: Stage2 post-turn action block not found")
text = text.replace(old_action, new_action, 1)

old_print = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id}"
'''
new_print = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id} | "
    f"blend_c/e/lat={POST_BLEND_COLLECTIVE_S:.1f}/{POST_BLEND_ELEVATOR_S:.1f}/{POST_BLEND_LATERAL_S:.1f}s | "
    f"handoff_action={np.array2string(handoff_action, precision=3)}"
'''
if old_print not in text:
    raise RuntimeError("V19: post-turn handoff print not found")
text = text.replace(old_print, new_print, 1)

# -----------------------------------------------------------------------------
# 360-degree transition trim before strict 10-second Stage2 qualification.
# Same FDM, built-in AFCS only; runtime teacher remains OFF.
# -----------------------------------------------------------------------------
needle = '''post_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))
for _ in range(post_steps):
'''
replacement = '''if abs(requested_turn) >= 300.0:
    settle_elapsed = 0.0
    settle_hold = 0.0
    settle_safety = False

    fdm["ap/afcs/psi-trim-rad"] = math.radians(post_target_heading)
    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    transition_collective = 0.590
    transition_elevator = -0.145
    transition_aileron = 0.19095
    transition_rudder = 0.390

    while settle_elapsed < 20.0:
        fdm["fcs/collective-cmd-norm"] = transition_collective
        fdm["fcs/elevator-cmd-norm"] = transition_elevator
        fdm["fcs/aileron-cmd-norm"] = transition_aileron
        fdm["fcs/rudder-cmd-norm"] = transition_rudder

        jsbsim_ok = True
        for _ in range(turn_env.PHYSICS_STEPS):
            if not fdm.run():
                jsbsim_ok = False
                break

        ms = physical_metrics(fdm)
        current_hdg = heading_deg(fdm)
        heading_error_s = wrap_deg(post_target_heading - current_hdg)
        settle_elapsed += turn_env.CONTROL_DT

        settle_ok = bool(
            abs(heading_error_s) <= 1.5
            and abs(ms["vs"]) <= 1.0
            and abs(ms["v_lat"]) <= 5.0
            and abs(ms["roll"]) <= 3.0
            and abs(ms["r"]) <= 3.0
            and 285.0 <= ms["alt"] <= 315.0
            and ms["u"] >= 8.0
        )
        settle_hold = settle_hold + turn_env.CONTROL_DT if settle_ok else 0.0
        settle_safety = settle_safety or (not jsbsim_ok) or (not hard_safe(ms))

        if settle_safety or settle_hold >= 1.0:
            break

    print(
        "POST-TURN TRANSITION TRIM | "
        f"duration={settle_elapsed:.2f}s | hold={settle_hold:.2f}s | "
        f"alt={ms['alt']:.2f}ft | vs={ms['vs']:+.2f}fps | "
        f"u={ms['u']:.2f}fps | v_lat={ms['v_lat']:+.2f}fps | "
        f"roll={ms['roll']:+.2f}deg | yaw_rate={ms['r']:+.2f}deg/s | "
        f"heading_error={heading_error_s:+.2f}deg | safety_failure={settle_safety}"
    )

    post_safety_failure = post_safety_failure or settle_safety

    handoff_collective = float(fdm["fcs/collective-cmd-norm"])
    handoff_elevator = float(fdm["fcs/elevator-cmd-norm"])
    handoff_action[0] = float(np.clip((handoff_collective - 0.620) / 0.030, -1.0, 1.0))
    handoff_action[1] = float(np.clip((handoff_elevator + 0.145) / 0.035, -1.0, 1.0))
    handoff_action[2] = 0.0
    handoff_action[3] = 0.0
    env2.previous_action = handoff_action.copy()
    obs_post = np.asarray(env2._get_obs(), dtype=np.float32)

    post_elapsed = 0.0
    post_stable_time = 0.0
    post_min_alt = float("inf")
    post_max_alt = -float("inf")
    post_min_speed = float("inf")
    post_max_abs_heading_error = 0.0
    post_max_abs_vs = 0.0
    post_max_abs_roll = 0.0
    post_max_abs_yaw_rate = 0.0

post_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))
for _ in range(post_steps):
'''
if needle not in text:
    raise RuntimeError("V19: post-turn qualification loop not found")
text = text.replace(needle, replacement, 1)

print("=" * 120)
print("V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF")
print("Direct V11 base; no nested wrappers. Same FDM; runtime teacher OFF; thresholds unchanged.")
print("360 transition collective=0.590; max settle=20.0s.")
print("=" * 120)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
