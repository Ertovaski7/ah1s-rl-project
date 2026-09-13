from pathlib import Path

SOURCE = Path("validate_full_mission_v11_final_gated.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

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
POST_HANDOFF_BLEND_S = 4.0

env2.fdm = fdm
'''
if old_a not in text:
    raise RuntimeError("Could not locate post-turn handoff start")
text = text.replace(old_a, new_a, 1)

old_b = '''env2.previous_action = np.zeros(4, dtype=np.float32)
'''
new_b = '''env2.previous_action = handoff_action.copy()
'''
if old_b not in text:
    raise RuntimeError("Could not locate previous_action reset")
text = text.replace(old_b, new_b, 1)

old_c = '''    action_post, _ = stage2_model.predict(obs_post, deterministic=True)
    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)
'''
new_c = '''    policy_action, _ = stage2_model.predict(obs_post, deterministic=True)
    policy_action = np.asarray(policy_action, dtype=np.float32)
    blend_alpha = float(np.clip(post_elapsed / POST_HANDOFF_BLEND_S, 0.0, 1.0))
    action_post = (1.0 - blend_alpha) * handoff_action + blend_alpha * policy_action
    action_post = np.clip(action_post, -1.0, 1.0).astype(np.float32)
    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)
'''
if old_c not in text:
    raise RuntimeError("Could not locate Stage2 post-turn action block")
text = text.replace(old_c, new_c, 1)

old_d = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id}"
'''
new_d = '''    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id} | "
    f"blend={POST_HANDOFF_BLEND_S:.1f}s | handoff_action={np.array2string(handoff_action, precision=3)}"
'''
if old_d not in text:
    raise RuntimeError("Could not locate post-turn handoff print")
text = text.replace(old_d, new_d, 1)

print("=" * 120)
print("FINAL FULL-MISSION V12 - SMOOTH TURN -> STAGE2 POLICY HANDOFF")
print("No model weights changed. Validation thresholds unchanged.")
print("Only the policy handoff is cross-faded for 4.0 seconds to suppress actuator discontinuity.")
print("=" * 120)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
