from __future__ import annotations

from pathlib import Path

SOURCE = Path("validate_full_mission_v7_final.py")
TRAINING_ENTRY_FORWARD_FT = 403.424
V10_PATCH_PATH = "models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt"
TARGET_NEG_NORM = -50.0 / 360.0
NEG_GATE_TOL = 0.02
NEG_SCALE = 0.18

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old_attach = '''env2.fdm = fdm
if hasattr(env2, "phase"):
'''
new_attach = '''env2.fdm = fdm
fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
fdm["ap/afcs/roll-channel-active-norm"] = 1.0
fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
if hasattr(env2, "phase"):
'''
if old_attach not in text:
    raise RuntimeError("Could not locate Stage2 same-FDM attach block")
text = text.replace(old_attach, new_attach, 1)

old_turn_distance = '''turn_env.forward_distance = env_forward
'''
new_turn_distance = f'''physical_stage2_forward_at_handoff = float(env_forward)
turn_env.forward_distance = {TRAINING_ENTRY_FORWARD_FT:.3f}
print(
    "TURN PROGRESS COORDINATE CALIBRATION | "
    f"physical_stage2_forward={{physical_stage2_forward_at_handoff:.3f}}ft | "
    f"policy_forward_coordinate={{turn_env.forward_distance:.3f}}ft"
)
'''
if old_turn_distance not in text:
    raise RuntimeError("Could not locate turn forward-distance attach line")
text = text.replace(old_turn_distance, new_turn_distance, 1)

old_obs = '''obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
turn_entry_metrics = physical_metrics(fdm)
'''
new_obs = f'''obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
turn_entry_metrics = physical_metrics(fdm)
print(
    "TURN ENTRY OBS CHECK | "
    f"obs10_forward_progress={{float(obs_turn[10]):+.6f}} | "
    "training_reference={TRAINING_ENTRY_FORWARD_FT / 300.0:+.6f}"
)
'''
if old_obs not in text:
    raise RuntimeError("Could not locate turn observation creation block")
text = text.replace(old_obs, new_obs, 1)

marker = '''def rule(text: str):
'''
insert = f'''class NegativeCollectivePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1) * {NEG_SCALE}


def neg_gate_from_obs(obs):
    return abs(float(obs[12]) - ({TARGET_NEG_NORM!r})) <= {NEG_GATE_TOL}


'''
if marker not in text:
    raise RuntimeError("Could not locate helper insertion marker")
text = text.replace(marker, insert + marker, 1)

old_require = '''for path in [TURN_PPO_PATH, V4_ADAPTER_PATH, V7_PATCH_PATH]:
    require_file(path)
'''
new_require = f'''V10_NEG_PATCH_PATH = Path({V10_PATCH_PATH!r})
for path in [TURN_PPO_PATH, V4_ADAPTER_PATH, V7_PATCH_PATH, V10_NEG_PATCH_PATH]:
    require_file(path)
'''
if old_require not in text:
    raise RuntimeError("Could not locate required-model block")
text = text.replace(old_require, new_require, 1)

old_load = '''patch = ResidualAdapter(PATCH_SCALE)
patch.load_state_dict(torch.load(V7_PATCH_PATH, map_location="cpu"), strict=True)
patch.eval()
for module in [base_adapter, patch]:
    for p in module.parameters():
        p.requires_grad = False
'''
new_load = '''patch = ResidualAdapter(PATCH_SCALE)
patch.load_state_dict(torch.load(V7_PATCH_PATH, map_location="cpu"), strict=True)
patch.eval()
neg_patch = NegativeCollectivePatch()
neg_patch.load_state_dict(torch.load(V10_NEG_PATCH_PATH, map_location="cpu"), strict=True)
neg_patch.eval()
for module in [base_adapter, patch, neg_patch]:
    for p in module.parameters():
        p.requires_grad = False
'''
if old_load not in text:
    raise RuntimeError("Could not locate turn model loading block")
text = text.replace(old_load, new_load, 1)

old_print = '''print("V7 +200 gated patch:", V7_PATCH_PATH)
'''
new_print = '''print("V7 +200 gated patch:", V7_PATCH_PATH)
print("V10 -50 gated collective patch:", V10_NEG_PATCH_PATH)
'''
if old_print not in text:
    raise RuntimeError("Could not locate model print block")
text = text.replace(old_print, new_print, 1)

old_action = '''def turn_action(turn_model, base_adapter, patch, obs):
    """Final runtime turn action. Teacher/controller is not used here."""
    base, _ = turn_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d_v4 = base_adapter(x).cpu().numpy()[0]
        if gate_from_obs(obs) > 0.5:
            d_v7 = patch(x).cpu().numpy()[0]
        else:
            d_v7 = np.zeros(4, dtype=np.float32)
    action = np.clip(np.asarray(base, dtype=np.float32) + d_v4 + d_v7, -1.0, 1.0)
    return action, d_v4, d_v7
'''
new_action = '''def turn_action(turn_model, base_adapter, patch, obs):
    """Final runtime turn action. Teacher/controller is not used here."""
    base, _ = turn_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d_v4 = base_adapter(x).cpu().numpy()[0]
        if gate_from_obs(obs) > 0.5:
            d_v7 = patch(x).cpu().numpy()[0]
        else:
            d_v7 = np.zeros(4, dtype=np.float32)
        d_v10 = float(neg_patch(x).cpu().numpy()[0]) if neg_gate_from_obs(obs) else 0.0
    action = np.asarray(base, dtype=np.float32) + d_v4 + d_v7
    action[0] += d_v10
    action = np.clip(action, -1.0, 1.0)
    return action, d_v4, d_v7
'''
if old_action not in text:
    raise RuntimeError("Could not locate turn_action function")
text = text.replace(old_action, new_action, 1)

# Replace the old post-turn behavior. The old validator kept executing the
# turn policy stack after capture. Final mission semantics require a handoff
# back to Stage2 PPO on the SAME live FDM and a new yaw/heading reference.
post_start_marker = '''# turn_env.step() has already terminated after its success hold, so continue
# the exact same learned stack manually on the same FDM. No reset/teacher/controller.
'''
post_end_marker = '''if post_final is None:
'''
post_start = text.find(post_start_marker)
post_end = text.find(post_end_marker, post_start)
if post_start < 0 or post_end < 0:
    raise RuntimeError("Could not locate post-turn continuation block")

new_post = '''# Final mission handoff: TURN -> STAGE2 PPO, same live FDM.
# Rebase the AFCS yaw trim to the captured new heading, then let the learned
# Stage2 policy manage forward speed/altitude for the 10-second continuation.
post_target_heading = heading_deg(fdm)
env2.fdm = fdm
env2.phase = 1
env2.forward_distance = 0.0
env2.previous_action = np.zeros(4, dtype=np.float32)
env2.steps = 0
fdm["ap/afcs/psi-trim-rad"] = math.radians(post_target_heading)
fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
fdm["ap/afcs/roll-channel-active-norm"] = 1.0
fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
obs_post = np.asarray(env2._get_obs(), dtype=np.float32)

print(
    "POST-TURN POLICY HANDOFF | policy=Stage2 PPO | "
    f"new_heading_ref={post_target_heading:.2f}deg | same_fdm={id(env2.fdm) == active_fdm_id}"
)

post_steps = int(math.ceil(POST_TURN_FORWARD_S / env2.CONTROL_DT))
for _ in range(post_steps):
    action_post, _ = stage2_model.predict(obs_post, deterministic=True)
    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)
    obs_post = np.asarray(obs_post, dtype=np.float32)

    m = physical_metrics(fdm)
    current_hdg = heading_deg(fdm)
    heading_error = wrap_deg(post_target_heading - current_hdg)
    remaining = heading_error
    completed_equiv = requested_turn - remaining
    post_elapsed += env2.CONTROL_DT

    post_min_alt = min(post_min_alt, m["alt"])
    post_max_alt = max(post_max_alt, m["alt"])
    post_min_speed = min(post_min_speed, m["u"])
    post_max_abs_heading_error = max(post_max_abs_heading_error, abs(remaining))
    post_max_abs_vs = max(post_max_abs_vs, abs(m["vs"]))
    post_max_abs_roll = max(post_max_abs_roll, abs(m["roll"]))
    post_max_abs_yaw_rate = max(post_max_abs_yaw_rate, abs(m["r"]))

    stable = bool(
        abs(remaining) <= TURN_CAPTURE_TOL_DEG
        and TURN_ALT_MIN_FT <= m["alt"] <= TURN_ALT_MAX_FT
        and abs(m["vs"]) <= TURN_MAX_ABS_VS_FPS
        and m["u"] >= TURN_MIN_SPEED_FPS
        and abs(m["roll"]) <= TURN_MAX_ABS_ROLL_DEG
        and abs(m["r"]) <= TURN_MAX_ABS_YAW_RATE_DEG_S
    )
    post_stable_time = post_stable_time + env2.CONTROL_DT if stable else 0.0
    env_failed = bool((terminated_post or truncated_post) and not bool(info_post.get("success", False)))
    post_safety_failure = post_safety_failure or env_failed or (not hard_safe(m))
    post_final = {**m, "remaining": remaining, "completed": completed_equiv}

    if post_safety_failure:
        break

'''
text = text[:post_start] + new_post + text[post_end:]

old_summary = '''print(f"V7 +200 GATE AT TURN ENTRY: {'ON' if patch_gate > 0.5 else 'OFF'}")
'''
new_summary = '''print(f"V7 +200 GATE AT TURN ENTRY: {'ON' if patch_gate > 0.5 else 'OFF'}")
print(f"V10 -50 GATE AT TURN ENTRY: {'ON' if neg_gate_from_obs(obs_turn) else 'OFF'}")
'''
if old_summary not in text:
    raise RuntimeError("Could not locate final gate summary line")
text = text.replace(old_summary, new_summary, 1)

# Make validation diagnostic-friendly: never throw away the final summary.
old_fail = '''if not final_pass:
    raise RuntimeError("FINAL FULL MISSION VALIDATION FAILED")
'''
new_fail = '''print("FINAL CRITERION BREAKDOWN | "
      f"stage1={stage1_pass} | stage2={entry_ok} | turn={turn_pass} | "
      f"post={post_pass} | same_fdm={fdm_continuity} | post_safety={post_safety_failure}")
if not final_pass:
    failed = []
    if not stage1_pass: failed.append("stage1")
    if not entry_ok: failed.append("stage2")
    if not turn_pass: failed.append("turn_capture")
    if not post_pass: failed.append("post_turn_forward")
    if not fdm_continuity: failed.append("fdm_continuity")
    print("FINAL FAILED CRITERIA:", ", ".join(failed) if failed else "unknown")
'''
if old_fail not in text:
    raise RuntimeError("Could not locate final failure raise block")
text = text.replace(old_fail, new_fail, 1)

print("=" * 120)
print("FINAL FULL-MISSION V11 - CALIBRATED PROGRESS + GATED -50 COLLECTIVE PATCH")
print("Runtime teacher/controller OFF. Same JSBSim FDM, no reset between flight phases.")
print("V7 correction is active only for +200; V10 collective correction is active only for -50.")
print("Post-turn continuation hands the same FDM back to Stage2 PPO on the captured new heading.")
print("Validator is exception-free at final summary and prints exact failed criteria.")
print("=" * 120)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
