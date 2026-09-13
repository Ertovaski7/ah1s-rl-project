from pathlib import Path

SOURCE = Path("validate_full_mission_v7_final.py")

text = SOURCE.read_text(encoding="utf-8")

stage2_start = '''# ============================================================================\n# 2 — STAGE 2: SAME FDM FORWARD FLIGHT\n# ============================================================================'''
stage3_start = '''# ============================================================================\n# 3 — FINAL TURN SYSTEM: SAME FDM'''

if stage2_start not in text or stage3_start not in text:
    raise RuntimeError("Could not locate Stage2/Stage3 blocks in validate_full_mission_v7_final.py")

prefix, rest = text.split(stage2_start, 1)
_, suffix = rest.split(stage3_start, 1)

stage2_block = r'''# ============================================================================
# 2 — STAGE 2: SAME FDM FORWARD FLIGHT
# IMPORTANT: reproduce the V7 full-entry training distribution on the SAME FDM.
# No teacher/controller is used. Stage2 PPO remains in control.
# ============================================================================
rule("2 — STAGE 2 PPO: SAME-FDM FORWARD FLIGHT -> V7-MATCHED ENTRY")

# Reference measured from diagnose_final_v7_entry_vs_training.py:
# forward_distance ~= 403.424 ft, obs[10] ~= 1.344747,
# alt ~= 302.94 ft, vs ~= -0.82 fps, speed ~= 11.89 fps.
FULL_ENTRY_FWD_MIN_FT = 390.0
FULL_ENTRY_FWD_MAX_FT = 430.0
FULL_ENTRY_ALT_MIN_FT = 299.0
FULL_ENTRY_ALT_MAX_FT = 303.0
FULL_ENTRY_MAX_ABS_VS_FPS = 1.25
FULL_ENTRY_MIN_SPEED_FPS = 8.0
FULL_ENTRY_MAX_SPEED_FPS = 18.0
FULL_ENTRY_MAX_ABS_ROLL_DEG = 8.0
FULL_ENTRY_MAX_ABS_PITCH_DEG = 8.0
FULL_ENTRY_EXTRA_MAX_S = 45.0
REFERENCE_FORWARD_FT = 403.424
REFERENCE_FORWARD_PROGRESS = 1.344747

sim_before_stage2 = fdm_get(fdm, "simulation/sim-time-sec")
env2 = HelicopterEnvStage2RefineMapped(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
env2.reset()  # disposable internal FDM only
env2.fdm = fdm

env2.phase = 1
env2.forward_distance = 0.0
if hasattr(env2, "target_heading"):
    env2.target_heading = float(mission_heading)
if hasattr(env2, "previous_action"):
    env2.previous_action = np.zeros(4, dtype=np.float32)
for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
    if hasattr(env2, attr):
        setattr(env2, attr, 0)

# CRITICAL: env2.reset() set these on its disposable FDM. Re-apply the exact
# Stage2 forward-flight AFCS configuration to the actual live mission FDM.
fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
fdm["ap/afcs/roll-channel-active-norm"] = 1.0
fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

if id(get_fdm(env2)) != active_fdm_id:
    raise RuntimeError("Stage2 same-FDM attach failed")
sim_after_stage2_attach = fdm_get(fdm, "simulation/sim-time-sec")
if np.isfinite(sim_before_stage2) and np.isfinite(sim_after_stage2_attach) and abs(sim_after_stage2_attach - sim_before_stage2) > 1e-9:
    raise RuntimeError("Active simulation clock changed during Stage2 attach")

obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
dt2 = env_control_dt(env2)
stage2_elapsed = 0.0

# --------------------------------------------------------------------------
# Phase A: normal Stage2 stepping until the original 160-ft forward entry.
# --------------------------------------------------------------------------
reached_160 = False
for step in range(int(STAGE2_MAX_TIME_S / dt2)):
    action2, _ = stage2_model.predict(obs2, deterministic=True)
    obs2, _, terminated, truncated, info2 = env2.step(action2)
    obs2 = np.asarray(obs2, dtype=np.float32)
    stage2_elapsed = (step + 1) * dt2

    if float(env2.forward_distance) >= ENTRY_FORWARD_FT:
        reached_160 = True
        break

    if truncated or (terminated and not bool(info2.get("success", False))):
        raise RuntimeError("Stage2 failed before initial 160-ft entry")

if not reached_160:
    raise RuntimeError("Stage2 did not reach initial 160-ft entry")

m160 = physical_metrics(fdm)
print_handoff(
    "STAGE2 160-FT CHECKPOINT",
    m160,
    env_forward=f"{float(env2.forward_distance):.1f}ft",
    obs_forward_progress=f"{float(obs2[10]):+.4f}" if len(obs2) > 10 else "n/a",
    afcs_pitch=f"{fdm_get(fdm, 'ap/afcs/pitch-channel-active-norm'):.2f}",
    afcs_roll=f"{fdm_get(fdm, 'ap/afcs/roll-channel-active-norm'):.2f}",
    afcs_yaw=f"{fdm_get(fdm, 'ap/afcs/yaw-channel-active-norm'):.2f}",
)

# --------------------------------------------------------------------------
# Phase B: exactly like HelicopterEnvTurnGoalFullEntry.reset(): keep the
# learned Stage2 policy flying straight on the SAME live FDM, bypassing
# Stage2 episode termination, until both the physical envelope AND the
# forward-progress distribution match the V7 training entry.
# --------------------------------------------------------------------------
entry_ready = False
extra_elapsed = 0.0
max_extra_steps = int(FULL_ENTRY_EXTRA_MAX_S / dt2)

for _ in range(max_extra_steps):
    # Keep exact Stage2 AFCS semantics active on the live mission FDM.
    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    action2, _ = stage2_model.predict(obs2, deterministic=True)
    env2._apply_action(action2)

    jsbsim_ok = True
    for _ in range(env2.PHYSICS_STEPS):
        if not fdm.run():
            jsbsim_ok = False
            break
    if not jsbsim_ok:
        raise RuntimeError("JSBSim stopped while matching V7 training entry")

    s2 = env2._raw_state()
    env2.forward_distance += float(s2["forward_velocity"]) * dt2
    stage2_elapsed += dt2
    extra_elapsed += dt2

    env_forward = float(env2.forward_distance)
    m2 = physical_metrics(fdm)

    physical_ready = bool(
        FULL_ENTRY_ALT_MIN_FT <= m2["alt"] <= FULL_ENTRY_ALT_MAX_FT
        and abs(m2["vs"]) <= FULL_ENTRY_MAX_ABS_VS_FPS
        and FULL_ENTRY_MIN_SPEED_FPS <= m2["u"] <= FULL_ENTRY_MAX_SPEED_FPS
        and abs(m2["roll"]) <= FULL_ENTRY_MAX_ABS_ROLL_DEG
        and abs(m2["pitch"]) <= FULL_ENTRY_MAX_ABS_PITCH_DEG
    )
    progress_ready = bool(FULL_ENTRY_FWD_MIN_FT <= env_forward <= FULL_ENTRY_FWD_MAX_FT)
    entry_ready = bool(physical_ready and progress_ready)

    if entry_ready:
        break

    if env_forward > FULL_ENTRY_FWD_MAX_FT:
        break

if not entry_ready:
    m2 = physical_metrics(fdm)
    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    raise RuntimeError(
        "Could not reproduce V7 training-entry envelope on live FDM: "
        f"fwd={float(env2.forward_distance):.1f}ft, obs10={float(obs2[10]):+.4f}, "
        f"alt={m2['alt']:.2f}, vs={m2['vs']:+.2f}, u={m2['u']:.2f}, "
        f"roll={m2['roll']:+.2f}, pitch={m2['pitch']:+.2f}"
    )

entry_state = snapshot(fdm, lat0, lon0, mission_heading)
obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
m2 = physical_metrics(fdm)
env_forward = float(env2.forward_distance)

entry_ok = bool(
    FULL_ENTRY_FWD_MIN_FT <= env_forward <= FULL_ENTRY_FWD_MAX_FT
    and FULL_ENTRY_ALT_MIN_FT <= m2["alt"] <= FULL_ENTRY_ALT_MAX_FT
    and abs(m2["vs"]) <= FULL_ENTRY_MAX_ABS_VS_FPS
    and FULL_ENTRY_MIN_SPEED_FPS <= m2["u"] <= FULL_ENTRY_MAX_SPEED_FPS
    and abs(m2["roll"]) <= FULL_ENTRY_MAX_ABS_ROLL_DEG
    and abs(m2["pitch"]) <= FULL_ENTRY_MAX_ABS_PITCH_DEG
    and hard_safe(m2)
)

print_phase(
    "STAGE2 V7-MATCHED ENTRY",
    entry_ok,
    elapsed=f"{stage2_elapsed:.2f}s",
    extra_after_160=f"{extra_elapsed:.2f}s",
    env_forward=f"{env_forward:.1f}ft",
    obs10=f"{float(obs2[10]):+.4f}",
)
print_handoff(
    "STAGE2 -> TURN",
    m2,
    fdm_id=id(get_fdm(env2)),
    geo_forward=f"{entry_state['forward_ft']:.1f}ft",
    env_forward=f"{env_forward:.1f}ft",
    obs_forward_progress=f"{float(obs2[10]):+.4f}",
    ref_forward=f"{REFERENCE_FORWARD_FT:.1f}ft",
    ref_obs10=f"{REFERENCE_FORWARD_PROGRESS:+.4f}",
    forward_delta=f"{env_forward - REFERENCE_FORWARD_FT:+.1f}ft",
    obs10_delta=f"{float(obs2[10]) - REFERENCE_FORWARD_PROGRESS:+.4f}",
    afcs_pitch=f"{fdm_get(fdm, 'ap/afcs/pitch-channel-active-norm'):.2f}",
    afcs_roll=f"{fdm_get(fdm, 'ap/afcs/roll-channel-active-norm'):.2f}",
    afcs_yaw=f"{fdm_get(fdm, 'ap/afcs/yaw-channel-active-norm'):.2f}",
)
if not entry_ok:
    raise RuntimeError("V7-matched turn-entry state failed acceptance criteria")


'''

patched = prefix + stage2_block + stage3_start + suffix

print("[V8] Patched validator: live Stage2 AFCS restored + V7 training-entry distribution matched before turn.")
exec(compile(patched, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
