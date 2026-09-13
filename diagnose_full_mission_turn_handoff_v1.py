from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
ENTRY_FORWARD_FT = 160.0
STAGE2_MAX_TIME_S = 70.0


def load_locked_defs():
    text = BASE_SOURCE.read_text(encoding="utf-8")
    prefix = text.split(MARKER, 1)[0]
    ns = {"__name__": "handoff_diag_locked", "__file__": str(BASE_SOURCE)}
    exec(compile(prefix, str(BASE_SOURCE), "exec"), ns)
    return ns


def fmt(x):
    return f"{float(x):+.6f}"


ns = load_locked_defs()
HelicopterEnvStage1Distill = ns["HelicopterEnvStage1Distill"]
HelicopterEnvStage2RefineMapped = ns["HelicopterEnvStage2RefineMapped"]
stage1_model = ns["stage1_model"]
stage2_model = ns["stage2_model"]
get_fdm = ns["get_fdm"]
heading_rad = ns["heading_rad"]
latitude_deg = ns["latitude_deg"]
longitude_deg = ns["longitude_deg"]
snapshot = ns["snapshot"]
env_control_dt = ns["env_control_dt"]
info_float = ns["info_float"]
AILERON_SCALE = ns["AILERON_SCALE"]
RUDDER_SCALE = ns["RUDDER_SCALE"]
HANDOFF_STABLE_TIME = ns["HANDOFF_STABLE_TIME"]
STAGE1_MAX_TIME = ns["STAGE1_MAX_TIME"]

print("=" * 120)
print("FULL-MISSION TURN HANDOFF DIAGNOSTIC")
print("Compare V5 training-reset entry vs true Stage1->Stage2 same-FDM entry")
print("=" * 120)

ref_env = HelicopterEnvTurnGoalV2(target_turn_deg=50.0)
ref_obs, _ = ref_env.reset(seed=42)
ref_raw = ref_env._raw_state()
ref_fd = float(ref_env.forward_distance)

env1 = HelicopterEnvStage1Distill(teacher_model_path=None, training_mode=False)
obs1, info1 = env1.reset()
fdm = get_fdm(env1)
mission_heading = heading_rad(fdm)
dt1 = env_control_dt(env1)
stable_time = 0.0

for _ in range(int(STAGE1_MAX_TIME / dt1)):
    a1, _ = stage1_model.predict(obs1, deterministic=True)
    obs1, _, terminated, truncated, info1 = env1.step(a1)
    alt = info_float(info1, "altitude")
    vs = info_float(info1, "vertical_speed")
    vn = info_float(info1, "vn", 0.0)
    ve = info_float(info1, "ve", 0.0)
    hs = float(np.hypot(vn, ve))
    drift = info_float(info1, "drift", 999.0)
    stable = 295.0 <= alt <= 305.0 and abs(vs) <= 0.50 and hs <= 1.0 and drift <= 3.0
    stable_time = stable_time + dt1 if stable else 0.0
    if stable_time >= HANDOFF_STABLE_TIME:
        break
    if truncated or (terminated and not bool(info1.get("success", False))):
        raise RuntimeError("Stage1 failed")

lat0 = latitude_deg(fdm)
lon0 = longitude_deg(fdm)
env2 = HelicopterEnvStage2RefineMapped(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
env2.reset()
env2.fdm = fdm
env2.phase = 1
env2.forward_distance = 0.0
if hasattr(env2, "target_heading"):
    env2.target_heading = float(mission_heading)
for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
    if hasattr(env2, attr):
        setattr(env2, attr, 0)
obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
dt2 = env_control_dt(env2)

for _ in range(int(STAGE2_MAX_TIME_S / dt2)):
    a2, _ = stage2_model.predict(obs2, deterministic=True)
    obs2, _, terminated, truncated, info2 = env2.step(a2)
    obs2 = np.asarray(obs2, dtype=np.float32)
    s = snapshot(fdm, lat0, lon0, mission_heading)
    if s["forward_ft"] >= ENTRY_FORWARD_FT:
        break
    if truncated or (terminated and not bool(info2.get("success", False))):
        raise RuntimeError("Stage2 failed")

live_env = HelicopterEnvTurnGoalV2(target_turn_deg=50.0)
live_env.fdm = fdm
live_env.phase = 1
live_env.turn_active = True
live_env.target_turn_deg = 50.0
live_env.cumulative_turn_deg = 0.0
live_env.prev_heading_deg = math.degrees(float(fdm["attitude/heading-true-rad"]))
live_env.turn_steps = 0
live_env.success_hold_s = 0.0
live_env.previous_turn_action = np.zeros(4, dtype=np.float32)
live_env.forward_distance = float(env2.forward_distance)
live_env.fdm["ap/afcs/roll-channel-active-norm"] = live_env.TURN_ROLL_AFCS
live_env.fdm["ap/afcs/yaw-channel-active-norm"] = live_env.TURN_YAW_AFCS
live_obs = np.asarray(live_env._get_obs(), dtype=np.float32)
live_raw = live_env._raw_state()

names = ["alt_error_norm", "alt_norm", "vs_norm", "fwd_v_norm", "lat_v_norm", "pitch_norm", "roll_norm", "p_rate", "q_rate", "r_rate", "forward_progress", "rpm_error_norm", "target_turn_norm", "remaining_norm", "cumulative_norm", "direction"]

print("\nOBSERVATION COMPARISON")
print(f"{'idx':>3} {'name':<22} {'reference':>12} {'full_mission':>12} {'delta':>12}")
for i, name in enumerate(names):
    d = float(live_obs[i] - ref_obs[i])
    print(f"{i:3d} {name:<22} {fmt(ref_obs[i]):>12} {fmt(live_obs[i]):>12} {fmt(d):>12}")

print("\nRAW STATE COMPARISON")
for key in ["altitude", "vertical_speed", "forward_velocity", "lateral_velocity", "pitch", "roll", "p_rate", "q_rate", "r_rate", "rotor_rpm"]:
    rv = float(ref_raw[key])
    lv = float(live_raw[key])
    print(f"{key:<20} ref={rv:+.6f} live={lv:+.6f} delta={lv-rv:+.6f}")

l2 = float(np.linalg.norm(live_obs[:12] - ref_obs[:12]))
max_abs = float(np.max(np.abs(live_obs[:12] - ref_obs[:12])))
print("\nREFERENCE forward_distance:", ref_fd)
print("FULL MISSION forward_distance:", float(live_env.forward_distance))
print(f"BASE OBS L2 DELTA: {l2:.6f}")
print(f"BASE OBS MAX ABS DELTA: {max_abs:.6f}")

ref_env.close()
live_env.fdm = None
env2.fdm = None
env1.close()
