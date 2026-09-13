from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry

# Reuse the locked Stage1/Stage2 definitions and helpers without executing
# the full validator mission body.
SOURCE = Path("validate_full_mission_v7_final.py")
MARKER = '# ============================================================================\n# 3 — FINAL TURN SYSTEM: SAME FDM'

text = SOURCE.read_text(encoding="utf-8")
if MARKER not in text:
    raise RuntimeError("Could not split final validator before turn phase")
prefix = text.split(MARKER, 1)[0]
ns = {"__name__": "entry_diag_defs", "__file__": str(SOURCE)}

# The prefix contains the interactive requested-turn read and mission execution,
# so use the older locked-definition source directly instead.
BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
LOCK_MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
base_text = BASE_SOURCE.read_text(encoding="utf-8")
locked_prefix = base_text.split(LOCK_MARKER, 1)[0]
locked = {"__name__": "diag_locked_defs", "__file__": str(BASE_SOURCE)}
exec(compile(locked_prefix, str(BASE_SOURCE), "exec"), locked)

HelicopterEnvStage1Distill = locked["HelicopterEnvStage1Distill"]
HelicopterEnvStage2RefineMapped = locked["HelicopterEnvStage2RefineMapped"]
stage1_model = locked["stage1_model"]
stage2_model = locked["stage2_model"]
get_fdm = locked["get_fdm"]
heading_rad = locked["heading_rad"]
latitude_deg = locked["latitude_deg"]
longitude_deg = locked["longitude_deg"]
snapshot = locked["snapshot"]
env_control_dt = locked["env_control_dt"]
info_float = locked["info_float"]
AILERON_SCALE = locked["AILERON_SCALE"]
RUDDER_SCALE = locked["RUDDER_SCALE"]
HANDOFF_STABLE_TIME = locked["HANDOFF_STABLE_TIME"]
STAGE1_MAX_TIME = locked["STAGE1_MAX_TIME"]

TARGET = 50.0
ENTRY_FORWARD_FT = 160.0
STAGE2_MAX_TIME_S = 70.0

print("=" * 120)
print("FINAL V7 ENTRY DIAGNOSTIC: TRAINING FULL-ENTRY vs TRUE STAGE1->STAGE2 ENTRY")
print("No model weights are changed.")
print("=" * 120)

# -----------------------------------------------------------------------------
# Reference: exact state/observation distribution V7 was trained/evaluated on.
# -----------------------------------------------------------------------------
ref = HelicopterEnvTurnGoalFullEntry(target_turn_deg=TARGET)
ref_obs, ref_info = ref.reset(seed=42)
ref_raw = ref._raw_state()
ref_forward = float(ref.forward_distance)

print("\nREFERENCE V7 TRAINING ENTRY")
print(f"forward_distance={ref_forward:.3f} ft")
print(f"alt={ref_raw['altitude']:.3f} ft | vs={ref_raw['vertical_speed']:+.3f} fps | "
      f"u={ref_raw['forward_velocity']:.3f} fps | lat_v={ref_raw['lateral_velocity']:+.3f} fps")
print(f"roll={math.degrees(ref_raw['roll']):+.3f} deg | pitch={math.degrees(ref_raw['pitch']):+.3f} deg | "
      f"p={math.degrees(ref_raw['p_rate']):+.3f} deg/s | q={math.degrees(ref_raw['q_rate']):+.3f} deg/s | "
      f"r={math.degrees(ref_raw['r_rate']):+.3f} deg/s | rpm={ref_raw['rotor_rpm']:.3f}")
ref.close()

# -----------------------------------------------------------------------------
# True mission: one active FDM, Stage1 then Stage2, stop at current final-validator
# handoff rule (env forward distance >=160 ft).
# -----------------------------------------------------------------------------
env1 = HelicopterEnvStage1Distill(teacher_model_path=None, training_mode=False)
obs1, info1 = env1.reset()
fdm = get_fdm(env1)
active_id = id(fdm)
mission_heading = heading_rad(fdm)
dt1 = env_control_dt(env1)
stable_time = 0.0

for _ in range(int(STAGE1_MAX_TIME / dt1)):
    a, _ = stage1_model.predict(obs1, deterministic=True)
    obs1, _, terminated, truncated, info1 = env1.step(a)
    alt = info_float(info1, "altitude")
    vs = info_float(info1, "vertical_speed")
    vn = info_float(info1, "vn", 0.0)
    ve = info_float(info1, "ve", 0.0)
    drift = info_float(info1, "drift", 999.0)
    stable = 295 <= alt <= 305 and abs(vs) <= 0.5 and np.hypot(vn, ve) <= 1.0 and drift <= 3.0
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
if hasattr(env2, "phase"):
    env2.phase = 1
if hasattr(env2, "forward_distance"):
    env2.forward_distance = 0.0
if hasattr(env2, "target_heading"):
    env2.target_heading = float(mission_heading)
for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
    if hasattr(env2, attr):
        setattr(env2, attr, 0)
if id(get_fdm(env2)) != active_id:
    raise RuntimeError("Same-FDM attach failed")

obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
dt2 = env_control_dt(env2)
entry = None
for _ in range(int(STAGE2_MAX_TIME_S / dt2)):
    a, _ = stage2_model.predict(obs2, deterministic=True)
    obs2, _, terminated, truncated, info2 = env2.step(a)
    obs2 = np.asarray(obs2, dtype=np.float32)
    s = snapshot(fdm, lat0, lon0, mission_heading)
    if float(getattr(env2, "forward_distance", 0.0)) >= ENTRY_FORWARD_FT:
        entry = s
        break
    if truncated or (terminated and not bool(info2.get("success", False))):
        raise RuntimeError("Stage2 failed")
if entry is None:
    raise RuntimeError("Stage2 entry not reached")

# Build a turn observation on the same active FDM without advancing physics.
turn = HelicopterEnvTurnGoalFullEntry(target_turn_deg=TARGET)
turn.fdm = fdm
turn.phase = 1
turn.turn_active = True
turn.target_turn_deg = TARGET
turn.cumulative_turn_deg = 0.0
turn.prev_heading_deg = math.degrees(float(fdm["attitude/heading-true-rad"]))
turn.turn_steps = 0
turn.success_hold_s = 0.0
turn.previous_turn_action = np.zeros(4, dtype=np.float32)
turn.forward_distance = float(getattr(env2, "forward_distance", ENTRY_FORWARD_FT))
turn._v2_previous_remaining = TARGET
turn.fdm["ap/afcs/roll-channel-active-norm"] = turn.TURN_ROLL_AFCS
turn.fdm["ap/afcs/yaw-channel-active-norm"] = turn.TURN_YAW_AFCS
live_obs = np.asarray(turn._get_obs(), dtype=np.float32)
live_raw = turn._raw_state()

names = [
    "alt_error_norm", "alt_norm", "vs_norm", "fwd_v_norm", "lat_v_norm",
    "pitch_norm", "roll_norm", "p_rate", "q_rate", "r_rate",
    "forward_progress", "rpm_error_norm", "target_turn_norm", "remaining_norm",
    "cumulative_norm", "direction",
]

print("\nTRUE FULL-MISSION ENTRY")
print(f"forward_distance={turn.forward_distance:.3f} ft")
print(f"alt={live_raw['altitude']:.3f} ft | vs={live_raw['vertical_speed']:+.3f} fps | "
      f"u={live_raw['forward_velocity']:.3f} fps | lat_v={live_raw['lateral_velocity']:+.3f} fps")
print(f"roll={math.degrees(live_raw['roll']):+.3f} deg | pitch={math.degrees(live_raw['pitch']):+.3f} deg | "
      f"p={math.degrees(live_raw['p_rate']):+.3f} deg/s | q={math.degrees(live_raw['q_rate']):+.3f} deg/s | "
      f"r={math.degrees(live_raw['r_rate']):+.3f} deg/s | rpm={live_raw['rotor_rpm']:.3f}")

print("\nOBSERVATION DELTA")
print(f"{'idx':>3s} {'name':24s} {'reference':>12s} {'live':>12s} {'delta':>12s}")
for i, name in enumerate(names):
    print(f"{i:3d} {name:24s} {ref_obs[i]:+12.6f} {live_obs[i]:+12.6f} {(live_obs[i]-ref_obs[i]):+12.6f}")

base_delta = live_obs[:12] - ref_obs[:12]
print(f"\nBASE OBS L2 DELTA: {np.linalg.norm(base_delta):.6f}")
print(f"BASE OBS MAX ABS DELTA: {np.max(np.abs(base_delta)):.6f}")
print(f"FORWARD PROGRESS DELTA: {(live_obs[10]-ref_obs[10]):+.6f}")
print(f"REFERENCE FWD: {ref_forward:.3f} ft | LIVE FWD: {turn.forward_distance:.3f} ft")
print("\nUse this output to decide whether the final validator must reproduce the V7 training-entry envelope before turn handoff.")
