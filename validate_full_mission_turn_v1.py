from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2, wrap_deg


BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
TURN_MODEL_PATH = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'

ENTRY_FORWARD_FT = 160.0
STAGE2_MAX_TIME_S = 70.0
POST_TURN_FORWARD_S = 10.0
TURN_CAPTURE_TOL_DEG = 1.5
POST_ALT_MIN_FT = 285.0
POST_ALT_MAX_FT = 315.0
POST_MIN_SPEED_FPS = 8.0
POST_MAX_ABS_VS_FPS = 1.5
POST_MAX_ABS_ROLL_DEG = 4.0
POST_MAX_ABS_YAW_RATE_DEG_S = 6.0
HARD_ALT_MIN_FT = 275.0
HARD_ALT_MAX_FT = 325.0


def rule(text: str):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def load_locked_defs():
    require_file(BASE_SOURCE)
    text = BASE_SOURCE.read_text(encoding="utf-8")
    if MARKER not in text:
        raise RuntimeError("Could not locate locked-definition split marker")
    prefix = text.split(MARKER, 1)[0]
    ns = {"__name__": "full_mission_locked_defs", "__file__": str(BASE_SOURCE)}
    exec(compile(prefix, str(BASE_SOURCE), "exec"), ns)
    return ns


def fdm_get(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


def heading_deg(fdm):
    for key in ["attitude/heading-true-rad", "attitude/psi-rad"]:
        v = fdm_get(fdm, key)
        if np.isfinite(v):
            return math.degrees(v)
    return 0.0


def state_metrics(fdm):
    return {
        "alt": fdm_get(fdm, "position/h-agl-ft"),
        "vs": fdm_get(fdm, "velocities/h-dot-fps"),
        "v": fdm_get(fdm, "velocities/u-aero-fps"),
        "roll": math.degrees(fdm_get(fdm, "attitude/roll-rad", 0.0)),
        "pitch": math.degrees(fdm_get(fdm, "attitude/pitch-rad", 0.0)),
        "yaw_rate": math.degrees(fdm_get(fdm, "velocities/r-rad_sec", 0.0)),
        "hdg": heading_deg(fdm),
    }


def hard_safe(m):
    return bool(
        HARD_ALT_MIN_FT <= m["alt"] <= HARD_ALT_MAX_FT
        and abs(m["roll"]) <= 18.0
        and abs(m["pitch"]) <= 15.0
        and abs(m["yaw_rate"]) <= 30.0
        and m["v"] >= 1.0
    )


def print_phase(name, ok, **vals):
    details = " | ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{name}: {'PASS' if ok else 'FAIL'}" + (f" | {details}" if details else ""))


require_file(TURN_MODEL_PATH)
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

turn_model = PPO.load(str(TURN_MODEL_PATH))

rule("AH-1S FULL MISSION VALIDATION - PPO RUNTIME / TEACHER OFF")
print("Runtime teacher/controller: OFF")
print("Stage1 PPO:", ns["STAGE1_MODEL_PATH"])
print("Stage2 PPO:", ns["STAGE2_MODEL_PATH"])
print("Turn PPO  :", TURN_MODEL_PATH)

try:
    requested_turn = float(input("Relative turn angle in degrees (e.g. -50, 50, 200, 360): ").strip())
except Exception as exc:
    raise RuntimeError("Invalid turn angle") from exc

if abs(requested_turn) < 1.0 or abs(requested_turn) > 720.0:
    raise ValueError("Use a non-zero relative turn between -720 and +720 degrees")


# =====================================================================
# STAGE 1 — TAKEOFF + 300 FT HOVER (LEARNED PPO)
# =====================================================================
rule("1 — STAGE 1 PPO: TAKEOFF -> 300 FT HOVER")

env1 = HelicopterEnvStage1Distill(teacher_model_path=None, training_mode=False)
obs1, info1 = env1.reset()
fdm = get_fdm(env1)
active_fdm_id = id(fdm)
mission_heading = heading_rad(fdm)
dt1 = env_control_dt(env1)
stable_time = 0.0
stage1_elapsed = 0.0

for _ in range(int(STAGE1_MAX_TIME / dt1)):
    action1, _ = stage1_model.predict(obs1, deterministic=True)
    obs1, _, terminated, truncated, info1 = env1.step(action1)
    stage1_elapsed += dt1

    alt = info_float(info1, "altitude")
    vs = info_float(info1, "vertical_speed")
    vn = info_float(info1, "vn", 0.0)
    ve = info_float(info1, "ve", 0.0)
    hs = float(np.hypot(vn, ve))
    drift = info_float(info1, "drift", 999.0)

    stable = bool(295.0 <= alt <= 305.0 and abs(vs) <= 0.50 and hs <= 1.0 and drift <= 3.0)
    stable_time = stable_time + dt1 if stable else 0.0
    if stable_time >= HANDOFF_STABLE_TIME:
        break
    if truncated or (terminated and not bool(info1.get("success", False))):
        raise RuntimeError("Stage1 failed before stable hover handoff")

stage1_pass = stable_time >= HANDOFF_STABLE_TIME
print_phase("STAGE1", stage1_pass, time=f"{stage1_elapsed:.2f}s", alt=f"{alt:.2f}ft", vs=f"{vs:+.3f}fps", hover=f"{stable_time:.2f}s")
if not stage1_pass:
    raise RuntimeError("Stage1 did not reach stable 300-ft hover")

lat0 = latitude_deg(fdm)
lon0 = longitude_deg(fdm)


# =====================================================================
# STAGE 2 — SAME FDM, FORWARD FLIGHT (LEARNED PPO)
# =====================================================================
rule("2 — STAGE 2 PPO: SAME-FDM FORWARD FLIGHT")

sim_before = fdm_get(fdm, "simulation/sim-time-sec")
env2 = HelicopterEnvStage2RefineMapped(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
env2.reset()  # build its internals on a disposable FDM
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

if id(get_fdm(env2)) != active_fdm_id:
    raise RuntimeError("Stage2 same-FDM attach failed")
sim_after = fdm_get(fdm, "simulation/sim-time-sec")
if np.isfinite(sim_before) and np.isfinite(sim_after) and abs(sim_after - sim_before) > 1e-9:
    raise RuntimeError("Active simulation clock changed during Stage2 attach")

obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
dt2 = env_control_dt(env2)
stage2_elapsed = 0.0
entry_state = None

for step in range(int(STAGE2_MAX_TIME_S / dt2)):
    action2, _ = stage2_model.predict(obs2, deterministic=True)
    obs2, _, terminated, truncated, info2 = env2.step(action2)
    obs2 = np.asarray(obs2, dtype=np.float32)
    stage2_elapsed = (step + 1) * dt2
    s = snapshot(fdm, lat0, lon0, mission_heading)

    if s["forward_ft"] >= ENTRY_FORWARD_FT:
        entry_state = s
        break
    if truncated or (terminated and not bool(info2.get("success", False))):
        raise RuntimeError("Stage2 failed before turn-entry distance")

if entry_state is None:
    raise RuntimeError("Stage2 did not reach turn-entry distance")

entry_ok = bool(
    285.0 <= entry_state["altitude_ft"] <= 315.0
    and abs(entry_state["vertical_speed_fps"]) <= 2.0
    and entry_state["forward_speed_fps"] >= 8.0
    and abs(math.degrees(entry_state["roll_rad"])) <= 8.0
)
print_phase(
    "STAGE2 ENTRY",
    entry_ok,
    forward=f"{entry_state['forward_ft']:.1f}ft",
    speed=f"{entry_state['forward_speed_fps']:.2f}fps",
    alt=f"{entry_state['altitude_ft']:.2f}ft",
    vs=f"{entry_state['vertical_speed_fps']:+.2f}fps",
)
if not entry_ok:
    raise RuntimeError("Turn-entry state failed acceptance criteria")


# =====================================================================
# TURN — SAME FDM, V5 PPO, TEACHER OFF
# =====================================================================
rule("3 — V5 GOAL-CONDITIONED PPO TURN: TEACHER OFF")

turn_env = HelicopterEnvTurnGoalV2(target_turn_deg=requested_turn)
turn_env.fdm = fdm
turn_env.phase = 1
turn_env.turn_active = True
turn_env.target_turn_deg = float(requested_turn)
turn_env.cumulative_turn_deg = 0.0
turn_env.prev_heading_deg = heading_deg(fdm)
turn_env.turn_steps = 0
turn_env.success_hold_s = 0.0
turn_env.previous_turn_action = np.zeros(4, dtype=np.float32)
turn_env.forward_distance = float(getattr(env2, "forward_distance", ENTRY_FORWARD_FT))
turn_env.fdm["ap/afcs/roll-channel-active-norm"] = turn_env.TURN_ROLL_AFCS
turn_env.fdm["ap/afcs/yaw-channel-active-norm"] = turn_env.TURN_YAW_AFCS

obs_turn = turn_env._get_obs()
turn_elapsed = 0.0
turn_info = None
turn_pass = False
next_print = 0.0
max_turn_s = max(70.0, abs(requested_turn) / 1.10 + 60.0)

for step in range(int(max_turn_s / turn_env.CONTROL_DT)):
    action, _ = turn_model.predict(obs_turn, deterministic=True)
    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action)
    turn_elapsed = (step + 1) * turn_env.CONTROL_DT

    if turn_elapsed >= next_print:
        print(
            f"t={turn_elapsed:6.1f}s | turn={turn_info['cumulative_turn_deg']:+8.2f} | "
            f"rem={turn_info['remaining_turn_deg']:+7.2f} | alt={turn_info['altitude']:7.2f} | "
            f"vs={turn_info['vertical_speed']:+6.2f} | v={turn_info['forward_velocity']:6.2f} | "
            f"hold={turn_info['success_hold_s']:4.1f}"
        )
        next_print += 5.0

    if terminated or truncated:
        turn_pass = bool(turn_info.get("success", False))
        break

if turn_info is None:
    raise RuntimeError("Turn phase produced no state")

print_phase(
    "TURN",
    turn_pass,
    requested=f"{requested_turn:+.2f}deg",
    completed=f"{turn_info['cumulative_turn_deg']:+.2f}deg",
    error=f"{turn_info['remaining_turn_deg']:+.2f}deg",
    hold=f"{turn_info['success_hold_s']:.2f}s",
    alt=f"{turn_info['altitude']:.2f}ft",
    speed=f"{turn_info['forward_velocity']:.2f}fps",
)
if not turn_pass:
    raise RuntimeError("V5 PPO turn did not satisfy capture criteria")


# =====================================================================
# POST-TURN — CONTINUE FORWARD ON NEW HEADING WITH SAME V5 PPO
# =====================================================================
rule("4 — POST-TURN FORWARD: NEW HEADING HOLD FOR 10 S")

post_elapsed = 0.0
post_stable = 0.0
post_min_alt = float("inf")
post_max_heading_error = 0.0
post_safety_failure = False
post_info = None

# turn_env.step() intentionally terminates after its 3-s success hold.
# Continue the SAME learned policy manually on the SAME live FDM; no reset,
# no teacher, and no controller substitution.
for step in range(int(POST_TURN_FORWARD_S / turn_env.CONTROL_DT) + 1):
    obs_turn = turn_env._get_obs()
    action, _ = turn_model.predict(obs_turn, deterministic=True)
    turn_env._apply_action(action)

    jsbsim_ok = True
    for _ in range(turn_env.PHYSICS_STEPS):
        if not fdm.run():
            jsbsim_ok = False
            break

    current_hdg = heading_deg(fdm)
    turn_env.cumulative_turn_deg += wrap_deg(current_hdg - turn_env.prev_heading_deg)
    turn_env.prev_heading_deg = current_hdg
    m = state_metrics(fdm)
    remaining = requested_turn - turn_env.cumulative_turn_deg
    post_elapsed = (step + 1) * turn_env.CONTROL_DT
    post_min_alt = min(post_min_alt, m["alt"])
    post_max_heading_error = max(post_max_heading_error, abs(remaining))

    stable = bool(
        abs(remaining) <= TURN_CAPTURE_TOL_DEG
        and POST_ALT_MIN_FT <= m["alt"] <= POST_ALT_MAX_FT
        and abs(m["vs"]) <= POST_MAX_ABS_VS_FPS
        and m["v"] >= POST_MIN_SPEED_FPS
        and abs(m["roll"]) <= POST_MAX_ABS_ROLL_DEG
        and abs(m["yaw_rate"]) <= POST_MAX_ABS_YAW_RATE_DEG_S
    )
    post_stable = post_stable + turn_env.CONTROL_DT if stable else 0.0
    post_safety_failure = post_safety_failure or (not jsbsim_ok) or (not hard_safe(m))
    post_info = {**m, "remaining": remaining}

    if post_safety_failure:
        break

post_pass = bool(
    not post_safety_failure
    and post_elapsed >= POST_TURN_FORWARD_S
    and abs(post_info["remaining"]) <= TURN_CAPTURE_TOL_DEG
    and POST_ALT_MIN_FT <= post_info["alt"] <= POST_ALT_MAX_FT
    and abs(post_info["vs"]) <= POST_MAX_ABS_VS_FPS
    and post_info["v"] >= POST_MIN_SPEED_FPS
    and abs(post_info["roll"]) <= POST_MAX_ABS_ROLL_DEG
    and abs(post_info["yaw_rate"]) <= POST_MAX_ABS_YAW_RATE_DEG_S
)

print_phase(
    "POST-TURN FORWARD",
    post_pass,
    time=f"{post_elapsed:.2f}s",
    remaining=f"{post_info['remaining']:+.2f}deg",
    alt=f"{post_info['alt']:.2f}ft",
    min_alt=f"{post_min_alt:.2f}ft",
    speed=f"{post_info['v']:.2f}fps",
    roll=f"{post_info['roll']:+.2f}deg",
    yaw_rate=f"{post_info['yaw_rate']:+.2f}deg/s",
)


# =====================================================================
# FINAL REPORT
# =====================================================================
rule("FULL MISSION RESULT")
full_pass = bool(stage1_pass and entry_ok and turn_pass and post_pass and not post_safety_failure)
print("FULL MISSION PASS:", full_pass)
print("TAKEOFF + 300 FT HOVER:", "PASS" if stage1_pass else "FAIL")
print("FORWARD ENTRY:", "PASS" if entry_ok else "FAIL")
print(f"REQUESTED TURN: {requested_turn:+.3f} deg")
print(f"COMPLETED TURN: {turn_env.cumulative_turn_deg:+.3f} deg")
print(f"FINAL TURN ERROR: {requested_turn - turn_env.cumulative_turn_deg:+.3f} deg")
print("TURN CAPTURE:", "PASS" if turn_pass else "FAIL")
print("POST-TURN FORWARD:", "PASS" if post_pass else "FAIL")
print(f"FINAL ALT: {post_info['alt']:.2f} ft")
print(f"FINAL SPEED: {post_info['v']:.2f} ft/s")
print("SAFETY FAILURE:", post_safety_failure)
print("TEACHER ACTIVE AT RUNTIME: False")
print("TURN POLICY:", TURN_MODEL_PATH)

turn_env.fdm = None
env2.fdm = None
env1.close()

if not full_pass:
    raise SystemExit(2)
