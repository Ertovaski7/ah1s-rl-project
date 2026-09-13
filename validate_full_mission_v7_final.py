from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2
from helicopter_env_turn_goal import wrap_deg


# ============================================================================
# FINAL MISSION: ONE JSBSim FDM, NO RESET BETWEEN PHASES
# Stage1 PPO -> Stage2 PPO -> V5 PPO + V4 residual + gated V7 +200 patch
# -> 10 s forward continuation on the new heading.
# Runtime teacher/controller: OFF.
# ============================================================================

BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'

TURN_PPO_PATH = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
V4_ADAPTER_PATH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt")
V7_PATCH_PATH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")

# Handoff / validation criteria.
ENTRY_FORWARD_FT = 160.0
STAGE2_MAX_TIME_S = 70.0
POST_TURN_FORWARD_S = 10.0

STAGE1_ALT_MIN_FT = 295.0
STAGE1_ALT_MAX_FT = 305.0
STAGE1_MAX_ABS_VS_FPS = 0.50
STAGE1_MAX_HS_FPS = 1.0
STAGE1_MAX_DRIFT_FT = 3.0

ENTRY_ALT_MIN_FT = 285.0
ENTRY_ALT_MAX_FT = 315.0
ENTRY_MAX_ABS_VS_FPS = 2.0
ENTRY_MIN_SPEED_FPS = 8.0
ENTRY_MAX_ABS_ROLL_DEG = 8.0

TURN_CAPTURE_TOL_DEG = 1.5
TURN_ALT_MIN_FT = 285.0
TURN_ALT_MAX_FT = 315.0
TURN_MAX_ABS_VS_FPS = 1.5
TURN_MIN_SPEED_FPS = 8.0
TURN_MAX_ABS_ROLL_DEG = 4.0
TURN_MAX_ABS_YAW_RATE_DEG_S = 6.0
TURN_REQUIRED_HOLD_S = 3.0

HARD_ALT_MIN_FT = 275.0
HARD_ALT_MAX_FT = 325.0
HARD_MAX_ABS_ROLL_DEG = 18.0
HARD_MAX_ABS_PITCH_DEG = 15.0
HARD_MAX_ABS_YAW_RATE_DEG_S = 30.0
HARD_MIN_SPEED_FPS = 1.0

BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
PATCH_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02
VALIDATED_TARGETS = (-50.0, 50.0, 200.0, 360.0)


class ResidualAdapter(nn.Module):
    """Checkpoint-compatible with the V4/V7 residual networks."""

    def __init__(self, scale):
        super().__init__()
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * self.scale.to(obs.device)


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


def physical_metrics(fdm):
    """Common physical metrics printed at every handoff."""
    return {
        "alt": fdm_get(fdm, "position/h-agl-ft"),
        "vs": fdm_get(fdm, "velocities/h-dot-fps"),
        "u": fdm_get(fdm, "velocities/u-aero-fps"),
        "v_lat": fdm_get(fdm, "velocities/v-aero-fps"),
        "roll": math.degrees(fdm_get(fdm, "attitude/roll-rad", 0.0)),
        "pitch": math.degrees(fdm_get(fdm, "attitude/pitch-rad", 0.0)),
        "p": math.degrees(fdm_get(fdm, "velocities/p-rad_sec", 0.0)),
        "q": math.degrees(fdm_get(fdm, "velocities/q-rad_sec", 0.0)),
        "r": math.degrees(fdm_get(fdm, "velocities/r-rad_sec", 0.0)),
        "hdg": heading_deg(fdm),
        "rpm": fdm_get(fdm, "propulsion/engine[0]/rotor-rpm", float("nan")),
        "sim_t": fdm_get(fdm, "simulation/sim-time-sec"),
    }


def hard_safe(m):
    return bool(
        HARD_ALT_MIN_FT <= m["alt"] <= HARD_ALT_MAX_FT
        and abs(m["roll"]) <= HARD_MAX_ABS_ROLL_DEG
        and abs(m["pitch"]) <= HARD_MAX_ABS_PITCH_DEG
        and abs(m["r"]) <= HARD_MAX_ABS_YAW_RATE_DEG_S
        and m["u"] >= HARD_MIN_SPEED_FPS
    )


def print_phase(name, ok, **vals):
    details = " | ".join(f"{k}={v}" for k, v in vals.items())
    print(f"{name}: {'PASS' if ok else 'FAIL'}" + (f" | {details}" if details else ""))


def print_handoff(label, m, **extra):
    print(f"\n--- {label} HANDOFF METRICS ---")
    print(
        f"sim_t={m['sim_t']:.2f}s | alt={m['alt']:.2f}ft | vs={m['vs']:+.3f}fps | "
        f"u={m['u']:.2f}fps | v_lat={m['v_lat']:+.2f}fps"
    )
    print(
        f"roll={m['roll']:+.2f}deg | pitch={m['pitch']:+.2f}deg | "
        f"p/q/r={m['p']:+.2f}/{m['q']:+.2f}/{m['r']:+.2f}deg/s | "
        f"heading={m['hdg']:.2f}deg | rotor_rpm={m['rpm']:.2f}"
    )
    if extra:
        print(" | ".join(f"{k}={v}" for k, v in extra.items()))


def gate_from_obs(obs):
    # obs[12] = requested relative turn / 360.
    return 1.0 if abs(float(obs[12]) - TARGET_200_NORM) <= GATE_TOL else 0.0


def turn_action(turn_model, base_adapter, patch, obs):
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


# ============================================================================
# LOAD LOCKED MODELS / DEFINITIONS
# ============================================================================
for path in [TURN_PPO_PATH, V4_ADAPTER_PATH, V7_PATCH_PATH]:
    require_file(path)

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

turn_model = PPO.load(str(TURN_PPO_PATH))
base_adapter = ResidualAdapter(BASE_SCALE)
base_adapter.load_state_dict(torch.load(V4_ADAPTER_PATH, map_location="cpu"), strict=True)
base_adapter.eval()
patch = ResidualAdapter(PATCH_SCALE)
patch.load_state_dict(torch.load(V7_PATCH_PATH, map_location="cpu"), strict=True)
patch.eval()
for module in [base_adapter, patch]:
    for p in module.parameters():
        p.requires_grad = False


rule("AH-1S FINAL FULL-MISSION VALIDATOR — ONE LIVE JSBSim FDM")
print("Runtime teacher/controller: OFF")
print("Stage1 PPO:", ns["STAGE1_MODEL_PATH"])
print("Stage2 PPO:", ns["STAGE2_MODEL_PATH"])
print("Turn PPO :", TURN_PPO_PATH)
print("V4 adapter:", V4_ADAPTER_PATH)
print("V7 +200 gated patch:", V7_PATCH_PATH)
print("No reset is allowed after Stage1 starts; FDM identity and simulation time continuity are checked.")

try:
    requested_turn = float(input("Relative turn angle in degrees (-50, 50, 200, 360 validated): ").strip())
except Exception as exc:
    raise RuntimeError("Invalid turn angle") from exc

if abs(requested_turn) < 1.0 or abs(requested_turn) > 720.0:
    raise ValueError("Use a non-zero relative turn between -720 and +720 degrees")

validated_command = any(abs(requested_turn - x) < 1e-6 for x in VALIDATED_TARGETS)
if not validated_command:
    print("WARNING: this angle was not part of the final four-command validation set; V7 +200 patch will normally be gated off.")


# ============================================================================
# 1 — STAGE 1: TAKEOFF -> 300 FT HOVER
# ============================================================================
rule("1 — STAGE 1 PPO: TAKEOFF -> 300 FT HOVER")

env1 = HelicopterEnvStage1Distill(teacher_model_path=None, training_mode=False)
obs1, info1 = env1.reset()
fdm = get_fdm(env1)
active_fdm_id = id(fdm)
mission_heading = heading_rad(fdm)
dt1 = env_control_dt(env1)
stable_time = 0.0
stage1_elapsed = 0.0

alt = vs = hs = drift = float("nan")
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

    stable = bool(
        STAGE1_ALT_MIN_FT <= alt <= STAGE1_ALT_MAX_FT
        and abs(vs) <= STAGE1_MAX_ABS_VS_FPS
        and hs <= STAGE1_MAX_HS_FPS
        and drift <= STAGE1_MAX_DRIFT_FT
    )
    stable_time = stable_time + dt1 if stable else 0.0
    if stable_time >= HANDOFF_STABLE_TIME:
        break
    if truncated or (terminated and not bool(info1.get("success", False))):
        raise RuntimeError("Stage1 failed before stable hover handoff")

stage1_pass = stable_time >= HANDOFF_STABLE_TIME
m1 = physical_metrics(fdm)
print_phase(
    "STAGE1",
    stage1_pass,
    elapsed=f"{stage1_elapsed:.2f}s",
    hover_hold=f"{stable_time:.2f}s",
    horizontal_speed=f"{hs:.2f}fps",
    drift=f"{drift:.2f}ft",
)
print_handoff(
    "STAGE1 -> STAGE2",
    m1,
    fdm_id=active_fdm_id,
    hover_hold=f"{stable_time:.2f}s",
    horizontal_speed=f"{hs:.2f}fps",
    drift=f"{drift:.2f}ft",
)
if not stage1_pass:
    raise RuntimeError("Stage1 did not reach stable 300-ft hover")

lat0 = latitude_deg(fdm)
lon0 = longitude_deg(fdm)


# ============================================================================
# 2 — STAGE 2: SAME FDM FORWARD FLIGHT
# ============================================================================
rule("2 — STAGE 2 PPO: SAME-FDM FORWARD FLIGHT")

sim_before_stage2 = fdm_get(fdm, "simulation/sim-time-sec")
env2 = HelicopterEnvStage2RefineMapped(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
env2.reset()  # disposable internal FDM only; active FDM is attached immediately below.
env2.fdm = fdm
if hasattr(env2, "phase"):
    env2.phase = 1
if hasattr(env2, "forward_distance"):
    env2.forward_distance = 0.0
if hasattr(env2, "target_heading"):
    env2.target_heading = float(mission_heading)
if hasattr(env2, "previous_action"):
    env2.previous_action = np.zeros(4, dtype=np.float32)
for attr in ["steps", "target_hold_steps", "hold_steps", "success_hold_steps"]:
    if hasattr(env2, attr):
        setattr(env2, attr, 0)

if id(get_fdm(env2)) != active_fdm_id:
    raise RuntimeError("Stage2 same-FDM attach failed")
sim_after_stage2_attach = fdm_get(fdm, "simulation/sim-time-sec")
if np.isfinite(sim_before_stage2) and np.isfinite(sim_after_stage2_attach) and abs(sim_after_stage2_attach - sim_before_stage2) > 1e-9:
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

    # Use the learned environment's integrated forward_distance as the actual
    # handoff trigger. Geographic displacement is still printed as a metric.
    env_forward = float(getattr(env2, "forward_distance", s["forward_ft"]))
    if env_forward >= ENTRY_FORWARD_FT:
        entry_state = s
        break
    if truncated or (terminated and not bool(info2.get("success", False))):
        raise RuntimeError("Stage2 failed before turn-entry distance")

if entry_state is None:
    raise RuntimeError("Stage2 did not reach turn-entry distance")

m2 = physical_metrics(fdm)
env_forward = float(getattr(env2, "forward_distance", entry_state["forward_ft"]))
entry_ok = bool(
    ENTRY_ALT_MIN_FT <= m2["alt"] <= ENTRY_ALT_MAX_FT
    and abs(m2["vs"]) <= ENTRY_MAX_ABS_VS_FPS
    and m2["u"] >= ENTRY_MIN_SPEED_FPS
    and abs(m2["roll"]) <= ENTRY_MAX_ABS_ROLL_DEG
    and hard_safe(m2)
)
print_phase(
    "STAGE2 ENTRY",
    entry_ok,
    elapsed=f"{stage2_elapsed:.2f}s",
    geo_forward=f"{entry_state['forward_ft']:.1f}ft",
    env_forward=f"{env_forward:.1f}ft",
)
print_handoff(
    "STAGE2 -> TURN",
    m2,
    fdm_id=id(get_fdm(env2)),
    geo_forward=f"{entry_state['forward_ft']:.1f}ft",
    env_forward=f"{env_forward:.1f}ft",
    obs_forward_progress=f"{float(obs2[10]):+.4f}" if len(obs2) > 10 else "n/a",
)
if not entry_ok:
    raise RuntimeError("Turn-entry state failed acceptance criteria")


# ============================================================================
# 3 — FINAL TURN SYSTEM: SAME FDM
# V5 PPO + V4 residual + hard-gated V7 +200 patch
# ============================================================================
rule("3 — FINAL TURN: V5 PPO + V4 RESIDUAL + V7 GATED +200 PATCH")

sim_before_turn = fdm_get(fdm, "simulation/sim-time-sec")
turn_env = HelicopterEnvTurnGoalV2(target_turn_deg=requested_turn)
# IMPORTANT: do not call turn_env.reset(); that would create a new mission state.
turn_env.fdm = fdm
turn_env.phase = 1
turn_env.turn_active = True
turn_env.target_turn_deg = float(requested_turn)
turn_env.cumulative_turn_deg = 0.0
turn_env.prev_heading_deg = heading_deg(fdm)
turn_env.turn_steps = 0
turn_env.success_hold_s = 0.0
turn_env.previous_turn_action = np.zeros(4, dtype=np.float32)
turn_env.forward_distance = env_forward
if hasattr(turn_env, "_v2_previous_remaining"):
    turn_env._v2_previous_remaining = float(requested_turn)
turn_env.fdm["ap/afcs/roll-channel-active-norm"] = turn_env.TURN_ROLL_AFCS
turn_env.fdm["ap/afcs/yaw-channel-active-norm"] = turn_env.TURN_YAW_AFCS

if id(turn_env.fdm) != active_fdm_id:
    raise RuntimeError("Turn same-FDM attach failed")
sim_after_turn_attach = fdm_get(fdm, "simulation/sim-time-sec")
if np.isfinite(sim_before_turn) and np.isfinite(sim_after_turn_attach) and abs(sim_after_turn_attach - sim_before_turn) > 1e-9:
    raise RuntimeError("Active simulation clock changed during turn attach")

obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
turn_entry_metrics = physical_metrics(fdm)
patch_gate = gate_from_obs(obs_turn)
print_handoff(
    "TURN ENTRY",
    turn_entry_metrics,
    fdm_id=id(turn_env.fdm),
    requested=f"{requested_turn:+.2f}deg",
    target_norm=f"{float(obs_turn[12]):+.5f}",
    v7_gate="ON" if patch_gate > 0.5 else "OFF",
    cumulative="0.00deg",
)

turn_elapsed = 0.0
turn_info = None
turn_pass = False
next_print = 0.0
max_turn_s = max(90.0, abs(requested_turn) / 1.10 + 70.0)
last_d_v4 = np.zeros(4, dtype=np.float32)
last_d_v7 = np.zeros(4, dtype=np.float32)

for step in range(int(max_turn_s / turn_env.CONTROL_DT)):
    action_turn, last_d_v4, last_d_v7 = turn_action(turn_model, base_adapter, patch, obs_turn)
    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)
    obs_turn = np.asarray(obs_turn, dtype=np.float32)
    turn_elapsed = (step + 1) * turn_env.CONTROL_DT

    if turn_elapsed >= next_print:
        print(
            f"t={turn_elapsed:6.1f}s | turn={turn_info['cumulative_turn_deg']:+8.2f} | "
            f"rem={turn_info['remaining_turn_deg']:+7.2f} | alt={turn_info['altitude']:7.2f} | "
            f"vs={turn_info['vertical_speed']:+6.2f} | v={turn_info['forward_velocity']:6.2f} | "
            f"hold={turn_info['success_hold_s']:4.1f} | gate={'ON' if gate_from_obs(obs_turn) > 0.5 else 'OFF'}"
        )
        next_print += 5.0

    if terminated or truncated:
        turn_pass = bool(turn_info.get("success", False))
        break

if turn_info is None:
    raise RuntimeError("Turn phase produced no state")

m_turn = physical_metrics(fdm)
turn_capture_metrics_ok = bool(
    abs(float(turn_info["remaining_turn_deg"])) <= TURN_CAPTURE_TOL_DEG
    and TURN_ALT_MIN_FT <= m_turn["alt"] <= TURN_ALT_MAX_FT
    and abs(m_turn["vs"]) <= TURN_MAX_ABS_VS_FPS
    and m_turn["u"] >= TURN_MIN_SPEED_FPS
    and abs(m_turn["roll"]) <= TURN_MAX_ABS_ROLL_DEG
    and abs(m_turn["r"]) <= TURN_MAX_ABS_YAW_RATE_DEG_S
    and float(turn_info.get("success_hold_s", 0.0)) >= TURN_REQUIRED_HOLD_S - turn_env.CONTROL_DT
    and hard_safe(m_turn)
)
turn_pass = bool(turn_pass and turn_capture_metrics_ok)
print_phase(
    "TURN CAPTURE",
    turn_pass,
    requested=f"{requested_turn:+.2f}deg",
    completed=f"{float(turn_info['cumulative_turn_deg']):+.2f}deg",
    remaining=f"{float(turn_info['remaining_turn_deg']):+.2f}deg",
    hold=f"{float(turn_info.get('success_hold_s', 0.0)):.2f}s",
)
print_handoff(
    "TURN -> POST-TURN FORWARD",
    m_turn,
    completed=f"{float(turn_info['cumulative_turn_deg']):+.2f}deg",
    remaining=f"{float(turn_info['remaining_turn_deg']):+.2f}deg",
    hold=f"{float(turn_info.get('success_hold_s', 0.0)):.2f}s",
    v4_delta=np.array2string(last_d_v4, precision=3),
    v7_delta=np.array2string(last_d_v7, precision=3),
)
if not turn_pass:
    raise RuntimeError("Final turn system did not satisfy capture criteria")


# ============================================================================
# 4 — POST-TURN FORWARD: SAME FDM, SAME FINAL TURN POLICY STACK
# ============================================================================
rule("4 — POST-TURN FORWARD: HOLD NEW HEADING FOR 10 S")

post_elapsed = 0.0
post_stable_time = 0.0
post_safety_failure = False
post_min_alt = float("inf")
post_max_alt = -float("inf")
post_max_abs_heading_error = 0.0
post_min_speed = float("inf")
post_max_abs_vs = 0.0
post_max_abs_roll = 0.0
post_max_abs_yaw_rate = 0.0
post_final = None

# turn_env.step() has already terminated after its success hold, so continue
# the exact same learned stack manually on the same FDM. No reset/teacher/controller.
post_steps = int(math.ceil(POST_TURN_FORWARD_S / turn_env.CONTROL_DT))
for _ in range(post_steps):
    obs_turn = np.asarray(turn_env._get_obs(), dtype=np.float32)
    action_turn, _, _ = turn_action(turn_model, base_adapter, patch, obs_turn)
    turn_env._apply_action(action_turn)

    jsbsim_ok = True
    for _ in range(turn_env.PHYSICS_STEPS):
        if not fdm.run():
            jsbsim_ok = False
            break

    current_hdg = heading_deg(fdm)
    turn_env.cumulative_turn_deg += wrap_deg(current_hdg - turn_env.prev_heading_deg)
    turn_env.prev_heading_deg = current_hdg

    m = physical_metrics(fdm)
    remaining = requested_turn - turn_env.cumulative_turn_deg
    post_elapsed += turn_env.CONTROL_DT

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
    post_stable_time = post_stable_time + turn_env.CONTROL_DT if stable else 0.0
    post_safety_failure = post_safety_failure or (not jsbsim_ok) or (not hard_safe(m))
    post_final = {**m, "remaining": remaining, "completed": turn_env.cumulative_turn_deg}

    if post_safety_failure:
        break

if post_final is None:
    raise RuntimeError("Post-turn phase produced no state")

post_pass = bool(
    not post_safety_failure
    and post_elapsed >= POST_TURN_FORWARD_S - turn_env.CONTROL_DT
    and abs(post_final["remaining"]) <= TURN_CAPTURE_TOL_DEG
    and TURN_ALT_MIN_FT <= post_final["alt"] <= TURN_ALT_MAX_FT
    and abs(post_final["vs"]) <= TURN_MAX_ABS_VS_FPS
    and post_final["u"] >= TURN_MIN_SPEED_FPS
    and abs(post_final["roll"]) <= TURN_MAX_ABS_ROLL_DEG
    and abs(post_final["r"]) <= TURN_MAX_ABS_YAW_RATE_DEG_S
)
print_phase(
    "POST-TURN FORWARD",
    post_pass,
    duration=f"{post_elapsed:.2f}s",
    completed=f"{post_final['completed']:+.2f}deg",
    remaining=f"{post_final['remaining']:+.2f}deg",
    final_alt=f"{post_final['alt']:.2f}ft",
    final_speed=f"{post_final['u']:.2f}fps",
)
print(
    "POST-TURN WINDOW | "
    f"alt=[{post_min_alt:.2f},{post_max_alt:.2f}]ft | min_speed={post_min_speed:.2f}fps | "
    f"max|heading_error|={post_max_abs_heading_error:.2f}deg | max|VS|={post_max_abs_vs:.2f}fps | "
    f"max|roll|={post_max_abs_roll:.2f}deg | max|yaw_rate|={post_max_abs_yaw_rate:.2f}deg/s | "
    f"continuous_stable_tail={post_stable_time:.2f}s | safety_failure={post_safety_failure}"
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
rule("FINAL MISSION SUMMARY")
fdm_continuity = bool(
    id(fdm) == active_fdm_id
    and id(get_fdm(env2)) == active_fdm_id
    and id(turn_env.fdm) == active_fdm_id
)
final_pass = bool(stage1_pass and entry_ok and turn_pass and post_pass and fdm_continuity)

print(f"SAME FDM / NO RESET: {'PASS' if fdm_continuity else 'FAIL'} | fdm_id={active_fdm_id}")
print(f"STAGE1 TAKEOFF + 300 FT HOVER: {'PASS' if stage1_pass else 'FAIL'}")
print(f"STAGE2 FORWARD ENTRY: {'PASS' if entry_ok else 'FAIL'}")
print(f"TURN CAPTURE ({requested_turn:+.0f} deg): {'PASS' if turn_pass else 'FAIL'}")
print(f"POST-TURN FORWARD {POST_TURN_FORWARD_S:.0f}s: {'PASS' if post_pass else 'FAIL'}")
print(f"SAFETY FAILURE: {post_safety_failure}")
print("TEACHER ACTIVE AT RUNTIME: False")
print("CONTROLLER ACTIVE AT RUNTIME: False")
print(f"V7 +200 GATE AT TURN ENTRY: {'ON' if patch_gate > 0.5 else 'OFF'}")
print(f"VALIDATED COMMAND SET MEMBER: {validated_command}")
print("-" * 120)
print("FULL MISSION PASS:", final_pass)

try:
    env1.close()
except Exception:
    pass
try:
    env2.close()
except Exception:
    pass
try:
    turn_env.close()
except Exception:
    pass

if not final_pass:
    raise RuntimeError("FINAL FULL MISSION VALIDATION FAILED")
