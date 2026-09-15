from __future__ import annotations

from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
TARGETS = [-50.0, 50.0, 200.0, 360.0]
BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
ENTRY_FORWARD_FT = 160.0
POLICY_FORWARD_COORD = 403.424


def load_locked_defs():
    text = BASE_SOURCE.read_text(encoding="utf-8")
    prefix = text.split(MARKER, 1)[0]
    ns = {"__name__": "v5_live_diag_defs", "__file__": str(BASE_SOURCE)}
    exec(compile(prefix, str(BASE_SOURCE), "exec"), ns)
    return ns


ns = load_locked_defs()
Env1 = ns["HelicopterEnvStage1Distill"]
Env2 = ns["HelicopterEnvStage2RefineMapped"]
stage1_model = ns["stage1_model"]
stage2_model = ns["stage2_model"]
get_fdm = ns["get_fdm"]
env_control_dt = ns["env_control_dt"]
info_float = ns["info_float"]
AILERON_SCALE = ns["AILERON_SCALE"]
RUDDER_SCALE = ns["RUDDER_SCALE"]
HANDOFF_STABLE_TIME = ns["HANDOFF_STABLE_TIME"]
STAGE1_MAX_TIME = ns["STAGE1_MAX_TIME"]

if not BASE_MODEL.exists():
    raise FileNotFoundError(BASE_MODEL)

base_model = PPO.load(str(BASE_MODEL))


def build_live_turn_entry(target):
    env1 = Env1(teacher_model_path=None, training_mode=False)
    obs1, _ = env1.reset(seed=SEED)
    fdm = get_fdm(env1)
    dt1 = env_control_dt(env1)
    stable_time = 0.0

    for _ in range(int(STAGE1_MAX_TIME / dt1)):
        action, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info = env1.step(action)
        alt = info_float(info, "altitude")
        vs = info_float(info, "vertical_speed")
        vn = info_float(info, "vn", 0.0)
        ve = info_float(info, "ve", 0.0)
        drift = info_float(info, "drift", 999.0)
        stable = (
            295.0 <= alt <= 305.0
            and abs(vs) <= 0.5
            and float(np.hypot(vn, ve)) <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt1 if stable else 0.0
        if stable_time >= HANDOFF_STABLE_TIME:
            break
        if truncated or (terminated and not bool(info.get("success", False))):
            raise RuntimeError("Stage1 failed while building live entry")

    if stable_time < HANDOFF_STABLE_TIME:
        raise RuntimeError("Stage1 hover handoff not reached")

    env2 = Env2(aileron_scale=AILERON_SCALE, rudder_scale=RUDDER_SCALE)
    env2.reset(seed=SEED)
    env2.fdm = fdm
    env2.phase = 1
    env2.forward_distance = 0.0
    env2.previous_action = np.zeros(4, dtype=np.float32)
    env2.steps = 0
    fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    fdm["ap/afcs/yaw-channel-active-norm"] = 1.0
    obs2 = np.asarray(env2._get_obs(), np.float32)
    dt2 = env_control_dt(env2)

    for _ in range(int(70.0 / dt2)):
        action, _ = stage2_model.predict(obs2, deterministic=True)
        obs2, _, terminated, truncated, _ = env2.step(action)
        obs2 = np.asarray(obs2, np.float32)
        if env2.forward_distance >= ENTRY_FORWARD_FT:
            break
        if terminated or truncated:
            raise RuntimeError("Stage2 failed before live turn entry")

    if env2.forward_distance < ENTRY_FORWARD_FT:
        raise RuntimeError("Stage2 did not reach live turn entry")

    turn = HelicopterEnvTurnGoalV2(target_turn_deg=float(target))
    turn.fdm = fdm
    turn.phase = 1
    turn.turn_active = True
    turn.target_turn_deg = float(target)
    turn.cumulative_turn_deg = 0.0
    turn.prev_heading_deg = turn._heading_deg()
    turn.turn_steps = 0
    turn.success_hold_s = 0.0
    turn.previous_turn_action = np.zeros(4, dtype=np.float32)
    turn.forward_distance = POLICY_FORWARD_COORD
    if hasattr(turn, "_v2_previous_remaining"):
        turn._v2_previous_remaining = float(target)
    turn.fdm["ap/afcs/roll-channel-active-norm"] = turn.TURN_ROLL_AFCS
    turn.fdm["ap/afcs/yaw-channel-active-norm"] = turn.TURN_YAW_AFCS
    obs = np.asarray(turn._get_obs(), np.float32)
    return env1, env2, turn, obs


def evaluate_target(target):
    env1, env2, env, obs = build_live_turn_entry(target)
    info = {}
    success = False
    try:
        max_steps = int(max(130.0, abs(target) / 1.0 + 70.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = base_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        for obj in (env, env2, env1):
            try:
                obj.close()
            except Exception:
                pass

    print(
        f"target={target:+7.1f} | pass={str(success):5s} | "
        f"done={float(info.get('cumulative_turn_deg', 0.0)):+8.2f} | "
        f"rem={float(info.get('remaining_turn_deg', 999.0)):+7.2f} | "
        f"alt={float(info.get('altitude', float('nan'))):7.2f} | "
        f"vs={float(info.get('vertical_speed', float('nan'))):+6.2f} | "
        f"v={float(info.get('forward_velocity', float('nan'))):6.2f} | "
        f"hold={float(info.get('success_hold_s', 0.0)):4.1f} | "
        f"safety={bool(info.get('safety_failure', False))}"
    )
    return success


print("=" * 120)
print("V5 BASE PPO — TRUE LIVE STAGE1 -> STAGE2 TURN ENTRY DIAGNOSTIC")
print("No V4/V7/V10 residuals. Runtime teacher/controller OFF.")
print("=" * 120)

passes = 0
for target in TARGETS:
    passes += int(evaluate_target(target))

print(f"\nV5 LIVE-ENTRY PASS COUNT: {passes}/4")
