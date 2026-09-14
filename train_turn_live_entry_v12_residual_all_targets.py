from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
TARGETS = [-50.0, 50.0, 200.0, 360.0]
BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
OUT_DIR = Path("models_turn_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ADAPTER = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt"
BEST_ADAPTER = OUT_DIR / "AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER_BEST.pt"

ENTRY_FORWARD_FT = 160.0
POLICY_FORWARD_COORD = 403.424
RESIDUAL_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
ROUNDS = 18
EPOCHS_PER_ROUND = 12
BATCH = 256
LR = 7e-5

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * RESIDUAL_SCALE.to(obs.device)


def load_locked_defs():
    text = BASE_SOURCE.read_text(encoding="utf-8")
    prefix = text.split(MARKER, 1)[0]
    ns = {"__name__": "v12_locked_defs", "__file__": str(BASE_SOURCE)}
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
        a, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info = env1.step(a)
        alt = info_float(info, "altitude")
        vs = info_float(info, "vertical_speed")
        vn = info_float(info, "vn", 0.0)
        ve = info_float(info, "ve", 0.0)
        drift = info_float(info, "drift", 999.0)
        stable = 295.0 <= alt <= 305.0 and abs(vs) <= 0.5 and float(np.hypot(vn, ve)) <= 1.0 and drift <= 3.0
        stable_time = stable_time + dt1 if stable else 0.0
        if stable_time >= HANDOFF_STABLE_TIME:
            break
        if truncated or (terminated and not bool(info.get("success", False))):
            raise RuntimeError("Stage1 failed while building live entry")
    if stable_time < HANDOFF_STABLE_TIME:
        raise RuntimeError("Stage1 hover not reached")

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
        a, _ = stage2_model.predict(obs2, deterministic=True)
        obs2, _, terminated, truncated, _ = env2.step(a)
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
    return env1, env2, turn, np.asarray(turn._get_obs(), np.float32)


def maneuver_teacher(state, remaining):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    desired_yaw_rate = float(np.clip(1.20 * remaining, -1.50, +1.50))
    rudder = -1.60 * (desired_yaw_rate - yaw_rate)
    if remaining >= 0.0:
        rudder = float(np.clip(rudder, -1.0, +0.20))
    else:
        rudder = float(np.clip(rudder, -0.20, +1.0))
    desired_roll = float(np.clip(0.25 * remaining, -5.0, +5.0))
    aileron = float(np.clip(0.35 * (desired_roll - roll), -1.0, +1.0))
    return aileron, rudder


def positive_terminal_hold(state, remaining, heading_integral):
    yaw_rate = math.degrees(state["r_rate"])
    roll = math.degrees(state["roll"])
    aileron = float(np.clip(0.45 * (0.0 - roll), -1.0, +1.0))
    desired_yaw_rate = float(np.clip(0.22 * remaining + 0.030 * heading_integral, -1.0, +1.0))
    rudder = float(np.clip(-0.75 * (desired_yaw_rate - yaw_rate), -1.0, +1.0))
    return aileron, rudder


def teacher_collective(target, altitude, vertical_speed):
    if target < 0.0:
        desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.00, +0.80))
        gain = 0.010 if vertical_speed < desired_vs else 0.030
        return float(np.clip(0.5840 + gain * (desired_vs - vertical_speed), 0.470, 0.610))
    desired_vs = float(np.clip(0.08 * (300.0 - altitude), -2.50, +0.15))
    gain = 0.045 if vertical_speed > desired_vs else 0.004
    return float(np.clip(0.5780 + gain * (desired_vs - vertical_speed), 0.460, 0.590))


def teacher_action(env, captured, capture_hold_s, heading_integral):
    s = env._raw_state()
    remaining = env.target_turn_deg - env.cumulative_turn_deg
    a0 = env.collective_to_action(teacher_collective(env.target_turn_deg, float(s["altitude"]), float(s["vertical_speed"])))
    a1 = float(np.clip(0.35 * (14.5 - float(s["forward_velocity"])), -1.0, +1.0))
    roll_deg = abs(math.degrees(s["roll"]))
    capture_ok = abs(remaining) <= 0.75 and roll_deg <= 3.0
    capture_hold_s = capture_hold_s + env.CONTROL_DT if capture_ok else 0.0
    if capture_hold_s >= 0.75:
        captured = True
    if captured and env.target_turn_deg > 0.0:
        heading_integral += float(np.clip(remaining, -5.0, +5.0)) * env.CONTROL_DT
        heading_integral = float(np.clip(heading_integral, -20.0, +20.0))
        a2, a3 = positive_terminal_hold(s, remaining, heading_integral)
    else:
        a2, a3 = maneuver_teacher(s, remaining)
    return np.array([a0, a1, a2, a3], dtype=np.float32), captured, capture_hold_s, heading_integral


def residual_action(adapter, obs):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d = adapter(x).cpu().numpy()[0]
    return np.clip(np.asarray(base, np.float32) + d, -1.0, 1.0)


def evaluate_one(adapter, target):
    env1, env2, env, obs = build_live_turn_entry(target)
    info = {}
    success = False
    try:
        max_steps = int(max(130.0, abs(target) / 1.10 + 80.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            action = residual_action(adapter, obs)
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        for e in (env, env2, env1):
            try: e.close()
            except Exception: pass
    return {
        "target": target,
        "success": success,
        "done": float(info.get("cumulative_turn_deg", 0.0)),
        "rem": float(info.get("remaining_turn_deg", 999.0)),
        "alt": float(info.get("altitude", float("nan"))),
        "vs": float(info.get("vertical_speed", float("nan"))),
        "hold": float(info.get("success_hold_s", 0.0)),
        "safety": bool(info.get("safety_failure", False)),
    }


def evaluate_all(adapter, label):
    rows = [evaluate_one(adapter, t) for t in TARGETS]
    passes = sum(int(r["success"]) for r in rows)
    safety = sum(int(r["safety"]) for r in rows)
    err = sum(abs(r["rem"]) for r in rows)
    score = passes * 10000.0 - safety * 500.0 - err
    print(f"\n{label} | PASS={passes}/4 | score={score:.2f}")
    for r in rows:
        print(f"  target={r['target']:+7.1f} | pass={str(r['success']):5s} | done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | vs={r['vs']:+6.2f} | hold={r['hold']:4.1f} | safety={r['safety']}")
    return rows, passes, score


def collect_shadow(adapter, target):
    env1, env2, env, obs = build_live_turn_entry(target)
    xs, ys, ws = [], [], []
    captured = False
    capture_hold_s = 0.0
    heading_integral = 0.0
    try:
        max_steps = int(max(130.0, abs(target) / 1.10 + 80.0) / env.CONTROL_DT)
        for _ in range(max_steps):
            teacher, captured, capture_hold_s, heading_integral = teacher_action(env, captured, capture_hold_s, heading_integral)
            base, _ = base_model.predict(obs, deterministic=True)
            desired_delta = np.clip(teacher - np.asarray(base, np.float32), -RESIDUAL_SCALE.numpy(), RESIDUAL_SCALE.numpy())
            remaining = env.target_turn_deg - env.cumulative_turn_deg
            s = env._raw_state()
            alt = float(s["altitude"])
            w = 1.0
            if abs(remaining) < 25.0: w *= 3.0
            if abs(remaining) < 8.0: w *= 2.0
            if alt < 288.0 or alt > 312.0: w *= 2.0
            xs.append(obs.copy())
            ys.append(desired_delta.astype(np.float32))
            ws.append(float(w))
            action = residual_action(adapter, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                break
    finally:
        for e in (env, env2, env1):
            try: e.close()
            except Exception: pass
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ws, np.float32)


def train(adapter, xs, ys, ws):
    xt = torch.as_tensor(xs, dtype=torch.float32)
    yt = torch.as_tensor(ys, dtype=torch.float32)
    wt = torch.as_tensor(ws, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR)
    idx = np.arange(len(xs))
    channel_w = torch.tensor([2.5, 1.0, 2.0, 2.0], dtype=torch.float32).reshape(1, 4)
    last = 0.0
    for _ in range(EPOCHS_PER_ROUND):
        rng.shuffle(idx)
        total = 0.0
        n = 0
        for st in range(0, len(idx), BATCH):
            bi = torch.as_tensor(idx[st:st+BATCH], dtype=torch.long)
            pred = adapter(xt[bi])
            loss = (((pred - yt[bi]) ** 2) * wt[bi] * channel_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.35)
            opt.step()
            total += float(loss.detach())
            n += 1
        last = total / max(1, n)
    return last


print("=" * 120)
print("V12 LIVE-ENTRY ALL-TARGET RESIDUAL TRAINING")
print("Frozen V5 PPO. Residual learns only on real Stage1 -> Stage2 live handoff states.")
print("Teacher labels training states only; runtime teacher/controller remains OFF.")
print("=" * 120)

adapter = ResidualAdapter()
with torch.no_grad():
    last = adapter.net[-2]
    last.weight.zero_()
    last.bias.zero_()

rows, best_passes, best_score = evaluate_all(adapter, "START V12 ZERO RESIDUAL")
torch.save(adapter.state_dict(), BEST_ADAPTER)

for round_idx in range(1, ROUNDS + 1):
    x_all, y_all, w_all = [], [], []
    for target in TARGETS:
        x, y, w = collect_shadow(adapter, target)
        x_all.append(x); y_all.append(y); w_all.append(w)
    xs = np.concatenate(x_all, axis=0)
    ys = np.concatenate(y_all, axis=0)
    ws = np.concatenate(w_all, axis=0)
    loss = train(adapter, xs, ys, ws)
    rows, passes, score = evaluate_all(adapter, f"V12 ROUND {round_idx}")
    print(f"round={round_idx}/{ROUNDS} rows={len(xs)} loss={loss:.7f} score={score:.2f}")
    if passes > best_passes or (passes == best_passes and score > best_score):
        best_passes, best_score = passes, score
        torch.save(adapter.state_dict(), BEST_ADAPTER)
        print("NEW V12 BEST:", BEST_ADAPTER)
    if passes == 4:
        print("4/4 LIVE-ENTRY TURN ACHIEVED - stopping immediately.")
        break

adapter.load_state_dict(torch.load(BEST_ADAPTER, map_location="cpu"), strict=True)
torch.save(adapter.state_dict(), OUT_ADAPTER)
evaluate_all(adapter, "FINAL V12 BEST")
print("V12 ADAPTER:", OUT_ADAPTER)
print("NOTE: runtime teacher/controller remains OFF; V5 PPO is frozen.")
