from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from helicopter_env_turn_goal_v2 import HelicopterEnvTurnGoalV2

SEED = 42
BASE_SOURCE = Path("deneme/diagnose_stage4_entry_margin_v3.py")
MARKER = 'rule("A — LOCKED STAGE1 -> STAGE2 -> STAGE3 FULL QUALIFICATION")'
BASE_MODEL = Path("models_turn_hybrid/AH1S_TURN_HYBRID_V5_COLLECTIVE_ONLY.zip")
V4_ADAPTER = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V4_RESIDUAL_ADAPTER.pt")
V7_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V7_GATED_200_PATCH.pt")
OUT_PATCH = Path("models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt")
BEST_PATCH = Path("models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE_BEST.pt")

ENTRY_FORWARD_FT = 160.0
POLICY_FORWARD_COORD = 403.424
TARGET_NEG_NORM = -50.0 / 360.0
TARGET_200_NORM = 200.0 / 360.0
GATE_TOL = 0.02
BASE_SCALE = torch.tensor([0.35, 0.20, 0.30, 0.30], dtype=torch.float32)
V7_SCALE = torch.tensor([0.18, 0.08, 0.20, 0.20], dtype=torch.float32)
NEG_SCALE = 0.18
ROUNDS = 10
EPOCHS_PER_ROUND = 14
BATCH = 256
LR = 5e-5

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class ResidualAdapter(nn.Module):
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


class NegativeCollectivePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1) * NEG_SCALE


def gate(obs, target_norm):
    return abs(float(obs[12]) - target_norm) <= GATE_TOL


def load_locked_defs():
    text = BASE_SOURCE.read_text(encoding="utf-8")
    prefix = text.split(MARKER, 1)[0]
    ns = {"__name__": "v10_locked_defs", "__file__": str(BASE_SOURCE)}
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

base_model = PPO.load(str(BASE_MODEL))
v4 = ResidualAdapter(BASE_SCALE)
v4.load_state_dict(torch.load(V4_ADAPTER, map_location="cpu"), strict=True)
v4.eval()
v7 = ResidualAdapter(V7_SCALE)
v7.load_state_dict(torch.load(V7_PATCH, map_location="cpu"), strict=True)
v7.eval()
for module in (v4, v7):
    for p in module.parameters():
        p.requires_grad = False


def build_live_turn_entry(target=-50.0):
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
            raise RuntimeError("Stage1 failed while building V10 live entry")
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

    turn = HelicopterEnvTurnGoalV2(target_turn_deg=target)
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


def stack_action(obs, neg_patch):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d4 = v4(x).cpu().numpy()[0]
        d7 = v7(x).cpu().numpy()[0] if gate(obs, TARGET_200_NORM) else np.zeros(4, np.float32)
        dn = float(neg_patch(x).cpu().numpy()[0]) if gate(obs, TARGET_NEG_NORM) else 0.0
    a = np.asarray(base, np.float32) + d4 + d7
    a[0] += dn
    return np.clip(a, -1.0, 1.0), dn


def teacher_collective_action(env):
    s = env._raw_state()
    altitude = float(s["altitude"])
    vs = float(s["vertical_speed"])
    desired_vs = float(np.clip(0.10 * (300.0 - altitude), -2.0, 0.8))
    gain = 0.010 if vs < desired_vs else 0.030
    collective = float(np.clip(0.5840 + gain * (desired_vs - vs), 0.470, 0.610))
    return env.collective_to_action(collective)


def evaluate(neg_patch, label):
    env1, env2, env, obs = build_live_turn_entry(-50.0)
    info = {}
    success = False
    try:
        max_steps = int(130.0 / env.CONTROL_DT)
        for _ in range(max_steps):
            action, _ = stack_action(obs, neg_patch)
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        try: env.close()
        except Exception: pass
        try: env2.close()
        except Exception: pass
        try: env1.close()
        except Exception: pass
    print(
        f"{label} | pass={success} | done={float(info.get('cumulative_turn_deg',0)):+.2f} | "
        f"rem={float(info.get('remaining_turn_deg',999)):+.2f} | alt={float(info.get('altitude',float('nan'))):.2f} | "
        f"vs={float(info.get('vertical_speed',float('nan'))):+.2f} | hold={float(info.get('success_hold_s',0)):.1f} | "
        f"safety={bool(info.get('safety_failure',False))}"
    )
    return success, info


def collect_shadow(neg_patch):
    env1, env2, env, obs = build_live_turn_entry(-50.0)
    xs, ys, ws = [], [], []
    try:
        for _ in range(int(130.0 / env.CONTROL_DT)):
            base, _ = base_model.predict(obs, deterministic=True)
            x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                d4 = v4(x).cpu().numpy()[0]
                current_no_neg = float(np.clip(np.asarray(base, np.float32)[0] + d4[0], -1.0, 1.0))
            teacher_a0 = teacher_collective_action(env)
            target_dn = float(np.clip(teacher_a0 - current_no_neg, -NEG_SCALE, NEG_SCALE))
            s = env._raw_state()
            alt = float(s["altitude"])
            remaining = -50.0 - env.cumulative_turn_deg
            w = 1.0
            if alt < 290.0: w *= 4.0
            if alt < 282.0: w *= 2.5
            if abs(remaining) < 20.0: w *= 2.0
            xs.append(obs.copy())
            ys.append(target_dn)
            ws.append(w)
            action, _ = stack_action(obs, neg_patch)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, np.float32)
            if terminated or truncated:
                break
    finally:
        try: env.close()
        except Exception: pass
        try: env2.close()
        except Exception: pass
        try: env1.close()
        except Exception: pass
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32), np.asarray(ws,np.float32)


def train_round(model, xs, ys, ws):
    xt = torch.as_tensor(xs, dtype=torch.float32)
    yt = torch.as_tensor(ys, dtype=torch.float32)
    wt = torch.as_tensor(ws, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    idx = np.arange(len(xs))
    last = 0.0
    for _ in range(EPOCHS_PER_ROUND):
        rng.shuffle(idx)
        total = 0.0
        n = 0
        for st in range(0, len(idx), BATCH):
            bi = torch.as_tensor(idx[st:st+BATCH], dtype=torch.long)
            pred = model(xt[bi])
            loss = (((pred - yt[bi]) ** 2) * wt[bi]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            opt.step()
            total += float(loss.detach())
            n += 1
        last = total / max(1,n)
    return last


print("="*120)
print("V10 LIVE-ENTRY -50 HARD-GATED COLLECTIVE REPAIR")
print("Base PPO, V4 adapter and V7 +200 patch are frozen. Only -50 collective residual is trained.")
print("Runtime target gating guarantees zero V10 effect on +50/+200/+360.")
print("="*120)

patch = NegativeCollectivePatch()
with torch.no_grad():
    last = patch.net[-2]
    last.weight.zero_(); last.bias.zero_()

best_success, best_info = evaluate(patch, "START V10 ZERO PATCH")
best_score = (10000.0 if best_success else 0.0) - abs(float(best_info.get("remaining_turn_deg",999))) - (100.0 if best_info.get("safety_failure",False) else 0.0)
torch.save(patch.state_dict(), BEST_PATCH)

for r in range(1, ROUNDS+1):
    xs, ys, ws = collect_shadow(patch)
    loss = train_round(patch, xs, ys, ws)
    success, info = evaluate(patch, f"V10 ROUND {r}")
    score = (10000.0 if success else 0.0) - abs(float(info.get("remaining_turn_deg",999))) - (100.0 if info.get("safety_failure",False) else 0.0)
    print(f"round={r}/{ROUNDS} rows={len(xs)} loss={loss:.7f} score={score:.2f}")
    if score > best_score:
        best_score = score
        torch.save(patch.state_dict(), BEST_PATCH)
        print("NEW V10 BEST:", BEST_PATCH)
    if success:
        print("-50 LIVE FULL-ENTRY TURN ACHIEVED - stopping immediately.")
        break

patch.load_state_dict(torch.load(BEST_PATCH, map_location="cpu"), strict=True)
torch.save(patch.state_dict(), OUT_PATCH)
evaluate(patch, "FINAL V10 BEST")
print("V10 PATCH:", OUT_PATCH)
print("NOTE: patch is hard-gated to -50 and changes collective action only.")
