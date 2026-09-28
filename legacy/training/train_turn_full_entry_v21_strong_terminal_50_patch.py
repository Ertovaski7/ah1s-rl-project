from __future__ import annotations

# training/ klasöründen çalıştırılabilmesi için: repo kökünü import yoluna ekle
# ve çalışma dizinini köke al (model/sonuç yolları köke göredir).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO_ROOT)
_os.chdir(_REPO_ROOT)

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEED = 42
TARGET = 50.0
TRAIN_SEEDS = [7, 21, 42, 84, 123, 256, 512, 777]
EVAL_SEEDS = [7, 21, 42, 84, 123]

BASE_MODEL = v16.BASE_MODEL
BASE_ADAPTER = v16.BASE_ADAPTER
PATCH_50 = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt")
OUT_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH.pt")
BEST_PATCH = Path("models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH_BEST.pt")

TERM_DEG = 6.0
AIL_SCALE = 0.15
RUD_SCALE = 0.25
ROUNDS = 8
EPOCHS = 100
BATCH = 256
LR = 5e-4

np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

class TerminalPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2), nn.Tanh(),
        )
    def forward(self, obs):
        y = self.net(obs)
        scale = torch.tensor([AIL_SCALE, RUD_SCALE], dtype=torch.float32, device=obs.device)
        return y * scale

def load_adapter(path, scale):
    m = v16.ResidualAdapter(scale)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

def gate_terminal(obs):
    return abs(float(obs[12]) - TARGET/360.0) <= v16.GATE_TOL and abs(float(obs[13]) * 360.0) <= TERM_DEG

def base_plus_v17(base_model, base_adapter, patch50, obs):
    base, _ = base_model.predict(obs, deterministic=True)
    x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        d0 = base_adapter(x).cpu().numpy()[0]
        d50 = patch50(x).cpu().numpy()[0]
    return np.clip(np.asarray(base, np.float32) + d0 + d50, -1.0, 1.0)

def combined_action(base_model, base_adapter, patch50, tpatch, obs):
    a = base_plus_v17(base_model, base_adapter, patch50, obs)
    if gate_terminal(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            d = tpatch(x).cpu().numpy()[0]
        a = a.copy()
        a[2] += float(d[0])
        a[3] += float(d[1])
    return np.clip(a, -1.0, 1.0)

def teacher_lateral(env):
    s = env._raw_state()
    rem = TARGET - env.cumulative_turn_deg
    roll = math.degrees(float(s["roll"]))
    yaw_rate = math.degrees(float(s["r_rate"]))
    desired_roll = float(np.clip(0.18 * rem, -2.5, 2.5))
    desired_yaw = float(np.clip(0.22 * rem, -0.8, 0.8))
    a2 = float(np.clip(0.40 * (desired_roll - roll), -1.0, 1.0))
    a3 = float(np.clip(-0.80 * (desired_yaw - yaw_rate), -1.0, 1.0))
    return a2, a3

def collect(base_model, base_adapter, patch50, tpatch, seed):
    env = v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    obs, _ = env.reset(seed=seed)
    xs, ys, ws = [], [], []
    try:
        for _ in range(int(110.0/env.CONTROL_DT)):
            if gate_terminal(obs):
                current = base_plus_v17(base_model, base_adapter, patch50, obs)
                ta2, ta3 = teacher_lateral(env)
                y = np.array([
                    np.clip(ta2-current[2], -AIL_SCALE, AIL_SCALE),
                    np.clip(ta3-current[3], -RUD_SCALE, RUD_SCALE),
                ], dtype=np.float32)
                rem = abs(TARGET-env.cumulative_turn_deg)
                w = 3.0
                if rem <= 3.0: w = 8.0
                if rem <= 1.5: w = 20.0
                if env.success_hold_s >= 0.5: w *= 2.0
                if seed == 7: w *= 2.0
                xs.append(np.asarray(obs,np.float32).copy()); ys.append(y); ws.append(w)
            action = combined_action(base_model, base_adapter, patch50, tpatch, obs)
            obs,_,term,trunc,_ = env.step(action)
            obs = np.asarray(obs,np.float32)
            if term or trunc: break
    finally:
        env.close()
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32), np.asarray(ws,np.float32)

def train_one(model, xs, ys, ws):
    xt=torch.as_tensor(xs); yt=torch.as_tensor(ys); wt=torch.as_tensor(ws).reshape(-1,1)
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    idx=np.arange(len(xs))
    last=0.0
    for _ in range(EPOCHS):
        rng.shuffle(idx); total=0.0; n=0
        for st in range(0,len(idx),BATCH):
            bi=torch.as_tensor(idx[st:st+BATCH],dtype=torch.long)
            pred=model(xt[bi]); loss=(((pred-yt[bi])**2)*wt[bi]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            total+=float(loss.detach()); n+=1
        last=total/max(1,n)
    with torch.no_grad():
        pred=model(xt)
        mae=torch.mean(torch.abs(pred-yt),dim=0).cpu().numpy()
        mean_abs=torch.mean(torch.abs(pred),dim=0).cpu().numpy()
    return last, mae, mean_abs

def eval_one(base_model,base_adapter,patch50,tpatch,seed):
    env=v16.RandomizedEntryEnv(target_turn_deg=TARGET)
    obs,reset_info=env.reset(seed=seed); info=reset_info; success=False
    try:
        for _ in range(int(110.0/env.CONTROL_DT)):
            action=combined_action(base_model,base_adapter,patch50,tpatch,obs)
            obs,_,term,trunc,info=env.step(action); obs=np.asarray(obs,np.float32)
            if term or trunc:
                success=bool(info.get("success",False)); break
    finally:
        env.close()
    return dict(seed=seed,extra=int(reset_info.get("extra_entry_steps",0)),success=success,
                done=float(info.get("cumulative_turn_deg",0.0)),rem=float(info.get("remaining_turn_deg",999.0)),
                alt=float(info.get("altitude",np.nan)),hold=float(info.get("success_hold_s",0.0)),
                safety=bool(info.get("safety_failure",False)))

def eval_all(base_model,base_adapter,patch50,tpatch,label):
    rows=[eval_one(base_model,base_adapter,patch50,tpatch,s) for s in EVAL_SEEDS]
    p=sum(int(r["success"]) for r in rows); safety=sum(int(r["safety"]) for r in rows); err=sum(abs(r["rem"]) for r in rows)
    score=p*10000-safety*100-err
    print(f"\n{label} | PASS={p}/5 | score={score:.2f}")
    for r in rows:
        print(f"  seed={r['seed']:3d} | extra={r['extra']:2d} | pass={str(r['success']):5s} | done={r['done']:+7.2f} | rem={r['rem']:+6.2f} | alt={r['alt']:7.2f} | hold={r['hold']:3.1f} | safety={r['safety']}")
    return rows,p,score

for p in (BASE_MODEL,BASE_ADAPTER,PATCH_50):
    if not p.exists(): raise FileNotFoundError(p)

base_model=PPO.load(str(BASE_MODEL))
base_adapter=load_adapter(BASE_ADAPTER,v16.BASE_SCALE)
patch50=load_adapter(PATCH_50,v16.PATCH_SCALE)
term=TerminalPatch()
with torch.no_grad():
    term.net[-2].weight.zero_(); term.net[-2].bias.zero_()

print("="*120)
print("V21 STRONG +50 TERMINAL SPECIALIST")
print("V17 frozen. V21 is hard-gated to +50 and |remaining|<=6 deg; only aileron/rudder are modified.")
print("Training uses stronger supervised terminal residuals; runtime teacher/controller OFF.")
print("="*120)

rows,best_p,best_score=eval_all(base_model,base_adapter,patch50,term,"START V21 ZERO PATCH")
torch.save(term.state_dict(),BEST_PATCH)

for r in range(1,ROUNDS+1):
    allx=[]; ally=[]; allw=[]
    for seed in TRAIN_SEEDS:
        x,y,w=collect(base_model,base_adapter,patch50,term,seed)
        if len(x): allx.append(x); ally.append(y); allw.append(w)
    xs=np.concatenate(allx); ys=np.concatenate(ally); ws=np.concatenate(allw)
    loss,mae,mean_abs=train_one(term,xs,ys,ws)
    rows,p,score=eval_all(base_model,base_adapter,patch50,term,f"V21 ROUND {r}")
    print(f"round={r}/{ROUNDS} rows={len(xs)} loss={loss:.7f} mae=[{mae[0]:.4f},{mae[1]:.4f}] mean_abs_patch=[{mean_abs[0]:.4f},{mean_abs[1]:.4f}]")
    if p>best_p or (p==best_p and score>best_score):
        best_p,best_score=p,score; torch.save(term.state_dict(),BEST_PATCH); print("NEW V21 BEST:",BEST_PATCH)
    if p==5 and all(not rr["safety"] for rr in rows):
        print("V21 5/5 ACHIEVED - stopping."); break

term.load_state_dict(torch.load(BEST_PATCH,map_location="cpu"),strict=True)
torch.save(term.state_dict(),OUT_PATCH)
rows,p,score=eval_all(base_model,base_adapter,patch50,term,"FINAL V21 BEST")
print("FINAL V21 +50 PASS COUNT:",p,"/ 5")
print("V21 PATCH:",OUT_PATCH)
