from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from helicopter_env_turn_goal_full_entry import HelicopterEnvTurnGoalFullEntry

SEED=42
START=Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V2_HEAD_REPAIRED.zip')
OUT=Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V3_CHANNEL_SELECTIVE')
TARGETS=[-50.0,50.0,200.0,360.0]
BATCH=256; LR=1e-5
rng=np.random.default_rng(SEED)
np.random.seed(SEED); torch.manual_seed(SEED)

def mteacher(s,rem):
    yr=math.degrees(s['r_rate']); roll=math.degrees(s['roll'])
    dyr=float(np.clip(1.2*rem,-1.5,1.5)); rud=-1.6*(dyr-yr)
    rud=float(np.clip(rud,-1,.2) if rem>=0 else np.clip(rud,-.2,1))
    dr=float(np.clip(.25*rem,-5,5)); ail=float(np.clip(.35*(dr-roll),-1,1))
    return ail,rud

def holdteacher(s,rem,integ):
    yr=math.degrees(s['r_rate']); roll=math.degrees(s['roll'])
    ail=float(np.clip(.45*(0-roll),-1,1))
    dyr=float(np.clip(.22*rem+.03*integ,-1,1))
    rud=float(np.clip(-.75*(dyr-yr),-1,1)); return ail,rud

def collective(target,alt,vs):
    if target<0:
        dvs=float(np.clip(.10*(300-alt),-2,.8)); g=.010 if vs<dvs else .030
        return float(np.clip(.584+g*(dvs-vs),.470,.610))
    dvs=float(np.clip(.08*(300-alt),-2.5,.15)); g=.045 if vs>dvs else .004
    return float(np.clip(.578+g*(dvs-vs),.460,.590))

def teacher(env,captured,hold,integ):
    s=env._raw_state(); rem=env.target_turn_deg-env.cumulative_turn_deg
    a0=env.collective_to_action(collective(env.target_turn_deg,s['altitude'],s['vertical_speed']))
    a1=float(np.clip(.35*(14.5-s['forward_velocity']),-1,1))
    ok=abs(rem)<=.75 and abs(math.degrees(s['roll']))<=3
    hold=hold+env.CONTROL_DT if ok else 0.0
    captured=captured or hold>=.75
    if captured and env.target_turn_deg>0:
        integ=float(np.clip(integ+np.clip(rem,-5,5)*env.CONTROL_DT,-20,20)); a2,a3=holdteacher(s,rem,integ)
    else: a2,a3=mteacher(s,rem)
    return np.array([a0,a1,a2,a3],np.float32),captured,hold,integ

def eval_one(model,target):
    env=HelicopterEnvTurnGoalFullEntry(target_turn_deg=target); obs,info=env.reset(seed=SEED); success=False
    try:
        for _ in range(int(max(90,abs(target)/1.1+70)/env.CONTROL_DT)):
            a,_=model.predict(obs,deterministic=True); obs,_,term,trunc,info=env.step(a)
            if term or trunc: success=bool(info.get('success',False)); break
    finally: env.close()
    return dict(target=target,success=success,done=float(info.get('cumulative_turn_deg',0)),rem=float(info.get('remaining_turn_deg',999)),alt=float(info.get('altitude',0)),safety=bool(info.get('safety_failure',False)))

def eval_all(model,label):
    rows=[eval_one(model,t) for t in TARGETS]; p=sum(r['success'] for r in rows)
    print('\n'+label,'PASS=',p,'/4')
    for r in rows: print(r)
    return p

def collect(target,n=2):
    O=[]; A=[]; W=[]
    for rid in range(n):
        env=HelicopterEnvTurnGoalFullEntry(target_turn_deg=target); obs,_=env.reset(seed=SEED+rid); c=False; h=0.; integ=0.
        try:
            for _ in range(int(max(90,abs(target)/1.1+70)/env.CONTROL_DT)):
                a,c,h,integ=teacher(env,c,h,integ); rem=env.target_turn_deg-env.cumulative_turn_deg
                w=8. if abs(rem)<8 else (3. if abs(rem)<25 else 1.)
                O.append(np.asarray(obs,np.float32).copy()); A.append(a.copy()); W.append(w)
                obs,_,term,trunc,_=env.step(a)
                if term or trunc: break
        finally: env.close()
    return np.asarray(O,np.float32),np.asarray(A,np.float32),np.asarray(W,np.float32)

def train_rows(model,O,A,W,rows,epochs,label):
    dev=model.device; ot=torch.as_tensor(O,dtype=torch.float32,device=dev); at=torch.as_tensor(A,dtype=torch.float32,device=dev); wt=torch.as_tensor(W,dtype=torch.float32,device=dev).reshape(-1,1)
    for p in model.policy.parameters(): p.requires_grad=False
    for p in model.policy.action_net.parameters(): p.requires_grad=True
    opt=torch.optim.Adam(model.policy.action_net.parameters(),lr=LR); ids=np.arange(len(O))
    for ep in range(1,epochs+1):
        rng.shuffle(ids)
        for st in range(0,len(ids),BATCH):
            ix=torch.as_tensor(ids[st:st+BATCH],dtype=torch.long,device=dev)
            with torch.no_grad():
                f=model.policy.extract_features(ot[ix]); f=f[0] if isinstance(f,tuple) else f; lat=model.policy.mlp_extractor.forward_actor(f)
            pred=model.policy.action_net(lat); mask=torch.zeros((1,4),device=dev)
            for r in rows: mask[0,r]=1
            loss=(((pred-at[ix])**2)*mask*wt[ix]).mean(); opt.zero_grad(set_to_none=True); loss.backward()
            if model.policy.action_net.weight.grad is not None:
                gm=torch.zeros_like(model.policy.action_net.weight.grad)
                for r in rows: gm[r]=1
                model.policy.action_net.weight.grad.mul_(gm)
            if model.policy.action_net.bias.grad is not None:
                gb=torch.zeros_like(model.policy.action_net.bias.grad)
                for r in rows: gb[r]=1
                model.policy.action_net.bias.grad.mul_(gb)
            torch.nn.utils.clip_grad_norm_(model.policy.action_net.parameters(),.15); opt.step()
        print(f'{label} epoch={ep}')
        if eval_all(model,f'{label} EPOCH {ep}')==4:
            model.save(str(OUT)); return True
        model.save(str(OUT))
    return False

if not START.exists(): raise FileNotFoundError(START)
print('='*120); print('FULL-ENTRY V3 CHANNEL-SELECTIVE REPAIR'); print('runtime teacher/controller OFF'); print('='*120)
model=PPO.load(str(START)); eval_all(model,'START V3')

O50,A50,W50=collect(50,3); O200,A200,W200=collect(200,1); O360,A360,W360=collect(360,1)
O=np.concatenate([O50,O200,O360]); A=np.concatenate([A50,A200,A360]); W=np.concatenate([W50*4,W200*.35,W360*.35])
print('\nPHASE1 collective-only +50 repair')
if not train_rows(model,O,A,W,[0],25,'P1'):
    Om,Am,Wm=collect(-50,1); O50,A50,W50=collect(50,1); O200,A200,W200=collect(200,3); O360,A360,W360=collect(360,2)
    O=np.concatenate([O200,Om,O360,O50]); A=np.concatenate([A200,Am,A360,A50]); W=np.concatenate([W200*4,Wm*.8,W360*1.5,W50])
    print('\nPHASE2 aileron/rudder-only +200 repair')
    train_rows(model,O,A,W,[2,3],40,'P2')

best=PPO.load(str(OUT)); p=eval_all(best,'FINAL V3'); print('\nFINAL FULL-ENTRY V3 PASS COUNT:',p,'/ 4'); print('FINAL MODEL:',OUT)
