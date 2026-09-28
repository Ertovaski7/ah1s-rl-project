from pathlib import Path
import numpy as np, torch, torch.nn as nn
from stable_baselines3 import PPO
import test_turn_full_entry_v16_randomized_entry_robustness as v16

SEEDS=[7,21,42,84,123]; TARGETS=[-50.0,50.0,200.0,360.0]
T50=50.0/360.0; T200=200.0/360.0; TERM=6.0
P50=Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V17_ROBUST_50_PATCH.pt')
P21=Path('models_turn_hybrid/AH1S_TURN_FULL_ENTRY_V21_STRONG_TERMINAL_50_PATCH.pt')

class TP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2),nn.Tanh())
    def forward(self,x):
        return self.net(x)*torch.tensor([0.15,0.25],dtype=torch.float32,device=x.device)

def load(path,scale):
    m=v16.ResidualAdapter(scale); m.load_state_dict(torch.load(path,map_location='cpu'),strict=True); m.eval()
    for p in m.parameters(): p.requires_grad=False
    return m

def is_t(obs,t): return abs(float(obs[12])-t)<=v16.GATE_TOL

def act(bm,ba,p200,p50,p21,obs):
    b,_=bm.predict(obs,deterministic=True); x=torch.as_tensor(obs,dtype=torch.float32).reshape(1,-1)
    with torch.no_grad():
        a=np.asarray(b,np.float32)+ba(x).cpu().numpy()[0]
        if is_t(obs,T50): a+=p50(x).cpu().numpy()[0]
        elif is_t(obs,T200): a+=p200(x).cpu().numpy()[0]
        if is_t(obs,T50) and abs(float(obs[13])*360.0)<=TERM:
            d=p21(x).cpu().numpy()[0]; a[2]+=float(d[0]); a[3]+=float(d[1])
    return np.clip(a,-1,1)

def one(bm,ba,p200,p50,p21,target,seed):
    env=v16.RandomizedEntryEnv(target_turn_deg=target); obs,ri=env.reset(seed=seed); info=ri; ok=False
    try:
        n=int(max(90.0,abs(target)/1.10+70.0)/env.CONTROL_DT)
        for _ in range(n):
            obs,_,te,tr,info=env.step(act(bm,ba,p200,p50,p21,obs)); obs=np.asarray(obs,np.float32)
            if te or tr: ok=bool(info.get('success',False)); break
    finally: env.close()
    return dict(seed=seed,target=target,extra=int(ri.get('extra_entry_steps',0)),ok=ok,done=float(info.get('cumulative_turn_deg',0)),rem=float(info.get('remaining_turn_deg',999)),alt=float(info.get('altitude',np.nan)),hold=float(info.get('success_hold_s',0)),safe=bool(info.get('safety_failure',False)))

def main():
    for p in (v16.BASE_MODEL,v16.BASE_ADAPTER,v16.PATCH_200,P50,P21):
        if not p.exists(): raise FileNotFoundError(p)
    print('='*120); print('V22 RANDOMIZED FULL-ENTRY TEST - V21 +50 TERMINAL SPECIALIST'); print('Runtime teacher/controller OFF'); print('='*120)
    bm=PPO.load(str(v16.BASE_MODEL)); ba=load(v16.BASE_ADAPTER,v16.BASE_SCALE); p200=load(v16.PATCH_200,v16.PATCH_SCALE); p50=load(P50,v16.PATCH_SCALE)
    p21=TP(); p21.load_state_dict(torch.load(P21,map_location='cpu'),strict=True); p21.eval()
    rows=[]
    for s in SEEDS:
        for t in TARGETS:
            r=one(bm,ba,p200,p50,p21,t,s); rows.append(r); print(f"seed={s:3d} | target={t:+7.1f} | extra={r['extra']:2d} | pass={str(r['ok']):5s} | done={r['done']:+8.2f} | rem={r['rem']:+7.2f} | alt={r['alt']:7.2f} | hold={r['hold']:3.1f} | safety={r['safe']}")
    print('\n'+'='*120); print('V22 SUMMARY'); print('='*120)
    total=safety=0
    for t in TARGETS:
        q=[r for r in rows if r['target']==t]; p=sum(int(r['ok']) for r in q); sf=sum(int(r['safe']) for r in q); total+=p; safety+=sf
        print(f"target={t:+7.1f} | pass={p}/5 | safety_fail={sf}/5 | max_abs_err={max(abs(r['rem']) for r in q):.2f}")
    print(f"\nRANDOMIZED-ENTRY TOTAL: PASS={total}/{len(rows)} | SAFETY_FAILURES={safety}/{len(rows)}")
    print('V22 ROBUSTNESS:', 'PASS' if total==len(rows) and safety==0 else 'PARTIAL/FAIL')

if __name__=='__main__': main()
