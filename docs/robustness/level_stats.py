"""Seviyelerden deterministik episode başarısı + komut türüne göre başarı (README 29.7).
Kullanım (repo kökünden): python docs/robustness/level_stats.py <model.zip> <episode> <seviyeler> [seed0=50000]
  ör. python docs/robustness/level_stats.py models_maneuver/maneuver_robust_final.zip 100 M5,S1,S2,S3"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo kökü
import numpy as np
from helicopter_env_maneuver import HelicopterEnvManeuver, load_maneuver_policy
from command_curriculum import command_axis
pol, cfg = load_maneuver_policy(sys.argv[1])
n = int(sys.argv[2]); levels = sys.argv[3].split(",")
seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
env = HelicopterEnvManeuver(level=levels[0], config=cfg)
print(f"model {sys.argv[1]} coll_scale {cfg.coll_scale}", flush=True)
for lvl in levels:
    ep_ok, term, cats = [], Counter(), defaultdict(list)
    for k in range(n):
        obs, info = env.reset(seed=seed0 + k, options=dict(level=lvl))
        done = False
        while not done:
            obs, r, te, tr, info = env.step(pol(obs))
            done = te or tr
        ep_ok.append(info["episode_success"]); term[info["termination"]] += 1
        for res in info["command_results"]:
            cats[res.get("category") or command_axis(res["cmd"])].append(res["success"])
    print(f"{lvl}: episode success {np.mean(ep_ok):.0%} ({n})  term {dict(term)}  " +
          " ".join(f"{c}={np.mean(v):.0%}({len(v)})" for c, v in sorted(cats.items())), flush=True)
