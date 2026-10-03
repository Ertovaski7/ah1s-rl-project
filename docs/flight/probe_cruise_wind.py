"""İleri uçuş komutları: rüzgârsız ↔ rüzgârlı ↔ türbülanslı (2026-10-02).
Aynı komutlar üç çevrede: sakin / 15 kt rüzgâr (sağ ön 45°) / 25 kt rüzgâr + orta türbülans + gust. Başlangıç 300 ft, 80 kt,
9300 lbs. Komutlar (her biri ayrı episode, önce 5 s tut): +30 kt, −30 kt, +90°, −90°, +200 ft, −200 ft, (+20 kt, +60°,
+100 ft) birleşik, tut 60 s. Ölçüler: başarı, banda giriş / son sınır, tutma süresindeki en büyük hatalar, pencere boyunca
en büyük irtifa sapması, roll std (salınım).
Kullanım: python docs/flight/probe_cruise_wind.py <out.json> <model1> [<model2> ...]
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

out = sys.argv[1]
ENVS = {"sakin": dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none"),
        "ruzgar15": dict(wind_kt=15.0, wind_dir_deg=45.0, turb_level="none"),
        "turb25": dict(wind_kt=25.0, wind_dir_deg=-120.0, turb_level="moderate", gust_rate_per_min=1.0, gust_kt=(5.0, 12.0))}
CMDS = {"hiz+30": EF.CR(du_kt=30), "hiz-30": EF.CR(du_kt=-30), "don+90": EF.CR(dpsi=90), "don-90": EF.CR(dpsi=-90),
        "irt+200": EF.CR(dh=200), "irt-200": EF.CR(dh=-200), "birlesik": EF.CR(du_kt=20, dpsi=60, dh=100),
        "tut60": dict(kind="cruise", hold=True, T=60.0)}
res = {}
for mp in sys.argv[2:]:
    m = PPO.load(mp if mp.startswith("/") else str(REPO / mp), device="cpu")
    ov = dict(getattr(m, "ah1s_env_overrides", None) or {}); ov.update({"torque_density_climb": False, "next_at_deadline": False})
    env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
    rows = []
    for en, phys in ENVS.items():
        for cn, cmd in CMDS.items():
            p = dict(phys); p.setdefault("wind_dir_relative", True)
            tasks = [EF.CHOLD, dict(cmd)] if cn != "tut60" else [dict(cmd)]
            opts = dict(level="F8", tasks=tasks, fuel=(400.0, 400.0), start_heading_deg=0.0, start_perturb=0.0, physics=p,
                        episode_s=150.0, start="cruise", start_speed_kt=80.0, start_alt_ft=300.0)
            obs, info = env.reset(seed=0, options=opts); done = False; tr = []
            while not done:
                obs, r, te, tr_, info = env.step(m.predict(obs, deterministic=True)[0]); done = te or tr_
                tr.append((info["t"], len(env.windows) - 1, info["err_altitude"], info["err_heading"], info.get("err_speed", 0.0), info["roll_deg"], info["pitch_deg"]))
            a = np.array(tr); wi = 0 if cn == "tut60" else 1
            c = info["command_results"][wi]; seg = a[a[:, 1] == wi]
            tail = seg[seg[:, 0] >= seg[-1, 0] - c["hold_s"]] if seg.size else seg
            rows.append(dict(env=en, cmd=cn, success=bool(c["success"]), coupling_ok=bool(c["coupling_ok"]),
                             settle_ratio=None if not np.isfinite(c["settle_s"]) else round(float(c["settle_s"] / c["deadline"]), 2),
                             max_alt_dev_ft=round(float(np.abs(seg[:, 2]).max()), 1), hold_alt_ft=round(float(np.abs(tail[:, 2]).max()), 1),
                             hold_hdg_deg=round(float(np.abs(tail[:, 3]).max()), 1), hold_u_fps=round(float(np.abs(tail[:, 4]).max()), 1),
                             roll_std=round(float(seg[:, 5].std()), 2), pitch_std=round(float(seg[:, 6].std()), 2), term=info["termination"]))
    tag = Path(mp).stem
    res[tag] = rows
    for en in ENVS:
        rr = [r for r in rows if r["env"] == en]
        print(f"{tag} {en:9s} başarı {sum(r['success'] for r in rr)}/{len(rr)} | irtifa sapması maks medyan {np.median([r['max_alt_dev_ft'] for r in rr]):.1f} ft "
              f"| tutma irtifa {np.median([r['hold_alt_ft'] for r in rr]):.1f} heading {np.median([r['hold_hdg_deg'] for r in rr]):.1f} hız {np.median([r['hold_u_fps'] for r in rr]):.1f} "
              f"| roll std {np.median([r['roll_std'] for r in rr]):.2f} | başarısız: {[r['cmd'] for r in rr if not r['success']]}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
