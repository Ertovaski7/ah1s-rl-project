"""Mükemmellik ölçümü (2026-10-03): (a) ileri uçuşta irtifa tutma (hızlanma dışı), (b) tırmanışta sallanmadan düz iz.

(a) 300 ft / 80 kt, 9300 lbs, sakin: 60 s tut, +90° dönüş, −180° dönüş, 100 → 60 kt yavaşlama, 15 kt yan rüzgârda +90°.
    Ölçü: pencere boyunca en büyük |irtifa hatası| ve RMS (komut verilmeyen eksen).
(b) tırmanışlar: yerden 300 ft'e kalkış, hover 100 → 500 ft, hover bob +40 ft, 80 kt'ta +300 ft.
    Ölçüler: yatay kayma (hover tırmanışında en büyük konum hatası), dikey hız salınımı (ḣ'nin 2 s kayan ortalamadan
    sapmasının RMS'i), roll / pitch std, hedef irtifayı aşma (overshoot), tırmanış sırasında heading sapması.
Kullanım: python docs/flight/probe_perfection.py <out.json> <model1> [<model2> ...]
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_command import CONTROL_DT
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

CALM = dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none")
SIDE = dict(wind_kt=15.0, wind_dir_deg=90.0, turb_level="none")
LEVEL = [("tut60", "cruise80@300", [dict(kind="cruise", hold=True, T=60.0)], CALM),
         ("don+90", "cruise80@300", [EF.CHOLD, EF.CR(dpsi=90)], CALM),
         ("don-180", "cruise80@300", [EF.CHOLD, EF.CR(dpsi=-180)], CALM),
         ("yavas100_60", "cruise100@300", [EF.CHOLD, EF.CR(u_kt=60)], CALM),
         ("don+90_yanruzgar", "cruise80@300", [EF.CHOLD, EF.CR(dpsi=90)], SIDE)]
CLIMB = [("kalkis300", "ground", [EF.TO(300)], CALM),
         ("hover100_500", "hover100", [EF.HOLD, dict(kind="climb_to", h=500.0)], CALM),
         ("bob+40", "hover50", [EF.HOLD, dict(kind="bob", dh=40.0)], CALM),
         ("ileri+300", "cruise80@300", [EF.CHOLD, EF.CR(dh=300)], CALM)]


def run(env, m, start, tasks, phys, ep_s=150.0):
    p = dict(phys); p.setdefault("wind_dir_relative", True)
    opts = dict(level="F8", tasks=[dict(t) for t in tasks], fuel=(400.0, 400.0), start_heading_deg=0.0, start_perturb=0.0,
                physics=p, episode_s=ep_s, **EF._start_opts(start))
    obs, info = env.reset(seed=0, options=opts); done = False; rows = []
    while not done:
        obs, r, te, tr, info = env.step(m.predict(obs, deterministic=True)[0]); done = te or tr
        rows.append((info["t"], len(env.windows) - 1, info["err_altitude"], info["err_xy"], info["vertical_speed"],
                     info["roll_deg"], info["pitch_deg"], info["err_heading"], info["altitude"]))
    return np.array(rows), info


res = {}
for mp in sys.argv[2:]:
    m = PPO.load(mp if mp.startswith("/") else str(REPO / mp), device="cpu")
    ov = dict(getattr(m, "ah1s_env_overrides", None) or {}); ov.update({"torque_density_climb": False, "next_at_deadline": False})
    env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
    out = {"level": [], "climb": []}
    for sid, start, tasks, phys in LEVEL:
        a, info = run(env, m, start, tasks, phys)
        wi = len(tasks) - 1
        seg = a[a[:, 1] == wi]
        c = info["command_results"][wi]
        out["level"].append(dict(id=sid, success=bool(c["success"]), max_dh_ft=round(float(np.abs(seg[:, 2]).max()), 1),
                                 rms_dh_ft=round(float(np.sqrt(np.mean(seg[:, 2] ** 2))), 2), term=info["termination"]))
    for sid, start, tasks, phys in CLIMB:
        a, info = run(env, m, start, tasks, phys)
        wi = len(tasks) - 1
        seg = a[a[:, 1] == wi]
        c = info["command_results"][wi]
        h_t = float(c["target"]["h"])
        # tırmanış evresi: pencere başından hedefe ilk varışa (|e_h| ≤ 3 ft) kadar
        k_arr = np.flatnonzero(np.abs(seg[:, 2]) <= 3.0)
        end = int(k_arr[0]) if k_arr.size else len(seg) - 1
        cl = seg[: max(end, 5)]
        n = max(3, int(round(2.0 / CONTROL_DT)))
        vs = cl[:, 4]
        vs_ma = np.convolve(vs, np.ones(n) / n, mode="same")
        osc = vs - vs_ma
        core = slice(n, max(n + 1, len(vs) - n))
        out["climb"].append(dict(
            id=sid, success=bool(c["success"]), climb_s=round(float(cl[-1, 0] - cl[0, 0]), 1),
            max_xy_drift_ft=round(float(seg[:, 3].max()), 1), vs_osc_rms=round(float(np.sqrt(np.mean(osc[core] ** 2))), 2),
            vs_mean=round(float(vs[core].mean()), 1), roll_std=round(float(cl[:, 5].std()), 2), pitch_std=round(float(cl[:, 6].std()), 2),
            overshoot_ft=round(float(max(0.0, (seg[:, 8] - h_t).max())), 1), max_hdg_dev=round(float(np.abs(cl[:, 7]).max()), 1),
            term=info["termination"]))
    res[Path(mp).stem] = out
    L, C = out["level"], out["climb"]
    print(f"{Path(mp).stem}: düz uçuş irtifa maks {[x['max_dh_ft'] for x in L]} (medyan {np.median([x['max_dh_ft'] for x in L]):.1f}) "
          f"başarı {sum(x['success'] for x in L)}/{len(L)}", flush=True)
    for x in C:
        print(f"    {x['id']:13s} ok={x['success']} {x['climb_s']:5.1f}s kayma {x['max_xy_drift_ft']:5.1f} ft  ḣ salınım {x['vs_osc_rms']:.2f} "
              f"(ort {x['vs_mean']:.1f}) roll/pitch std {x['roll_std']:.2f}/{x['pitch_std']:.2f}  aşma {x['overshoot_ft']:.1f} ft  heading {x['max_hdg_dev']:.1f}°", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
