"""Probe C — ileri uçuşta hızlanma / yavaşlamada irtifa kaybı ve zamanlama.
Kullanım: python probe_accel.py <model.zip> <out.json> ['{"rotor_dt_mode":"sim"}']
Senaryolar (sakin, 9300 lbs): 300 ft / 20 kt → 120 kt (Δ+100), 300 ft / 120 kt → 20 kt (Δ−100), 100 ft hover → 100 kt
(dh 0), 300 ft / 60 kt → 100 kt. Ölçüler: pencere içinde en büyük irtifa sapması (−: kayıp), hız hatasının %90'ının
kapandığı süre, banda giriş / son sınır, başarı, en büyük pitch.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

model_path, out = sys.argv[1], sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
m = PPO.load(str(REPO / model_path), device="cpu")
ov = dict(getattr(m, "ah1s_env_overrides", None) or {})
ov.update({"torque_density_climb": False, "next_at_deadline": False})
ov.update(extra)
env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
SC = [("acc_20_120", "cruise20@300", [EF.CHOLD, EF.CR(u_kt=120)], 150.0),
      ("dec_120_20", "cruise120@300", [EF.CHOLD, EF.CR(u_kt=20)], 150.0),
      ("hover_to_100", "hover100", [EF.HOLD, EF.ACC(100, 0)], 150.0),
      ("acc_60_100", "cruise60@300", [EF.CHOLD, EF.CR(u_kt=100)], 120.0),
      ("land_300", "hover300", [EF.HOLD, EF.LAND], 150.0),
      ("land_50_wind", "hover50", [EF.HOLD, EF.LAND], 120.0)]
res = {"model": model_path, "env": ov, "rows": []}
for sid, start, tasks, ep_s in SC:
    phys = dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none") if sid != "land_50_wind" else dict(wind_kt=15.0, wind_dir_deg=45.0, turb_level="light")
    opts = dict(level="F8", tasks=[dict(t) for t in tasks], fuel=(400.0, 400.0), start_heading_deg=0.0, start_perturb=0.0,
                physics=phys, episode_s=ep_s, **EF._start_opts(start))
    obs, info = env.reset(seed=0, options=opts)
    rows, done = [], False
    while not done:
        obs, r, te, tr, info = env.step(m.predict(obs, deterministic=True)[0])
        done = te or tr
        rows.append((info["t"], len(env.windows) - 1, info["altitude"], info["err_altitude"], info.get("err_speed", 0.0),
                     info["pitch_deg"], info["airspeed_kt"], info["vertical_speed"], info["skid_height"]))
    a = np.array(rows)
    cres = info["command_results"]
    w = cres[1]                                   # ikinci pencere (hızlanma / iniş)
    seg = a[a[:, 1] == 1]
    d = dict(id=sid, success=bool(w["success"]), settle_s=None if not np.isfinite(w["settle_s"]) else round(float(w["settle_s"]), 1),
             deadline=round(w["deadline"], 1), term=info["termination"])
    if w["kind"] == "land":
        d.update(touchdown_vs=w["touchdown_vs"], xy_td=round(float(w["final_err"]["xy"]), 2))
        # kızak 20 → 0 ft arasında alçalma hızı profili (en büyük sink her 5 ft bandında)
        prof = {}
        for lo in (15, 10, 5, 0):
            k = (seg[:, 8] >= lo) & (seg[:, 8] < lo + 5)
            prof[f"{lo}-{lo+5}ft"] = round(float(-seg[k, 7].min()), 2) if k.any() else None
        d["max_sink_by_band"] = prof
    else:
        e_h = seg[:, 3]                           # hedef − ölçülen: + → helikopter aşağıda
        d.update(max_alt_loss_ft=round(float(e_h.max()), 1), max_alt_gain_ft=round(float(-e_h.min()), 1),
                 max_pitch=round(float(np.abs(seg[:, 5]).max()), 1))
        eu0 = abs(seg[0, 4])
        k90 = np.flatnonzero(np.abs(seg[:, 4]) <= 0.1 * eu0)
        d["t90_speed_s"] = round(float(seg[k90[0], 0] - seg[0, 0]), 1) if k90.size else None
        # ilk 10 s'de irtifa kaybı ve hız kazancı
        k10 = seg[:, 0] - seg[0, 0] <= 10.0
        d["loss_first10s_ft"] = round(float(e_h[k10].max()), 1)
        d["dv_first10s_kt"] = round(float(abs(seg[k10, 6][-1] - seg[0, 6])), 1)
    res["rows"].append(d)
    print(d, flush=True)
Path(out).write_text(json.dumps(res, indent=1, default=float))
