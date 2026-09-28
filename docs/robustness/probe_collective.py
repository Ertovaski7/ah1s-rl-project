"""Kök neden probu: 55° yatışlı 180° dönüşte irtifayı tutmak için collective yetkisi yetiyor mu? (README 29.5)

Basit dış döngü (policy değil, yalnızca ölçüm): yatış komutu 55° (hata < 15° olunca 0), yunuslama ← hız hatası,
collective ← irtifa hatası + dikey hız (±1 → trim ± coll_scale), pedal 0. Her hız × coll_scale için:
en büyük irtifa kaybı, ortalama / en büyük collective, collective doygunluğu, en düşük rpm, 180°'ye varış süresi.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo kökü (docs/robustness/ → ../..)
import numpy as np  # noqa: E402

from helicopter_env_maneuver import HelicopterEnvManeuver, ManeuverEnvConfig  # noqa: E402

KT = 1.6878
rows = []
for u_kt in (60, 80, 100):
    for cs in (0.25, 0.45):
        env = HelicopterEnvManeuver(level="M5", config=ManeuverEnvConfig(coll_scale=cs))
        u0, h0 = u_kt * KT, 900.0
        obs, info = env.reset(seed=0, options=dict(level="M5", start_alt_ft=h0, start_speed_fps=u0, start_heading_deg=0.0,
                                                   commands=[(1.0, {"heading": 180.0})], episode_s=26.0))
        h_start = info["altitude"]
        hs, colls, a0s, rpms, t180 = [], [], [], [], None
        done, rolling = False, True
        while not done:
            e_h = info["err_altitude"]
            vs = info["vertical_speed"]
            e_u = info.get("err_speed", 0.0)
            e_psi = info["err_heading"]
            if info["t"] < 1.2:
                a2 = 0.0
            else:
                rolling = rolling and abs(e_psi) > 15.0
                a2 = 55.0 / 60.0 if rolling else 0.0
            a0 = float(np.clip(0.06 * e_h - 0.10 * vs, -1, 1))
            a1 = float(np.clip(0.04 * e_u, -1, 1))
            obs, r, term, trunc, info = env.step(np.array([a0, a1, a2, 0.0]))
            done = term or trunc
            if info["t"] >= 1.0:
                hs.append(info["altitude"]); colls.append(info["controls"][0]); a0s.append(abs(a0)); rpms.append(info["rotor_rpm"])
                if t180 is None and info["t"] > 1.5 and abs(info["err_heading"]) < 2.0:
                    t180 = info["t"] - 1.0
        hs = np.array(hs)
        rows.append((u_kt, cs, h_start - hs.min(), float(np.mean(colls)), float(np.max(colls)), float(np.mean(np.array(a0s) > 0.98)),
                     float(np.nanmin(rpms)), t180, info.get("termination")))
        print(f"{u_kt:3d} kt  coll ±{cs:.2f}:  en büyük irtifa kaybı {rows[-1][2]:6.1f} ft   collective ort {rows[-1][3]:.3f} "
              f"en büyük {rows[-1][4]:.3f}   doygun %{100 * rows[-1][5]:4.0f}   en düşük rpm {rows[-1][6]:5.1f}   "
              f"180°'ye {('%.1f s' % t180) if t180 else '—'}   bitiş {rows[-1][8]}", flush=True)

# Çalıştırma (repo kökünden): python docs/robustness/probe_collective.py  → probe_collective.txt
