"""Rüzgârda hover → ileri uçuş (2026-10-03, kullanıcı gözlemi: "rüzgârda ileri git komutunda geri geri gidiyor").

Senaryo: 100 ft hover, 9100 lbs; uçuş başında hava canlı değişir: 20 kt rüzgâr 8 yönden (buruna göre 0 = karşıdan …
180 = arkadan; 4 kt/s rampa), orta türbülans + gust; 15 s sonra canlı uygulamanın yolundan (flight_commands.route_command, strict) "hız 80 kt" komutu.
Ölçüler (komuttan sonra): ilk 6 s'de burun yönündeki en küçük yer hızı (kt; − = geri gidiyor), yer hızının 10 kt'ı geçtiği
an, ilk 10 s'de ileri gidilen yol (ft), en büyük |pitch|, en düşük
irtifa, 80 kt'ın %90'ına varış süresi, pencere başarısı, episode sonu. Sakin hava ile 20 → 120 kt ve 100 ft hover → 80 kt
için ayrıca: pitch profili ve en büyük irtifa kaybı; ileri uçuşta (40–80 kt) yana kayma |β|.

Kullanım: python docs/flight/probe_wind_accel.py <out.json> <model1> [<model2> ...]   (model:ENV_JSON ile ek ayar)
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from flight_commands import apply, route_command
from helicopter_env_command import CONTROL_DT
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight

KT = 1.6878099
DIRS = (0, 45, 90, 135, 180, 225, 270, 315)


def run(env, m, wdir, wind_kt=20.0, turb="moderate", gusts=True, t_cmd=15.0, dur=75.0, seed=0, start="hover",
        u0=60.0, alt=100.0, cmd=None):
    # Sakin havada başla, hava canlı uygulamadaki gibi uçuşta değişsin (physics_ext.set_live: rüzgâr 4 kt/s rampa →
    # 20 kt'a 5 s). Neden: reset PID'i bazı yönlerde 20 kt'ta hover'ı kuramıyor (v = 1.8 ft/s'de kalıyor).
    phys = dict(wind_kt=0.0, wind_dir_deg=0.0, wind_dir_relative=True, turb_level="none")
    opts = dict(tasks=[dict(kind="hold")] if start == "hover" else [dict(kind="cruise", hold=True)], episode_s=dur,
                live=True, start=start, start_alt_ft=alt, start_speed_kt=u0, fuel=(300.0, 300.0), start_perturb=0.0,
                start_heading_deg=0.0, physics=phys)
    obs, info = env.reset(seed=seed, options=opts)
    if wind_kt > 0.0 or turb != "none" or gusts:
        env.ext.set_live(env.fdm, wind_kt=wind_kt, wind_from_deg=(info["heading_deg"] + wdir) % 360.0, turb_level=turb,
                         gusts=gusts)
        env.cfg = env._cfg_for_turb(turb)
    done, issued, rows = False, False, []
    while not done:
        t = env.steps * CONTROL_DT
        if not issued and t >= t_cmd:
            apply(env, route_command(env, strict=True, **(cmd or dict(speed_kt=80.0))))
            issued = True
            k0 = len(env.windows) - 1
        obs, r, te, tr, info = env.step(m.predict(obs, deterministic=True)[0])
        done = te or tr
        if issued:
            psi = math.radians(info["heading_deg"])
            vn, ve = float(env.fdm["velocities/v-north-fps"]), float(env.fdm["velocities/v-east-fps"])
            rows.append((info["t"] - t_cmd, (vn * math.cos(psi) + ve * math.sin(psi)) / KT, info["pitch_deg"],
                         info["altitude"], info["airspeed_kt"], info.get("lateral_airspeed", 0.0)))
    a = np.array(rows)
    w = env.windows[k0] if issued and k0 < len(env.windows) else None
    tgt = float((cmd or {}).get("speed_kt", 80.0))
    k90 = np.flatnonzero(a[:, 4] >= 0.9 * tgt)
    first6 = a[:, 0] <= 6.0
    fast = (a[:, 4] >= 40.0) & (a[:, 4] <= 80.0)
    beta = np.degrees(np.arctan2(np.abs(a[fast, 5]), a[fast, 4] * KT)) if fast.any() else np.array([np.nan])
    k10 = np.flatnonzero(a[:, 1] >= 10.0)
    first10 = a[:, 0] <= 10.0
    dist10 = float(np.sum(a[first10, 1]) * KT * CONTROL_DT)          # ilk 10 s'de burun yönünde yer mesafesi (ft)
    return dict(min_gs_fwd_6s_kt=round(float(a[first6, 1].min()), 1), t_gs10_s=round(float(a[k10[0], 0]), 1) if k10.size else None,
                dist10_ft=round(dist10, 0), max_pitch=round(float(np.abs(a[:, 2]).max()), 1),
                min_alt_ft=round(float(a[:, 3].min()), 1), t90_s=round(float(a[k90[0], 0]), 1) if k90.size else None,
                success=bool(w["success"]) if w is not None else None, term=info.get("termination"),
                beta_med_deg=round(float(np.nanmedian(beta)), 1), beta_max_deg=round(float(np.nanmax(beta)), 1))


def main():
    out, res = sys.argv[1], {}
    for spec in sys.argv[2:]:
        mp, _, extra = spec.partition(":")
        m = PPO.load(mp if mp.startswith("/") else str(REPO / mp), device="cpu")
        ov = dict(getattr(m, "ah1s_env_overrides", None) or {})
        ov.update({"torque_density_climb": False, "next_at_deadline": False})
        if extra:
            ov.update(json.loads(extra))
        env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
        name = Path(mp).stem + ("+" + extra if extra else "")
        r = dict(wind={}, calm={})
        for d in DIRS:
            r["wind"][d] = run(env, m, d)
        r["calm"]["hover_to_80"] = run(env, m, 0, wind_kt=0.0, turb="none", gusts=False)
        r["calm"]["c20_to_120"] = run(env, m, 0, wind_kt=0.0, turb="none", gusts=False, start="cruise", u0=20.0, alt=300.0,
                                      cmd=dict(speed_kt=120.0), dur=100.0)
        res[name] = r
        W = r["wind"]
        print(f"{name}: rüzgârda başarı {sum(bool(x['success']) for x in W.values())}/{len(W)}, düşme "
              f"{sum(x['term'] not in (None, 'time_limit') for x in W.values())}, en kötü ilk-6 s yer hızı "
              f"{min(x['min_gs_fwd_6s_kt'] for x in W.values()):+.1f} kt, en büyük pitch {max(x['max_pitch'] for x in W.values()):.0f}°, "
              f"en düşük irtifa {min(x['min_alt_ft'] for x in W.values()):.0f} ft", flush=True)
        for d, x in W.items():
            print(f"    rüzgâr {d:3d}°: {x}", flush=True)
        for k, x in r["calm"].items():
            print(f"    sakin {k}: {x}", flush=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
