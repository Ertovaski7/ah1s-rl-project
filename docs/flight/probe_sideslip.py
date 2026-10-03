"""Yana kayma (burun ↔ yer izi farkı) ölçümü (2026-10-03, kullanıcı gözlemi: "arkadaki iz düz gidiyor ama helikopter o
yöne bakmıyor"). Sakin havada: (a) ileri uçuşta sabit hız (40 / 60 / 80 / 100 / 120 kt, 300 ft, 40 s tut), (b) 300 ft
20 kt'tan 120 kt'a hızlanma, (c) 100 ft hover'dan 80 kt'a hızlanma. Ölçü: |β| = atan(|yana hava hızı| / ileri hava hızı)
medyan ve en büyük (hava hızı ≥ 30 kt iken); sakin havada burun ile yer izi arasındaki fark β'ya eşit.

Kullanım: python docs/flight/probe_sideslip.py <out.json> <model1> [<model2> ...]   (model:ENV_JSON ile ek ayar)
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

KT = 1.6878099
CALM = dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none")
SC = [(f"tut{u}", f"cruise{u}@300", [dict(kind="cruise", hold=True, T=40.0)]) for u in (40, 60, 80, 100, 120)]
SC += [("hizlan20_120", "cruise20@300", [EF.CHOLD, EF.CR(u_kt=120)]), ("hover_80", "hover100", [EF.HOLD, EF.ACC(80, 0)])]


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
        r = {}
        for sid, start, tasks in SC:
            opts = dict(level="F8", tasks=[dict(t) for t in tasks], fuel=(400.0, 400.0), start_heading_deg=0.0,
                        start_perturb=0.0, physics=dict(CALM, wind_dir_relative=True), episode_s=90.0, **EF._start_opts(start))
            obs, info = env.reset(seed=0, options=opts)
            done, b = False, []
            while not done:
                obs, rr, te, tr, info = env.step(m.predict(obs, deterministic=True)[0])
                done = te or tr
                if info["airspeed_kt"] >= 30.0:
                    b.append(math.degrees(math.atan2(abs(info["lateral_airspeed"]), info["airspeed_kt"] * KT)))
            b = np.array(b) if b else np.array([np.nan])
            r[sid] = dict(beta_med=round(float(np.nanmedian(b)), 2), beta_max=round(float(np.nanmax(b)), 2),
                          term=info.get("termination"))
        res[name] = r
        print(name, {k: (v["beta_med"], v["beta_max"]) for k, v in r.items()}, flush=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
