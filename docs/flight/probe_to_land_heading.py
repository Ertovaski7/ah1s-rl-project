"""Kalkış / inişte heading ve yatay hız (2026-10-03, kullanıcı isteği: kalkış ve inişte Δheading ≈ 0, Δhız ≈ 0, yalnızca
irtifa değişsin). Senaryolar: yerden 100 ft'e kalkış, 100 ft hover'dan iniş; sakin, 15 kt sağdan + hafif türbülans, 25 kt
sol-arka + orta türbülans + gust. Ölçüler: en büyük |heading sapması|, en büyük yatay yer hızı (kt), en büyük konum hatası,
temas hızı, başarı.

Kullanım: python docs/flight/probe_to_land_heading.py <out.json> <model[:ENV_JSON]> [...]
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

KT = 1.6878099
ENVS = {"sakin": dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none"),
        "15kt_sag": dict(wind_kt=15.0, wind_dir_deg=90.0, turb_level="light"),
        "25kt_turb": dict(wind_kt=25.0, wind_dir_deg=-120.0, turb_level="moderate", gust_rate_per_min=1.0, gust_kt=(5.0, 12.0))}
SC = [("kalkis100", "ground", [EF.TO(100)]), ("inis100", "hover100", [EF.HOLD, EF.LAND])]


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
        for en, phys in ENVS.items():
            for sid, start, tasks in SC:
                opts = dict(level="F8", tasks=[dict(t) for t in tasks], fuel=(300.0, 300.0), start_heading_deg=0.0,
                            start_perturb=0.0, physics=dict(phys, wind_dir_relative=True), episode_s=90.0, **EF._start_opts(start))
                try:
                    obs, info = env.reset(seed=0, options=opts)
                except RuntimeError as exc:
                    r[f"{sid}/{en}"] = dict(error=str(exc)[:60]); continue
                done, R, k = False, [], len(tasks) - 1
                while not done:
                    obs, rr, te, tr, info = env.step(m.predict(obs, deterministic=True)[0])
                    done = te or tr
                    if len(env.windows) > k and not env.windows[k]["closed"]:
                        psi_t = env.windows[k]["target"]["psi"]
                        R.append((abs(((info["heading_deg"] - psi_t + 180) % 360) - 180), info["ground_speed"] / KT,
                                  info["err_xy"]))
                a = np.array(R) if R else np.zeros((1, 3))
                c = info["command_results"][k] if len(info["command_results"]) > k else {}
                r[f"{sid}/{en}"] = dict(max_hdg_deg=round(float(a[:, 0].max()), 1), max_gs_kt=round(float(a[:, 1].max()), 1),
                                        max_xy_ft=round(float(a[:, 2].max()), 1), success=bool(c.get("success")),
                                        touchdown_vs=c.get("touchdown_vs"), term=info.get("termination"))
        res[name] = r
        print(name, flush=True)
        for k2, v in r.items():
            print(f"    {k2:22s} {v}", flush=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
