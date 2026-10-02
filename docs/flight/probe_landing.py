"""İniş stres takımı (güvenlik taraması): rüzgâr / türbülans / gust / ağırlık kombinasyonlarında iniş.
Kullanım: python probe_landing.py <out.json> <model1> [<model2> ...]
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO
from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
import evaluate_flight as EF

out = sys.argv[1]
CASES = []
for wn, phys in (("15kt_yan_hafif", dict(wind_kt=15.0, wind_dir_deg=90.0, turb_level="light")),
                 ("25kt_orta_gust", dict(wind_kt=25.0, wind_dir_deg=-120.0, turb_level="moderate", gust_rate_per_min=1.0, gust_kt=(5.0, 12.0))),
                 ("20kt_arka_hafif_gust", dict(wind_kt=20.0, wind_dir_deg=180.0, turb_level="light", gust_rate_per_min=1.0, gust_kt=(5.0, 12.0))),
                 ("sakin", dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none"))):
    for h, fuel in ((300.0, (600.0, 600.0)), (50.0, (300.0, 300.0)), (150.0, (450.0, 450.0))):
        CASES.append((f"inis{h:.0f}_{wn}_{8500+sum(fuel):.0f}", h, fuel, phys))
res = {}
for mp in sys.argv[2:]:
    m = PPO.load(mp if mp.startswith("/") else str(REPO / mp), device="cpu")
    ov = dict(getattr(m, "ah1s_env_overrides", None) or {}); ov.update({"torque_density_climb": False, "next_at_deadline": False})
    env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
    rows = []
    for sid, h, fuel, phys in CASES:
        for seed in (0, 1):
            p = dict(phys); p.setdefault("wind_dir_relative", True)
            opts = dict(level="F8", tasks=[EF.HOLD, EF.LAND], fuel=fuel, start_heading_deg=0.0, start_perturb=0.0, physics=p,
                        episode_s=200.0, start="hover", start_alt_ft=h)
            obs, info = env.reset(seed=seed, options=opts); done = False; hs_min_v = []
            while not done:
                obs, r, te, tr, info = env.step(m.predict(obs, deterministic=True)[0]); done = te or tr
                if info["skid_height"] < 5.0 and info["wow"] == 0: hs_min_v.append(info["vertical_speed"])
            w = [c for c in info["command_results"] if c["kind"] == "land"]
            w = w[0] if w else None
            rows.append(dict(id=sid, seed=seed, term=info["termination"], ok=bool(w and w["success"]),
                             touchdown=None if not w else w["touchdown_vs"], xy=None if not w else round(w["final_err"]["xy"], 1),
                             sink_last5ft=None if not hs_min_v else round(float(-min(hs_min_v)), 2)))
    tag = Path(mp).stem
    n_ok = sum(r["ok"] for r in rows); unsafe = [(r["id"], r["seed"], r["term"]) for r in rows if r["term"] not in ("time_limit", "fuel_exhausted")]
    td = [r["touchdown"] for r in rows if r["touchdown"] is not None]
    res[tag] = dict(rows=rows, n_ok=n_ok, n=len(rows), unsafe=unsafe, td_median=float(np.median(td)) if td else None, td_min=min(td) if td else None,
                    hard=[(r["id"], r["seed"], round(r["touchdown"], 2)) for r in rows if r["touchdown"] is not None and r["touchdown"] < -4.0])
    print(tag, f"iniş {n_ok}/{len(rows)}", "güvensiz:", unsafe, "temas medyan/min:", round(res[tag]["td_median"], 2), round(res[tag]["td_min"], 2), "sert:", res[tag]["hard"], flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
