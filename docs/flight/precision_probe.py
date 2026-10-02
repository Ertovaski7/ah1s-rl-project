"""
PRECISION PROBE (2026-10-01) — hover hassasiyeti: yerinde dönüş ve pirouette (RL değil; ölçüm)
==============================================================================================

Repo kökünden:
  python docs/flight/precision_probe.py models_flight/flight_v2.zip models_flight/flight_final.zip \
      --json docs/flight/precision_v2.json

Sakin hava, 10 ft kızak yüksekliği (CG 16.3 ft); yerinde dönüş +180°, −180°, −270°, +360° ve pirouette (100 ft yarıçap,
45 s + 4 s rampa) saat yönü / tersi; her biri 8800 ve 9700 lbs. Ölçütler F10 seviyesinin toleranslarıyla (hover
hassasiyeti ×0.5: dönüşte konum ≤ 10 ft, pirouette'te hareketli hedefe ≤ 10 ft). "Kayma" = görev penceresi açıkken
hedef noktadan en büyük yatay uzaklık (err_xy); pirouette'te hedef çember üzerinde ilerleyen nokta.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from stable_baselines3 import PPO  # noqa: E402

from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight  # noqa: E402

TASKS = (dict(kind="turn", dpsi=180.0), dict(kind="turn", dpsi=-180.0), dict(kind="turn", dpsi=-270.0),
         dict(kind="turn", dpsi=360.0),
         dict(kind="pirouette", radius=100.0, circle_s=45.0, direction=1.0),
         dict(kind="pirouette", radius=100.0, circle_s=45.0, direction=-1.0))
FUELS = ((150.0, 150.0), (600.0, 600.0))                    # 8800 / 9700 lbs


def probe(path) -> dict:
    m = PPO.load(str(path), device="cpu")
    ov = dict(getattr(m, "ah1s_env_overrides", None) or {})
    env = HelicopterEnvFlight(level="F10", config=FlightEnvConfig(**ov))
    res = []
    for task in TASKS:
        for fuel in FUELS:
            obs, _ = env.reset(seed=0, options=dict(
                level="F10", start="hover", start_alt_ft=16.3, start_heading_deg=0.0, start_perturb=0.0, fuel=fuel,
                physics=dict(wind_kt=0.0, turb_level="none"), tasks=[dict(kind="hold"), task], live=False,
                episode_s=90.0))
            done, xy, yaw = False, [], []
            info = {}
            while not done:
                obs, r, term, trunc, info = env.step(m.predict(obs, deterministic=True)[0])
                done = term or trunc
                if env.windows[-1]["kind"] == task["kind"] and not env.windows[-1]["closed"]:
                    xy.append(info["err_xy"])
                    yaw.append(abs(info["yaw_rate_dps"]))
            c = [x for x in info["command_results"] if x["kind"] == task["kind"]][0]
            res.append(dict(kind=task["kind"], arg=task.get("dpsi", task.get("direction")), weight=sum(fuel) + 8500.0,
                            success=bool(c["success"]), max_xy=float(max(xy)), max_yaw=float(max(yaw))))
    t = [r for r in res if r["kind"] == "turn"]
    p = [r for r in res if r["kind"] == "pirouette"]
    return dict(model=str(path), rows=res,
                turn=dict(ok=sum(r["success"] for r in t), n=len(t), xy_mean=float(np.mean([r["max_xy"] for r in t])),
                          xy_max=float(max(r["max_xy"] for r in t)), yaw_mean=float(np.mean([r["max_yaw"] for r in t]))),
                pirouette=dict(ok=sum(r["success"] for r in p), n=len(p),
                               xy_mean=float(np.mean([r["max_xy"] for r in p])), xy_max=float(max(r["max_xy"] for r in p))))


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--json", default=None, help="sonuçları bu dosyaya yaz")
    args = ap.parse_args(argv)
    out = []
    for path in args.models:
        s = probe(path)
        out.append(s)
        t, p = s["turn"], s["pirouette"]
        print(f"{Path(path).name:<24} dönüş: başarı {t['ok']}/{t['n']}, kayma ort {t['xy_mean']:5.1f} ft (en büyük "
              f"{t['xy_max']:5.1f}), en büyük yaw hızı ort {t['yaw_mean']:4.1f}°/s | pirouette: başarı {p['ok']}/{p['n']}, "
              f"uzaklık ort {p['xy_mean']:5.1f} ft (en büyük {p['xy_max']:5.1f})", flush=True)
    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("kaydedildi:", args.json)


if __name__ == "__main__":
    main()
