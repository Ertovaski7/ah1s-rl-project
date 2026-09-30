"""
Tek ajanlı uçuş env'inin (`helicopter_env_flight.py`) yapılabilirlik ve ödül tutarlılığı kontrolü — RL DEĞİL.

Kural tabanlı pilot (`scripted_pilot.py`, PID) sabit senaryoları ve curriculum seviyelerinden örneklenen episode'ları
uçar; aynı episode'ları sıfır action (kumandalar trimde, çizelgeyle) da uçar. Sorular:
  (1) görevler, başarı bantları ve süre hedefleri bu fizikte yapılabilir mi (kategori başına başarı, oturma süresi / T);
  (2) ödül görevi yapanı yapmayanın üstünde tutuyor mu (getiri: pilot ↔ sıfır action).
Eğitimde kullanılmaz; policy'ye action önermez.

  python docs/flight/check_flight_scripted.py --scenarios all --levels F2,F4,F5,F8 --seeds 6 --json docs/flight/check_scripted.json
  python docs/flight/check_flight_scripted.py --scenarios cruise_calm --trace 2      # 2 s'de bir durum satırı
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight  # noqa: E402
from scripted_pilot import ScriptedPilot  # noqa: E402

KT = 1.6878099


def env_opts(wind_kt=0.0, wind_dir=0.0, turb="none", gusts=False):
    d = dict(wind_kt=float(wind_kt), wind_dir_deg=float(wind_dir), wind_dir_relative=True, turb_level=turb)
    if gusts:
        d.update(gust_rate_per_min=1.0, gust_kt=(5.0, 10.0))
    return d


CRUISE_TASKS = [dict(kind="cruise", hold=True), dict(kind="cruise", du_kt=20.0), dict(kind="cruise", dpsi=90.0),
                dict(kind="cruise", dh=-150.0), dict(kind="cruise", du_kt=-25.0, dpsi=-60.0, dh=100.0),
                dict(kind="stop", decel=2.5), dict(kind="land")]
ACCEL_TASKS = [dict(kind="hold"), dict(kind="cruise", u_kt=80.0, dh=100.0, accel=True), dict(kind="cruise", dpsi=-120.0),
               dict(kind="stop", decel=2.5), dict(kind="land")]
CHAIN_TASKS = [dict(kind="takeoff", h=40.0), dict(kind="cruise", u_kt=60.0, dh=150.0, accel=True),
               dict(kind="cruise", du_kt=30.0), dict(kind="cruise", dpsi=90.0), dict(kind="cruise", dh=-100.0),
               dict(kind="stop", decel=2.5), dict(kind="land")]

SCENARIOS = {
    "cruise_calm": dict(start="cruise", start_speed_kt=60.0, start_alt_ft=300.0, tasks=CRUISE_TASKS, physics=env_opts()),
    "cruise_wind15_light": dict(start="cruise", start_speed_kt=60.0, start_alt_ft=300.0, tasks=CRUISE_TASKS,
                                physics=env_opts(15.0, 45.0, "light", gusts=True)),
    "cruise_wind20_moderate": dict(start="cruise", start_speed_kt=60.0, start_alt_ft=300.0, tasks=CRUISE_TASKS,
                                   physics=env_opts(20.0, 120.0, "moderate", gusts=True)),
    "accel_calm": dict(start="hover", start_alt_ft=50.0, tasks=ACCEL_TASKS, physics=env_opts()),
    "accel_wind15_light": dict(start="hover", start_alt_ft=50.0, tasks=ACCEL_TASKS,
                               physics=env_opts(15.0, 200.0, "light", gusts=True)),
    "chain_calm": dict(start="ground", tasks=CHAIN_TASKS, physics=env_opts()),
    "chain_wind20_moderate": dict(start="ground", tasks=CHAIN_TASKS, physics=env_opts(20.0, 300.0, "moderate", gusts=True)),
}


def run_episode(env, mode: str, seed: int, options: dict | None = None, trace: float = 0.0):
    obs, info = env.reset(seed=seed, options=options)
    pilot = ScriptedPilot(env)
    done, ret, parts = False, 0.0, collections.Counter()
    last_trace = -1e9
    while not done:
        a = pilot() if mode == "scripted" else np.zeros(4, dtype=np.float32)
        obs, r, te, tr, info = env.step(a)
        ret += r
        for k, v in info["reward_parts"].items():
            parts[k] += env.cfg.reward_scale * v
        done = te or tr
        if trace > 0 and info["t"] - last_trace >= trace - 1e-9:
            last_trace = info["t"]
            print(f"  t={info['t']:6.1f} {info.get('task_category', '-'):12s} h={info['altitude']:6.1f} "
                  f"(*{info['target']['h']:6.1f}) u={info['airspeed_kt']:5.1f} kt (*{info['u_target_kt']:5.1f}) "
                  f"ψ={info['heading_unwrapped']:7.1f} (*{info['target']['psi']:7.1f}) vs={info['vertical_speed']:5.1f} "
                  f"vy={info['lateral_airspeed']:5.1f} xy={info['err_xy']:6.1f} φ={info['roll_deg']:5.1f} "
                  f"θ={info['pitch_deg']:5.1f} psi={info['torque_psi']:5.1f} rpm={info['rotor_rpm']:5.1f} "
                  f"c={np.round(info['controls'], 3)}", flush=True)
    ps = info.get("physics_summary", {})
    res = [dict(category=c["category"], success=bool(c["success"]), settle_s=c["settle_s"], T=round(c["T"], 1),
                deadline=round(c["deadline"], 1), on_time=bool(c["on_time"]), coupling_ok=bool(c["coupling_ok"]),
                max_err={k: round(float(v), 1) for k, v in c["max_abs_err"].items()}) for c in info["command_results"]]
    return dict(mode=mode, seed=seed, level=info["level"], ret=round(ret, 1), termination=info["termination"],
                episode_success=bool(info["episode_success"]), commands_ok=info["commands_ok"],
                commands_total=info["commands_total"], results=res, t_end=round(info["t"], 1),
                env_stage=env.env_stage, peak_psi=round(float(ps.get("peak_psi", float("nan"))), 1),
                over56_s=round(float(ps.get("t_over_limit", 0.0)), 1), min_rpm=round(float(ps.get("min_rpm", float("nan"))), 1),
                fuel_used=round(float(ps.get("fuel_used_lbs", 0.0)), 1),
                parts={k: round(v, 1) for k, v in parts.items()})


def show(r: dict, name: str):
    cats = " ".join(f"{c['category']}:{'✓' if c['success'] else '✗'}"
                    + (f"({c['settle_s']:.0f}/{c['deadline']:.0f}s)" if c['settle_s'] == c['settle_s'] else "")
                    for c in r["results"])
    print(f"{name:24s} {r['mode']:8s} s{r['seed']:<3d} {r['level']:3s} {r['env_stage']:6s} getiri {r['ret']:7.1f} "
          f"{r['commands_ok']}/{r['commands_total']} {r['termination']:22s} tepe {r['peak_psi']:5.1f} psi "
          f">56 {r['over56_s']:4.1f}s rpm≥{r['min_rpm']:5.1f}  {cats}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="all", help="virgülle; 'all' ya da 'none'")
    ap.add_argument("--levels", default="", help="örn. F2,F4,F5,F8")
    ap.add_argument("--seeds", type=int, default=4, help="seviye başına episode")
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--zero", action="store_true", help="aynı episode'ları sıfır action ile de uç")
    ap.add_argument("--trace", type=float, default=0.0)
    ap.add_argument("--env", default="{}", help="FlightEnvConfig alanları (JSON)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    cfg = FlightEnvConfig(**json.loads(args.env))
    modes = ["scripted"] + (["zero"] if args.zero else [])
    out = []
    t0 = time.time()
    names = list(SCENARIOS) if args.scenarios == "all" else [n for n in args.scenarios.split(",") if n and n != "none"]
    for name in names:
        sc = SCENARIOS[name]
        env = HelicopterEnvFlight(level="F8", config=cfg)
        opts = dict(sc, live=False)
        for mode in modes:
            r = run_episode(env, mode, 1, dict(opts, start_heading_deg=30.0, fuel=(300.0, 300.0)), args.trace)
            r["scenario"] = name
            show(r, name)
            out.append(r)
    for lvl in [x for x in args.levels.split(",") if x]:
        env = HelicopterEnvFlight(level=lvl, config=cfg)
        for i in range(args.seeds):
            for mode in modes:
                r = run_episode(env, mode, args.seed0 + i, None, args.trace)
                r["scenario"] = f"level {lvl}"
                show(r, f"level {lvl}")
                out.append(r)
    # özet: kategori başına başarı (pilot)
    agg = collections.defaultdict(lambda: [0, 0])
    for r in out:
        if r["mode"] != "scripted":
            continue
        for c in r["results"]:
            agg[c["category"]][0] += int(c["success"])
            agg[c["category"]][1] += 1
    print("\npilot, kategori başına başarı: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(agg.items())))
    if args.zero:
        pr = [(a["ret"], b["ret"]) for a, b in zip(out[0::2], out[1::2])]
        print(f"getiri pilot > sıfır action: {sum(p > z for p, z in pr)}/{len(pr)} "
              f"(ort. {np.mean([p for p, _ in pr]):.1f} ↔ {np.mean([z for _, z in pr]):.1f})")
    print(f"({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(summary={k: v for k, v in agg.items()}, episodes=out), indent=1,
                                              ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
