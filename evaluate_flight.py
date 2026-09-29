from __future__ import annotations

"""
EVALUATE FLIGHT — tek ajanlı sürekli uçuşun sabit senaryo takımı (helicopter_env_flight.py)
==========================================================================================

Eğitimde kullanılmayan sabit senaryolar (deterministik policy). Takım:

  zincir (12) : kısa / uzun görev zinciri × sakin / 15 kt rüzgâr / 25 kt rüzgâr + orta türbülans × hafif / ağır
                  kısa : yerden kalkış 50 ft → 60 kt'a hızlanma (+100 ft) → duruş → iniş
                  uzun : kalkış 30 ft → 80 kt (+200 ft) → +20 kt → +90° → −100 ft → (−30 kt, −60°, +50 ft) → +180°
                         → duruş → iniş
                  hafif 8800 lbs (2 × 150 lbs yakıt), ağır 9700 lbs (2 × 600 lbs; OGE hover ~53 psi, sınıra yakın)
  öğe (9)     : hover 30 s (sakin; 15 kt yandan rüzgâr + hafif türbülans), ileri uçuşta hız 60 → 100 → 40 kt,
                  80 kt'ta +90° / −180° dönüş, 80 kt'ta +300 / −300 ft, hover'dan 100 kt'a hızlanma, 100 kt'tan duruş,
                  300 ft'ten iniş

Her senaryo için: güvenli mi (başarısızlık yok), görev başarısı (env'in bandı + süre hedefi + tutma + kuplaj),
**ADS-33 benzeri sınıf** (her görev penceresi: istenen = env başarısı; yeterli = bantlar 2×, son sınır 1.5×, kuplaj
sınırları 2× — ADS-33E'deki desired / adequate ayrımı gibi; Cooper-Harper değil), tutma doğruluğu (tutma süresindeki
en büyük hatalar), tork (en yüksek psi, 50 / 56 psi üstü süre, 56 üstü en uzun kesintisiz süre, en düşük rotor devri),
yakıt (harcanan lbs, ortalama akış lb/h), bitişteki ağırlık.

Kullanım
  python evaluate_flight.py --model runs/fl_v1/models/best.zip --json docs/flight/eval_best.json
  python evaluate_flight.py --model ... --only uzun_sakin_hafif hover_ruzgar
  python evaluate_flight.py --scripted                     # kural tabanlı PID pilot (karşılaştırma; RL değil)
  python evaluate_flight.py --model ... --levels F3,F6,F9 --episodes 20 --no-scenarios
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helicopter_env_command import CONTROL_DT  # noqa: E402

KT = 1.6878099
TO = lambda h: dict(kind="takeoff", h=float(h))                                          # noqa: E731
ACC = lambda kt, dh: dict(kind="cruise", u_kt=float(kt), dh=float(dh), accel=True)      # noqa: E731
CR = lambda **k: dict(kind="cruise", **{a: float(b) for a, b in k.items()})              # noqa: E731
STOP = dict(kind="stop", decel=2.5)
LAND = dict(kind="land")
HOLD = dict(kind="hold")
CHOLD = dict(kind="cruise", hold=True)

CHAIN_SHORT = [TO(50), ACC(60, 100), STOP, LAND]
CHAIN_LONG = [TO(30), ACC(80, 200), CR(du_kt=20), CR(dpsi=90), CR(dh=-100), CR(du_kt=-30, dpsi=-60, dh=50),
              CR(dpsi=180), STOP, LAND]
ENVS = {
    "sakin": dict(wind_kt=0.0, wind_dir_deg=0.0, turb_level="none"),
    "ruzgar": dict(wind_kt=15.0, wind_dir_deg=45.0, turb_level="none"),
    "turb": dict(wind_kt=25.0, wind_dir_deg=-120.0, turb_level="moderate", gust_rate_per_min=1.0, gust_kt=(5.0, 12.0)),
}
ENV_TITLE = {"sakin": "sakin", "ruzgar": "15 kt rüzgâr (sağ ön 45°)", "turb": "25 kt rüzgâr (sol arka) + orta türbülans + gust"}
WEIGHTS = {"hafif": (150.0, 150.0), "agir": (600.0, 600.0)}

# (id, grup, başlık, başlangıç, görevler, yakıt, çevre, süre sınırı s)
SCENARIOS = []
for _cn, _tasks, _title in (("kisa", CHAIN_SHORT, "kısa zincir"), ("uzun", CHAIN_LONG, "uzun zincir")):
    for _en in ENVS:
        for _wn, _fuel in WEIGHTS.items():
            SCENARIOS.append((f"{_cn}_{_en}_{_wn}", f"zincir/{_cn}", f"{_title}, {ENV_TITLE[_en]}, "
                              f"{8500 + sum(_fuel):.0f} lbs", "ground", _tasks, _fuel, _en, 600.0))
SCENARIOS += [
    ("hover_sakin", "öğe", "100 ft hover, 30 s", "hover100", [HOLD], (300.0, 300.0), "sakin", 40.0),
    ("hover_ruzgar", "öğe", "50 ft hover, 30 s; 15 kt yandan rüzgâr + hafif türbülans", "hover50", [HOLD], (300.0, 300.0),
     dict(wind_kt=15.0, wind_dir_deg=90.0, turb_level="light"), 40.0),
    ("hiz_60_100_40", "öğe", "300 ft / 60 kt: → 100 kt → 40 kt", "cruise60@300", [CHOLD, CR(du_kt=40), CR(du_kt=-60)],
     (300.0, 300.0), "sakin", 180.0),
    ("donus_80kt", "öğe", "300 ft / 80 kt: +90°, −180°", "cruise80@300", [CHOLD, CR(dpsi=90), CR(dpsi=-180)],
     (300.0, 300.0), "sakin", 180.0),
    ("tirmanis_80kt", "öğe", "300 ft / 80 kt: +300 ft, −300 ft", "cruise80@300", [CHOLD, CR(dh=300), CR(dh=-300)],
     (300.0, 300.0), "sakin", 180.0),
    ("hizlanma_100", "öğe", "50 ft hover → 100 kt (+150 ft)", "hover50", [HOLD, ACC(100, 150)], (300.0, 300.0), "sakin",
     150.0),
    ("durus_100", "öğe", "300 ft / 100 kt → duruş (hover)", "cruise100@300", [CHOLD, STOP], (300.0, 300.0), "sakin",
     150.0),
    ("inis_300", "öğe", "300 ft hover → pad'e iniş", "hover300", [HOLD, LAND], (300.0, 300.0), "sakin", 150.0),
    ("hizlanma_ruzgar", "öğe", "50 ft hover → 80 kt, 20 kt arkadan rüzgâr + hafif türbülans", "hover50",
     [HOLD, ACC(80, 150)], (300.0, 300.0), dict(wind_kt=20.0, wind_dir_deg=180.0, turb_level="light"), 150.0),
]
GROUPS = ("zincir/kisa", "zincir/uzun", "öğe")


# ---------------------------------------------------------------------------------------------------------------------
def load_policy(model_path):
    from stable_baselines3 import PPO
    m = PPO.load(str(model_path), device="cpu")
    return (lambda o: m.predict(o, deterministic=True)[0]), dict(getattr(m, "ah1s_env_overrides", None) or {})


def make_env(model_path=None, env_overrides: dict | None = None, level="F9", scripted=False):
    """(env, policy(obs) → action, ayarlar). scripted=True → kural tabanlı pilot (docs/flight/scripted_pilot.py)."""
    from helicopter_env_flight import FlightEnvConfig, HelicopterEnvFlight
    ov = {}
    pol = None
    if not scripted:
        pol, ov = load_policy(model_path)
    ov.update(env_overrides or {})
    env = HelicopterEnvFlight(level=level, config=FlightEnvConfig(**ov))
    if scripted:
        sys.path.insert(0, str(REPO_ROOT / "docs" / "flight"))
        from scripted_pilot import ScriptedPilot
        pilot = ScriptedPilot(env)
        pol = lambda o: pilot()                                                          # noqa: E731
        pol.pilot = pilot
    return env, pol, ov


def _start_opts(start: str) -> dict:
    if start == "ground":
        return dict(start="ground")
    if start.startswith("hover"):
        return dict(start="hover", start_alt_ft=float(start[5:] or 100.0))
    kt, h = start[6:].split("@")
    return dict(start="cruise", start_speed_kt=float(kt), start_alt_ft=float(h))


def _x2_cfg(env, k=2.0):
    """ADS-33 'yeterli' bandı: env'in (türbülansa göre ölçeklenmiş) bantları × k."""
    from helicopter_env_flight import TOL_SCALAR, TOL_TUPLE
    c = env.cfg
    upd = {f: float(getattr(c, f)) * k for f in TOL_SCALAR}
    upd.update({f: tuple(float(v) * k for v in getattr(c, f)) for f in TOL_TUPLE})
    return replace(c, **upd)


def run_one(env, policy, sc, seed=0):
    sid, group, title, start, tasks, fuel, env_name, episode_s = sc
    phys = dict(ENVS[env_name] if isinstance(env_name, str) else env_name)
    phys.setdefault("wind_dir_relative", True)
    opts = dict(level="F9", tasks=[dict(t) for t in tasks], fuel=fuel, start_heading_deg=0.0, start_perturb=0.0,
                physics=phys, episode_s=episode_s, live=episode_s <= 40.0, **_start_opts(start))
    obs, info = env.reset(seed=seed, options=opts)
    if hasattr(policy, "pilot"):
        policy.pilot.reset()
    cfg2 = _x2_cfg(env)
    rows, done = [], False
    fuel0 = float(info["fuel_lbs"])
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        s = env._state()
        e = env._errors(s)
        wi = len(env.windows) - 1
        ins2 = False
        if env.windows and not term:
            w = env.windows[-1]
            cfg0, tol0 = env.cfg, w["tol"]
            env.cfg, w["tol"] = cfg2, (2.0 * tol0[0], 2.0 * tol0[1])
            try:
                ins2 = bool(env._inside(s, e, w))
            finally:
                env.cfg, w["tol"] = cfg0, tol0
        rows.append((info["t"], wi, info["altitude"], info["err_xy"], abs(info["err_altitude"]), abs(info["err_heading"]),
                     abs(info["err_speed"]), info["ground_speed"], info["vertical_speed"], info["airspeed_kt"],
                     info["lateral_airspeed"], info["torque_psi"], info["rotor_rpm"], info["fuel_lbs"], float(ins2),
                     info["roll_deg"], info["pitch_deg"], info["weight_lbs"]))
    a = np.array(rows)
    res = info.get("command_results") or []
    ends = [c["t"] for c in res[1:]] + [float(a[-1, 0])]
    wins = []
    for i, c in enumerate(res):
        k = a[:, 1] == i
        seg = a[k]
        cls = "istenen" if c["success"] else "başarısız"
        cut_at_end = (i == len(res) - 1 and c["cmd"].get("auto") and not c["success"]
                      and info.get("termination") == "time_limit")      # canlı modda süre bitince yarım kalan oto-tutma
        if c["interrupted"] or cut_at_end:
            cls = "kesildi"
        elif not c["success"] and seg.size:
            # yeterli: 2× bantta (tutma süresi kadar kesintisiz), girişi ≤ 1.5 × son sınır, kuplaj 2× sınırda
            hold_n = int(round(c["hold_s"] / CONTROL_DT))
            ins = seg[:, 14] > 0.5
            ok_t = None
            run_ = 0
            for j in range(len(ins)):
                run_ = run_ + 1 if ins[j] else 0
                if run_ >= hold_n:
                    ok_t = seg[j - hold_n + 1, 0] - c["t"]
                    break
            lims = dict(zip(("u", "psi", "h"), env.cfg.coupling_cruise)) if c["kind"] == "cruise" else \
                dict(zip(("xy", "h", "psi"), env.cfg.coupling_limits))           # (türbülansa göre ölçekli)
            coup_ok = all(c["max_abs_err"].get(ax, 0.0) <= 2.0 * lim for ax, lim in lims.items()
                          if ax not in c.get("active", tuple(lims)))
            if ok_t is not None and ok_t <= 1.5 * c["deadline"] and coup_ok:
                cls = "yeterli"
        tail = seg[seg[:, 0] >= min(ends[i], seg[-1, 0] if seg.size else 0) - c["hold_s"]] if seg.size else seg
        wins.append(dict(
            category=c["category"], kind=c["kind"], success=bool(c["success"]), ads33=cls, T=round(c["T"], 1),
            deadline=round(c["deadline"], 1),
            settle_s=None if not np.isfinite(c["settle_s"]) else round(float(c["settle_s"]), 1),
            coupling_ok=bool(c["coupling_ok"]), touchdown_vs=None if c["touchdown_vs"] is None else round(c["touchdown_vs"], 2),
            max_err={k2: round(float(v), 1) for k2, v in c["max_abs_err"].items()},
            hold_err=None if not tail.size else dict(xy=round(float(tail[:, 3].max()), 1), h=round(float(tail[:, 4].max()), 1),
                                                    psi=round(float(tail[:, 5].max()), 1),
                                                    u_fps=round(float(tail[:, 6].max()), 1))))
    q, rpm = a[:, 11], a[:, 12]
    over = q > 56.0
    run_, longest = 0, 0
    for o in over:
        run_ = run_ + 1 if o else 0
        longest = max(longest, run_)
    dur = float(a[-1, 0])
    used = fuel0 - float(a[-1, 13])
    ps = info.get("physics_summary") or {}
    keep = [w for w in wins if w["ads33"] != "kesildi" or w["category"] == "interrupted"]
    wins = keep
    return dict(
        id=sid, group=group, title=title, env=env_name if isinstance(env_name, str) else "özel",
        safe=info.get("termination") in ("time_limit", "fuel_exhausted"), termination=info.get("termination"),
        episode_ok=bool(info.get("episode_success")), n_ok=sum(w["success"] for w in wins), n=len(wins),
        n_adequate=sum(w["ads33"] in ("istenen", "yeterli") for w in wins), windows=wins, duration=dur,
        torque_max=float(q.max()), torque_p95=float(np.percentile(q, 95)), t_over50_s=float((q > 50.0).sum() * CONTROL_DT),
        t_over56_s=float(over.sum() * CONTROL_DT), longest_over56_s=float(longest * CONTROL_DT), rpm_min=float(rpm.min()),
        fuel_used_lbs=used, fuel_flow_lbh=used / max(dur, 1e-6) * 3600.0, weight_end=float(a[-1, 17]),
        max_airspeed_kt=float(a[:, 9].max()), max_roll=float(np.abs(a[:, 15]).max()), max_pitch=float(np.abs(a[:, 16]).max()),
        events=[e.get("event", str(e)) for e in ps.get("events", [])], tol_scale=float(env.tol_scale))


def summarize(results):
    s = {}
    for g in GROUPS + (None,):
        rs = [r for r in results if g is None or r["group"] == g]
        if not rs:
            continue
        s[g or "toplam"] = dict(safe=[sum(r["safe"] for r in rs), len(rs)], episode_ok=[sum(r["episode_ok"] for r in rs), len(rs)],
                                tasks_ok=[sum(r["n_ok"] for r in rs), sum(r["n"] for r in rs)],
                                tasks_adequate=[sum(r["n_adequate"] for r in rs), sum(r["n"] for r in rs)])
    for en in ENVS:
        rs = [r for r in results if r["env"] == en and r["group"].startswith("zincir")]
        if rs:
            s[f"zincir_{en}"] = dict(safe=[sum(r["safe"] for r in rs), len(rs)], episode_ok=[sum(r["episode_ok"] for r in rs), len(rs)],
                                     tasks_ok=[sum(r["n_ok"] for r in rs), sum(r["n"] for r in rs)])
    cats = defaultdict(lambda: [0, 0, 0])
    for r in results:
        for w in r["windows"]:
            if w["ads33"] == "kesildi":
                continue
            cats[w["category"]][0] += int(w["ads33"] == "istenen")
            cats[w["category"]][1] += int(w["ads33"] in ("istenen", "yeterli"))
            cats[w["category"]][2] += 1
    s["kategori"] = {k: dict(istenen=v[0], yeterli_veya_iyi=v[1], n=v[2]) for k, v in sorted(cats.items())}
    s["tork"] = dict(senaryo_56_ustu=[sum(r["t_over56_s"] > 0 for r in results), len(results)],
                     maks_psi=float(max(r["torque_max"] for r in results)),
                     sure_50_ustu_s=float(sum(r["t_over50_s"] for r in results)),
                     sure_56_ustu_s=float(sum(r["t_over56_s"] for r in results)),
                     en_uzun_56_ustu_s=float(max(r["longest_over56_s"] for r in results)),
                     sure_toplam_s=float(sum(r["duration"] for r in results)),
                     rpm_min=float(min(r["rpm_min"] for r in results)))
    s["yakit"] = dict(harcanan_lbs=float(sum(r["fuel_used_lbs"] for r in results)),
                      ort_akis_lbh=float(np.mean([r["fuel_flow_lbh"] for r in results])))
    return s


SHORT = {"takeoff": "kalk", "accel": "hızl", "cruise_u": "hız", "cruise_psi": "dön", "cruise_h": "irt", "cruise_mix": "kar",
         "cruise_hold": "tut", "stop": "dur", "land": "iniş", "hold": "hovr", "interrupted": "kes"}


def fmt(r):
    tick = lambda b: "✓" if b else "✗"          # noqa: E731
    mark = {"istenen": "✓", "yeterli": "~", "başarısız": "✗", "kesildi": "·"}
    tasks = " ".join(f"{SHORT.get(w['category'], w['category'][:4])}{mark[w['ads33']]}"
                     + (f"{w['settle_s']:.0f}/{w['deadline']:.0f}" if w["settle_s"] is not None else "") for w in r["windows"])
    return (f"{r['id']:22s} {tick(r['safe']):>3s} {tick(r['episode_ok']):>3s} {r['n_ok']:>2d}/{r['n']:<2d} {r['n_adequate']:>2d} "
            f"{r['torque_max']:5.1f} {r['t_over50_s']:5.0f} {r['t_over56_s']:5.1f} {r['longest_over56_s']:4.1f} {r['rpm_min']:4.0f} "
            f"{r['fuel_used_lbs']:5.0f} {r['fuel_flow_lbh']:4.0f} {r['max_airspeed_kt']:4.0f} {r['max_roll']:3.0f}°  {tasks}  "
            f"{'' if r['safe'] else r['termination']}")


HEADER = (f"{'senaryo':22s} {'güv':>3s} {'tüm':>3s} {'görev':>5s} {'yet':>2s} {'psi':>5s} {'>50s':>5s} {'>56s':>5s} "
          f"{'kes.':>4s} {'rpm':>4s} {'yakıt':>5s} {'lb/h':>4s} {'Vmax':>4s} {'φmax':>4s}  "
          f"görevler (✓ istenen, ~ yeterli, ✗, · kesildi; oturma/son sınır s)")


def evaluate(model_path=None, only=None, verbose=True, env_overrides=None, scripted=False, seed=0):
    env, pol, ov = make_env(model_path, env_overrides, scripted=scripted)
    results = []
    if verbose:
        who = "kural tabanlı PID pilot (RL değil)" if scripted else f"model {Path(model_path).name}"
        print(f"\nTEK AJANLI UÇUŞ — {who}  env ayarları {ov or 'varsayılan (FlightEnvConfig)'}")
        print(HEADER)
    for sc in SCENARIOS:
        if only and sc[0] not in only:
            continue
        r = run_one(env, pol, sc, seed=seed)
        results.append(r)
        if verbose:
            print(fmt(r), flush=True)
    s = summarize(results)
    if verbose:
        for g in GROUPS + tuple(f"zincir_{e}" for e in ENVS) + ("toplam",):
            if g in s:
                x = s[g]
                print(f"  {g:14s} güvenli {x['safe'][0]}/{x['safe'][1]}   tüm görevler {x['episode_ok'][0]}/{x['episode_ok'][1]}   "
                      f"görev {x['tasks_ok'][0]}/{x['tasks_ok'][1]}"
                      + (f"   (yeterli ya da iyi {x['tasks_adequate'][0]}/{x['tasks_adequate'][1]})" if "tasks_adequate" in x else ""))
        print("  kategori (istenen / yeterli+ / n): " + "  ".join(
            f"{k} {v['istenen']}/{v['yeterli_veya_iyi']}/{v['n']}" for k, v in s["kategori"].items()))
        tq = s["tork"]
        print(f"  tork: {tq['senaryo_56_ustu'][0]}/{tq['senaryo_56_ustu'][1]} senaryoda 56 psi aşıldı; 50 psi üstü "
              f"{tq['sure_50_ustu_s']:.0f} s, 56 üstü {tq['sure_56_ustu_s']:.0f} s / {tq['sure_toplam_s']:.0f} s (en uzun "
              f"kesintisiz {tq['en_uzun_56_ustu_s']:.1f} s); en yüksek {tq['maks_psi']:.1f} psi; en düşük devir {tq['rpm_min']:.0f}")
        print(f"  yakıt: toplam {s['yakit']['harcanan_lbs']:.0f} lbs, ortalama akış {s['yakit']['ort_akis_lbh']:.0f} lb/h")
    return results, s


def level_stats(model_path, levels, n, seed0=70000, env_overrides=None, scripted=False):
    env, pol, _ = make_env(model_path, env_overrides, level=levels[0], scripted=scripted)
    out = {}
    for lvl in levels:
        ok, term, cats, stages = [], Counter(), defaultdict(list), Counter()
        for k in range(n):
            obs, info = env.reset(seed=seed0 + k, options=dict(level=lvl))
            if hasattr(pol, "pilot"):
                pol.pilot.reset()
            stages[env.env_stage] += 1
            done = False
            while not done:
                obs, r, te, tr, info = env.step(pol(obs))
                done = te or tr
            ok.append(bool(info["episode_success"]))
            term[info["termination"]] += 1
            for c in info["command_results"]:
                cats[c["category"]].append(bool(c["success"]))
        out[lvl] = dict(episode=float(np.mean(ok)), n=n, termination=dict(term), env_stages=dict(stages),
                        categories={c: [float(np.mean(v)), len(v)] for c, v in sorted(cats.items())})
        print(f"{lvl}: episode başarısı {np.mean(ok):.0%} ({n})  çevre {dict(stages)}  bitiş {dict(term)}  " +
              " ".join(f"{c}={np.mean(v):.0%}({len(v)})" for c, v in sorted(cats.items())), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scripted", action="store_true", help="kural tabanlı PID pilot (karşılaştırma)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--levels", default=None, help="ayrıca bu seviyelerden rastgele episode (virgülle, ör. F3,F6,F9)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no-scenarios", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--env", default=None, help="env ayarları (JSON), modelin taşıdıklarının üstüne yazılır")
    args = ap.parse_args(argv)
    if not args.scripted and not args.model:
        ap.error("--model ya da --scripted gerekli")
    env_ov = json.loads(args.env) if args.env else None
    t0 = time.time()
    out = dict(model=str(args.model) if not args.scripted else "scripted_pilot")
    if not args.no_scenarios:
        res, summ = evaluate(args.model, args.only, env_overrides=env_ov, scripted=args.scripted, seed=args.seed)
        out.update(results=res, summary=summ)
    if args.levels:
        print()
        out["levels"] = level_stats(args.model, [x.strip() for x in args.levels.split(",") if x.strip()], args.episodes,
                                    env_overrides=env_ov, scripted=args.scripted)
    out["env"] = env_ov
    print(f"\n({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"kaydedildi: {args.json}")


if __name__ == "__main__":
    main()
