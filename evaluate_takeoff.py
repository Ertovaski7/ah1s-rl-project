from __future__ import annotations

"""
EVALUATE TAKEOFF — kalkış / hover / iniş ajanının sabit senaryolarla testi ve dört kumandanın kullanımı
=====================================================================================================

Eğitimde kullanılmayan sabit senaryolar (28) (deterministik policy, `helicopter_env_takeoff.py`):

  kalkış     : yerden 15 / 50 / 150 / 500 / 1000 ft hover
  manevra    : 100 ft hover'da yerinde dönüş (+90°, −180°, +360°), ileri / geri / sağa / sola kayma, bob-up / down
  hedef      : kalkış sürerken yeni irtifa (800 → 300 ft, 200 → 600 ft, 500 → 20 ft alçak hover)
  iniş       : 50 ft'ten, 300 ft'ten, yana kaydıktan sonra
  ağırlık    : tam yakıt (10280 lbs) kalkış + dönüş, tam yakıtla iniş (100 ft ve 300 ft'ten); öne / arkaya kaymış CG
  bozucu     : 100 ft hover'da lateral cyclic darbesi, pedal darbesi, 10 ft/s yana itki, +10° yatış, 8° burun yukarı

Her koşu için: görev başarısı (süre hedefi + tutma + kuplaj), oturma / son sınır, kalkış anı, en büyük sürüklenme,
heading sapması, açılar, tırmanış hızı, iniş temas hızı ve **dört kumandanın kullanımı** (trimden en büyük / RMS
sapma; kalkışta collective ile pedal arasındaki ilişki = tork telafisi).

Kullanım
  python evaluate_takeoff.py --model models_takeoff/takeoff_final.zip
  python evaluate_takeoff.py --model runs/to/models/latest.zip --only kalkis_1000 inis_300 --json out.json
  python evaluate_takeoff.py --model ... --levels K2,K5,K7,K9 --episodes 50      # seviyelerden rastgele episode
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TO = lambda h: dict(kind="takeoff", h=float(h))           # noqa: E731
TURN = lambda d: dict(kind="turn", dpsi=float(d))         # noqa: E731
MOVE = lambda x, y: dict(kind="move", dx=float(x), dy=float(y))   # noqa: E731
BOB = lambda d: dict(kind="bob", dh=float(d))             # noqa: E731
LAND = dict(kind="land")
HOLD = dict(kind="hold")

# (id, grup, başlık, seviye (çeviklik / süre hedefi), başlangıç, görevler, yakıt (tank0, tank1))
SCENARIOS = [
    ("kalkis_15", "kalkış", "Yerden 15 ft alçak hover", "K2", "ground", [TO(15)], (0, 0)),
    ("kalkis_50", "kalkış", "Yerden 50 ft hover", "K3", "ground", [TO(50)], (0, 0)),
    ("kalkis_150", "kalkış", "Yerden 150 ft hover", "K3", "ground", [TO(150)], (0, 0)),
    ("kalkis_500", "kalkış", "Yerden 500 ft hover", "K4", "ground", [TO(500)], (0, 0)),
    ("kalkis_1000", "kalkış", "Yerden 1000 ft hover", "K4", "ground", [TO(1000)], (0, 0)),
    ("donus_90", "manevra", "100 ft hover: yerinde +90° dönüş", "K5", "hover", [HOLD, TURN(90)], (0, 0)),
    ("donus_180", "manevra", "100 ft hover: yerinde −180° dönüş", "K5", "hover", [HOLD, TURN(-180)], (0, 0)),
    ("donus_360", "manevra", "100 ft hover: yerinde +360° dönüş (pirouette)", "K5", "hover", [HOLD, TURN(360)], (0, 0)),
    ("kayma_ileri_geri", "manevra", "100 ft hover: 50 ft ileri, sonra 50 ft geri", "K5", "hover",
     [HOLD, MOVE(50, 0), MOVE(-50, 0)], (0, 0)),
    ("kayma_yan", "manevra", "100 ft hover: 40 ft sağa, sonra 40 ft sola (sidestep)", "K5", "hover",
     [HOLD, MOVE(0, 40), MOVE(0, -40)], (0, 0)),
    ("bob_up_down", "manevra", "30 ft hover: +40 ft bob-up, −40 ft bob-down", "K5", "hover30",
     [HOLD, BOB(40), BOB(-40)], (0, 0)),
    ("kombine", "manevra", "Kalkış 80 ft → +90° → 30 ft sağa → +30 ft → −180°", "K5", "ground",
     [TO(80), TURN(90), MOVE(0, 30), BOB(30), TURN(-180)], (0, 0)),
    ("hedef_800_300", "hedef", "Kalkış 800 ft, %30'da yeni hedef 300 ft", "K6", "ground",
     [TO(800), dict(kind="climb_to", h=300.0, _frac=0.3)], (0, 0)),
    ("hedef_200_600", "hedef", "Kalkış 200 ft, %50'de yeni hedef 600 ft", "K6", "ground",
     [TO(200), dict(kind="climb_to", h=600.0, _frac=0.5)], (0, 0)),
    ("hedef_500_20", "hedef", "Kalkış 500 ft, %40'ta alçak hover'a (20 ft) dön", "K6", "ground",
     [TO(500), dict(kind="climb_to", h=20.0, _frac=0.4)], (0, 0)),
    ("inis_50", "iniş", "Kalkış 50 ft → pad'e iniş", "K7", "ground", [TO(50), LAND], (0, 0)),
    ("inis_300", "iniş", "300 ft hover'dan iniş", "K7", "hover300", [HOLD, LAND], (0, 0)),
    ("inis_kayma", "iniş", "Kalkış 40 ft → 30 ft sola kay → iniş", "K7", "ground", [TO(40), MOVE(0, -30), LAND], (0, 0)),
    ("agir_kalkis", "ağırlık", "Tam yakıt (10280 lbs): kalkış 500 ft + 180° dönüş", "K8", "ground",
     [TO(500), TURN(180)], (890, 890)),
    ("agir_inis", "ağırlık", "Tam yakıt (10280 lbs): kalkış 100 ft → pad'e iniş", "K8", "ground",
     [TO(100), LAND], (890, 890)),
    ("agir_inis_300", "ağırlık", "Tam yakıt (10280 lbs): 300 ft hover'dan iniş", "K8", "hover300", [HOLD, LAND], (890, 890)),
    ("cg_on", "ağırlık", "CG önde (tank0 dolu): kalkış 100 ft, 40 ft ileri, iniş", "K8", "ground",
     [TO(100), MOVE(40, 0), LAND], (890, 0)),
    ("cg_arka", "ağırlık", "CG arkada (tank1 dolu): kalkış 100 ft, 40 ft geri, iniş", "K8", "ground",
     [TO(100), MOVE(-40, 0), LAND], (0, 890)),
    ("bozucu_lat", "bozucu", "100 ft hover: lateral cyclic +0.3 darbesi (0.8 s)", "K8", "hover",
     [HOLD, dict(kind="recover", dist="kick", axis=2, mag=0.3, dur=0.8)], (0, 0)),
    ("bozucu_pedal", "bozucu", "100 ft hover: pedal −0.35 darbesi (0.8 s)", "K8", "hover",
     [HOLD, dict(kind="recover", dist="kick", axis=3, mag=-0.35, dur=0.8)], (0, 0)),
    ("bozucu_itki", "bozucu", "100 ft hover: 10 ft/s yana itki", "K8", "hover",
     [HOLD, dict(kind="recover", dist="push", dn=0.0, de=10.0, dw=0.0)], (0, 0)),
    ("bozucu_yatis", "bozucu", "100 ft hover: +10° yatış bozucusu", "K8", "hover",
     [HOLD, dict(kind="recover", dist="tilt", dphi=10.0, dtheta=0.0)], (0, 0)),
    ("bozucu_burun", "bozucu", "100 ft hover: 8° burun yukarı bozucusu", "K8", "hover",
     [HOLD, dict(kind="recover", dist="tilt", dphi=0.0, dtheta=8.0)], (0, 0)),
]
GROUPS = ("kalkış", "manevra", "hedef", "iniş", "ağırlık", "bozucu")
CH = ("collective", "long. cyclic", "lat. cyclic", "pedal")


def load(path):
    from stable_baselines3 import PPO
    m = PPO.load(str(path), device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def run_one(env, policy, sc, seed=0):
    sid, group, title, level, start, tasks, fuel = sc
    opts = dict(level=level, tasks=[dict(t) for t in tasks], fuel=fuel, start_heading_deg=0.0, start_perturb=0.0)
    if start.startswith("hover"):
        opts.update(start="hover", start_alt_ft=float(start[5:] or 100.0))
    else:
        opts.update(start="ground")
    obs, info = env.reset(seed=seed, options=opts)
    trim = env.trim
    rows, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        rows.append((info["t"], info["altitude"], info["err_xy"], info["err_heading"], info["roll_deg"],
                     info["pitch_deg"], info["vertical_speed"], info["ground_speed"], info["wow"], *info["controls"]))
        done = term or trunc
    a = np.array(rows)
    res = info.get("command_results") or []
    ctrl = a[:, 9:13] - trim
    lift = float(a[np.argmax(a[:, 8] == 0), 0]) if start == "ground" and (a[:, 8] == 0).any() else None
    # kalkışta tork telafisi: collective ile pedal arasındaki korelasyon (ilk 8 s)
    corr = None
    if start == "ground":
        k = a[:, 0] <= 8.0
        if k.sum() > 10 and np.std(a[k, 9]) > 1e-4 and np.std(a[k, 12]) > 1e-4:
            corr = float(np.corrcoef(a[k, 9], a[k, 12])[0, 1])
    td = [c["touchdown_vs"] for c in res if c["kind"] == "land" and c["touchdown_vs"] is not None]
    return dict(
        id=sid, group=group, title=title, level=level, safe=info.get("termination") == "time_limit",
        termination=info.get("termination"), episode_ok=bool(info.get("episode_success")),
        n_ok=sum(c["success"] for c in res), n=len(res),
        windows=[dict(kind=c["kind"], success=c["success"], T=round(c["T"], 1), deadline=round(c["deadline"], 1),
                      settle_s=None if not np.isfinite(c["settle_s"]) else round(c["settle_s"], 1),
                      interrupted=c["interrupted"], coupling_ok=c["coupling_ok"],
                      max_err={k: round(v, 1) for k, v in c["max_abs_err"].items()},
                      touchdown_vs=None if c["touchdown_vs"] is None else round(c["touchdown_vs"], 2)) for c in res],
        liftoff_s=lift, max_drift=float(a[:, 2].max()), max_roll=float(np.abs(a[:, 4]).max()),
        max_pitch=float(np.abs(a[:, 5]).max()), max_climb=float(a[:, 6].max()), max_sink=float(-a[:, 6].min()),
        touchdown_vs=td[0] if td else None,
        ctrl_max=[float(x) for x in np.abs(ctrl).max(axis=0)], ctrl_rms=[float(x) for x in np.sqrt((ctrl ** 2).mean(axis=0))],
        coll_pedal_corr=corr, duration=float(a[-1, 0]))


def summarize(results):
    s = {}
    for g in GROUPS + (None,):
        rs = [r for r in results if g is None or r["group"] == g]
        if not rs:
            continue
        s[g or "toplam"] = dict(safe=[sum(r["safe"] for r in rs), len(rs)], episode_ok=[sum(r["episode_ok"] for r in rs), len(rs)],
                                tasks_ok=[sum(r["n_ok"] for r in rs), sum(r["n"] for r in rs)])
    s["kumanda_rms"] = dict(zip(CH, np.mean([r["ctrl_rms"] for r in results], axis=0).round(3).tolist()))
    s["kumanda_max"] = dict(zip(CH, np.max([r["ctrl_max"] for r in results], axis=0).round(3).tolist()))
    corr = [r["coll_pedal_corr"] for r in results if r["coll_pedal_corr"] is not None]
    s["kalkis_coll_pedal_corr"] = float(np.mean(corr)) if corr else None
    return s


def fmt(r):
    tick = lambda b: "✓" if b else "✗"          # noqa: E731
    w = r["windows"]
    tasks = " ".join(f"{x['kind'][:4]}{'✓' if x['success'] else '✗'}"
                     + (f"{x['settle_s']:.0f}/{x['deadline']:.0f}" if x["settle_s"] is not None and not x["interrupted"] else "")
                     for x in w)
    td = f"{r['touchdown_vs']:+.1f}" if r["touchdown_vs"] is not None else "—"
    lift = f"{r['liftoff_s']:.1f}" if r["liftoff_s"] is not None else "—"
    return (f"{r['id']:18s} {tick(r['safe']):>3s} {tick(r['episode_ok']):>3s} {r['n_ok']:>2d}/{r['n']:<2d} {lift:>5s} "
            f"{r['max_drift']:5.1f} {r['max_roll']:4.0f}° {r['max_pitch']:3.0f}° {r['max_climb']:5.1f} {td:>5s}  "
            f"{' '.join(f'{x:.2f}' for x in r['ctrl_max'])}  {tasks}  {'' if r['safe'] else r['termination']}")


HEADER = (f"{'senaryo':18s} {'güv':>3s} {'tüm':>3s} {'görev':>5s} {'kalk':>5s} {'drift':>5s} {'maxφ':>5s} {'maxθ':>4s} "
          f"{'tırm':>5s} {'temas':>5s}  {'kumanda max |Δ| (coll lon lat ped)':34s}  görevler (oturma/son sınır s)")


def evaluate(model_path, only=None, verbose=True):
    from helicopter_env_takeoff import HelicopterEnvTakeoff
    env = HelicopterEnvTakeoff(level="K9")
    pol = load(model_path)
    results = []
    if verbose:
        print(f"\nKALKIŞ / HOVER / İNİŞ — model {Path(model_path).name}")
        print(HEADER)
    for sc in SCENARIOS:
        if only and sc[0] not in only:
            continue
        r = run_one(env, pol, sc)
        results.append(r)
        if verbose:
            print(fmt(r), flush=True)
    s = summarize(results)
    if verbose:
        for g in GROUPS + ("toplam",):
            if g in s:
                x = s[g]
                print(f"  {g:8s} güvenli {x['safe'][0]}/{x['safe'][1]}   tüm görevler {x['episode_ok'][0]}/{x['episode_ok'][1]}   "
                      f"görev {x['tasks_ok'][0]}/{x['tasks_ok'][1]}")
        print("  kumanda kullanımı (trimden sapma, RMS / en büyük): " + "  ".join(
            f"{c} {s['kumanda_rms'][c]:.3f}/{s['kumanda_max'][c]:.2f}" for c in CH))
        if s["kalkis_coll_pedal_corr"] is not None:
            print(f"  kalkışta collective–pedal korelasyonu (tork telafisi): {s['kalkis_coll_pedal_corr']:+.2f}")
    return results, s


def level_stats(model_path, levels, n, seed0=70000):
    from helicopter_env_takeoff import HelicopterEnvTakeoff
    env = HelicopterEnvTakeoff(level=levels[0])
    pol = load(model_path)
    out = {}
    for lvl in levels:
        ok, term, cats = [], Counter(), defaultdict(list)
        for k in range(n):
            obs, info = env.reset(seed=seed0 + k, options=dict(level=lvl))
            done = False
            while not done:
                obs, r, te, tr, info = env.step(pol(obs))
                done = te or tr
            ok.append(bool(info["episode_success"]))
            term[info["termination"]] += 1
            for c in info["command_results"]:
                cats[c["category"]].append(bool(c["success"]))
        out[lvl] = dict(episode=float(np.mean(ok)), n=n, termination=dict(term),
                        categories={c: [float(np.mean(v)), len(v)] for c, v in sorted(cats.items())})
        print(f"{lvl}: episode başarısı {np.mean(ok):.0%} ({n})  bitiş {dict(term)}  " +
              " ".join(f"{c}={np.mean(v):.0%}({len(v)})" for c, v in sorted(cats.items())), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--levels", default=None, help="ayrıca bu seviyelerden rastgele episode (virgülle, ör. K2,K5,K9)")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--no-scenarios", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = dict(model=str(args.model))
    if not args.no_scenarios:
        res, summ = evaluate(args.model, args.only)
        out.update(results=res, summary=summ)
    if args.levels:
        print()
        out["levels"] = level_stats(args.model, [x.strip() for x in args.levels.split(",") if x.strip()], args.episodes)
    print(f"\n({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"kaydedildi: {args.json}")


if __name__ == "__main__":
    main()
