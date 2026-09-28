from __future__ import annotations

"""
EVALUATE ROBUSTNESS — manevra ajanı için zorlayıcı test senaryoları
===================================================================

Eğitimde KULLANILMAYAN, sabit (deterministik) senaryolar. İki grup:

  kesilen : manevra bitmeden gelen / ters yöne çeviren komutlar — slalomda ani ters
            dönüş, dönüş ortasında ani tırmanma, hızlanırken ani fren, tırmanırken ani
            dalış, art arda hızlı komutlar, birleşik komutun tersi.
  sınır   : zarf sınırları — hover (0–5 kt), 95–100 kt, alçak irtifa (250–400 ft),
            sınırda ters dönüş, en yüksek performans (100 kt'ta birleşik komut).
  rastgele: S1 / S2 seviyelerinin kendi takviminden sabit seed'li 10 episode (eğitim
            dağılımı; seed'ler eğitimde kullanılanlardan ayrı).

Heading içeren senaryolar aynaları (sağ ↔ sol) ile de koşulur (helikopter asimetrik:
kuyruk rotoru, tork).

Komutlar ölçülen duruma göre uygulanır (env ile aynı); kesilen komut kendi süre hedefiyle
yargılanmaz (başarı = komut verilmeyen eksenler kuplaj sınırında), kesen komutun süre
hedefi o anki hareketi (ters yöne yaw hızı, dikey hız, ivme) hesaba katar
(maneuver_curriculum.dynamic_time_target).

Ölçüler (koşu başına)
  güvenli     : güvenlik ihlali yok (uçuş bitmedi)
  son komut   : son komut zamanında oturdu + 5 s tuttu + kuplaj
  tüm komutlar: env'in episode başarısı (kesilenler dahil, kendi kuralıyla)
  tepki       : kesen komuttan sonra hareketin (yaw hızı / dikey hız / ivme) yeni yöne dönme süresi
  max |φ| |θ| |p| |r|, en düşük irtifa, doygunluk (|a| > 0.95 adım oranı)

Kullanım
  python evaluate_robustness.py --model models_maneuver/maneuver_M5_final.zip
  python evaluate_robustness.py --model runs/rob/models/latest.zip --compare models_maneuver/maneuver_M5_final.zip
  python evaluate_robustness.py --model ... --only slalom_ters_60kt --json out.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helicopter_env_command import CONTROL_DT  # noqa: E402

KT = 1.6878
AXES = ("heading", "speed", "altitude")


def H(x):
    return {"heading": float(x)}


def V(x):
    return {"speed": float(x)}


def A(x):
    return {"altitude": float(x)}


def C(h=0.0, v=0.0, a=0.0):
    return {"heading": float(h), "speed": float(v), "altitude": float(a)}


# (id, grup, başlık, başlangıç (ft, ft/s), süre s, [(t, komut), ...])
SCENARIOS = [
    # ---------------- kesilen / ters komutlar ----------------
    ("slalom_ters_60kt", "kesilen", "Slalom, her 2.5 s'de ters yön (60 kt)", (800, 60 * KT), 32,
     [(3, H(90)), (5.5, H(-90)), (8, H(90)), (10.5, H(-90)), (13, H(45))]),
    ("slalom_ters_hover", "kesilen", "Hover'da pedal dönüşü, 3 s sonra ters (5 kt)", (800, 5 * KT), 30,
     [(3, H(180)), (6, H(-180)), (9, H(90))]),
    ("donus_ani_tirmanis", "kesilen", "180° dönüşün ortasında +150 ft (60 kt)", (800, 60 * KT), 35,
     [(3, H(180)), (7, A(150))]),
    ("donus_ani_alcalma_fren", "kesilen", "−120° dönüşte −100 ft ve −30 ft/s (80 kt)", (900, 80 * KT), 35,
     [(3, H(-120)), (6, C(0, -30, -100))]),
    ("hizlan_ani_fren", "kesilen", "+40 ft/s hızlanırken −60 ft/s fren (30 kt)", (800, 30 * KT), 35,
     [(3, V(40)), (6, V(-60))]),
    ("tirman_ani_dalis", "kesilen", "+150 ft tırmanırken −200 ft dalış (60 kt)", (800, 60 * KT), 35,
     [(3, A(150)), (6, A(-200))]),
    ("bobup_ters_hover", "kesilen", "Bob-up +100 ft, 2 s sonra −150 ft (5 kt)", (800, 5 * KT), 30,
     [(3, A(100)), (5, A(-150))]),
    ("ardisik_hizli", "kesilen", "6 komut, 2 s arayla (60 kt)", (800, 60 * KT), 40,
     [(3, H(45)), (5, A(80)), (7, H(-90)), (9, V(-30)), (11, C(90, 0, -60)), (13, V(30))]),
    ("birlesik_ters", "kesilen", "Birleşik komut, 3 s sonra tam tersi (60 kt)", (800, 60 * KT), 35,
     [(3, C(90, 20, 100)), (6, C(-90, -20, -100))]),
    ("cift_kesme", "kesilen", "+90°, 2 s sonra +90° daha, 2 s sonra −180° (60 kt)", (800, 60 * KT), 32,
     [(3, H(90)), (5, H(90)), (7, H(-180))]),
    ("slalom_sert_60kt", "kesilen", "Slalom, her 1.5 s'de ters yön (60 kt)", (800, 60 * KT), 30,
     [(3, H(90)), (4.5, H(-90)), (6, H(90)), (7.5, H(-90)), (9, H(90)), (10.5, H(-45))]),
    ("testere_180", "kesilen", "+180°, 3 s sonra −180°, 3 s sonra +180° (60 kt)", (800, 60 * KT), 38,
     [(3, H(180)), (6, H(-180)), (9, H(180))]),
    ("ardisik_100kt", "kesilen", "5 komut, 2 s arayla (98 kt)", (900, 165.0), 42,
     [(3, H(60)), (5, A(100)), (7, H(-120)), (9, V(-40)), (11, C(90, 0, -80))]),
    ("dikey_testere", "kesilen", "−100 / +150 / −100 ft, 2 s arayla (380 ft, 30 kt)", (380, 30 * KT), 35,
     [(3, A(-100)), (5, A(150)), (7, A(-100))]),
    ("hiz_testere_hover", "kesilen", "+40 / −40 / +40 ft/s, 2 s arayla (5 kt)", (800, 5 * KT), 32,
     [(3, V(40)), (5, V(-40)), (7, V(40))]),
    ("birlesik_ters_100kt", "kesilen", "98 kt: +90° −30 ft/s +100 ft, 3 s sonra −90° −100 ft", (900, 165.0), 40,
     [(3, C(90, -30, 100)), (6, C(-90, 0, -100))]),
    # ---------------- zarf sınırları ----------------
    ("hover_sinir", "sınır", "0 kt: 180° pedal, +150 ft, +50 ft/s, tekrar 0 kt", (800, 0.5), 66,
     [(3, H(180)), (16, A(150)), (30, V(50)), (45, V(-50))]),
    ("hiz_sinir_95kt", "sınır", "95 kt: 100 kt'a çık, 180° dönüş, +150 ft, −50 ft/s", (900, 160.0), 78,
     [(3, V(10)), (18, H(180)), (40, A(150)), (58, V(-50))]),
    ("alcak_irtifa", "sınır", "350 ft: 250 ft'e in, orada 180° dönüş, +150, −150 ft (60 kt)", (350, 60 * KT), 76,
     [(3, A(-100)), (18, H(180)), (40, A(150)), (58, A(-150))]),
    ("hiz_sinir_ters", "sınır", "100 kt'ta 180° dönüş, 4 s sonra ters", (900, 168.0), 38,
     [(3, H(180)), (7, H(-180))]),
    ("hover_hizlan_fren", "sınır", "5 kt: +50 ft/s, 4 s sonra dur (0 kt)", (800, 5 * KT), 35,
     [(3, V(50)), (7, V(-55))]),
    ("alcak_ani_dalis", "sınır", "400 ft: +100 ft, 3 s sonra −180 ft (30 kt)", (400, 30 * KT), 32,
     [(3, A(100)), (6, A(-180))]),
    ("max_performans", "sınır", "98 kt: +90°, +150 ft, −50 ft/s birlikte", (900, 165.0), 40,
     [(3, C(90, -50, 150))]),
]


def mirror(cmds):
    out = []
    for t, c in cmds:
        c2 = dict(c)
        if "heading" in c2:
            c2["heading"] = -c2["heading"]
        out.append((t, c2))
    return out


RANDOM_SEEDS = {"S1": (7001, 7002, 7003, 7004, 7005), "S2": (8001, 8002, 8003, 8004, 8005)}


def expand(only=None, no_mirror=False, random_eps=True):
    runs = []
    for sid, group, title, (h0, u0), dur, cmds in SCENARIOS:
        if only and sid not in only:
            continue
        runs.append(dict(id=sid, group=group, title=title, h0=float(h0), u0=float(u0), dur=float(dur), cmds=cmds, mirror=False))
        if not no_mirror and any(c.get("heading") for _, c in cmds):
            runs.append(dict(id=sid + "~ayna", group=group, title=title + " (ayna)", h0=float(h0), u0=float(u0),
                             dur=float(dur), cmds=mirror(cmds), mirror=True))
    if random_eps:                    # eğitim dağılımından sabit seed'li rastgele zorlayıcı episode'lar
        for lvl, seeds in RANDOM_SEEDS.items():
            for sd in seeds:
                sid = f"rastgele_{lvl}_{sd}"
                if only and sid not in only and "rastgele" not in only:
                    continue
                runs.append(dict(id=sid, group="rastgele", title=f"{lvl} seviyesinden rastgele episode (seed {sd})",
                                 level=lvl, seed=sd, cmds=None, mirror=False))
    return runs


def load(path):
    """(deterministik policy, modelin env ayarı) — ör. dayanıklı model collective yetkisi 0.45 ile eğitildi."""
    from helicopter_env_maneuver import load_maneuver_policy
    return load_maneuver_policy(path)


def reaction_time(rows, t_cmd, new_cmd, prev_cmd):
    """Kesen komuttan sonra hareketin yeni yöne dönmesi (s). Yalnızca ters işaretli eksen için."""
    t = rows[:, 0]
    for axis, col, thr in (("heading", 3, 1.0), ("altitude", 4, 1.0), ("speed", 9, 0.5)):
        d_new, d_old = new_cmd.get(axis, 0.0), (prev_cmd or {}).get(axis, 0.0)
        if d_new and d_old and d_new * d_old < 0:
            post = t >= t_cmd
            x = rows[post, col] * math.copysign(1.0, d_new)
            k = np.flatnonzero(x > thr)
            return float(t[post][k[0]] - t_cmd) if k.size else float("nan")
    return None


def run_one(env, policy, sc):
    if sc.get("cmds") is None:        # rastgele: seviyenin kendi takvimi ve başlangıcı
        obs, info = env.reset(seed=sc["seed"], options=dict(level=sc["level"]))
    else:
        items = [(t, dict(c)) for t, c in sc["cmds"]]
        obs, info = env.reset(seed=0, options=dict(level="M5", start_alt_ft=sc["h0"], start_speed_fps=sc["u0"],
                                                   start_heading_deg=0.0, commands=items, episode_s=sc["dur"]))
    rows, acts, done = [], [], False
    u_prev = info["speed"]
    while not done:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        u_dot = (info["speed"] - u_prev) / CONTROL_DT
        u_prev = info["speed"]
        rows.append((info["t"], info["roll_deg"], info["pitch_deg"], info["yaw_rate_dps"], info["vertical_speed"],
                     info["speed"], info["altitude"], info["err_heading"], info["err_altitude"], u_dot))
        acts.append(np.clip(np.asarray(a, dtype=float), -1, 1))
        done = term or trunc
    rows, acts = np.array(rows), np.array(acts)
    res = info.get("command_results") or []
    term_reason = info.get("termination")
    safe = term_reason == "time_limit"
    last = res[-1] if res else {}
    react = []
    for k in range(1, len(res)):
        if res[k - 1].get("interrupted"):
            rt = reaction_time(rows, res[k]["t"], res[k]["cmd"], res[k - 1]["cmd"])
            if rt is not None:
                react.append(rt)
    # açılar komut öncesine göre değil mutlak (manevrada anlamlı olan mutlak yatış / yunuslama)
    out = dict(
        id=sc["id"], group=sc["group"], title=sc["title"], safe=bool(safe), termination=term_reason,
        episode_ok=bool(info.get("episode_success", False)), final_ok=bool(last.get("success", False)) and safe,
        n_cmd=len(res), n_interrupted=sum(bool(w.get("interrupted")) for w in res),
        windows=[dict(t=w["t"], cmd={a: round(v, 1) for a, v in w["cmd"].items() if v}, T=round(w["T"], 2),
                      deadline=round(w["deadline"], 2), allow_s=round(w.get("allow_s", 0.0), 2),
                      settle_s=None if not np.isfinite(w["settle_s"]) else round(w["settle_s"], 2),
                      on_time=w["on_time"], success=w["success"], interrupted=bool(w.get("interrupted")),
                      coupling_ok=w.get("coupling_ok", True),
                      max_abs_err={a: round(v, 1) for a, v in w["max_abs_err"].items()}) for w in res],
        late_s=(None if not res or res[-1]["on_time"] or not np.isfinite(res[-1]["settle_s"])
                else round(res[-1]["settle_s"] - res[-1]["deadline"], 2)),
        reaction_s=[round(x, 2) for x in react],
        max_roll=float(np.abs(rows[:, 1]).max()), max_pitch=float(np.abs(rows[:, 2]).max()),
        max_r=float(np.abs(rows[:, 3]).max()), max_vs=float(np.abs(rows[:, 4]).max()),
        min_alt=float(rows[:, 6].min()), sat_frac=float(np.mean(np.abs(acts) > 0.95)),
        act_rate=float(np.mean(np.abs(np.diff(acts, axis=0)))) if len(acts) > 1 else 0.0,
    )
    return out


def summarize(results):
    def rate(key, group=None):
        rs = [r for r in results if group is None or r["group"] == group]
        return sum(r[key] for r in rs), len(rs)
    s = {}
    for g in (None, "kesilen", "sınır", "rastgele"):
        name = g or "toplam"
        s[name] = {k: rate(k, g) for k in ("safe", "final_ok", "episode_ok")}
    rts = [x for r in results for x in r["reaction_s"] if np.isfinite(x)]
    s["tepki_ort_s"] = float(np.mean(rts)) if rts else float("nan")
    return s


def fmt_row(r):
    w = r["windows"]
    last = w[-1] if w else {}
    tick = lambda b: "✓" if b else "✗"          # noqa: E731
    settle = "—" if last.get("settle_s") is None else f"{last['settle_s']:.1f}"
    return (f"{r['id']:30s} {tick(r['safe']):>4s} {tick(r['final_ok']):>5s} {tick(r['episode_ok']):>5s} "
            f"{r['n_interrupted']:>3d}/{r['n_cmd']:<3d} {settle:>6s}/{last.get('deadline', float('nan')):<5.1f} "
            f"{r['max_roll']:5.0f}° {r['max_pitch']:4.0f}° {r['max_r']:4.0f} {r['min_alt']:6.0f} "
            f"{r['sat_frac']:5.0%} {('/'.join(f'{x:.1f}' for x in r['reaction_s']) or '—'):>10s}  "
            f"{'' if r['safe'] else r['termination']}")


HEADER = (f"{'senaryo':30s} {'güv':>4s} {'son':>5s} {'tüm':>5s} {'kes':>7s} {'oturma/son sınır':>13s} "
          f"{'maxφ':>6s} {'maxθ':>5s} {'max r':>5s} {'min h':>6s} {'doyg.':>5s} {'tepki (s)':>10s}")


def evaluate(model_path, runs, level="M5", verbose=True):
    from helicopter_env_maneuver import HelicopterEnvManeuver
    pol, cfg = load(model_path)
    env = HelicopterEnvManeuver(level=level, config=cfg)
    results = []
    if verbose:
        print(f"\nDAYANIKLILIK — model {Path(model_path).name} ({len(runs)} koşu, süre hedefi {level} çevikliği, "
              f"collective yetkisi ±{cfg.coll_scale:g})")
        print(HEADER)
    for sc in runs:
        r = run_one(env, pol, sc)
        results.append(r)
        if verbose:
            print(fmt_row(r), flush=True)
    s = summarize(results)
    if verbose:
        for g in ("kesilen", "sınır", "rastgele", "toplam"):
            x = s[g]
            if not x["safe"][1]:
                continue
            print(f"  {g:8s} güvenli {x['safe'][0]}/{x['safe'][1]}   son komut {x['final_ok'][0]}/{x['final_ok'][1]}   "
                  f"tüm komutlar {x['episode_ok'][0]}/{x['episode_ok'][1]}")
        print(f"  kesen komuta tepki (ters eksen) ortalama {s['tepki_ort_s']:.2f} s")
    return results, s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare", default=None, help="ikinci model (ör. önceki): aynı senaryolar, özet karşılaştırma")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-random", action="store_true", help="S1/S2'den sabit seed'li rastgele episode'ları atla")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    runs = expand(args.only, args.no_mirror, not args.no_random)
    t0 = time.time()
    res, summ = evaluate(args.model, runs)
    out = dict(model=str(args.model), results=res, summary=summ)
    if args.compare:
        res2, summ2 = evaluate(args.compare, runs)
        out.update(compare=str(args.compare), compare_results=res2, compare_summary=summ2)
        print("\nKARŞILAŞTIRMA (güvenli / son komut / tüm komutlar)")
        for g in ("kesilen", "sınır", "rastgele", "toplam"):
            a, b = summ[g], summ2[g]
            if not a["safe"][1]:
                continue
            f = lambda x: f"{x[0]:2d}/{x[1]}"          # noqa: E731
            print(f"  {g:8s} {Path(args.model).name:28s} {f(a['safe'])} {f(a['final_ok'])} {f(a['episode_ok'])}   |   "
                  f"{Path(args.compare).name:28s} {f(b['safe'])} {f(b['final_ok'])} {f(b['episode_ok'])}")
    print(f"\n({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"kaydedildi: {args.json}")


if __name__ == "__main__":
    main()
