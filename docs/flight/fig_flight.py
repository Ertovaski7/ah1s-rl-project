"""
Tek ajanlı uçuş şekilleri (docs/flight/):

  fig_flight_chain.png   : bir görev zinciri uçuşu (varsayılan: uzun zincir, 15 kt rüzgâr, 9700 lbs) — irtifa, hava hızı,
                           heading, tork, yakıt, yer izi; görev pencereleri dikey çizgilerle
  fig_flight_training.png: eğitim — seviye ve eğitim başarısı (progress.csv), deterministik değerlendirme (eval.csv)

  python docs/flight/fig_flight.py --model models_flight/flight_final.zip \
      --runs runs/fl_v3:2765704 runs/fl_v6:500000 runs/fl_v8 runs/fl_v9     # soy: her koşu öncekinin modelinden
  python docs/flight/fig_flight.py --scripted          # PID pilotla (şekil kodunu denemek için)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import evaluate_flight as EF  # noqa: E402

# dataviz referans paleti (açık zemin): kategorik sıra sabit; durum renkleri yalnızca sınırlar için
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
WARN, CRIT = "#eda100", "#e34948"
KT = 1.6878099


def _style(ax, title, unit):
    ax.set_title(f"{title}  ", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.text(1.0, 1.02, unit, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def record_flight(env, policy, sc, seed=0):
    sid, group, title, start, tasks, fuel, env_name, episode_s = sc
    phys = dict(EF.ENVS[env_name] if isinstance(env_name, str) else env_name)
    phys.setdefault("wind_dir_relative", True)
    opts = dict(level="F8", tasks=[dict(t) for t in tasks], fuel=fuel, start_heading_deg=0.0, start_perturb=0.0,
                physics=phys, episode_s=episode_s, live=False, **EF._start_opts(start))
    obs, info = env.reset(seed=seed, options=opts)
    if hasattr(policy, "pilot"):
        policy.pilot.reset()
    rows, done = [], False
    while not done:
        obs, r, te, tr, info = env.step(policy(obs))
        done = te or tr
        tg = info["target"]
        rows.append(dict(t=info["t"], h=info["altitude"], h_t=tg["h"], u=info["airspeed_kt"],
                         u_t=info["u_target_kt"] if info.get("cruise") else np.nan, u_ff=info["trim_speed_kt"],
                         gs=info["ground_speed"] / KT, psi=info["heading_unwrapped"], psi_t=tg["psi"],
                         tq=info["torque_psi"], fuel=info["fuel_lbs"], w=info["weight_lbs"], n=info["north"],
                         e=info["east"], k=len(env.windows) - 1, rpm=info["rotor_rpm"]))
    res = info.get("command_results") or []
    return rows, res, info


def plot_chain(rows, res, info, title, out: Path):
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    t = a["t"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 10.5), constrained_layout=True)
    marks = [(c["t"], c["category"], c["success"]) for c in res]

    def windows(ax, label=False):
        for tc, cat, ok in marks:
            ax.axvline(tc, color=MUTED, linewidth=0.7, linestyle=":")
            if label:
                ax.text(tc + 1, 1.0, f"{EF.SHORT.get(cat, cat)}{'✓' if ok else '✗'}", transform=ax.get_xaxis_transform(),
                        fontsize=7, color=INK2, va="top", rotation=90)

    ax = axs[0, 0]
    ax.plot(t, a["h_t"], color=INK2, linestyle="--", linewidth=1.2, label="hedef")
    ax.plot(t, a["h"], color=BLUE, linewidth=1.6, label="irtifa (CG AGL)")
    windows(ax, label=True)
    _style(ax, "İrtifa", "ft")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axs[0, 1]
    ax.plot(t, a["u_t"], color=INK2, linestyle="--", linewidth=1.2, label="hedef hava hızı")
    ax.plot(t, a["u_ff"], color=YELLOW, linewidth=1.0, label="trim çizelgesi (referans hızı)")
    ax.plot(t, a["gs"], color=AQUA, linewidth=1.0, label="yer hızı")
    ax.plot(t, a["u"], color=BLUE, linewidth=1.6, label="hava hızı (ileri)")
    windows(ax)
    _style(ax, "Hız", "kt")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axs[1, 0]
    ax.plot(t, a["psi_t"], color=INK2, linestyle="--", linewidth=1.2, label="hedef")
    ax.plot(t, a["psi"], color=BLUE, linewidth=1.6, label="heading (sarılmamış)")
    windows(ax)
    _style(ax, "Heading", "°")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axs[1, 1]
    ax.axhline(56.0, color=CRIT, linewidth=1.0, linestyle="--")
    ax.axhline(50.0, color=WARN, linewidth=1.0, linestyle="--")
    ax.text(t[-1], 56.4, "56 psi (%100, güç tavanı)", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(t[-1], 50.4, "50 psi (sürekli)", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.plot(t, a["tq"], color=BLUE, linewidth=1.2)
    windows(ax)
    _style(ax, "Tork göstergesi", "psi")

    ax = axs[2, 0]
    ax.plot(t, a["fuel"], color=BLUE, linewidth=1.6)
    windows(ax)
    _style(ax, "Yakıt (iki tank)", "lbs")
    used = a["fuel"][0] - a["fuel"][-1]
    ax.text(0.02, 0.06, f"harcanan {used:.0f} lbs · ortalama {used / max(t[-1], 1) * 3600:.0f} lb/h · "
                        f"ağırlık {a['w'][0]:.0f} → {a['w'][-1]:.0f} lbs", transform=ax.transAxes, fontsize=8, color=INK2)
    ax.set_xlabel("t (s)", fontsize=8, color=INK2)

    ax = axs[2, 1]
    ax.plot(a["e"], a["n"], color=BLUE, linewidth=1.4)
    ax.plot(a["e"][0], a["n"][0], "o", color=AQUA, markersize=7, label="kalkış")
    ax.plot(a["e"][-1], a["n"][-1], "s", color=ORANGE, markersize=7, label="son")
    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, "Yer izi", "ft (doğu → / kuzey ↑)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ph = info.get("physics_summary", {}).get("params", {})
    wind = f"rüzgâr {ph.get('wind_kt', 0):.0f} kt ({ph.get('wind_from_deg', 0):.0f}°'den)" if ph.get("wind_kt") else "rüzgâr yok"
    fig.suptitle(f"{title} — {wind}, türbülans {ph.get('turb_level', 'none')} · bitiş: {info.get('termination')} · "
                 f"görev {sum(c['success'] for c in res)}/{len(res)}", fontsize=11, color=INK, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"kaydedildi: {out}")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _run_spec(spec: str) -> tuple[Path, int | None]:
    """'klasör' ya da 'klasör:en_çok_adım' (koşunun yalnızca o adıma kadarki kısmı soya girer; ör. fl_v3:2765704)."""
    if ":" in spec and spec.rsplit(":", 1)[1].isdigit():
        d, m = spec.rsplit(":", 1)
        return Path(d), int(m)
    return Path(spec), None


def plot_training(run_specs: list[str], out: Path):
    """Soydaki koşuları (her biri bir öncekinin modelinden başlar) adım ekseninde uç uca ekler."""
    prog, ev, bounds = [], [], []
    offset = 0
    for spec in run_specs:
        d, mx = _run_spec(spec)
        p = [r for r in _read_csv(d / "progress.csv") if mx is None or int(r["timesteps"]) <= mx]
        e = [r for r in _read_csv(d / "eval.csv") if mx is None or int(r["timesteps"]) <= mx]
        for r in p + e:
            r["_steps"] = offset + int(r["timesteps"])
        prog += p
        ev += e
        bounds.append((offset, d.name))
        if p:
            offset = (offset + mx) if mx is not None else prog[-1]["_steps"]
    if not prog:
        print("progress.csv yok — eğitim şekli atlandı")
        return
    levels = []
    for r in prog:
        if r["level"] not in levels:
            levels.append(r["level"])
    x = np.array([r["_steps"] for r in prog]) / 1e6
    lvl = np.array([levels.index(r["level"]) for r in prog])
    succ = np.array([float(r["success_rate"]) for r in prog])
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)

    def runs(ax, label=False):
        for b, name in bounds[1:] if not label else bounds:
            if b > 0:
                ax.axvline(b / 1e6, color=MUTED, linewidth=0.8, linestyle=":")
            if label:
                ax.text(b / 1e6 + 0.03, 0.98, name, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")

    ax = axs[0]
    ax.step(x, lvl, where="post", color=BLUE, linewidth=1.6)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    runs(ax, label=True)
    _style(ax, "Seviye", "curriculum")
    ax = axs[1]
    ax.plot(x, 100 * succ, color=BLUE, linewidth=1.2)
    runs(ax)
    _style(ax, "Eğitim başarısı (stokastik policy, son 100 episode, mevcut seviye)", "%")
    ax.set_ylim(-3, 103)
    ax = axs[2]
    if ev:
        cols = []
        for r in ev:
            cols += [c for c in r if c.startswith("success_") and c != "success_current" and c not in cols]
        cols = sorted(cols, key=lambda c: next((i for i, lv in enumerate(levels) if lv == c[8:]), 99))
        colors = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        xe = np.array([r["_steps"] for r in ev]) / 1e6
        for k, c in enumerate(cols):
            y = np.array([float(r[c]) if r.get(c) not in (None, "", "None") else np.nan for r in ev])
            ax.plot(xe, 100 * y, color=colors[k % len(colors)], linewidth=1.4, marker="o", markersize=3.5,
                    label=c.replace("success_", ""))
        cur = [(r["_steps"] / 1e6, float(r["success_current"]), r.get("current_level", "")) for r in ev
               if r.get("success_current") not in (None, "", "None") and f"success_{r.get('current_level')}" not in r]
        if cur:
            ax.plot([c[0] for c in cur], [100 * c[1] for c in cur], linestyle="none", marker="s", markersize=4.5,
                    color=INK2, label="mevcut seviye (ör. F7, F8)")
        ax.legend(fontsize=8, frameon=False, ncol=len(cols) + 1, loc="lower right")
    runs(ax)
    _style(ax, "Deterministik değerlendirme (12 episode / seviye, sabit seed)", "%")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("adım (milyon; soydaki koşular uç uca)", fontsize=8, color=INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"kaydedildi: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--scenario", default="uzun_ruzgar_agir")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="eğitim koşu klasörleri, soy sırasıyla; 'klasör:adım' o adımdan sonrasını atar")
    ap.add_argument("--out-prefix", default=str(HERE / "fig_flight"))
    args = ap.parse_args(argv)
    if args.model or args.scripted:
        env, pol, _ = EF.make_env(args.model, scripted=args.scripted)
        sc = next(s for s in EF.SCENARIOS if s[0] == args.scenario)
        rows, res, info = record_flight(env, pol, sc)
        who = "PID pilot (RL değil)" if args.scripted else Path(args.model).name
        plot_chain(rows, res, info, f"{sc[2]} — {who}", Path(f"{args.out_prefix}_chain{'_pid' if args.scripted else ''}.png"))
    if args.runs:
        plot_training(args.runs, Path(f"{args.out_prefix}_training.png"))


if __name__ == "__main__":
    main()
