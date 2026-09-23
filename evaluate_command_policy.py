from __future__ import annotations

"""
EVALUATE COMMAND POLICY — eğitilmiş PPO Δ komutlarını ne kadar iyi uyguluyor?
============================================================================

Her test: helikopter HelicopterEnvCommand ile (300 ft / 15 ft/s, rastgele
heading) başlar, t = 5 s'de TEK bir komut verilir, 60 s uçulur. Policy
deterministik (keşif gürültüsü yok). Eğitimdeki başarı kriteriyle aynı ölçülür,
ayrıca basamak cevabı (step response) metrikleri çıkarılır.

Metrikler (komut verilen eksen için)
  final   : son 10 s ortalama hata          overshoot: hedefi aşma, |Δ|'nın %'si
  settle  : banda (±tol) girip bir daha çıkmadığı an (komuttan itibaren, s)
  rise    : %10 → %90 süresi (s)
  kuplaj  : diğer eksenlerde en büyük sapma (heading komutunda irtifa / hız);
            birleşik komutta, komut verilen diğer eksen için son 10 s hatası (ör. hea→+0.2)
  max|roll|, max|r| (yaw rate)

Başarı kriteri eğitimdekiyle aynı (v2: son 10 s tolerans + komut verilmeyen eksende
kuplaj sınırı; `--env-config v1` ile eski kriter).

Kullanım (Colab, repo kökünde)
  %run evaluate_command_policy.py --model runs/cmd_v2/models/level_00_H1.zip --level H1
  %run evaluate_command_policy.py --model .../latest.zip --commands heading:+5 heading:-5 heading:+10
  %run evaluate_command_policy.py --model ... --level H1 --zero-baseline    # a=0 ile karşılaştır
  %run evaluate_command_policy.py --model ... --start-alt 800 --start-speed 22 --commands heading:+90
  # tek uçuşta art arda komutlar (arayüzün yapacağı gibi):
  %run evaluate_command_policy.py --model ... --mission 5:heading:+90 40:speed:+8 75:altitude:+100
Çıktılar: results_command_eval/<model adı>/ (summary.csv, summary.json, fig_*.png)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from command_curriculum import AXES, DEFAULT_LEVELS, find_level
from helicopter_env_command import CONTROL_DT, HelicopterEnvCommand

CMD_T = 5.0
TOL = {"heading": 1.5, "speed": 1.5, "altitude": 10.0}
UNIT = {"heading": "°", "speed": "ft/s", "altitude": "ft"}


def default_commands(level_name: str) -> list[dict]:
    """Seviyenin ana ekseni (en olası eksen; birleşik seviyede hepsi): tam büyüklük ve yarısı, iki yön."""
    lv = DEFAULT_LEVELS[find_level(level_name)]
    ranges = lv.ranges()
    axes = list(ranges) if lv.combined else [max(lv.axis_probs, key=lv.axis_probs.get)]
    cmds = []
    for a in axes:
        m = ranges[a].max_abs
        for mag in (m, m / 2.0):
            for sgn in (+1, -1):
                c = {k: 0.0 for k in AXES}
                c[a] = sgn * mag
                cmds.append(c)
    if lv.combined:                       # üç eksen aynı anda
        cmds.append({a: ranges[a].max_abs / 2.0 for a in AXES if a in ranges})
    return cmds


def parse_commands(items: list[str]) -> list[dict]:
    out = []
    for it in items:
        c = {k: 0.0 for k in AXES}
        for part in it.split(","):
            k, v = part.split(":")
            k = {"h": "altitude", "alt": "altitude", "v": "speed", "psi": "heading", "hdg": "heading"}.get(k, k)
            c[k] = float(v)
        out.append(c)
    return out


def run_case(env, policy, cmd: dict, seed: int, episode_s: float, start: dict | None = None):
    opts = dict(commands=[(CMD_T, cmd)], episode_s=episode_s, **(start or {}))
    obs, info = env.reset(seed=seed, options=opts)
    rows = []
    done = False
    while not done:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        rows.append(dict(t=info["t"], eh=info["err_heading"], ev=info["err_speed"], ea=info["err_altitude"],
                         roll=info["roll_deg"], r=info["yaw_rate_dps"], h=info["altitude"], u=info["speed"],
                         coll=info["controls"][0], elev=info["controls"][1], ail=info["controls"][2],
                         rud=info["controls"][3]))
        done = term or trunc
    return rows, info


def metrics(rows, cmd, info):
    a = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    t = a["t"]
    res = info.get("command_results") or []
    t_issue = float(res[0]["t"]) if res else CMD_T
    if res:                                     # env güvenlik için işareti çevirmiş/kırpmış olabilir
        cmd = dict(res[0]["cmd"])
    after = t >= t_issue + CONTROL_DT - 1e-9      # komutun verildiği adımdan sonrası
    last = t >= t[-1] - 10.0
    key = {"heading": "eh", "speed": "ev", "altitude": "ea"}
    axis = max(AXES, key=lambda k: abs(cmd[k]))
    d = cmd[axis]
    e = a[key[axis]]
    y = d - e                                   # kat edilen yol (hedef = d)
    active = [k for k in AXES if abs(cmd[k]) > 1e-9]
    label = (" ".join(f"{k[:3]}{cmd[k]:+g}" for k in active) if len(active) > 1
             else f"{axis[:3]} {d:+g}{UNIT[axis]}")
    m = dict(axis=axis, delta=d, cmd_label=label, success=bool(info.get("episode_success", False)),
             termination=info.get("termination", "?"), t_issue=t_issue)
    m["final"] = float(np.mean(e[last]))
    frac = y[after] / d if abs(d) > 1e-9 else np.zeros(after.sum())
    m["overshoot_pct"] = float(max(0.0, np.max(frac) - 1.0) * 100.0) if frac.size else float("nan")
    ta = t[after] - t_issue
    i10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else None
    i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
    m["rise_s"] = float(ta[i90] - ta[i10]) if i10 is not None and i90 is not None else float("nan")
    out = np.abs(e[after]) > TOL[axis]
    m["settle_s"] = (float(ta[np.where(out)[0][-1] + 1]) if out.any() and not out[-1] else
                     (0.0 if not out.any() else float("nan")))
    for other in AXES:
        if other != axis:
            m[f"max_{other}_err"] = float(np.max(np.abs(a[key[other]][after])))
            if other in active:                   # birleşik komut: bu eksen de komutlu, kuplaj değil
                m[f"final_{other}"] = float(np.mean(a[key[other]][last]))
    m["max_roll"] = float(np.max(np.abs(a["roll"])))
    m["max_yaw_rate"] = float(np.max(np.abs(a["r"])))
    return m


def make_figures(cases, out: Path, title: str, show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
    ax = ax.ravel()
    for i, (cmd, seed, rows, info, m, tag) in enumerate(cases):
        a = {k: np.array([x[k] for x in rows]) for k in rows[0]}
        c = pal[i % len(pal)]
        axis, d = m["axis"], m["delta"]
        e = a[{"heading": "eh", "speed": "ev", "altitude": "ea"}[axis]]
        lab = f"{tag}Δ{m['cmd_label']} (s{seed}) {'✓' if m['success'] else '✗'}"
        ls = "--" if tag else "-"
        resp = (d - e) / d if d else 0 * e
        resp = np.where(a["t"] >= m["t_issue"] + CONTROL_DT - 1e-9, resp, 0.0)   # komuttan önce 0
        ax[0].plot(a["t"] - CMD_T, resp, color=c, lw=1.8, ls=ls, label=lab)
        ax[1].plot(a["t"] - CMD_T, a["eh"], color=c, lw=1.5, ls=ls)
        ax[2].plot(a["t"] - CMD_T, a["ea"], color=c, lw=1.5, ls=ls)
        ax[3].plot(a["t"] - CMD_T, a["ev"], color=c, lw=1.5, ls=ls)
        ax[4].plot(a["t"] - CMD_T, a["roll"], color=c, lw=1.5, ls=ls)
        ax[5].plot(a["t"] - CMD_T, a["rud"] - a["rud"][0], color=c, lw=1.2, ls=ls)
    titles = ["normalize cevap (1 = hedef)", "heading hatası [°]", "irtifa hatası [ft]", "hız hatası [ft/s]",
              "roll [°]", "rudder − trim"]
    for x, tt in zip(ax, titles):
        x.set_title(tt, fontsize=10)
        x.axvline(0, color="#9a9994", lw=0.8, ls=":")
        x.grid(alpha=0.25)
        x.set_xlabel("komuttan sonra t [s]")
    ax[0].axhline(1.0, color="#52514e", lw=1, ls="--")
    ax[0].legend(fontsize=7, loc="lower right", ncol=2 if len(cases) > 12 else 1)
    for k, tol in ((1, TOL["heading"]), (2, TOL["altitude"]), (3, TOL["speed"])):
        ax[k].axhspan(-tol, tol, color="#9a9994", alpha=0.15, lw=0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_step_responses.png", dpi=110)
    if show:
        try:
            from IPython.display import display
            display(fig)
        except Exception:        # noqa: BLE001
            pass
    plt.close(fig)


def parse_mission(items: list[str]) -> list[tuple[float, dict]]:
    """'5:heading:+90'  '40:speed:+8,altitude:-50'  →  [(5.0, {...}), (40.0, {...})]"""
    sched = []
    for it in items:
        t, rest = it.split(":", 1)
        sched.append((float(t), parse_commands([rest])[0]))
    return sorted(sched, key=lambda x: x[0])


def run_mission(env, policy, sched, seed, start, out: Path, title: str, show: bool):
    """Tek FDM'de art arda komutlar (arayüzün yapacağı gibi); zaman serisi grafiği."""
    episode_s = sched[-1][0] + 45.0
    obs, info = env.reset(seed=seed, options=dict(commands=sched, episode_s=episode_s, **start))
    rec = []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        rec.append(dict(t=info["t"], hdg=info["heading_unwrapped"], hdg_ref=info["ref"]["heading"],
                        h=info["altitude"], h_ref=info["ref"]["altitude"], u=info["speed"], u_ref=info["ref"]["speed"],
                        roll=info["roll_deg"], pitch=info["pitch_deg"], c=info["controls"]))
        done = term or trunc
    print(f"\nGÖREV: {len(sched)} komut, {episode_s:.0f} s | bitiş: {info['termination']} | "
          f"komut başarısı {info.get('commands_ok', 0)}/{info.get('commands_total', len(sched))}")
    for res in info.get("command_results", []):
        cm = ",".join(f"{k[:3]}{v:+g}" for k, v in res["cmd"].items() if v)
        fe = res["final_err"]
        print(f"   t={res['t']:6.1f}s {cm:28s} {'✓' if res['success'] else '✗'}  son hata: ψ {fe['heading']:+.2f}° "
              f"v {fe['speed']:+.2f} ft/s h {fe['altitude']:+.2f} ft")
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = {k: np.array([x[k] for x in rec]) for k in rec[0] if k != "c"}
    ctrl = np.array([x["c"] for x in rec])
    fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    ax[0].plot(a["t"], a["hdg"] - a["hdg"][0], color="#2a78d6", lw=2, label="heading (başlangıca göre)")
    ax[0].plot(a["t"], a["hdg_ref"] - a["hdg"][0], color="#52514e", ls="--", lw=1, label="komut")
    ax[1].plot(a["t"], a["h"], color="#1baf7a", lw=2, label="irtifa")
    ax[1].plot(a["t"], a["h_ref"], color="#52514e", ls="--", lw=1, label="komut")
    ax[2].plot(a["t"], a["u"], color="#eb6834", lw=2, label="ileri hız u")
    ax[2].plot(a["t"], a["u_ref"], color="#52514e", ls="--", lw=1, label="komut")
    for i, (nm, col) in enumerate(zip(("collective", "elevator", "aileron", "rudder"),
                                      ("#1baf7a", "#4a3aa7", "#2a78d6", "#e34948"))):
        ax[3].plot(a["t"], ctrl[:, i] - ctrl[0, i], color=col, lw=1.2, label=f"{nm} − başlangıç")
    for t_cmd, _ in sched:
        for x in ax:
            x.axvline(t_cmd, color="#9a9994", lw=0.8, ls=":")
    for x, tt in zip(ax, ("Δheading [°]", "irtifa [ft]", "hız [ft/s]", "kumandalar")):
        x.set_title(tt, fontsize=10, loc="left")
        x.grid(alpha=0.25)
        x.legend(fontsize=8, loc="upper right")
    ax[3].set_xlabel("t [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_mission.png", dpi=110)
    if show:
        try:
            from IPython.display import display
            display(fig)
        except Exception:        # noqa: BLE001
            pass
    plt.close(fig)
    return info


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False, help="PPO .zip (yoksa yalnızca --zero-baseline)")
    p.add_argument("--level", default="H1", help="varsayılan komut seti bu seviyeden")
    p.add_argument("--commands", nargs="*", default=None, help="ör. heading:+5 speed:-3 heading:+10,altitude:+20")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    p.add_argument("--episode-s", type=float, default=60.0)
    p.add_argument("--zero-baseline", action="store_true", help="a = 0 (hiç kumanda oynatmama) ile karşılaştır")
    p.add_argument("--start-alt", type=float, default=None, help="başlangıç irtifası [ft] (varsayılan: seviyeninki)")
    p.add_argument("--start-speed", type=float, default=None, help="başlangıç hızı [ft/s] (varsayılan: seviyeninki)")
    p.add_argument("--env-config", choices=["v2", "v1"], default="v2",
                   help="başarı kriteri: v2 (kuplaj sınırlı, varsayılan) ya da v1")
    p.add_argument("--mission", nargs="*", default=None,
                   help="tek uçuşta art arda komutlar: '5:heading:+90' '40:speed:+8,altitude:-50' ...")
    p.add_argument("--out", default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)

    cmds = parse_commands(args.commands) if args.commands else default_commands(args.level)
    policies = []
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        policies.append(("", lambda o: model.predict(o, deterministic=True)[0]))
    if args.zero_baseline or not args.model:
        policies.append(("a=0 ", lambda o: np.zeros(4, dtype=np.float32)))
    name = Path(args.model).stem if args.model else "zero_baseline"
    out = Path(args.out or f"results_command_eval/{name}")
    out.mkdir(parents=True, exist_ok=True)

    show = not args.no_show
    try:
        from IPython import get_ipython
        show = show and get_ipython() is not None
    except Exception:            # noqa: BLE001
        show = False

    from helicopter_env_command import CommandEnvConfig
    env = HelicopterEnvCommand(level=args.level,
                               config=CommandEnvConfig.v1() if args.env_config == "v1" else CommandEnvConfig())
    start = {}
    if args.start_alt is not None:
        start["start_alt_ft"] = args.start_alt
    if args.start_speed is not None:
        start["start_speed_fps"] = args.start_speed
    if args.mission:
        tag, pol = policies[0]
        return run_mission(env, pol, parse_mission(args.mission), args.seeds[0], start, out,
                           f"{name} — art arda komutlar (tek FDM)", show)
    cases, rows_out = [], []
    print("=" * 118)
    print(f"EVALUATE — model={args.model or '-'} | {len(cmds)} komut × {len(args.seeds)} seed | komut t={CMD_T:.0f}s, "
          f"episode {args.episode_s:.0f}s | tolerans ψ±{TOL['heading']}° v±{TOL['speed']} h±{TOL['altitude']}"
          f"{' | başlangıç ' + str(start) if start else ''}")
    print("=" * 118)
    print(f"{'politika':8s} {'komut':>21s} {'seed':>4s} {'ok':>3s} {'final':>7s} {'OS%':>6s} {'rise':>6s} "
          f"{'settle':>7s} {'kuplaj':>22s} {'max|roll|':>9s} {'bitiş':>18s}")
    for tag, pol in policies:
        for cmd in cmds:
            for seed in args.seeds:
                rows, info = run_case(env, pol, cmd, seed, args.episode_s, start)
                m = metrics(rows, cmd, info)
                cases.append((cmd, seed, rows, info, m, tag))
                others = [k for k in AXES if k != m["axis"]]
                coup = " ".join(f"{o[:3]}→{m[f'final_{o}']:+.1f}" if f"final_{o}" in m
                                else f"{o[:3]}={m[f'max_{o}_err']:.1f}" for o in others)
                fmt = lambda x, nd=1: "—" if not np.isfinite(x) else f"{x:.{nd}f}"
                print(f"{tag or 'PPO':8s} {m['cmd_label']:>21s} "
                      f"{seed:4d} {'✓' if m['success'] else '✗':>3s} {m['final']:+7.2f} {fmt(m['overshoot_pct']):>6s} "
                      f"{fmt(m['rise_s']):>6s} {fmt(m['settle_s']):>7s} {coup:>22s} {m['max_roll']:9.1f} "
                      f"{m['termination']:>18s}")
                rows_out.append(dict(policy=tag.strip() or "PPO", seed=seed, **{f"cmd_{k}": v for k, v in cmd.items()},
                                     **m))
    if any(k.startswith("final_") for r in rows_out for k in r):
        print("\n(birleşik komut satırı: 'hea→+0.2' = komut verilen diğer eksenin son 10 s ortalama hatası; "
              "'alt=1.2' = komut verilmeyen eksende en büyük sapma, yani kuplaj)")
    for tag, _ in policies:
        sub = [r for r in rows_out if r["policy"] == (tag.strip() or "PPO")]
        st = [r["settle_s"] for r in sub if np.isfinite(r["settle_s"])]
        print(f"\n{tag.strip() or 'PPO'}: başarı {sum(r['success'] for r in sub)}/{len(sub)} | "
              f"ort. |final| {np.mean([abs(r['final']) for r in sub]):.2f} | "
              f"ort. settle {f'{np.mean(st):.1f}s' if st else '—'} ({len(st)}/{len(sub)} oturdu)")
    keys = sorted({k for r in rows_out for k in r})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    with open(out / "summary.json", "w") as f:
        json.dump(rows_out, f, indent=2, default=float)
    make_figures(cases, out, f"{name} — {len(cmds)} komut", show)
    print(f"\nçıktılar: {out}/")
    return rows_out


if __name__ == "__main__":
    main()
