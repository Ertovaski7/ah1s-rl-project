"""Probe (c): türbülans şiddet seviyelerinde bozucunun büyüklüğü — sabit kumandada (açık döngü) açısal hız / konum
sapmaları ve basit bir PID tutucuyla (kapalı döngü) hover / düz uçuş tutma doğruluğu. Hangi seviyeler öğrenilebilir?

Koşullar: H0 = OGE hover 100 ft AGL, sakin hava; H15 = aynı, 15 kt karşıdan ortalama rüzgâr; F60 = 60 kt düz uçuş,
300 ft AGL, rüzgârsız. Seviyeler: none / light / moderate / severe (MIL-F-8785C alçak irtifa: W20 = 15 / 30 / 45 kt).
Backend'ler: dryden_agl (bizim, yükseklik AGL), jsbsim_tustin, jsbsim_milspec (JSBSim, yükseklik MSL → Edwards'ta
her zaman orta irtifa dalı). Her kombinasyon N tohumla (seed).

Açık döngü: tutucu oturur → türbülans başlar → kumandalar trimde DONDURULUR, 10 s: en büyük |Δφ|, |Δθ|, |Δψ|,
|p|, |q|, |r|, yatay kayma, irtifa değişimi (türbülanssız taban çizgisiyle birlikte: dondurulmuş kumandada kayma).
Kapalı döngü: tutucu (ölçüm PID'i, policy değil) 120 s tutar: konum / irtifa / heading (/ hız) hatası RMS ve en
büyüğü, kalkış env'inin hover bandında (±6 ft konum, ±3 ft irtifa, ±3°) geçen zaman oranı, kumanda hızı RMS,
doygunluk. Türbülans hızları (toplam rüzgâr − ortalama) σ olarak raporlanır.

Çıktı: probe_c_turbulence.csv, probe_c_turbulence.txt, fig_c_turbulence.png
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np

import probe_common  # noqa: E402  (outp: çıktı adı eki)
from probe_common import (CONTROL_DT, GROUND_MSL_FT, KT_TO_FPS, OUT, Holder, Rig, env_config_from_args,
                          settle_and_measure, sfd_trim)
from helicopter_env_command import wrap_deg  # noqa: E402

LEVELS = ("none", "light", "moderate", "severe")
BACKENDS = ("dryden_agl", "jsbsim_tustin", "jsbsim_milspec")
CONDS = {"H0": dict(h=100.0, u_kt=0.0, wind_kt=0.0), "H15": dict(h=100.0, u_kt=0.0, wind_kt=15.0),
         "F60": dict(h=300.0, u_kt=60.0, wind_kt=0.0)}


def make(cond: str, backend: str, level: str, env_config=None):
    c = CONDS[cond]
    phys = {"turb": dict(enable=True, backend=backend, levels=(level,))}
    rig = Rig(fuel=(0.0, 0.0), physics=phys, env_config=env_config)
    if c["wind_kt"]:
        rig.set_wind(c["wind_kt"], 0.0)
    u_air = (c["u_kt"] + c["wind_kt"]) * KT_TO_FPS
    rig.teleport(c["h"], 0.0, u_air=u_air, ctrl=sfd_trim(u_air / KT_TO_FPS, GROUND_MSL_FT + c["h"]))
    s = rig.state()
    if c["u_kt"] == 0.0:
        hold = Holder(c["h"], 0.0, mode="ground", n0=s["n"], e0=s["e"])
    else:
        hold = Holder(c["h"], 0.0, mode="air", u=c["u_kt"] * KT_TO_FPS)
    settle_and_measure(rig, hold, t_min=30.0, t_max=200.0, avg_s=2.0, tol=dict(dctrl=0.004, drpm=0.3))
    return rig, hold


def run_case(cond, backend, level, seed, env_config=None, t_open=10.0, t_closed=120.0) -> dict:
    out = dict(cond=cond, backend=backend, level=level, seed=seed)
    # ---- açık döngü ----
    rig, hold = make(cond, backend, level, env_config)
    s0 = rig.state()
    frozen = hold.ctrl.copy()
    rig.start_physics(1000 + seed, 0.0)
    mx = dict(dphi=0.0, dth=0.0, dpsi=0.0, p=0.0, q=0.0, r=0.0, dxy=0.0, dh=0.0)
    gust = []
    for _ in range(int(round(t_open / CONTROL_DT))):
        rig.step(frozen)
        s = rig.state()
        mx["dphi"] = max(mx["dphi"], abs(math.degrees(s["phi"] - s0["phi"])))
        mx["dth"] = max(mx["dth"], abs(math.degrees(s["theta"] - s0["theta"])))
        mx["dpsi"] = max(mx["dpsi"], abs(wrap_deg(s["psi_deg"] - s0["psi_deg"])))
        for k in ("p", "q", "r"):
            mx[k] = max(mx[k], abs(math.degrees(s[k])))
        if CONDS[cond]["u_kt"] == 0.0:
            mx["dxy"] = max(mx["dxy"], math.hypot(s["n"] - s0["n"], s["e"] - s0["e"]))
        else:                                                    # ileri uçuşta: yana sapma (rotadan)
            mx["dxy"] = max(mx["dxy"], abs(s["e"] - s0["e"]))
        mx["dh"] = max(mx["dh"], abs(s["h"] - s0["h"]))
        gust.append((s["gust_n"], s["gust_e"], s["gust_d"]))
    out.update({f"ol_{k}": v for k, v in mx.items()})
    # ---- kapalı döngü ----
    rig, hold = make(cond, backend, level, env_config)
    s0 = rig.state()
    rig.start_physics(2000 + seed, 0.0)
    e_xy, e_h, e_psi, e_u, ctrls, band = [], [], [], [], [], []
    for _ in range(int(round(t_closed / CONTROL_DT))):
        s = rig.state()
        c = hold(s)
        rig.step(c)
        ctrls.append(c.copy())
        gust.append((s["gust_n"], s["gust_e"], s["gust_d"]))
        if CONDS[cond]["u_kt"] == 0.0:
            exy = math.hypot(s["n"] - hold.n0, s["e"] - hold.e0)
        else:
            exy = abs(s["e"] - s0["e"])
            e_u.append(s["u_air"] - hold.u_t)
        e_xy.append(exy)
        e_h.append(s["h"] - hold.h0)
        e_psi.append(wrap_deg(s["psi_deg"] - hold.psi0))
        if CONDS[cond]["u_kt"] == 0.0:          # kalkış env'i hover bandı
            band.append(exy <= 6.0 and abs(e_h[-1]) <= 3.0 and abs(e_psi[-1]) <= 3.0)
        else:                                   # manevra env'i toleransları: hız 2 ft/s, irtifa 10 ft, heading 2°
            band.append(abs(e_u[-1]) <= 2.0 and abs(e_h[-1]) <= 10.0 and abs(e_psi[-1]) <= 2.0)
    ctrls = np.asarray(ctrls)
    dc = np.diff(ctrls, axis=0) / CONTROL_DT
    g = np.asarray(gust)
    out.update(cl_xy_rms=float(np.sqrt(np.mean(np.square(e_xy)))), cl_xy_max=float(np.max(e_xy)),
               cl_h_rms=float(np.sqrt(np.mean(np.square(e_h)))), cl_h_max=float(np.max(np.abs(e_h))),
               cl_psi_rms=float(np.sqrt(np.mean(np.square(e_psi)))), cl_psi_max=float(np.max(np.abs(e_psi))),
               cl_u_rms=float(np.sqrt(np.mean(np.square(e_u)))) if e_u else float("nan"),
               cl_u_max=float(np.max(np.abs(e_u))) if e_u else float("nan"),
               cl_band_frac=float(np.mean(band)),
               cl_dctrl_rms=[float(x) for x in np.sqrt(np.mean(dc ** 2, axis=0))],
               cl_sat_frac=float(np.mean((ctrls[:, 0] >= 0.999) | (ctrls[:, 0] <= 0.001)
                                         | np.any(np.abs(ctrls[:, 1:]) >= 0.999, axis=1))),
               gust_sig_n=float(np.std(g[:, 0])), gust_sig_e=float(np.std(g[:, 1])), gust_sig_d=float(np.std(g[:, 2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--power-cap", type=float, default=0.0, help="repo uçağında güç tavanı, psi (0 → yok)")
    ap.add_argument("--conds", default="H0,H15,F60")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    args = ap.parse_args()
    env_config = env_config_from_args(args.aircraft, args.power_cap)
    rows = []
    for cond in args.conds.split(","):
        for backend in args.backends.split(","):
            for level in LEVELS:
                if level == "none" and backend != BACKENDS[0]:
                    continue                                     # taban çizgisi bir kez
                for seed in range(args.seeds):
                    r = run_case(cond, backend, level, seed, env_config)
                    rows.append(r)
                agg = summarize([r for r in rows if r["cond"] == cond and r["backend"] == backend
                                 and r["level"] == level])
                print(f"{cond:4s} {backend:14s} {level:9s} σ(n,e,d) {agg['gust_sig']} | açık 10 s: Δφ {agg['ol_dphi']:.1f}° "
                      f"Δθ {agg['ol_dth']:.1f}° Δψ {agg['ol_dpsi']:.1f}° p/q/r {agg['ol_p']:.1f}/{agg['ol_q']:.1f}/"
                      f"{agg['ol_r']:.1f} °/s kayma {agg['ol_dxy']:.1f} ft Δh {agg['ol_dh']:.1f} ft | PID 120 s: konum "
                      f"{agg['cl_xy_rms']:.1f}/{agg['cl_xy_max']:.1f} ft irtifa {agg['cl_h_rms']:.1f}/{agg['cl_h_max']:.1f} "
                      f"heading {agg['cl_psi_rms']:.1f}/{agg['cl_psi_max']:.1f}° bant %{agg['cl_band_frac'] * 100:.0f} "
                      f"doygunluk %{agg['cl_sat_frac'] * 100:.1f}", flush=True)
    flat = []
    for r in rows:
        d = dict(r)
        d["cl_dctrl_rms"] = ";".join(f"{x:.3f}" for x in r["cl_dctrl_rms"])
        flat.append(d)
    with open(probe_common.outp("probe_c_turbulence.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(flat[0]))
        wr.writeheader()
        wr.writerows(flat)
    write_report(rows)
    plot(rows)


def summarize(rr: list[dict]) -> dict:
    keys = [k for k in rr[0] if k.startswith(("ol_", "cl_")) and k != "cl_dctrl_rms"]
    agg = {k: float(np.nanmean([r[k] for r in rr])) for k in keys}
    agg["gust_sig"] = "/".join(f"{np.mean([r[f'gust_sig_{a}'] for r in rr]):.1f}" for a in "ned")
    agg["dctrl"] = np.mean([r["cl_dctrl_rms"] for r in rr], axis=0)
    return agg


def _f(x, w=5, d=1):
    """sayı; ıraksayan (NaN / çok büyük) değerler 'ırak' diye."""
    return f"{x:{w}.{d}f}" if np.isfinite(x) and abs(x) < 1e4 else f"{'ırak':>{w}s}"


def write_report(rows):
    lines = ["Probe (c) — türbülans şiddeti: açık döngü (10 s, kumanda trimde dondurulmuş) ve PID tutucu (120 s)",
             "(tohum ortalaması; bant: hover'da kalkış env'i bandı ±6 ft konum / ±3 ft irtifa / ±3°, F60'ta manevra env'i",
             " toleransları ±2 ft/s hız / ±10 ft irtifa / ±2°; kayma = hover'da yatay, F60'ta rotadan yana; F60'ta 'konum'",
             " sütunu yerine hız hatası rms/max ft/s — tutucunun yana rota döngüsü yok; 'ırak' = sayısal olarak ıraksadı)",
             ""]
    for cond in CONDS:
        lines.append(f"== {cond}: {CONDS[cond]}")
        lines.append("backend         seviye    | σ türb. n/e/d ft/s | açık: Δφ Δθ Δψ ° | p q r °/s | kayma Δh ft "
                     "| PID: " + ("hız hatası rms/max ft/s" if CONDS[cond]["u_kt"] > 0 else "konum rms/max ft")
                     + " | irtifa rms/max ft | heading rms/max ° | bant % | kumanda hızı rms (coll lon lat ped /s)")
        for backend in BACKENDS:
            for level in LEVELS:
                rr = [r for r in rows if r["cond"] == cond and r["backend"] == backend and r["level"] == level]
                if not rr:
                    continue
                a = summarize(rr)
                fwd = CONDS[cond]["u_kt"] > 0.0
                pos = (_f(a['cl_u_rms']) + " " + _f(a['cl_u_max'])) if fwd else (_f(a['cl_xy_rms']) + " " + _f(a['cl_xy_max']))
                lines.append(f"{backend:15s} {level:9s} | {a['gust_sig']:18s} | {_f(a['ol_dphi'], 4)} {_f(a['ol_dth'], 4)} "
                             f"{_f(a['ol_dpsi'], 4)} | {_f(a['ol_p'], 4)} {_f(a['ol_q'], 4)} {_f(a['ol_r'], 4)} | "
                             f"{_f(a['ol_dxy'])} {_f(a['ol_dh'])} | {pos} | "
                             f"{_f(a['cl_h_rms'], 4)} {_f(a['cl_h_max'])} | {_f(a['cl_psi_rms'], 4)} {_f(a['cl_psi_max'])} | "
                             f"{a['cl_band_frac'] * 100:5.1f} | " + " ".join(_f(x, 5, 3) for x in a["dctrl"]))
        lines.append("")
    (probe_common.outp("probe_c_turbulence.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"dryden_agl": "#2a6fdb", "jsbsim_tustin": "#d9480f", "jsbsim_milspec": "#2f9e44"}
    metrics = (("ol_p", "açık döngü: en büyük |p|, °/s"), ("ol_dxy", "açık döngü 10 s: kayma, ft"),
               ("cl_xy_max", "PID: en büyük konum hatası (F60: hız, ft/s)"), ("cl_h_max", "PID: en büyük irtifa hatası, ft"),
               ("cl_band_frac", "PID: bantta zaman oranı"), ("gust_sig_d", "türbülans σ dikey, ft/s"))
    conds = [c for c in CONDS if any(r["cond"] == c for r in rows)]
    fig, axes = plt.subplots(len(conds), len(metrics), figsize=(3.3 * len(metrics), 2.9 * len(conds)), squeeze=False)
    x = np.arange(len(LEVELS))
    for i, cond in enumerate(conds):
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i][j]
            for b, col in cols.items():
                ys = []
                for level in LEVELS:
                    mm = "cl_u_max" if (m == "cl_xy_max" and CONDS[cond]["u_kt"] > 0) else m
                    rr = [r[mm] for r in rows if r["cond"] == cond and r["level"] == level
                          and (r["backend"] == b or level == "none")]
                    v = np.mean(rr) if rr else np.nan
                    ys.append(v if np.isfinite(v) and abs(v) < 1e4 else np.nan)
                ax.plot(x, ys, "-o", color=col, ms=4, label=b)
            ax.set_xticks(x, LEVELS, fontsize=8)
            if m in ("ol_p", "ol_dxy", "cl_xy_max", "cl_h_max"):
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(cond)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Probe (c): türbülans seviyeleri — H0 hover sakin, H15 hover 15 kt karşı rüzgâr, F60 60 kt düz uçuş")
    fig.tight_layout()
    fig.savefig(probe_common.outp("fig_c_turbulence.png"), dpi=110)


if __name__ == "__main__":
    main()
