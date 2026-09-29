"""Probe (a): tam depoyla (2 × 890 lbs, 10,280 lbs) sürekli OGE hover ve 60 kt düz uçuşta yakıtın bitme süresi.

Tutucu (ölçüm amaçlı PID, policy değil) 300 ft AGL'de konumu (hover) ya da 60 kt ileri hava hızını tutar; PhysicsExt
yakıtı her kontrol adımında tanklardan düşer (iki tanktan eşit), tanklar bitince motor ayrılır → süre kaydedilir.
Yakıt modelleri: chart (TM 55-1520-234-10 şekil 7-8), sfc_gauge (0.568 lb/shp/h × psi'den shp), sfc_rotor (0.568 ×
modelin fiziksel ana + kuyruk rotoru gücü). Karşılaştırma: kullanıcının verdiği gerçek dayanıklılık (~2–2.5 saat) ve
el kitabından hesaplanan (hover şekli 7-5 + seyir şekli 7-8 + aynı yakıt akışı doğruları; bkz. manual_reference()).

Çıktı: probe_a_fuel_endurance.csv (özet), probe_a_timeseries.csv (10 s'de bir), fig_a_fuel_endurance.png, .txt
Kullanım: python docs/physics_ext/probe_a_fuel_endurance.py [--aircraft repo] [--models chart,sfc_gauge,sfc_rotor]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np

from probe_common import KT_TO_FPS, OUT, Holder, Rig, run_hold, settle_and_measure, sfd_trim
from physics_ext import fuel_flow_chart_lbh

MODELS = {"chart": dict(model="chart"), "sfc_gauge": dict(model="sfc", power_source="gauge"),
          "sfc_rotor": dict(model="sfc", power_source="rotor")}


def endurance_run(cond: str, fuel_model: str, env_config=None, max_h: float = 10.0, log_every_s: float = 60.0):
    phys = {"fuel": dict(enable=True, **MODELS[fuel_model]), "torque": dict(enable=True)}
    rig = Rig(fuel=(890.0, 890.0), physics=phys, env_config=env_config)
    h0, psi0 = 300.0, 0.0
    u = 60.0 * KT_TO_FPS if cond == "60kt" else 0.0
    rig.teleport(h0, psi0, u_air=u, ctrl=sfd_trim(u / KT_TO_FPS, 2583.5, 10280.0))
    s = rig.state()
    hold = (Holder(h0, psi0, mode="ground", n0=s["n"], e0=s["e"], weight=10280.0) if cond == "hover"
            else Holder(h0, psi0, mode="air", u=u, weight=10280.0))
    ext = rig.ext
    ext.reset(rig.fdm, np.random.default_rng(0), psi0)
    series = []
    next_log = 0.0
    t_start = rig.t
    wall = time.time()
    while rig.t - t_start < max_h * 3600.0:
        s = rig.state()
        c = hold(s)
        if not rig.step(c):
            raise RuntimeError("JSBSim durdu")
        t = rig.t - t_start
        if t >= next_log or ext.engine_is_out:
            inf = ext.info()
            series.append(dict(cond=cond, fuel_model=fuel_model, t_s=t, fuel_lbs=s["fuel"], weight=s["weight"],
                               cg_x=s["cg_x"], cg_z=s["cg_z"], psi=s["psi_gauge"], rpm=s["rpm"],
                               ff_lbh=inf["fuel_flow_lbh"], shp=inf["shaft_shp"], p_rotor_hp=s["p_rotor_hp"],
                               h=s["h"], u_air=s["u_air"], c0=c[0], c1=c[1], c2=c[2], c3=c[3],
                               theta_deg=math.degrees(s["theta"]), phi_deg=math.degrees(s["phi"])))
            next_log += log_every_s
        if ext.engine_is_out:
            break
    sm = ext.summary()
    out = dict(cond=cond, fuel_model=fuel_model, t_empty_h=(sm["engine_out_t"] or np.nan) / 3600.0,
               fuel_used=sm["fuel_used_lbs"], t_over_56_min=sm["t_over_limit"] / 60.0,
               t_over_50_min=sm["t_over_cont"] / 60.0, peak_psi=sm["peak_psi"], min_rpm=sm["min_rpm"],
               events=";".join(e["kind"] for e in sm["events"]), wall_s=time.time() - wall)
    return out, series


# ---------------------------------------------------------------------------------------------------------------
# El kitabı referansı (gerçek AH-1S, T53-L-703): tork gereksinimi × yakıt akışı doğrusu → dayanıklılık
# ---------------------------------------------------------------------------------------------------------------
# Şekil 7-5 (hover, OGE, sakin hava), yoğunluk irtifası ≈ 2600 ft (modelin 300 ft AGL'si, standart gün): GW eğrilerinin
# tork okuması (psi) — 400 dpi tarama, 1 psi ızgarası üzerinde elle okundu (±0.5 psi).
MANUAL_HOVER_OGE = dict(gw=(8000.0, 8500.0, 9000.0, 9500.0, 10000.0), psi=(40.9, 44.1, 47.5, 51.4, 54.4))
# Şekil 7-8 sayfa 7 (seyir, 4 TOW, 2000 ft, +15 °C): TAS 60 kt'ta GW eğrilerinin torku (psi, ±1 psi)
MANUAL_CRUISE_60KT = dict(gw=(6000.0, 7000.0, 8000.0, 9000.0, 10000.0), psi=(18.1, 20.1, 22.25, 25.25, 28.6))
# aynı şekil, TAS × GW tork tablosu (psi, ±1 psi) — model doğrulaması (seyir gücü) için
MANUAL_CRUISE_TAS = (40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 110.0, 120.0)
MANUAL_CRUISE_PSI = {6000: (19.9, 18.1, 18.1, 18.5, 19.4, 24.5, 28.1, 32.5),
                     7000: (23.25, 20.4, 20.1, 20.5, 21.4, 25.75, 29.6, 33.75),
                     8000: (26.25, 22.75, 22.25, 22.5, 23.4, 27.25, 31.0, 35.0),
                     9000: (29.9, 26.0, 25.25, 25.25, 26.1, 30.0, 33.25, 37.0),
                     10000: (33.9, 30.0, 28.6, 28.4, 29.2, 32.25, 36.25, 40.0)}


def _fit(tab, deg: int = 1):
    c = np.polyfit(tab["gw"], tab["psi"], deg)
    return lambda w: float(np.polyval(c, w))


def manual_endurance(psi_of_gw, gw0: float = 10000.0, fuel_lbs: float = 1700.0, h_p: float = 2583.0) -> float:
    """dW/dt = −W_f(psi(W)); gw0'dan fuel_lbs yakana kadar (saat). psi_of_gw: GW → psi (el kitabından)."""
    w, t, dt = gw0, 0.0, 10.0
    while w > gw0 - fuel_lbs:
        ff = fuel_flow_chart_lbh(float(psi_of_gw(w)), 324.0, h_p)
        w -= ff * dt / 3600.0
        t += dt
    return t / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--models", default="chart,sfc_gauge,sfc_rotor")
    ap.add_argument("--conds", default="hover,60kt")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        report_and_plot()
        return
    env_config = None
    if args.aircraft:
        from helicopter_env_takeoff import TakeoffEnvConfig
        env_config = TakeoffEnvConfig(aircraft=args.aircraft)
    summary, series = [], []
    for cond in args.conds.split(","):
        for fm in args.models.split(","):
            out, ser = endurance_run(cond, fm, env_config)
            summary.append(out)
            series += ser
            print(f"{cond:6s} {fm:10s} yakıt bitti: {out['t_empty_h']:.2f} h  yakılan {out['fuel_used']:.0f} lbs  "
                  f"56 psi üstü {out['t_over_56_min']:.1f} dk  50 üstü {out['t_over_50_min']:.1f} dk  "
                  f"tepe {out['peak_psi']:.1f} psi  olay {out['events']}  ({out['wall_s']:.0f} s)", flush=True)
    for name, rows in (("probe_a_fuel_endurance.csv", summary), ("probe_a_timeseries.csv", series)):
        with open(OUT / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    report_and_plot()


def _read(name):
    with open(OUT / name) as fh:
        return list(csv.DictReader(fh))


def report_and_plot():
    summ, ts = _read("probe_a_fuel_endurance.csv"), _read("probe_a_timeseries.csv")
    hov, c60 = _fit(MANUAL_HOVER_OGE, 1), _fit(MANUAL_CRUISE_60KT, 2)
    ref = {("hover", 10280.0, 1780.0): manual_endurance(hov, 10280.0, 1780.0),
           ("hover", 10000.0, 1700.0): manual_endurance(hov, 10000.0, 1700.0),
           ("60kt", 10280.0, 1780.0): manual_endurance(c60, 10280.0, 1780.0),
           ("60kt", 10000.0, 1700.0): manual_endurance(c60, 10000.0, 1700.0)}
    lines = ["Probe (a) — tam depoyla yakıtın bitme süresi (300 ft AGL, 2 × 890 lbs, eşit çekim, stok uçak)", "",
             "koşul  yakıt modeli | bitiş saat | 56 psi üstü dk | 50 psi üstü dk | tepe psi | olay"]
    for r in summ:
        lines.append(f"{r['cond']:6s} {r['fuel_model']:11s} | {float(r['t_empty_h']):10.2f} | "
                     f"{float(r['t_over_56_min']):14.1f} | {float(r['t_over_50_min']):14.1f} | "
                     f"{float(r['peak_psi']):8.1f} | {r['events']}")
    lines += ["", "El kitabı ile hesaplanan (aynı yakıt akışı doğruları; tork gereksinimi el kitabının şekillerinden):",
              f"  hover OGE  10,280 → 8,500 lbs (1780 lbs, modelle aynı): {ref[('hover', 10280.0, 1780.0)]:.2f} h",
              f"  hover OGE  10,000 → 8,300 lbs (1700 lbs JP-4, gerçek azami GW): {ref[('hover', 10000.0, 1700.0)]:.2f} h",
              f"  60 KTAS    10,280 → 8,500 lbs: {ref[('60kt', 10280.0, 1780.0)]:.2f} h",
              f"  60 KTAS    10,000 → 8,300 lbs: {ref[('60kt', 10000.0, 1700.0)]:.2f} h",
              "  Kullanıcının verdiği gerçek dayanıklılık: ~2–2.5 saat (rezervsiz; seyir / hover karışık)."]
    # model psi(GW) ↔ el kitabı
    lines += ["", "tork gereksinimi, model ↔ el kitabı (psi):"]
    for cond, fit in (("hover", hov), ("60kt", c60)):
        rr = [r for r in ts if r["cond"] == cond and r["fuel_model"] == "chart"]
        for w in (10200.0, 9500.0, 9000.0, 8600.0):
            near = min(rr, key=lambda r: abs(float(r["weight"]) - w))
            lines.append(f"  {cond:5s} W {float(near['weight']):6.0f}: model {float(near['psi']):5.1f}  el kitabı "
                         f"{fit(float(near['weight'])):5.1f}")
    (OUT / "probe_a_fuel_endurance.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physics_ext import FUEL_CHART_ALT_FT, T53_703_TO_SFC, shaft_power_gauge_shp
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0][0]
    cols = {"chart": "#2a6fdb", "sfc_gauge": "#d9480f", "sfc_rotor": "#2f9e44"}
    for cond, ls in (("hover", "-"), ("60kt", "--")):
        for fm, col in cols.items():
            rr = [r for r in ts if r["cond"] == cond and r["fuel_model"] == fm]
            if rr:
                ax.plot([float(r["t_s"]) / 3600 for r in rr], [float(r["fuel_lbs"]) for r in rr], ls, color=col,
                        label=f"{cond} · {fm}")
    for (cond, gw0, fuel), t in ref.items():
        if gw0 == 10280.0:
            ax.plot([t], [0.0], "k^" if cond == "hover" else "kv", ms=9,
                    label=f"el kitabı {cond} (aynı yakıt): {t:.2f} h")
    ax.axvspan(2.0, 2.5, color="0.85", zorder=0, label="gerçek dayanıklılık ~2–2.5 h")
    ax.set_xlabel("süre, saat")
    ax.set_ylabel("toplam yakıt, lbs")
    ax.set_title("yakıt — zaman (tam depo 1780 lbs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[0][1]
    psi = np.linspace(0, 62, 100)
    for h, c in zip(FUEL_CHART_ALT_FT, ("#1f3b73", "#2a6fdb", "#74a9ff", "#b7d1ff")):
        ax.plot(psi, [fuel_flow_chart_lbh(x, 324.0, h) for x in psi], color=c, lw=1.5, label=f"el kitabı {h:.0f} ft")
    ax.plot(psi, [T53_703_TO_SFC * shaft_power_gauge_shp(x, 324.0) for x in psi], color="#d9480f", lw=1.5,
            label="sabit SFC 0.568 × (psi → shp)")
    import csv as _csv
    with open(OUT / "fuel_chart_digitized.csv") as fh:
        dig = list(_csv.DictReader(fh))
    ax.plot([float(r["psi"]) for r in dig], [float(r["fuel_flow_lbh"]) for r in dig], "k.", ms=4,
            label="taramadan okunan noktalar")
    ax.axvline(50, color="orange", lw=1)
    ax.axvline(56, color="red", lw=1)
    ax.set_xlabel("tork, psi")
    ax.set_ylabel("yakıt akışı, lb/h")
    ax.set_title("yakıt akışı ↔ tork (TM 55-1520-234-10 şekil 7-8, +15 °C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1][0]
    for cond, fit, col in (("hover", hov, "#2a6fdb"), ("60kt", c60, "#2f9e44")):
        rr = [r for r in ts if r["cond"] == cond and r["fuel_model"] == "chart" and float(r["t_s"]) >= 60.0
              and float(r["rpm"]) > 300.0]                      # tutucunun ilk oturması ve motor ayrıldıktan sonrası hariç
        w = np.array([float(r["weight"]) for r in rr])
        ax.plot(w, [float(r["psi"]) for r in rr], "-", color=col, label=f"model {cond}")
        tab = MANUAL_HOVER_OGE if cond == "hover" else MANUAL_CRUISE_60KT
        ax.plot(tab["gw"], tab["psi"], "o", color=col, mfc="none", label=f"el kitabı {cond}")
    ax.axhline(50, color="orange", lw=1)
    ax.axhline(56, color="red", lw=1)
    ax.set_xlabel("ağırlık, lbs")
    ax.set_ylabel("tork, psi")
    ax.set_title("tork gereksinimi: model (sürekli koşu) ↔ el kitabı")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1][1]
    try:
        tab = json.loads((OUT / "trim_table_airspeed.json").read_text())
        cm = plt.get_cmap("viridis")
        for j, (w, line) in enumerate(zip(tab["weight_lbs"], tab["torque_psi"])):
            ax.plot(tab["u_kt"], line, "-", color=cm(j / 3), label=f"model {w} lbs (300 ft, stok)")
    except FileNotFoundError:
        pass
    cm2 = plt.get_cmap("autumn")
    for j, (w, vals) in enumerate(MANUAL_CRUISE_PSI.items()):
        if w < 8000:
            continue
        ax.plot(MANUAL_CRUISE_TAS, vals, "o--", color=cm2(j / 5), ms=4, lw=0.8, label=f"el kitabı {w} lbs (4 TOW)")
    ax.set_xlabel("hava hızı, kt")
    ax.set_ylabel("tork, psi")
    ax.set_title("seyir torku: model ↔ el kitabı (2000 ft, +15 °C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Probe (a): yakıt tüketimi ve dayanıklılık — model ↔ AH-1S el kitabı")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a_fuel_endurance.png", dpi=110)


if __name__ == "__main__":
    main()
