from __future__ import annotations

"""
EVALUATE ADS-33 — kalkış / hover ajanının ADS-33E-PRF görev öğeleriyle (MTE) karnesi
====================================================================================

ADS-33E-PRF (ABD Ordusu, askerî helikopterler için uçuş kalitesi standardı) pilotun belirli görev öğelerini (Mission
Task Element, MTE) belirli toleranslar içinde yapıp yapamadığını "desired" (istenen) ve "adequate" (yeterli) performans
sınırlarıyla tanımlar. Bu script aynı ölçütleri RL ajanına uygular (pilot yerine ajan; performans kategorisi —
Cooper-Harper pilot puanı değil). Rüzgâr yok (standart MTE'ler en kritik yönden 10–15 kt rüzgârla da uçulur;
rüzgâr aşamasında tekrarlanacak).

MTE'ler (hover rejimi; ileri uçuş MTE'leri — depart/abort, lateral reposition, slalom — manevra ajanı / görev zinciri
ile sonra):
  hover      6–10 kt'lık kayarak yaklaşmadan (< 20 ft, hedef burna göre 45°) hedef noktada dur; 30 s sabit hover
  turn       10 ft'te yerinde 180° dönüş (iki yön)
  vertical   15 ft hover → 40 ft (+25) → 15 ft; iki duruşta da sabitlen
  pirouette  100 ft yarıçaplı çember boyunca yana uçuş, burun merkeze; başlangıç noktasında 5 s içinde sabit, 5 s tut
  landing    20 ft hover'dan pad'e iniş

Toleranslar `MTE_STANDARDS` sözlüğünde (kaynakları yanında; "doğrulanmalı" olanlar standardın aslıyla kontrol
edilmeli). Konum toleransları başlangıç heading'ine hizalı yer eksenlerinde (boyuna / yanal) ayrı ayrı.

Kullanım
  python evaluate_ads33.py --model models_takeoff/takeoff_final.zip
  python evaluate_ads33.py --model runs/tq_v1/models/best.zip --json docs/ads33/karne.json
  python evaluate_ads33.py --model ... --only hover landing --env '{"aircraft": "repo"}'
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
from takeoff_curriculum import GROUND_H_FT  # noqa: E402

KT = 1.68781
SRC_NASA = "NASA Ames 2023 (Altamirano vd., ADS-33 MTE tablosu)"
SRC_DLR = "DLR CH-53G ADS-33E değerlendirmesi (Höfinger vd.)"
MTE_STANDARDS = {
    "hover": dict(desired=dict(lon=3.0, lat=3.0, alt=2.0, hdg=5.0, t=5.0), adequate=dict(lon=6.0, lat=6.0, alt=4.0, hdg=10.0, t=8.0),
                  hold_s=30.0, speed_kt=8.0, hs_ft=10.0, approach_ft=25.0, source=SRC_NASA),
    "turn": dict(desired=dict(lon=3.0, lat=3.0, alt=3.0, hdg=3.0, t=10.0), adequate=dict(lon=6.0, lat=6.0, alt=5.0, hdg=5.0, t=15.0),
                 hold_s=5.0, hs_ft=10.0, source="ADS-33E-PRF hovering turn (değerler hatırlanan — doğrulanmalı)"),
    "vertical": dict(desired=dict(lon=3.0, lat=3.0, alt=3.0, hdg=5.0, t=13.0), adequate=dict(lon=6.0, lat=6.0, alt=6.0, hdg=10.0, t=18.0),
                     hs_ft=15.0, dh_ft=25.0, source=SRC_NASA + "; yukarı + aşağı tek süre (varsayım — doğrulanmalı)"),
    "pirouette": dict(desired=dict(radial=10.0, alt=3.0, hdg=10.0, t=45.0, stab=5.0), adequate=dict(radial=15.0, alt=10.0, hdg=15.0, t=60.0, stab=10.0),
                      hold_s=5.0, radius_ft=100.0, hs_ft=10.0, circle_s=38.0, source=SRC_DLR + " (+ hatırlanan süreler — doğrulanmalı)"),
    "landing": dict(desired=dict(lon=1.0, lat=0.5, hdg=5.0, t=10.0), adequate=dict(lon=3.0, lat=3.0, hdg=10.0, t=15.0),
                    hs_ft=20.0, source=SRC_DLR + " (adequate süre varsayım)"),
}
CATS = ("desired", "adequate")


# ---------------------------------------------------------------------------------------------------------------------
class Runner:
    """Tek FDM'de ajanı uçurur; hedefleri dışarıdan değiştirebilir, durumu kaydeder."""

    def __init__(self, model_path, env_overrides=None, deterministic=True):
        # model gözlem boyutu 42 → tek ajanlı uçuş env'i (helicopter_env_flight; sakin hava, yakıt tüketimi açık),
        # yoksa kalkış env'i (modelin env ayarlarıyla)
        from stable_baselines3 import PPO
        from helicopter_env_flight import OBS_DIM_F, FlightEnvConfig, HelicopterEnvFlight
        m = PPO.load(str(model_path), device="cpu")
        self.flight = int(m.observation_space.shape[0]) == OBS_DIM_F
        if self.flight:
            ov = dict(getattr(m, "ah1s_env_overrides", None) or {})
            ov.update(env_overrides or {})
            self.env = HelicopterEnvFlight(level="F8", config=FlightEnvConfig(**ov))
            self.policy = lambda o: m.predict(o, deterministic=deterministic)[0]      # noqa: E731
            self.overrides = ov
        else:
            from evaluate_takeoff import make_env
            self.env, self.policy, self.overrides = make_env(model_path, env_overrides, level="K9")
        self.obs = None
        self.rows = []

    def reset(self, hs: float, heading: float = 0.0, fuel=(0.0, 0.0), settle_s: float = 6.0, seed: int = 0):
        env = self.env
        opts = dict(start="hover", start_alt_ft=GROUND_H_FT + hs, start_heading_deg=heading, fuel=fuel, start_perturb=0.0,
                    tasks=[dict(kind="hold")], live=True, episode_s=400.0)
        if self.flight:                                   # ADS-33 MTE'leri sakin havada; eğitimin ağırlık aralığı 8800 lbs'den
            opts.update(physics=dict(wind_kt=0.0, turb_level="none"),
                        fuel=tuple(max(150.0, float(x)) for x in fuel))
        self.obs, info = env.reset(seed=seed, options=opts)
        self.rows = []
        self.run(settle_s, record=False)
        s = env._state()
        self.ref = dict(n=float(env.target["n"]), e=float(env.target["e"]), h=float(env.target["h"]),
                        psi=float(env.target["psi"]), psi0=float(env.psi_unwrap), t=self.t)
        return s

    @property
    def t(self) -> float:
        return self.env.steps * CONTROL_DT

    def step(self, record=True, on_step=None):
        env = self.env
        if on_step is not None:
            on_step(self)
        self.obs, r, term, trunc, info = env.step(self.policy(self.obs))
        if record:
            s = env._state()
            self.rows.append(dict(t=self.t, n=s["n"], e=s["e"], h=s["h"], hs=s["hs"], vs=s["vs"], vh=s["vh"],
                                  psi=float(env.psi_unwrap), wow=s["wow"], wfrac=s["wfrac"], torque=s["torque_psi"],
                                  rpm=s["rpm"], tn=float(env.target["n"]), te=float(env.target["e"]),
                                  th=float(env.target["h"]), tpsi=float(env.target["psi"])))
        return term or trunc, info

    def run(self, seconds: float, record=True, on_step=None, until=None):
        n = int(round(seconds / CONTROL_DT))
        info = {}
        for _ in range(n):
            done, info = self.step(record, on_step)
            if done:
                return True, info
            if until is not None and until(self):
                break
        return False, info


def axes_err(rows, n0, e0, psi0_deg):
    """(boyuna, yanal) konum hatası: başlangıç heading'ine hizalı eksenlerde, hedef noktaya göre."""
    ps = math.radians(psi0_deg)
    dn = np.array([r["n"] for r in rows]) - n0
    de = np.array([r["e"] for r in rows]) - e0
    return dn * math.cos(ps) + de * math.sin(ps), -dn * math.sin(ps) + de * math.cos(ps)


def rate(ok_desired: bool, ok_adequate: bool) -> str:
    return "desired" if ok_desired else ("adequate" if ok_adequate else "yetersiz")


def first_stable(t, inside, hold_s, t_from=0.0):
    """inside[k] True olan ve hold_s boyunca kesintisiz süren ilk aralığın başlangıç zamanı (yoksa None)."""
    t = np.asarray(t)
    inside = np.asarray(inside, dtype=bool)
    start = None
    for k in range(len(t)):
        if t[k] < t_from:
            continue
        if inside[k]:
            if start is None:
                start = t[k]
            if t[k] - start >= hold_s - 1e-9:
                return start
        else:
            start = None
    return None


# ---------------------------------------------------------------------------------------------------------------------
def mte_hover(R: Runner, side: int = 1, fuel=(0.0, 0.0)) -> dict:
    """6–10 kt ile burna göre 45° yönde kayarak yaklaş; hedef noktada dur ve 30 s sabit hover (ADS-33E 3.11.1)."""
    S = MTE_STANDARDS["hover"]
    env = R.env
    R.reset(S["hs_ft"], fuel=fuel)
    psi0 = R.ref["psi0"]
    course = math.radians(psi0 + 45.0 * side)
    v = S["speed_kt"] * KT
    env._set_ic_from_state(dn=v * math.cos(course), de=v * math.sin(course))      # kayarak yaklaşma (başlangıç hızı)
    d = S["approach_ft"]
    env.queue_task(dict(kind="move", dx=d * math.cos(math.radians(45.0)), dy=side * d * math.sin(math.radians(45.0))))
    R.run(CONTROL_DT * 2)
    t0 = R.rows[0]["t"]
    tn, te = env.target["n"], env.target["e"]
    done, _ = R.run(S["hold_s"] + 12.0)
    rows = R.rows
    t = np.array([r["t"] for r in rows]) - t0
    lon, lat = axes_err(rows, tn, te, psi0)
    alt = np.array([r["h"] for r in rows]) - R.ref["h"]
    hdg = np.array([r["psi"] for r in rows]) - psi0
    res = dict(mte="hover", side="sağ" if side > 0 else "sol", weight=float(env.setup_info["weight"]), crashed=done)
    for cat in CATS:
        tol = S[cat]
        inside = (np.abs(lon) <= tol["lon"]) & (np.abs(lat) <= tol["lat"]) & (np.abs(alt) <= tol["alt"]) & (np.abs(hdg) <= tol["hdg"])
        ts = first_stable(t, inside, S["hold_s"])
        res[f"t_stab_{cat}"] = ts
        res[f"ok_{cat}"] = bool(ts is not None and ts <= tol["t"] and not done)
    # tutma penceresindeki en büyük sapmalar (adequate sabitlenmesinden itibaren 30 s): hangi tolerans taşıyor
    ta = res["t_stab_adequate"] if res["t_stab_adequate"] is not None else 8.0
    k = (t >= ta) & (t <= ta + S["hold_s"])
    ahead = lon * math.cos(math.radians(45.0)) + side * lat * math.sin(math.radians(45.0))   # yaklaşma yönünde (+ = geçti)
    res.update(hold_lon=float(np.abs(lon[k]).max()), hold_lat=float(np.abs(lat[k]).max()),
               hold_alt=float(np.abs(alt[k]).max()), hold_hdg=float(np.abs(hdg[k]).max()),
               overshoot_ft=float(max(0.0, ahead.max())), torque_max=float(max(r["torque"] for r in rows)))
    res["rating"] = rate(res["ok_desired"], res["ok_adequate"])
    return res


def mte_turn(R: Runner, dpsi: float = 180.0, fuel=(0.0, 0.0)) -> dict:
    """10 ft'te yerinde 180° dönüş; konum / irtifa tut, son heading'de sabitlen."""
    S = MTE_STANDARDS["turn"]
    env = R.env
    R.reset(S["hs_ft"], fuel=fuel)
    n0, e0, psi0 = R.ref["n"], R.ref["e"], R.ref["psi0"]
    env.queue_task(dict(kind="turn", dpsi=float(dpsi)))
    done, _ = R.run(S["adequate"]["t"] + S["hold_s"] + 8.0)
    rows = R.rows
    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    lon, lat = axes_err(rows, n0, e0, psi0)
    alt = np.array([r["h"] for r in rows]) - R.ref["h"]
    hdg_err = np.array([r["psi"] for r in rows]) - (psi0 + dpsi)
    res = dict(mte="turn", side=f"{dpsi:+.0f}°", weight=float(env.setup_info["weight"]), crashed=done)
    for cat in CATS:
        tol = S[cat]
        ts = first_stable(t, np.abs(hdg_err) <= tol["hdg"], S["hold_s"])
        end = (ts + S["hold_s"]) if ts is not None else t[-1]
        k = t <= end
        pos_ok = bool((np.abs(lon[k]) <= tol["lon"]).all() and (np.abs(lat[k]) <= tol["lat"]).all()
                      and (np.abs(alt[k]) <= tol["alt"]).all())
        res[f"t_stab_{cat}"] = ts
        res[f"ok_{cat}"] = bool(ts is not None and ts <= tol["t"] and pos_ok and not done)
    res.update(max_lon=float(np.abs(lon).max()), max_lat=float(np.abs(lat).max()), max_alt=float(np.abs(alt).max()),
               torque_max=float(max(r["torque"] for r in rows)))
    res["rating"] = rate(res["ok_desired"], res["ok_adequate"])
    return res


def mte_vertical(R: Runner, fuel=(0.0, 0.0)) -> dict:
    """15 ft hover → +25 ft → 15 ft; her duruşta sabitlen (|irtifa hatası| ≤ tol, |ḣ| ≤ 1 ft/s, 1 s)."""
    S = MTE_STANDARDS["vertical"]
    env = R.env
    R.reset(S["hs_ft"], fuel=fuel)
    n0, e0, psi0, h0 = R.ref["n"], R.ref["e"], R.ref["psi0"], R.ref["h"]
    env.queue_task(dict(kind="bob", dh=S["dh_ft"]))
    state = dict(phase="up", t_up=None, streak=0.0, errs=[], top_err=None)

    def watch(r):
        # tepede (adequate toleransıyla) 1 s sabit kalınca aşağı komutu; desired için o 1 s'deki en büyük hata ≤ 3 ft
        if state["phase"] != "up" or not r.rows:
            return
        x = r.rows[-1]
        err = abs(x["h"] - (h0 + S["dh_ft"]))
        ok = err <= S["adequate"]["alt"] and abs(x["vs"]) <= 1.0
        state["streak"] = state["streak"] + CONTROL_DT if ok else 0.0
        state["errs"] = (state["errs"] + [err])[-int(round(1.0 / CONTROL_DT)):] if ok else []
        if state["streak"] >= 1.0:
            state["phase"], state["t_up"], state["top_err"] = "down", x["t"], max(state["errs"])
            env.queue_task(dict(kind="bob", dh=-S["dh_ft"]))

    done, _ = R.run(S["adequate"]["t"] + 15.0, on_step=watch)
    rows = R.rows
    t = np.array([r["t"] for r in rows]) - rows[0]["t"]
    lon, lat = axes_err(rows, n0, e0, psi0)
    hdg = np.array([r["psi"] for r in rows]) - psi0
    h = np.array([r["h"] for r in rows])
    vs = np.array([r["vs"] for r in rows])
    t_up = None if state["t_up"] is None else state["t_up"] - rows[0]["t"]
    res = dict(mte="vertical", side="+25/−25 ft", weight=float(env.setup_info["weight"]), crashed=done,
               t_top=t_up, top_err=state["top_err"])
    for cat in CATS:
        tol = S[cat]
        ts = None
        if t_up is not None:
            ts0 = first_stable(t, (np.abs(h - h0) <= tol["alt"]) & (np.abs(vs) <= 1.0), 1.0, t_from=t_up)
            ts = None if ts0 is None else ts0 + 1.0                       # sabitlenme 1 s sürdüğünde biter
        end = ts if ts is not None else t[-1]
        k = t <= end
        pos_ok = bool((np.abs(lon[k]) <= tol["lon"]).all() and (np.abs(lat[k]) <= tol["lat"]).all()
                      and (np.abs(hdg[k]) <= tol["hdg"]).all())
        top_ok = state["top_err"] is not None and state["top_err"] <= tol["alt"]
        res[f"t_done_{cat}"] = ts
        res[f"ok_{cat}"] = bool(ts is not None and ts <= tol["t"] and pos_ok and top_ok and not done)
    res.update(max_lon=float(np.abs(lon).max()), max_lat=float(np.abs(lat).max()), max_hdg=float(np.abs(hdg).max()),
               max_climb=float(vs.max()), max_sink=float(-vs.min()), torque_max=float(max(r["torque"] for r in rows)))
    res["rating"] = rate(res["ok_desired"], res["ok_adequate"])
    return res


def mte_pirouette(R: Runner, direction: int = 1, fuel=(0.0, 0.0), circle_s: float | None = None, lead_deg=8.0) -> dict:
    """100 ft yarıçaplı çember: burun merkeze, yana uçuş; hedef noktası çember üzerinde sabit açısal hızla ilerler
    (ajan eğitimde hareketli hedef görmedi — hedef her adımda env.target'a yazılır)."""
    S = MTE_STANDARDS["pirouette"]
    env = R.env
    R.reset(S["hs_ft"], fuel=fuel)
    n0, e0, psi0 = R.ref["n"], R.ref["e"], R.ref["psi0"]
    rad = S["radius_ft"]
    ps = math.radians(psi0)
    cn, ce = n0 + rad * math.cos(ps), e0 + rad * math.sin(ps)            # merkez burnun 100 ft önünde
    T = float(circle_s or S["circle_s"])
    w = direction * 360.0 / T                                             # °/s (+ → saat yönü, yukarıdan)
    t_start = R.t

    def carrot(r):
        tt = r.t - t_start
        ang = min(360.0, abs(w) * tt + lead_deg) * math.copysign(1.0, w) if tt < T else 360.0 * math.copysign(1.0, w)
        a = math.radians(psi0 + 180.0 + ang)                              # merkezden helikoptere bakış açısı
        env.target["n"], env.target["e"] = cn + rad * math.cos(a), ce + rad * math.sin(a)
        env.target["psi"] = psi0 + ang                                    # burun merkeze (açı kadar döner)

    done, _ = R.run(T + S["adequate"]["stab"] + S["hold_s"] + 10.0, on_step=carrot)
    rows = R.rows
    t = np.array([r["t"] for r in rows]) - t_start
    n = np.array([r["n"] for r in rows])
    e = np.array([r["e"] for r in rows])
    radial = np.hypot(n - cn, e - ce) - rad
    bearing = np.degrees(np.arctan2(ce - e, cn - n))                      # helikopterden merkeze
    hdg_err = (np.array([r["psi"] for r in rows]) - bearing + 180.0) % 360.0 - 180.0
    alt = np.array([r["h"] for r in rows]) - R.ref["h"]
    back = np.hypot(n - n0, e - e0)
    res = dict(mte="pirouette", side="saat yönü" if direction > 0 else "saat tersi", weight=float(env.setup_info["weight"]),
               crashed=done)
    k_circ = t <= T
    for cat in CATS:
        tol = S[cat]
        # dönüş: tur bitince başlangıç noktasına (radyal toleransla) varış; sonra stab içinde sabit hover, 5 s tut
        arr = np.flatnonzero((t >= T * 0.9) & (back <= tol["radial"]))
        t_arr = float(t[arr[0]]) if arr.size else None
        ts = None
        if t_arr is not None:
            inside = (back <= min(tol["radial"], 10.0)) & (np.abs(alt) <= tol["alt"]) & (np.abs(hdg_err) <= tol["hdg"]) \
                & (np.array([r["vh"] for r in rows]) <= 2.0)
            ts = first_stable(t, inside, S["hold_s"], t_from=t_arr)
        circ_ok = bool((np.abs(radial[k_circ]) <= tol["radial"]).all() and (np.abs(alt[k_circ]) <= tol["alt"]).all()
                       and (np.abs(hdg_err[k_circ]) <= tol["hdg"]).all())
        res[f"t_arrive_{cat}"] = t_arr
        res[f"ok_{cat}"] = bool(t_arr is not None and t_arr <= tol["t"] and ts is not None and ts - t_arr <= tol["stab"]
                                and circ_ok and not done)
    res.update(max_radial=float(np.abs(radial[k_circ]).max()), max_alt=float(np.abs(alt[k_circ]).max()),
               max_hdg=float(np.abs(hdg_err[k_circ]).max()), circle_s=T, torque_max=float(max(r["torque"] for r in rows)))
    res["rating"] = rate(res["ok_desired"], res["ok_adequate"])
    return res


def mte_landing(R: Runner, fuel=(0.0, 0.0)) -> dict:
    """20 ft hover'dan pad'e iniş: 10 ft'in altına inince temas süresi, temas noktasının pad'e göre konumu."""
    S = MTE_STANDARDS["landing"]
    env = R.env
    R.reset(S["hs_ft"], fuel=fuel)
    n0, e0, psi0 = R.ref["n"], R.ref["e"], R.ref["psi0"]
    env.queue_task(dict(kind="land"))
    done, info = R.run(60.0)
    rows = R.rows
    t = np.array([r["t"] for r in rows])
    hs = np.array([r["hs"] for r in rows])
    wow = np.array([r["wow"] for r in rows])
    lon, lat = axes_err(rows, n0, e0, psi0)
    hdg = np.array([r["psi"] for r in rows]) - psi0
    below = np.flatnonzero(hs <= 10.0)
    td = np.flatnonzero(wow == 4)
    res = dict(mte="landing", side="—", weight=float(env.setup_info["weight"]), crashed=done)
    if below.size and td.size:
        k = int(td[0])
        t10 = float(t[k] - t[below[0]])
        vs_td = float(rows[k - 1]["vs"]) if k > 0 else 0.0
        res.update(t_from_10ft=t10, lon=float(lon[k]), lat=float(lat[k]), hdg=float(hdg[k]), touchdown_vs=vs_td)
        for cat in CATS:
            tol = S[cat]
            res[f"ok_{cat}"] = bool(abs(lon[k]) <= tol["lon"] and abs(lat[k]) <= tol["lat"] and abs(hdg[k]) <= tol["hdg"]
                                    and t10 <= tol["t"] and vs_td >= -4.0 and not done)
    else:
        res.update(ok_desired=False, ok_adequate=False)
    res["torque_max"] = float(max(r["torque"] for r in rows))
    res["rating"] = rate(res["ok_desired"], res["ok_adequate"])
    return res


PLAN = {
    "hover": [("hover", dict(side=1)), ("hover", dict(side=-1))],
    "turn": [("turn", dict(dpsi=180.0)), ("turn", dict(dpsi=-180.0))],
    "vertical": [("vertical", {})],
    "pirouette": [("pirouette", dict(direction=1)), ("pirouette", dict(direction=-1))],
    "landing": [("landing", {})],
}
FN = dict(hover=mte_hover, turn=mte_turn, vertical=mte_vertical, pirouette=mte_pirouette, landing=mte_landing)


def fmt(r: dict) -> str:
    mark = {"desired": "●  desired", "adequate": "◐  adequate", "yetersiz": "○  yetersiz"}[r["rating"]]
    extra = {
        "hover": lambda: f"sabitlenme {r['t_stab_desired'] if r['t_stab_desired'] is not None else '—'} / "
                         f"{r['t_stab_adequate'] if r['t_stab_adequate'] is not None else '—'} s, hedefi geçme "
                         f"{r['overshoot_ft']:.1f} ft; 30 s tutmada maks boyuna {r['hold_lon']:.1f}, yanal {r['hold_lat']:.1f}, "
                         f"irtifa {r['hold_alt']:.1f} ft, heading {r['hold_hdg']:.1f}°",
        "turn": lambda: f"±3° içinde {r['t_stab_desired'] if r['t_stab_desired'] is not None else '—'} s, "
                        f"maks boyuna {r['max_lon']:.1f} ft, yanal {r['max_lat']:.1f} ft, irtifa {r['max_alt']:.1f} ft",
        "vertical": lambda: f"tepe {r['t_top'] if r['t_top'] is not None else '—'} s, bitiş "
                            f"{r['t_done_desired'] if r['t_done_desired'] is not None else '—'} s, "
                            f"maks yatay {max(r['max_lon'], r['max_lat']):.1f} ft, heading {r['max_hdg']:.1f}°",
        "pirouette": lambda: f"dönüş {r['t_arrive_adequate'] if r['t_arrive_adequate'] is not None else '—'} s, "
                             f"maks radyal {r['max_radial']:.1f} ft, irtifa {r['max_alt']:.1f} ft, burun {r['max_hdg']:.1f}°",
        "landing": lambda: (f"10 ft→temas {r.get('t_from_10ft', float('nan')):.1f} s, boyuna {r.get('lon', float('nan')):+.1f} ft, "
                            f"yanal {r.get('lat', float('nan')):+.1f} ft, temas {r.get('touchdown_vs', float('nan')):+.1f} ft/s"),
    }[r["mte"]]()
    extra = extra.replace("None", "—")
    return f"{r['mte']:10s} {r['side']:12s} {r['weight']:6.0f}  {mark:12s}  {extra}  tork maks {r['torque_max']:.0f} psi"


def evaluate(model_path, only=None, env_overrides=None, fuels=((0.0, 0.0),), verbose=True):
    R = Runner(model_path, env_overrides)
    out = []
    if verbose:
        kind = "tek ajanlı uçuş env'i (F8, sakin hava, yakıt ≥ 150 lbs/tank) " if R.flight else ""
        print(f"\nADS-33 MTE KARNESİ — model {Path(model_path).name}  env {kind}{R.overrides or 'stok'}")
    for name, runs in PLAN.items():
        if only and name not in only:
            continue
        for fuel in fuels:
            for fn_name, kw in runs:
                r = FN[fn_name](R, fuel=tuple(fuel), **kw)
                r = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
                out.append(r)
                if verbose:
                    print(fmt(r), flush=True)
    cnt = {c: sum(r["rating"] == c for r in out) for c in ("desired", "adequate", "yetersiz")}
    if verbose:
        print(f"  toplam: desired {cnt['desired']}, adequate {cnt['adequate']}, yetersiz {cnt['yetersiz']} / {len(out)}")
    return out, cnt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", nargs="*", default=None, choices=list(PLAN))
    ap.add_argument("--env", default=None, help="env ayarları (JSON), modelinkilerin üstüne")
    ap.add_argument("--heavy", action="store_true", help="ayrıca 9700 lbs (tank başına 600 lbs)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    fuels = ((0.0, 0.0), (600.0, 600.0)) if args.heavy else ((0.0, 0.0),)
    res, cnt = evaluate(args.model, args.only, json.loads(args.env) if args.env else None, fuels)
    print(f"({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dict(model=str(args.model), env=args.env, standards=MTE_STANDARDS,
                                                   results=res, summary=cnt), indent=1, ensure_ascii=False, default=str))
        print(f"kaydedildi: {args.json}")


if __name__ == "__main__":
    main()
