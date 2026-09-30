"""Physics-ext probe'ları için ortak düzenek: env'lerle aynı FDM kurulumu + ölçüm amaçlı tutucu (policy değil).

`Rig`: HelicopterEnvTakeoff'un kendi kurulum fonksiyonları (aynı JSBSim dt, rotor warm-up, AFCS yalnızca SAS, yakıt)
ile tek bir FDM; rüzgâr ve `PhysicsExt` kancaları kontrol adımının etrafında. `Holder`: irtifa / heading / hız (hava
ya da yer ekseninde) ya da yer konumu tutan kademeli PID (reset PID'lerinin genelleştirilmişi: hover kazançları
`_settle_hover`, ileri uçuş kazançları `_settle_m`; hava hızına göre harmanlanır), ileri besleme = SFD trim tabloları
(steady_flight_data.xml) — ama yer hızı yerine seçilen hıza göre bakılır. `settle_and_measure`: oturunca ortalama.
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jsbsim  # noqa: E402
import numpy as np  # noqa: E402

from helicopter_env_command import CONTROL_DT, wrap_deg  # noqa: E402
from helicopter_env_takeoff import CTRL_HI, CTRL_LO, HOVER_TRIM, HelicopterEnvTakeoff, TakeoffEnvConfig  # noqa: E402
from physics_ext import KT_TO_FPS, air_ground_velocities, fuel_total, read_torque_psi, shaft_power_rotor_hp  # noqa: E402

OUT = Path(__file__).resolve().parent


SUFFIX = ""                        # çıktı dosya adı eki: stok uçak "", repo uçağı "_repo" / "_repo_cap56"


def outp(name: str) -> Path:
    """docs/physics_ext/<ad><SUFFIX>.<uzantı> (stok ve repo uçağı sonuçları yan yana kalsın)."""
    stem, dot, ext = name.rpartition(".")
    return OUT / (f"{stem}{SUFFIX}.{ext}" if dot else f"{name}{SUFFIX}")


def env_config_from_args(aircraft: str | None, power_cap_psi: float = 0.0) -> TakeoffEnvConfig | None:
    """--aircraft repo [--power-cap 56] → TakeoffEnvConfig (kalibre yer etkisi + güç tavanı); None → stok.
    Çıktı dosyalarının ekini de ayarlar."""
    global SUFFIX
    if not aircraft:
        SUFFIX = ""
        return None
    SUFFIX = f"_{aircraft}" + (f"_cap{power_cap_psi:g}" if power_cap_psi > 0 else "")
    return TakeoffEnvConfig(aircraft=aircraft, power_cap_psi=float(power_cap_psi))
GROUND_MSL_FT = 2283.5            # reset00.xml (Edwards AFB)


# ------------------------------------------------------------------------------------------------------------------
# SFD tabloları (steady_flight_data.xml): fcs/automatic/<k>-trim-cmd-norm, ap/afcs/automatic/<φ,θ>-trim-rad
# satır = v_dir_kts (JSBSim'de yer hızı × sign(u)), sütun = MSL irtifa
# ------------------------------------------------------------------------------------------------------------------
def _load_sfd():
    path = Path(jsbsim.get_default_root_dir()) / "aircraft" / "ah1s" / "Systems" / "steady_flight_data.xml"
    root = ET.parse(path).getroot()
    tabs = {}
    for fn in root.iter("fcs_function"):
        name = fn.get("name")
        td = fn.find(".//tableData")
        if td is None:
            continue
        rows = [ln.split() for ln in td.text.strip().splitlines() if ln.strip()]
        cols = np.array([float(x) for x in rows[0]])
        v = np.array([float(r[0]) for r in rows[1:]])
        data = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
        tabs[name] = (v, cols, data)
    return tabs


SFD = _load_sfd()
SFD_CMD = ("fcs/automatic/collective-trim-cmd-norm", "fcs/automatic/pitch-trim-cmd-norm",
           "fcs/automatic/roll-trim-cmd-norm", "fcs/automatic/yaw-trim-cmd-norm")


def sfd_lookup(name: str, v_kts: float, h_msl_ft: float) -> float:
    v, cols, data = SFD[name]
    vv = float(np.clip(v_kts, v[0], v[-1]))
    hh = float(np.clip(h_msl_ft, cols[0], cols[-1]))
    col = np.array([np.interp(hh, cols, data[i]) for i in range(len(v))])
    return float(np.interp(vv, v, col))


def sfd_trim(v_kts: float, h_msl_ft: float, weight_lbs: float = 8500.0) -> np.ndarray:
    c = np.array([sfd_lookup(k, v_kts, h_msl_ft) for k in SFD_CMD])
    c[0] += 0.057 * (weight_lbs - 8500.0) / 1000.0                  # hover probu (+0.057 / 1000 lbs)
    return c


# ------------------------------------------------------------------------------------------------------------------
# FDM düzeneği
# ------------------------------------------------------------------------------------------------------------------
class Rig:
    """physics: physics_ext config (dict) → env'in kendi katmanı (env.ext; _run_plain her adımda before_step çağırır).
    env_config: uçak ayarları (aircraft="repo", power_cap_psi, ...); physics burada verilirse o kullanılır."""

    def __init__(self, fuel=(0.0, 0.0), physics=None, env_config: TakeoffEnvConfig | None = None):
        cfg = env_config or TakeoffEnvConfig()
        if physics is not None:
            from dataclasses import replace
            cfg = replace(cfg, physics=physics)
        self.env = HelicopterEnvTakeoff(level="K1", config=cfg)
        env = self.env
        env._create_fdm()
        env._set_fuel(fuel)
        self.fdm = env.fdm
        self.ext = env.ext
        env._warmup_rotor()
        f = self.fdm
        env._lat0, env._lon0 = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        env._set_sas(0.0)
        self.t = 0.0
        self.wind_ned = (0.0, 0.0)

    def set_wind(self, speed_kt: float, from_deg: float):
        a = math.radians(from_deg)
        w = speed_kt * KT_TO_FPS
        self.wind_ned = (-w * math.cos(a), -w * math.sin(a))
        f = self.fdm
        f["atmosphere/wind-north-fps"], f["atmosphere/wind-east-fps"] = self.wind_ned
        f["atmosphere/wind-down-fps"] = 0.0

    def teleport(self, h_agl_ft: float, psi_deg: float, u_air=0.0, v_air=0.0, ctrl=None):
        """Havaya taşı: gövde ekseninde HAVA hızı (u, v) + rüzgâr → IC yer hızı. Rotor dönmeye devam eder."""
        f = self.fdm
        ps = math.radians(psi_deg)
        wn, we = self.wind_ned
        ug = u_air + wn * math.cos(ps) + we * math.sin(ps)
        vg = v_air - wn * math.sin(ps) + we * math.cos(ps)
        if ctrl is not None:
            self.env._write_controls(ctrl)
        vals = {"ic/lat-geod-rad": self.env._lat0, "ic/long-gc-rad": self.env._lon0, "ic/h-agl-ft": h_agl_ft,
                "ic/phi-rad": 0.0, "ic/theta-rad": 0.0, "ic/psi-true-rad": ps % (2 * math.pi),
                "ic/u-fps": ug, "ic/v-fps": vg, "ic/w-fps": 0.0,
                "ic/p-rad_sec": 0.0, "ic/q-rad_sec": 0.0, "ic/r-rad_sec": 0.0}
        for k, x in vals.items():
            f[k] = x
        f["fcs/rpm-governor-active-norm"] = 1.0
        if not f.run_ic():
            raise RuntimeError("run_ic başarısız")
        # run_ic rüzgârı IC rüzgârıyla (0) eziyor → yeniden yaz
        f["atmosphere/wind-north-fps"], f["atmosphere/wind-east-fps"] = self.wind_ned
        f["atmosphere/wind-down-fps"] = 0.0
        self.env._set_sas(psi_deg)

    def start_physics(self, seed: int = 0, heading_deg: float = 0.0, options: dict | None = None):
        """Ölçüm başlangıcı: katmanın bölüm parametreleri + rüzgâr + türbülans / gust / yakıt sayacı."""
        self.ext.begin_episode(np.random.default_rng(seed), heading_deg, options)
        self.ext.attach(self.fdm)
        self.ext.start_disturbances(self.fdm)

    def state(self) -> dict:
        f = self.fdm
        s = self.env._state()
        s.update(air_ground_velocities(f))
        s["psi_gauge"] = read_torque_psi(f)
        s["p_rotor_hp"] = shaft_power_rotor_hp(f)
        s["p_eng_hp"] = float(f["propulsion/engine/power-hp"])
        s["h_msl"] = float(f["position/h-sl-ft"])
        s["vt"] = float(f["velocities/vt-fps"])
        s["cg_x"] = float(f["inertia/cg-x-in"])
        s["cg_z"] = float(f["inertia/cg-z-in"])
        s["fuel"] = fuel_total(f)
        s["gust_n"] = float(f["atmosphere/total-wind-north-fps"]) - self.wind_ned[0]
        s["gust_e"] = float(f["atmosphere/total-wind-east-fps"]) - self.wind_ned[1]
        s["gust_d"] = float(f["atmosphere/total-wind-down-fps"])
        s["rot_coll"] = float(f["propulsion/engine/collective-ctrl-rad"])
        s["rot_lon"] = float(f["propulsion/engine/longitudinal-ctrl-rad"])
        s["rot_lat"] = float(f["propulsion/engine/lateral-ctrl-rad"])
        s["rot_tail"] = float(f["propulsion/engine[1]/antitorque-ctrl-rad"])
        return s

    def step(self, ctrl) -> bool:
        self.env._write_controls(np.clip(ctrl, CTRL_LO, CTRL_HI))
        ok = self.env._run_plain()                              # içinde env.ext.before_step
        self.ext.after_step(self.fdm)
        self.t += CONTROL_DT
        return ok


# ------------------------------------------------------------------------------------------------------------------
# Tutucu (yalnızca ölçüm; policy'ye action önermez)
# ------------------------------------------------------------------------------------------------------------------
class Holder:
    """mode="air": gövde ekseninde hava hızı (u, v) hedefi; mode="ground": yer konumu (n0, e0) tut (hover).
    ff: "sfd_air" (SFD tabloları ileri HAVA hızıyla), "sfd_ground" (JSBSim'deki gibi yer hızıyla), "hover" (sabit)."""

    def __init__(self, h_agl: float, psi_deg: float, mode: str = "air", u=0.0, v=0.0, n0=0.0, e0=0.0,
                 ff: str = "sfd_air", weight: float = 8500.0, ff_table=None):
        self.h0, self.psi0, self.mode = h_agl, psi_deg, mode
        self.u_t, self.v_t, self.n0, self.e0 = u, v, n0, e0
        self.ff, self.weight, self.ff_table = ff, weight, ff_table
        self.I = np.zeros(4)
        self.ctrl = np.array(HOVER_TRIM, dtype=np.float64)

    def feedforward(self, s: dict) -> np.ndarray:
        if self.ff_table is not None:
            return np.asarray(self.ff_table(s), dtype=np.float64)
        if self.ff == "hover":
            c = np.array(HOVER_TRIM, dtype=np.float64)
            c[0] += 0.057 * (self.weight - 8500.0) / 1000.0
            return c
        v = s["u_air"] if self.ff == "sfd_air" else math.copysign(s["vh"], s["u"] if s["u"] else 1.0)
        return sfd_trim(v / KT_TO_FPS, s["h_msl"], self.weight)

    def __call__(self, s: dict) -> np.ndarray:
        dt = CONTROL_DT
        ff = self.feedforward(s)
        # hava hızına göre kazanç harmanı: 0 → hover (_settle_hover), 1 → ileri uçuş (_settle_m)
        x = float(np.clip((abs(s["u_air"]) - 20.0 * KT_TO_FPS) / (20.0 * KT_TO_FPS), 0.0, 1.0))
        if self.mode == "ground":
            ps = math.radians(s["psi_deg"])
            dn, de = self.n0 - s["n"], self.e0 - s["e"]
            xf, yr = dn * math.cos(ps) + de * math.sin(ps), -dn * math.sin(ps) + de * math.cos(ps)
            eu = float(np.clip(0.12 * xf, -6, 6)) - s["u_gnd"]
            ev = float(np.clip(0.12 * yr, -6, 6)) - s["v_gnd"]
        else:
            eu, ev = self.u_t - s["u_air"], self.v_t - s["v_air"]
        eh = self.h0 - s["h"]
        evs = float(np.clip(0.4 * eh, -6, 6)) - s["vs"]
        eps = wrap_deg(self.psi0 - s["psi_deg"])
        k_u = (1 - x) * 0.004 + x * 0.002
        self.I[0] = np.clip(self.I[0] + 0.01 * evs * dt, -0.3, 0.3)
        self.I[1] = np.clip(self.I[1] - k_u * eu * dt, -0.3, 0.3)
        self.I[2] = np.clip(self.I[2] + k_u * ev * dt, -0.3, 0.3)
        self.I[3] = np.clip(self.I[3] - 0.003 * eps * dt, -0.3, 0.3)
        th_ref = float(np.clip(-((1 - x) * 0.02 + x * 0.012) * eu, -0.25, 0.25)) + self.I[1]
        ph_ref = float(np.clip(((1 - x) * 0.03 + x * 0.02) * ev, -0.25, 0.25)) + self.I[2]
        kth, kq = (1 - x) * 3.0 + x * 1.2, (1 - x) * 1.6 + x * 0.6
        c = np.array([ff[0] + 0.03 * evs + self.I[0],
                      ff[1] + kth * (s["theta"] - th_ref) + kq * s["q"],
                      ff[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"],
                      ff[3] - 0.02 * eps + 0.8 * s["r"] + self.I[3]])
        self.ctrl = np.clip(c, CTRL_LO, CTRL_HI)
        return self.ctrl


def run_hold(rig: Rig, hold: Holder, seconds: float, record: bool = False):
    rows = []
    n = int(round(seconds / CONTROL_DT))
    for _ in range(n):
        s = rig.state()
        c = hold(s)
        if record:
            rows.append(dict(t=rig.t, **{k: s[k] for k in REC_KEYS}, c0=c[0], c1=c[1], c2=c[2], c3=c[3]))
        if not rig.step(c):
            raise RuntimeError("JSBSim durdu")
    return rows


REC_KEYS = ("h", "vs", "n", "e", "u_air", "v_air", "w_air", "u_gnd", "v_gnd", "vh", "phi", "theta", "psi_deg", "p",
            "q", "r", "rpm", "psi_gauge", "p_rotor_hp", "p_eng_hp", "weight", "fuel", "cg_x", "cg_z", "vt", "gust_n",
            "gust_e", "gust_d", "rot_coll", "rot_lon", "rot_lat", "rot_tail")


def settle_and_measure(rig: Rig, hold: Holder, t_min=20.0, t_max=150.0, avg_s=8.0, tol=None) -> dict:
    """Otur (hız / irtifa / heading / açısal hız + kumanda oynaması küçük + rotor devri sabit) → avg_s ortalaması.
    Rotor devri: governor PID'inin integrali yavaş (ki 0.0012) → hover'da devir ~100 s'de 321 → 323.5 rpm kayıyor,
    collective trimi bununla ±0.003 değişiyor; bu yüzden devir de oturmalı."""
    tol = dict(dict(vs=0.3, uv=0.5, h=2.0, psi=0.5, rate=0.01, dctrl=0.004, drpm=0.1), **(tol or {}))
    hist, rpms = [], []
    t0 = rig.t
    stable = 0.0
    while rig.t - t0 < t_max:
        s = rig.state()
        c = hold(s)
        hist.append(c.copy())
        rpms.append(s["rpm"])
        if not rig.step(c):
            raise RuntimeError("JSBSim durdu")
        if hold.mode == "ground":
            uv_err = math.hypot(s["u_gnd"], s["v_gnd"])
        else:
            uv_err = math.hypot(hold.u_t - s["u_air"], hold.v_t - s["v_air"])
        recent = np.asarray(hist[-40:])
        ok = (abs(s["vs"]) < tol["vs"] and uv_err < tol["uv"] and abs(hold.h0 - s["h"]) < tol["h"]
              and abs(wrap_deg(hold.psi0 - s["psi_deg"])) < tol["psi"]
              and max(abs(s["p"]), abs(s["q"]), abs(s["r"])) < tol["rate"]
              and len(recent) >= 40 and float(np.ptp(recent, axis=0).max()) < tol["dctrl"]
              and len(rpms) >= 134 and abs(rpms[-1] - rpms[-134]) < tol["drpm"])
        stable = stable + CONTROL_DT if ok else 0.0
        if rig.t - t0 >= t_min and stable >= 3.0:
            break
    converged = stable >= 3.0
    rows = run_hold(rig, hold, avg_s, record=True)
    keys = [k for k in rows[0] if k != "t"]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    m["ctrl_std"] = float(np.max([np.std([r[f"c{i}"] for r in rows]) for i in range(4)]))
    m["converged"] = converged
    m["settle_s"] = rig.t - t0 - avg_s
    return m
