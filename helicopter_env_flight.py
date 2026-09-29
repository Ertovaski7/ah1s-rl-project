from __future__ import annotations

"""
HELICOPTER ENV FLIGHT — tek ajan: kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δ komutları → duruş → iniş
=============================================================================================================

`helicopter_env_takeoff.py`'nin (kalkış / hover / iniş, dört kumanda doğrudan, AFCS yalnızca SAS) üstüne kurulu.
Aynı action hattı: a → filtre (α 0.5) → expo (0.2a + 0.8a³) → trim ± aralık (0.6, 1, 1, 1) → hız sınırı.

Farklar (2026-09-29, Aşama 2):
* Fizik baştan açık: repo uçağı (kalibre yer etkisi), güç tavanı %100 tork (56 psi), tork gözlemi + cezası (kalkış
  env'inin K10 ayarları), yakıt tüketimi (physics_ext, el kitabı grafiği; iki tanktan eşit; bitince motor ayrılır →
  bölüm kesilir). Rüzgâr / gust / türbülans seviyenin çevre aşamasından (flight_curriculum.ENV_STAGES).
* Trim HAVA HIZINA göre çizelgelenir: `trim_table.AirspeedTrimTable` (probe e; gövde ekseninde ileri hava hızı ×
  ağırlık; u = 0'da hover trimi). Tabloya süzülmüş hava hızı girer (τ 2 s): türbülans kumandayı titretmesin. Trim
  kayınca filtre durumu bir sonraki adımda yeni trime göre yeniden hesaplanır (kalkış env'inin hattı): sabit action'da
  fiziksel kumanda trimle birlikte kayar (ileri besleme), a-uzayında sıçrama olmaz.
* Yeni görevler (flight_curriculum):
    cruise — ileri uçuşta hava hızı (u*, gövde ekseninde ileri), heading ψ*, irtifa h* hedefleri; Δ'lar ölçülen duruma
             göre, komut verilmeyen eksen önceki hedefine devam eder. Konum hedefi helikopteri izler (konum hatası 0).
             Hover'dan verilen hız komutu = hızlanma (ileri uçuşa geçiş).
    stop   — ileri uçuştan duruş: referans noktası yer izi boyunca mevcut yer hızından sabit yavaşlamayla durur
             (depart'ın hareketli hedefi, tersine); sonra o noktada hover bandı.
* Gözlem 41 = kalkış env'inin 29'u (ileri uçuşta konum hatası 0; takvim gecikmesinin ilk elemanı hız ekseni) + tork
  (1) + hava hızı ileri / yana, yer hızı ileri / yana (4; ileri uçuşa uygun ölçek; farkları rüzgârı dolaylı gösterir)
  + ileri uçuş bayrağı, hız hatası (ince / kaba), hedef hava hızı (4) + hareketli hedefin hızı (2) + ağırlık (1).
  Rüzgârın kendisi verilmez.
* Ödül: hover görevlerinde kalkış env'inin ödülü (tork cezası dahil); ileri uçuşta hız / irtifa / heading çekirdekleri
  + yönlendirme (dikey hız, dönüş hızı, ivme) + ilerleme − takvim gecikmesi − süre aşımı − kuplaj − yana kayma − açı /
  oran − kumanda hızı − tork. Başarı: hız ±4 ft/s (süzülmüş hava hızı, τ 1 s), irtifa ±12 ft, heading ±3°, dikey hız
  ±4 ft/s, `cruise_hold_s` boyunca.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flight_curriculum import (CRUISE_MAX_ALT_FT, CRUISE_MAX_KT, CRUISE_MIN_ALT_FT, CRUISE_MIN_KT, DEFAULT_FLIGHT_LEVELS,
                               FlightLevel, cruise_yaw_rate_dps, find_flight_level, flight_time_target)
from helicopter_env_command import CONTROL_DT, wrap_deg
from helicopter_env_takeoff import (ACTIVE_AXES, CTRL_HI, CTRL_LO, GROUND_H_FT, MAX_HOVER_H_FT, MIN_HOVER_H_FT,
                                    OBS_DIM_T, HelicopterEnvTakeoff, TakeoffEnvConfig, deadline_of, expo, expo_inv)
from physics_ext import air_ground_velocities, fuel_total, torque_penalty
from trim_table import AirspeedTrimTable

KT = 1.6878099
REPO = Path(__file__).resolve().parent
FLIGHT_EXTRA_OBS = 11
OBS_DIM_F = OBS_DIM_T + 1 + FLIGHT_EXTRA_OBS            # 41 (torque_obs açık, track_obs kapalı)


@dataclass
class FlightEnvConfig(TakeoffEnvConfig):
    # --- fizik (baştan açık) ----------------------------------------------------------------------------
    aircraft: str = "repo"
    power_cap_psi: float = 56.0
    torque_obs: bool = True
    pen_torque_cont: float = 0.3
    pen_torque_over: float = 2.0
    pen_rpm_low: float = 1.0
    torque_aware_climb: bool = True
    physics: dict | None = field(default_factory=lambda: {"fuel": {"enable": True}})
    # --- trim çizelgesi ---------------------------------------------------------------------------------
    trim_table: str = "docs/physics_ext/trim_table_airspeed_repo_cap56.json"
    trim_schedule: bool = True              # False → sabit hover trimi (kalkış env'i gibi; karşılaştırma için)
    trim_u_tau_s: float = 2.0               # trim tablosuna giren hava hızının süzgeci
    # --- ileri uçuş: başarı -----------------------------------------------------------------------------
    u_meas_tau_s: float = 1.0               # hız hatası / başarı için süzülmüş hava hızı (türbülansta ±2 ft/s tutulamaz)
    tol_u_fps: float = 4.0
    tol_psi_cruise_deg: float = 3.0
    tol_h_cruise_ft: float = 12.0
    tol_vs_cruise_fps: float = 4.0
    coupling_cruise: tuple = (15.0, 10.0, 40.0)   # komut verilmeyen eksen: hız ft/s, heading °, irtifa ft
    # --- ileri uçuş: ödül ----------------------------------------------------------------------------------
    w_u: float = 1.0
    kernel_u: tuple = (2.0, 15.0)
    kernel_h_cruise: tuple = (5.0, 40.0)
    kernel_psi_cruise: tuple = (2.0, 15.0)
    w_accel: float = 0.4                    # ivme yönlendirmesi: istenen u̇ = clip(k·e_u, ±ivme)
    guide_k_u: float = 0.3
    kernel_accel: tuple = (1.0, 4.0)
    progress_min_scale_u: float = 10.0
    sched_scale_u: float = 15.0
    pen_side: float = 0.3                   # · (v_air / 10)², hava hızı 30 kt üstünde (koordineli uçuş)
    cruise_att_roll_deg: float = 35.0       # açı cezası eşikleri (hover'da 20° / 15°)
    cruise_att_pitch_deg: float = 20.0
    # --- güvenlik ----------------------------------------------------------------------------------------------
    max_roll_cruise_deg: float = 60.0       # ileri uçuşta (hava hızı > 30 kt)
    max_airspeed_kt: float = 130.0
    min_hs_at_speed_ft: float = 8.0         # hava hızı > 40 kt iken kızak yüksekliği bunun altına inmesin
    max_speed_fps: float = 60.0             # hover görevlerinde yer hızı (kalkış env'i); cruise / stop'ta hava hızı sınırı
    # --- reset ---------------------------------------------------------------------------------------------------
    cruise_settle_max_s: float = 90.0
    max_episode_s: float = 600.0


class HelicopterEnvFlight(HelicopterEnvTakeoff):
    """Kalkış env'i + ileri uçuş (cruise), duruş (stop), hava hızı trim çizelgesi, rüzgâr / türbülans / yakıt."""
    metadata = {"render_modes": []}

    def __init__(self, level=0, levels=None, config: FlightEnvConfig | None = None, rehearsal: bool = False):
        levels = list(levels or DEFAULT_FLIGHT_LEVELS)
        cfg = config or FlightEnvConfig()
        if not cfg.torque_obs or cfg.track_obs:
            raise ValueError("flight env: torque_obs=True, track_obs=False (kendi hedef hızı gözlemi var)")
        super().__init__(level=0, levels=levels, config=cfg, rehearsal=rehearsal)
        self.level_index = find_flight_level(level, self.levels)
        path = Path(cfg.trim_table)
        self.trim_tab = AirspeedTrimTable(path if path.is_absolute() else REPO / path)
        self.obs_dim = OBS_DIM_F
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)
        self.u_trim = 0.0                   # trim çizelgesinin süzülmüş hava hızı (ft/s)
        self.u_meas = 0.0                   # başarı / hız hatası için süzülmüş hava hızı (ft/s)
        self.u_dot = 0.0
        self._u_prev = 0.0
        self.cruise = None                  # ileri uçuş hedefi: dict(u=ft/s) — pencere cruise iken
        self.env_stage = "E0"

    # =================================================================
    # trim / hava hızı
    # =================================================================

    def _air(self) -> dict:
        return air_ground_velocities(self.fdm)

    def _trim_for(self, u_fps: float, weight: float, h_agl: float) -> np.ndarray:
        if not self.cfg.trim_schedule:
            return np.asarray(self.cfg.hover_trim, dtype=np.float64)
        c = self.trim_tab(u_fps / KT, weight, h_agl)
        return np.clip(c, CTRL_LO, CTRL_HI)

    def _update_trim(self, s: dict):
        """Süzgeçler (hava hızı) + trim çizelgesi. step() başında, action hattından ÖNCE çağrılır."""
        cfg = self.cfg
        u_air = float(self.fdm["velocities/u-aero-fps"])
        a_t = 1.0 - math.exp(-CONTROL_DT / cfg.trim_u_tau_s)
        self.u_trim += a_t * (u_air - self.u_trim)
        self.trim = self._trim_for(self.u_trim, s["weight"], s["h"])

    def _ige_coll(self, weight: float) -> float:
        """İniş ödülünde IGE collective'i: tablonun hover trimi (ağırlığı içeriyor) − 0.04 + yer etkisi farkı."""
        return float(self.trim_tab(0.0, weight)[0]) - 0.04 + self._ige_coll_offset()

    # =================================================================
    # RESET
    # =================================================================

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if "level" in options:
            self.set_level(options["level"])
        rng, cfg = self.np_random, self.cfg
        ep_idx = self.level_index
        lv0 = self.levels[ep_idx]
        if (self.rehearsal and "level" not in options and "tasks" not in options and lv0.rehearse
                and rng.random() < lv0.p_rehearse):
            ep_idx = find_flight_level(lv0.rehearse[int(rng.integers(len(lv0.rehearse)))], self.levels)
        lv: FlightLevel = self.levels[ep_idx]
        st = lv.sample_start(rng)
        start = options.get("start", st["start"])
        if start not in ("ground", "hover", "touch", "low", "cruise"):
            raise ValueError(f"bilinmeyen başlangıç: {start}")
        air = start in ("hover", "low", "cruise")
        h0 = float(options.get("start_alt_ft", st["h"] if air else GROUND_H_FT))
        if air and "start_alt_ft" not in options and st["start"] != start:
            h0 = (float(rng.uniform(*lv.cruise_start_alt_ft)) if start == "cruise"
                  else float(rng.uniform(*lv.hover_start_alt_ft)) if start == "hover"
                  else GROUND_H_FT + float(rng.uniform(*lv.low_hover_hs_ft)))
        u0_kt = float(options.get("start_speed_kt", st.get("u_kt", rng.uniform(*lv.cruise_start_kt))
                                  if start == "cruise" else 0.0))
        psi0 = float(options["start_heading_deg"]) % 360.0 if "start_heading_deg" in options else float(rng.uniform(0, 360))
        fuel = tuple(options["fuel"]) if options.get("fuel") is not None else lv.sample_fuel(rng)
        # reset yardımcıları (_teleport_ground, _settle_hover) self.trim'i 8500 lbs hover trimi sanıyor (ağırlığı kendileri
        # ekliyor): önceki bölümün (ör. 80 kt) trimi kalmasın
        self.trim = self._trim_for(0.0, 8500.0, 300.0)
        c_touch = None
        if start == "touch":
            c_touch = (float(options.get("touch_coll", rng.uniform(*lv.touch_coll))) + 0.057 * sum(fuel) / 1000.0
                       + self._ige_coll_offset())
        # çevre: seviyenin çevre aşaması (options["physics"] verilirse o; değerlendirme / canlı)
        if "physics" in options:
            self.env_stage, env_opts = "custom", dict(options["physics"] or {})
        else:
            self.env_stage, env_opts = lv.sample_env(rng)
        self.ext.begin_episode(rng, psi0, env_opts)
        errors, info = [], {}
        for attempt in range(3):
            try:
                self._create_fdm()
                self._set_fuel(fuel)
                self.ext.attach(self.fdm)
                self._warmup_rotor()
                f = self.fdm
                self._lat0, self._lon0 = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
                if start in ("ground", "touch"):
                    self._teleport_ground(psi0)
                    if start == "touch":
                        self._raise_collective(c_touch)
                    info = dict(mode=start)
                    break
                if start == "cruise":
                    ok, info = self._settle_cruise(h0, u0_kt * KT, psi0, fuel)
                else:
                    ok, info = self._settle_hover(h0, psi0, fuel)
                if ok:
                    info["mode"] = start
                    break
                errors.append(info.get("reason", "?"))
            except Exception as exc:                            # noqa: BLE001
                errors.append(str(exc))
        else:
            raise RuntimeError("başlangıç koşulu kurulamadı: " + " | ".join(errors))
        self.setup_info = dict(info, errors=errors, start=start, h0=h0, psi0=psi0, fuel=fuel, touch_coll=c_touch,
                               start_speed_kt=u0_kt, env_stage=self.env_stage,
                               weight=float(self.fdm["inertia/weight-lbs"]), cg_x_in=float(self.fdm["inertia/cg-x-in"]))
        self.setup_info.pop("ctrl", None)
        pert = float(options.get("start_perturb", lv.start_perturb)) if start == "hover" else 0.0
        if pert > 0.0:
            a = float(rng.uniform(0, 2 * math.pi))
            self._set_ic_from_state(dn=3.0 * pert * math.cos(a), de=3.0 * pert * math.sin(a),
                                    dw=float(rng.uniform(-1.5, 1.5)) * pert,
                                    dphi=float(rng.uniform(-3, 3)) * pert, dtheta=float(rng.uniform(-3, 3)) * pert)
        return self._handover(lv, options, start, ep_idx)

    def _settle_cruise(self, h0: float, u0: float, psi0: float, fuel) -> tuple[bool, dict]:
        """Yalnızca reset: ileri uçuşta başlangıç için hava hızı / irtifa / heading / yana hava hızı tutan PID (policy'ye
        action önermez). İleri besleme = trim tablosu. Rüzgârda IC yer hızı = hava hızı + rüzgâr."""
        cfg = self.cfg
        f = self.fdm
        w = 8500.0 + float(sum(fuel))
        ff = self._trim_for(u0, w, h0)
        self._write_controls(ff)
        ps = math.radians(psi0)
        wn, we = self.ext.wind_ned
        ug = u0 + wn * math.cos(ps) + we * math.sin(ps)
        vg = -wn * math.sin(ps) + we * math.cos(ps)
        vals = {"ic/lat-geod-rad": self._lat0, "ic/long-gc-rad": self._lon0, "ic/h-agl-ft": h0, "ic/phi-rad": 0.0,
                "ic/theta-rad": 0.0, "ic/psi-true-rad": ps % (2 * math.pi), "ic/u-fps": ug, "ic/v-fps": vg,
                "ic/w-fps": 0.0, "ic/p-rad_sec": 0.0, "ic/q-rad_sec": 0.0, "ic/r-rad_sec": 0.0}
        for k, x in vals.items():
            f[k] = x
        if not f.run_ic():
            raise RuntimeError("run_ic() başarısız (cruise)")
        self._set_sas(psi0)
        integ = np.zeros(4)
        t, hold = 0.0, 0.0
        c = ff.copy()
        while t < cfg.cruise_settle_max_s:
            s = self._state()
            a = self._air()
            ff = self._trim_for(a["u_air"], s["weight"], s["h"])
            eu, ev = u0 - a["u_air"], -a["v_air"]
            evs = float(np.clip(0.4 * (h0 - s["h"]), -6, 6)) - s["vs"]
            eps = wrap_deg(psi0 - s["psi_deg"])
            x = float(np.clip((abs(a["u_air"]) - 20.0 * KT) / (20.0 * KT), 0.0, 1.0))
            k_u = (1 - x) * 0.004 + x * 0.002
            integ[0] = np.clip(integ[0] + 0.01 * evs * CONTROL_DT, -0.3, 0.3)
            integ[1] = np.clip(integ[1] - k_u * eu * CONTROL_DT, -0.3, 0.3)
            integ[2] = np.clip(integ[2] + k_u * ev * CONTROL_DT, -0.3, 0.3)
            integ[3] = np.clip(integ[3] - 0.003 * eps * CONTROL_DT, -0.3, 0.3)
            th_ref = float(np.clip(-((1 - x) * 0.02 + x * 0.012) * eu, -0.25, 0.25)) + integ[1]
            ph_ref = float(np.clip(((1 - x) * 0.03 + x * 0.02) * ev, -0.25, 0.25)) + integ[2]
            kth, kq = (1 - x) * 3.0 + x * 1.2, (1 - x) * 1.6 + x * 0.6
            c = np.array([ff[0] + 0.03 * evs + integ[0], ff[1] + kth * (s["theta"] - th_ref) + kq * s["q"],
                          ff[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"], ff[3] - 0.02 * eps + 0.8 * s["r"] + integ[3]])
            c = np.clip(c, CTRL_LO, CTRL_HI)
            self._write_controls(c)
            if not self._run_plain():
                return False, dict(reason="jsbsim_stopped")
            t += CONTROL_DT
            ok = (abs(h0 - s["h"]) < 3.0 and abs(s["vs"]) < 1.0 and abs(eu) < 1.5 and abs(a["v_air"]) < 1.5
                  and abs(eps) < 1.0 and max(abs(s["p"]), abs(s["q"]), abs(s["r"])) < 0.03)
            hold = hold + CONTROL_DT if ok else 0.0
            if hold >= 2.0:
                return True, dict(settle_s=t)
        s = self._state()
        return False, dict(reason=f"ileri uçuş kurulamadı ({cfg.cruise_settle_max_s:.0f} s): h={s['h']:.1f}")

    def _handover(self, lv: FlightLevel, options: dict, start: str, ep_idx: int | None = None):
        # süzgeçler mevcut hava hızıyla başlasın; trim bu hıza göre (filtre durumu kumanda − trim'den)
        a = self._air()
        self.u_trim = self.u_meas = self._u_prev = a["u_air"]
        self.u_dot = 0.0
        s0 = self._state()
        self.trim = self._trim_for(self.u_trim, s0["weight"], s0["h"])
        self.cruise = None
        if start == "cruise" and "tasks" not in options:        # takvim ileri uçuş başlangıcına göre (hover'ınki değil)
            options = dict(options, tasks=lv.sample_schedule(self.np_random, "cruise"))
        base_start = "hover" if start == "cruise" else start     # hedef irtifa = mevcut irtifa
        obs, info = super()._handover(lv, options, base_start, ep_idx)
        info["setup"]["start"] = start
        return self._obs(self._state(), self._errors(self._state())), info

    # =================================================================
    # GÖREVLER
    # =================================================================

    def _cruise_prev(self) -> dict | None:
        """Önceki pencere ileri uçuş hedefiyse onun hedefleri (komut verilmeyen eksenler devam eder)."""
        if self.cruise is not None and self.windows and self.windows[-1]["kind"] == "cruise":
            return dict(u=self.cruise["u"], h=self.target["h"], psi=self.target["psi"])
        return None

    def _new_window(self, d: dict, t: float, s: dict) -> dict:
        kind = d["kind"]
        if kind not in ("cruise", "stop"):
            self.cruise = None
            return super()._new_window(d, t, s)
        cfg, lv = self.cfg, self.ep_level
        tg = self.target
        self.track = None
        if kind == "stop":
            self.cruise = None
            v0 = math.hypot(s["vn"], s["ve"])
            c = math.atan2(s["ve"], s["vn"]) if v0 > 1.0 else math.radians(s["psi_deg"])
            dec = float(d.get("decel", 2.5))
            tg.update(n=s["n"], e=s["e"], psi=self.psi_unwrap,
                      h=float(np.clip(d.get("h", s["h"]), MIN_HOVER_H_FT, MAX_HOVER_H_FT)))
            self.track = dict(n0=s["n"], e0=s["e"], c=c, v=0.0, v0=v0, a=dec, t0=t, lag_ft=0.0, prev_lag=0.0,
                              t_end=t + v0 / dec)
            T = float(d["T"]) if d.get("T") else flight_time_target("stop", lv, v0_fps=v0, decel=dec)
            active, category = ("xy", "h", "psi"), "stop"
            e = self._errors(s)
            e0 = {"xy": e["xy"], "h": abs(e["h"]), "psi": abs(e["psi"]), "u": 0.0}
            hold = lv.hold_s
        else:
            prev = self._cruise_prev()
            u_m, h_m = self.u_meas, s["h"]
            base_u = prev["u"] if prev else u_m
            base_h = prev["h"] if prev else h_m
            base_psi = prev["psi"] if prev else self.psi_unwrap
            axes = []
            if "u_kt" in d:
                u_t = float(d["u_kt"]) * KT
                axes.append("u")
            elif "du_kt" in d:
                du = float(d["du_kt"]) * KT
                if not CRUISE_MIN_KT * KT <= u_m + du <= CRUISE_MAX_KT * KT:
                    du = -du
                u_t = u_m + du
                axes.append("u")
            else:
                u_t = base_u
            u_t = float(np.clip(u_t, CRUISE_MIN_KT * KT if (axes or prev) else -30.0 * KT, CRUISE_MAX_KT * KT)) \
                if not d.get("hold") else u_m
            if "dpsi" in d:
                psi_t = self.psi_unwrap + float(d["dpsi"])
                axes.append("psi")
            else:
                psi_t = base_psi if not d.get("hold") else self.psi_unwrap
            if "h" in d:
                h_t = float(d["h"])
                axes.append("h")
            elif "dh" in d:
                dh = float(d["dh"])
                h_t = h_m + dh
                if d.get("accel"):                               # geçiş: en az 100 ft'e tırman
                    h_t = max(h_t, CRUISE_MIN_ALT_FT)
                elif not CRUISE_MIN_ALT_FT <= h_t <= CRUISE_MAX_ALT_FT:
                    # zarf dışı → ters yöne; o da dışarıdaysa (ör. alçakta) zarfın sınırına
                    h_alt = h_m - dh
                    h_t = h_alt if CRUISE_MIN_ALT_FT <= h_alt <= CRUISE_MAX_ALT_FT else float(
                        np.clip(h_t, CRUISE_MIN_ALT_FT, CRUISE_MAX_ALT_FT))
                axes.append("h")
            else:
                h_t = base_h if not d.get("hold") else h_m
            h_t = float(np.clip(h_t, MIN_HOVER_H_FT, MAX_HOVER_H_FT))
            tg.update(n=s["n"], e=s["e"], h=h_t, psi=psi_t)
            self.cruise = dict(u=u_t)
            if d.get("hold"):
                active, category = ("u", "h", "psi"), "cruise_hold"
            else:
                active = tuple(axes) if axes else ("u", "h", "psi")
                if d.get("accel"):
                    category = "accel"
                elif len(axes) > 1:
                    category = "cruise_mix"
                else:
                    category = "cruise_" + (axes[0] if axes else "hold")
            T = float(d["T"]) if d.get("T") else (
                flight_time_target("cruise", lv, du_fps=u_t - u_m, dpsi_deg=psi_t - self.psi_unwrap, dh_ft=h_t - h_m,
                                   v_fps=u_m) if not d.get("hold") else lv.hold_T_s)
            e = self._errors(s)
            e0 = {"xy": 0.0, "h": abs(e["h"]), "psi": abs(e["psi"]), "u": abs(e["u"])}
            hold = lv.cruise_hold_s
        w = dict(t=t, kind=kind, category=category, task={k: v for k, v in d.items() if not k.startswith("_")}, T=T,
                 h_start=s["h"], deadline=deadline_of(T), allow_s=0.0, hold_s=hold, active=active, target=dict(tg),
                 e0=e0, streak=0.0, entry_t=None, settle_s=float("nan"), on_time=False, success=False, closed=False,
                 complete=False, interrupted=False, coupling_ok=True,
                 max_err={"xy": 0.0, "h": 0.0, "psi": 0.0, "u": 0.0}, touchdown_vs=None, final_err=None,
                 tol=self._tol(tg["h"]), u_target=(self.cruise["u"] if self.cruise else 0.0))
        return w

    def _init_progress(self, s: dict):
        e2 = self._errors(s)
        self.prev_abs_err = {"xy": e2["xy"], "h": abs(e2["h"]), "psi": abs(e2["psi"]), "u": abs(e2["u"])}

    def _in_cruise(self) -> bool:
        return self.cruise is not None and bool(self.windows) and self.windows[-1]["kind"] == "cruise"

    def _errors(self, s: dict) -> dict:
        e = super()._errors(s)
        e["u"] = (self.cruise["u"] - self.u_meas) if self.cruise is not None else 0.0
        return e

    def _schedule_lag(self, e: dict) -> tuple[np.ndarray, float, float]:
        if not self.windows or self.windows[-1]["kind"] != "cruise":
            return super()._schedule_lag(e)
        w = self.windows[-1]
        t = self.steps * CONTROL_DT
        tau = (t - w["t"]) / max(w["T"], 1e-6)
        lag = np.zeros(3)
        scales = (self.cfg.sched_scale_u, self.cfg.sched_scale[1], self.cfg.sched_scale[2])
        for k, a in enumerate(("u", "h", "psi")):
            if a not in w["active"] or w["category"] == "cruise_hold":
                continue
            sched = w["e0"][a] * max(0.0, 1.0 - max(0.0, tau))
            lag[k] = math.copysign(max(0.0, abs(e[a]) - sched), e[a]) / scales[k]
        return lag, float(tau), float(w["T"])

    def _inside(self, s: dict, e: dict, w: dict) -> bool:
        cfg = self.cfg
        if w["kind"] == "cruise":
            a = self._air()
            return (s["wow"] == 0 and abs(e["u"]) <= cfg.tol_u_fps and abs(e["h"]) <= cfg.tol_h_cruise_ft
                    and abs(e["psi"]) <= cfg.tol_psi_cruise_deg and abs(s["vs"]) <= cfg.tol_vs_cruise_fps
                    and abs(a["v_air"]) <= 2.0 * cfg.tol_u_fps)
        if w["kind"] == "stop":
            tr = self.track
            if tr is None or self.steps * CONTROL_DT < tr["t_end"] - 1e-9:
                return False
            n_f, e_f, _, _ = self._track_ref(tr["t_end"] + 1.0)
            tol_h, tol_xy = w["tol"]
            return (s["wow"] == 0 and math.hypot(s["n"] - n_f, s["e"] - e_f) <= tol_xy and abs(e["h"]) <= tol_h
                    and abs(e["psi"]) <= cfg.tol_psi_deg and s["vh"] <= cfg.tol_v_fps and abs(s["vs"]) <= cfg.tol_vs_fps)
        return super()._inside(s, e, w)

    def _update_window(self, s: dict, e: dict, t: float):
        w = self.windows[-1]
        if w["kind"] != "cruise" or w["closed"]:
            return super()._update_window(s, e, t)
        cfg = self.cfg
        for k, lim in zip(("u", "psi", "h"), cfg.coupling_cruise):
            val = abs(e[k])
            w["max_err"][k] = max(w["max_err"][k], val)
            if k not in w["active"] and val > lim:
                w["coupling_ok"] = False
        if self._inside(s, e, w):
            if w["streak"] == 0.0:
                w["entry_t"] = t
            w["streak"] += CONTROL_DT
        else:
            w["streak"], w["entry_t"] = 0.0, None
        elapsed = t - w["t"]
        if w["entry_t"] is not None:
            w["settle_s"] = w["entry_t"] - w["t"]
            w["on_time"] = w["settle_s"] <= w["deadline"] + 1e-9
        else:
            w["settle_s"], w["on_time"] = float("nan"), False
        if not w["complete"]:
            if w["on_time"] and w["streak"] >= w["hold_s"] - 1e-9:
                w["complete"] = True
                self._close_window(w)
                self._schedule_next(t)
            elif elapsed > w["deadline"] and not w["on_time"]:
                w["complete"] = True
                self._close_window(w)
                self._schedule_next(w["t"] + w["deadline"] + w["hold_s"])

    def _schedule_next(self, t_next: float):
        if (not self.pending and not self.auto_end and self.windows and self.windows[-1]["kind"] == "cruise"
                and self._prev_wow == 0):
            # canlı: son ileri uçuş komutu bitince aynı hız / heading / irtifada devam (hover tut değil)
            self.pending.append(dict(kind="cruise", auto=True))
            self.next_issue_t = t_next + float(self.np_random.uniform(*self.cfg.gap_s))
            return
        super()._schedule_next(t_next)

    def _result(self, w: dict) -> dict:
        r = super()._result(w)
        if w.get("category"):
            r["category"] = "interrupted" if w["interrupted"] else w["category"]
        r["u_target_kt"] = w.get("u_target", 0.0) / KT
        fe = w.get("final_err") or {}
        r["final_err"]["speed"] = float(fe.get("u", 0.0))
        r["max_abs_err"] = dict(w["max_err"])
        return r

    # =================================================================
    # OBS
    # =================================================================

    def _obs(self, s: dict, e: dict) -> np.ndarray:
        base = super()._obs(s, e)                                # 30 (29 + tork)
        a = self._air()
        cr = self._in_cruise()
        vf, vr = self._target_vel_body(s)
        extra = np.array([
            a["u_air"] / 100.0, a["v_air"] / 30.0, a["u_gnd"] / 100.0, a["v_gnd"] / 30.0,
            1.0 if cr else 0.0, float(np.clip(e["u"] / 10.0, -3.0, 3.0)), e["u"] / 60.0,
            (self.cruise["u"] / 100.0) if self.cruise is not None else 0.0,
            vf / 100.0, vr / 100.0, (s["weight"] - 9000.0) / 1000.0,
        ], dtype=np.float64)
        extra = np.clip(np.nan_to_num(extra, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        return np.concatenate([base, extra.astype(np.float32)])

    # =================================================================
    # STEP
    # =================================================================

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
        t_now = self.steps * CONTROL_DT
        s_pre = self._state()
        self._issue_due_tasks(t_now, s_pre)
        self._update_trim(s_pre)                                 # hava hızı çizelgesi (filtre durumu aşağıda güncellenir)

        filt = self.filt + cfg.action_filter_alpha * (a - self.filt)
        target_c = np.clip(self.trim + self.rng_ctrl * expo(filt, cfg.expo), CTRL_LO, CTRL_HI)
        step_max = np.asarray(cfg.rate_limit) * CONTROL_DT
        new_c = self.ctrl + np.clip(target_c - self.ctrl, -step_max, step_max)
        self._dctrl = (new_c - self.ctrl) / (step_max + 1e-9)
        self.ctrl = new_c
        self.filt = np.clip(expo_inv(np.clip((self.ctrl - self.trim) / self.rng_ctrl, -1.0, 1.0), cfg.expo), -1.0, 1.0)
        out = self.ctrl.copy()
        if self.kick is not None:
            ax, mag, t_end = self.kick
            if t_now < t_end:
                out[ax] += mag
            else:
                self.kick = None
        self._write_controls(np.clip(out, CTRL_LO, CTRL_HI))

        ok = self._run_plain()
        self.steps += 1
        ext_events = self.ext.after_step(self.fdm)
        t_after = self.steps * CONTROL_DT
        s = self._state()
        air = self._air()
        a_m = 1.0 - math.exp(-CONTROL_DT / cfg.u_meas_tau_s)
        self.u_meas += a_m * (air["u_air"] - self.u_meas)
        self.u_dot += 0.3 * ((air["u_air"] - self._u_prev) / CONTROL_DT - self.u_dot)
        self._u_prev = air["u_air"]
        if self._in_cruise():                                    # ileri uçuşta konum hedefi helikopteri izler
            self.target["n"], self.target["e"] = s["n"], s["e"]
        self._update_track(t_after, s)
        finite = all(np.isfinite([s[k] for k in ("h", "vs", "u", "v", "phi", "theta", "p", "q", "r", "rpm", "n", "e")]))
        if finite:
            self.psi_unwrap += wrap_deg(s["psi_deg"] - self._psi_prev_meas)
            self._psi_prev_meas = s["psi_deg"]
            if s["wow"] == 0:
                self._airborne_once = True
        e = self._errors(s) if finite else dict(self._last_err)
        if finite:
            self._last_err = dict(e)

        reward, parts = self._reward(s, e, a) if finite else (0.0, {})
        self.failure = self._safety(s, e, ok, finite)
        terminated = self.failure is not None
        if terminated:
            reward -= cfg.fail_penalty
        reward *= cfg.reward_scale
        if self.windows and not terminated and finite:
            w_open = self.windows[-1] if not self.windows[-1]["closed"] else None
            self._update_window(s, e, t_after)
            if w_open is not None and w_open["closed"] and w_open["success"]:
                reward += cfg.w_task_success * cfg.reward_scale
        self._prev_wow, self._prev_vs = s["wow"], s["vs"]

        fuel_out = self.ext.engine_is_out and cfg.fuel_exhausted_ends
        truncated = (not terminated) and (self.steps >= self.max_steps or t_after >= self.episode_end_t - 1e-9
                                          or fuel_out)
        info = dict(level=self.levels[self._ep_level].name, level_index=self._ep_level, reward_parts=parts,
                    controls=[float(x) for x in out], failure=self.failure, **self._info_state(s, e))
        if self.ext.armed:
            info["physics"] = self.ext.info(self.fdm)
            if ext_events:
                info["physics_events"] = ext_events
        if self.windows:
            w = self.windows[-1]
            info.update(task_kind=w["kind"], task_category=w.get("category", w["kind"]), cmd_T=w["T"],
                        cmd_deadline=w["deadline"], cmd_elapsed=t_after - w["t"])
        if terminated or truncated:
            for w in self.windows:
                self._close_window(w)
            n_ok = sum(int(w["success"]) for w in self.windows)
            info["episode_success"] = bool(not terminated and self.windows and n_ok == len(self.windows)
                                           and not self.pending)
            info["commands_ok"] = n_ok
            info["commands_total"] = len(self.windows) + len(self.pending)
            info["command_results"] = [self._result(w) for w in self.windows]
            info["termination"] = self.failure or ("fuel_exhausted" if fuel_out else "time_limit")
            if self.ext.armed:
                info["physics_summary"] = self.ext.summary()
        self.prev_action = a
        obs = self._obs(s, e) if finite else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # =================================================================
    # REWARD / GÜVENLİK
    # =================================================================

    def _side_penalty(self) -> float:
        """Koordineli uçuş: yana hava hızı cezası · (v_air / 10)², hava hızı 15 → 30 kt arasında 0 → 1 ağırlıkla
        (hover'da yana hava hızı anlamlı değil: rüzgârda heading komutluyken yana hava hızı olur)."""
        air = self._air()
        x = float(np.clip((abs(air["u_air"]) - 15.0 * KT) / (15.0 * KT), 0.0, 1.0))
        return x * self.cfg.pen_side * (air["v_air"] / 10.0) ** 2

    def _reward(self, s: dict, e: dict, a: np.ndarray):
        if not self._in_cruise():
            r, parts = super()._reward(s, e, a)
            side = 0.0
            if self.windows and self.windows[-1]["kind"] == "stop":  # duruşun başında hâlâ hızlıyken koordineli
                side = self._side_penalty()
                r -= side
            parts["side"] = -side
            return float(r), parts
        cfg, lv = self.cfg, self.ep_level
        w = self.windows[-1]
        K = self._kernel2
        track = (cfg.w_u * K(e["u"], cfg.kernel_u) + cfg.w_h * K(e["h"], cfg.kernel_h_cruise)
                 + cfg.w_psi * K(e["psi"], cfg.kernel_psi_cruise))
        vs_des = float(np.clip(cfg.guide_k_h * e["h"], -1.25 * lv.cruise_descent_fps, 1.25 * lv.cruise_climb_fps))
        r_max = 1.25 * cruise_yaw_rate_dps(lv, max(abs(self.u_meas), abs(self.cruise["u"])))
        r_des = float(np.clip(cfg.guide_k_psi * e["psi"], -r_max, r_max))
        a_des = float(np.clip(cfg.guide_k_u * e["u"], -1.25 * lv.cruise_decel_fps2, 1.25 * lv.cruise_accel_fps2))
        guide = (cfg.w_vs * K(s["vs"] - vs_des, cfg.kernel_vs) + cfg.w_r * K(math.degrees(s["r"]) - r_des, cfg.kernel_r)
                 + cfg.w_accel * K(self.u_dot - a_des, cfg.kernel_accel))
        progress = 0.0
        for k, mn in (("u", cfg.progress_min_scale_u), ("h", cfg.progress_min_scale[1]),
                      ("psi", cfg.progress_min_scale[2])):
            cur = abs(e[k])
            scale = max(mn, w["e0"].get(k, 0.0))
            progress += (self.prev_abs_err.get(k, cur) - cur) / scale
            self.prev_abs_err[k] = cur
        progress *= cfg.progress_weight
        lag, tau, T = self._schedule_lag(e)
        sched = cfg.pen_sched * float(np.sum(np.minimum(1.0, np.abs(lag)) ** 2))
        late = 0.0
        if not w["closed"] and self.steps * CONTROL_DT - w["t"] > w["deadline"] and not self._inside(s, e, w):
            late = cfg.pen_late
        phi, th = abs(math.degrees(s["phi"])), abs(math.degrees(s["theta"]))
        att = cfg.pen_att * ((max(0.0, phi - cfg.cruise_att_roll_deg) / 10.0) ** 2
                             + (max(0.0, th - cfg.cruise_att_pitch_deg) / 10.0) ** 2)
        rate = cfg.pen_rate * ((s["p"] / 0.5) ** 2 + (s["q"] / 0.5) ** 2 + (s["r"] / 1.0) ** 2)
        smooth = (cfg.pen_dctrl * float(np.mean(self._dctrl ** 2)) * 0.1
                  + cfg.pen_sat * float(np.mean(np.maximum(np.abs(a) - cfg.sat_threshold, 0.0))))
        couple = 0.0
        if not w["closed"]:
            for k, lim in zip(("u", "psi", "h"), cfg.coupling_cruise):
                if k in w["active"]:
                    continue
                x = min(1.0, max(0.0, abs(e[k]) - cfg.couple_soft_frac * lim) / lim)
                couple += cfg.pen_couple * x * x
        side = self._side_penalty()
        sink = 0.0
        if 0.0 < s["hs"] < 60.0 and s["wow"] == 0:
            allow = cfg.sink_base_fps + cfg.sink_slope * s["hs"]
            sink = cfg.pen_sink * (max(0.0, -s["vs"] - allow) / 3.0) ** 2
        torque = torque_penalty(s["torque_psi"], s["rpm"], cfg.pen_torque_cont, cfg.pen_torque_over, cfg.pen_rpm_low,
                                cfg.torque_cont_psi, cfg.torque_max_psi, cfg.rpm_low_warn)
        r = track + guide + progress - sched - late - att - rate - smooth - couple - side - sink - torque
        return float(r), dict(track=track, guide=guide, progress=progress, schedule=sched, late=late, attitude=att,
                              rate=rate, smooth=smooth, coupling=couple, side=-side, sink=-sink, torque=-torque)

    def _flight_window(self) -> bool:
        """İleri uçuş / duruş penceresi (güvenlik sınırları ileri uçuşunkiler)."""
        return bool(self.windows) and self.windows[-1]["kind"] in ("cruise", "stop")

    def _safety(self, s: dict, e: dict, ok: bool, finite: bool) -> str | None:
        """Hover görevlerinde kalkış env'inin sınırları (yer hızı ≤ 60 ft/s dahil — rüzgârda hover'da hava hızı büyük
        olabilir, yer hızı değil). İleri uçuş / duruş pencerelerinde: hava hızı ≤ 130 kt, yatış ≤ 60° (hava hızı > 30 kt),
        hızlıyken alçak uçuş ve yerle temas, hareketli / izleyen hedeften uzaklık."""
        cfg = self.cfg
        if not self._flight_window():
            return super()._safety(s, e, ok, finite)
        if not ok:
            return "jsbsim_stopped"
        if not finite:
            return "non_finite_state"
        air = self._air()
        fast = abs(air["u_air"]) > 30.0 * KT
        if s["tail"]:
            return "tail_strike"
        if s["wow"] > 0:
            # yerle temas: yer hızıyla (rüzgâra karşı yavaş temas kaza değil) — hızlıysa kaza, değilse hover kuralları
            return "ground_contact_at_speed" if s["vh"] > 15.0 else super()._safety(s, e, ok, finite)
        phi, th = abs(math.degrees(s["phi"])), abs(math.degrees(s["theta"]))
        if phi > (cfg.max_roll_cruise_deg if fast else cfg.max_roll_deg):
            return "roll_limit"
        if th > cfg.max_pitch_deg:
            return "pitch_limit"
        if max(abs(math.degrees(s["p"])), abs(math.degrees(s["q"]))) > cfg.max_rate_dps:
            return "rate_limit"
        if abs(math.degrees(s["r"])) > cfg.max_yaw_rate_dps:
            return "yaw_rate_limit"
        if not cfg.rpm_limits[0] <= s["rpm"] <= cfg.rpm_limits[1]:
            return "rotor_rpm"
        if abs(air["u_air"]) > cfg.max_airspeed_kt * KT or s["vh"] > cfg.max_airspeed_kt * KT + 60.0:
            return "speed_limit"
        if s["vh"] > 40.0 * KT and s["hs"] < cfg.min_hs_at_speed_ft:
            return "low_altitude"
        if e["xy"] > cfg.flyaway_track_ft:
            return "position_deviation"
        top = max(self.target["h"], self.windows[-1]["h_start"] if self.windows else self.target["h"])
        if s["h"] > top + cfg.alt_over_ft:
            return "altitude_deviation"
        return None

    def _info_state(self, s: dict, e: dict) -> dict:
        d = super()._info_state(s, e)
        a = self._air()
        d.update(airspeed_kt=a["u_air"] / KT, lateral_airspeed=a["v_air"], ground_speed_fwd=a["u_gnd"],
                 ground_speed_lat=a["v_gnd"], u_meas_kt=self.u_meas / KT, cruise=self._in_cruise(),
                 u_target_kt=(self.cruise["u"] / KT if self.cruise is not None else float("nan")),
                 err_speed=e.get("u", 0.0), trim=[float(x) for x in self.trim], env_stage=self.env_stage,
                 fuel_lbs=fuel_total(self.fdm), wind_n=self.ext.wind_ned[0], wind_e=self.ext.wind_ned[1])
        return d


# =====================================================================
# HIZLI TEST:  python helicopter_env_flight.py
# =====================================================================

if __name__ == "__main__":
    import time
    for lvl in ("F1", "F3", "F5", "F6", "F9"):
        env = HelicopterEnvFlight(level=lvl)
        t0 = time.time()
        obs, info = env.reset(seed=1)
        print(f"{lvl} reset {time.time() - t0:.2f}s start={info['setup']['start']} h={info['altitude']:.1f} "
              f"v={info['airspeed_kt']:.1f} kt W={info['weight_lbs']:.0f} çevre={info['env_stage']} "
              f"obs={obs.shape} görevler={[w.get('category', w['kind']) for w in env.windows]} + "
              f"{[d['kind'] for d in env.pending]}")
        t0 = time.time()
        done, ret, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, info = env.step(np.zeros(4))
            ret += r
            n += 1
            done = term or trunc
        print(f"   a=0: {n} adım ({n / (time.time() - t0):.0f} adım/s) return={ret:.1f} bitiş={info['termination']} "
              f"h={info['altitude']:.1f} v={info['airspeed_kt']:.1f} kt yakıt={info['fuel_lbs']:.0f} "
              f"sonuç={[(c['category'], c['success']) for c in info['command_results']]}")
