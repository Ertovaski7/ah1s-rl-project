"""
Kural tabanlı pilot (PID) — `helicopter_env_flight.py` görevlerinin yapılabilirlik / ödül tutarlılığı kontrolü için.
RL DEĞİL: eğitimde kullanılmaz, policy'ye action önermez (öğretmen–öğrenci yok). Yalnızca şunu sınar: görevler, başarı
bantları ve süre hedefleri bu fizikte (güç tavanı, rüzgâr / türbülans, yakıt) kural tabanlı bir pilotla yapılabiliyor mu
ve ödül görevi yapanı yapmayanın üstünde tutuyor mu.

Yapı (probe tutucusunun ve reset PID'lerinin genelleştirilmişi): kademeli PID, kazançlar hava hızına göre hover ↔ ileri
uçuş arasında harmanlanır; ileri besleme = env'in kendi trim çizelgesi (env.trim) + tablonun trim attitude'u.
  hover görevleri (hold / takeoff / turn / move / bob / land / recover) ve stop: hedef konum (+ hareketli hedefin hızı)
      → istenen yer hızı → attitude → cyclic; irtifa → dikey hız → collective; heading → dönüş hızı → pedal.
  cruise: hava hızı → pitch attitude (ivme sınırlı); irtifa → dikey hız → collective; heading → koordineli dönüş
      (yatış = atan(r·V/g)) → lateral cyclic; yana hava hızı → pedal.
  tork sınırlayıcı: gösterge 53 psi üstünde collective'i kıs (pilot gibi).
Action'a çeviri: kumanda → (c − trim) / aralık → expo⁻¹ → action filtresinin tersi (a = 2·hedef − filtre).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from helicopter_env_command import CONTROL_DT  # noqa: E402
from helicopter_env_takeoff import CTRL_HI, CTRL_LO, GROUND_H_FT, expo_inv  # noqa: E402
from takeoff_curriculum import LAND_PROFILE_K, LAND_PROFILE_MIN_FPS  # noqa: E402
from flight_curriculum import cruise_yaw_rate_dps  # noqa: E402

KT = 1.6878099
G = 32.174


class ScriptedPilot:
    def __init__(self, env, torque_soft_psi: float = 53.0):
        self.env = env
        self.torque_soft = torque_soft_psi
        self.reset()

    def reset(self):
        self.I = np.zeros(4)
        self._coll_ground = None

    # ------------------------------------------------------------------------------------------------------------
    def controls(self) -> np.ndarray:
        env, cfg = self.env, self.env.cfg
        lv = env.ep_level
        s = env._state()
        air = env._air()
        w = env.windows[-1] if env.windows else None
        kind = w["kind"] if w is not None else "hold"
        dt = CONTROL_DT
        ff = np.asarray(env.trim, dtype=np.float64).copy()
        u_kt = air["u_air"] / KT
        th_tab, ph_tab = np.radians(env.trim_tab.attitude(u_kt, s["weight"]))
        x = float(np.clip((abs(air["u_air"]) - 20.0 * KT) / (20.0 * KT), 0.0, 1.0))
        e = env._errors(s)
        cruise = env._in_cruise()

        # ---------- dikey ----------
        if kind == "land" and w is not None and not w["closed"]:
            vs_des = -float(np.clip(LAND_PROFILE_K * max(0.0, s["hs"]), LAND_PROFILE_MIN_FPS, lv.descent_fps))
        elif cruise:
            vs_des = float(np.clip(0.3 * e["h"], -lv.cruise_descent_fps, lv.cruise_climb_fps))
        else:
            up = env.climb_fps(lv) if kind in ("takeoff", "climb_to", "bob") else 3.0
            dn = lv.descent_fps if kind in ("climb_to", "bob") else 3.0
            vs_des = float(np.clip(0.3 * e["h"], -dn, up))
        evs = vs_des - s["vs"]
        self.I[0] = np.clip(self.I[0] + 0.01 * evs * dt, -0.3, 0.3)
        coll = ff[0] + 0.03 * evs + self.I[0] - 0.02 * max(0.0, s["torque_psi"] - self.torque_soft)

        # ---------- boylamsal / yanal / yön ----------
        if cruise:
            u_t = env.cruise["u"]
            eu = u_t - air["u_air"]
            k_u = (1 - x) * 0.004 + x * 0.002
            self.I[1] = np.clip(self.I[1] - k_u * eu * dt, -0.15, 0.15)
            lim = 0.12 if x > 0.5 else 0.2
            th_ref = th_tab + float(np.clip(-((1 - x) * 0.02 + x * 0.012) * eu, -lim, 0.10)) + self.I[1]
            # heading: koordineli dönüş (hızlı) ↔ pedal (yavaş)
            V = max(abs(air["u_air"]), 1.0)
            r_max = cruise_yaw_rate_dps(lv, max(V, abs(u_t)))
            r_des = float(np.clip(0.6 * e["psi"], -r_max, r_max))            # °/s
            ph_turn = math.atan(math.radians(r_des) * V / G)
            ev_g = -air["v_gnd"]                                             # yavaşken yer izini tut
            self.I[2] = np.clip(self.I[2] + (1 - x) * 0.004 * ev_g * dt, -0.15, 0.15)
            ph_ref = ph_tab + x * ph_turn + (1 - x) * float(np.clip(0.03 * ev_g, -0.2, 0.2)) + self.I[2]
            self.I[3] = np.clip(self.I[3] - ((1 - x) * 0.003 * e["psi"] + x * 0.002 * air["v_air"]) * dt, -0.3, 0.3)
            ped_hover = 0.014 * (math.degrees(s["r"]) - float(np.clip(1.0 * e["psi"], -8.0, 8.0)))
            ped_fwd = -0.01 * air["v_air"] + 0.8 * (s["r"] - math.radians(r_des))
            ped = ff[3] + (1 - x) * ped_hover + x * ped_fwd + self.I[3]
        else:
            vf, vr = env._target_vel_body(s)
            v_lim = lv.move_fps if kind == "move" else 6.0
            u_des = vf + float(np.clip(0.12 * e["fwd"], -v_lim, v_lim))
            v_des = vr + float(np.clip(0.12 * e["right"], -v_lim, v_lim))
            if kind == "land" and s["wow"] > 0:
                u_des = v_des = 0.0
            eu, ev = u_des - s["ug"], v_des - s["vg"]
            k_u = (1 - x) * 0.004 + x * 0.002
            self.I[1] = np.clip(self.I[1] - k_u * eu * dt, -0.15, 0.15)
            self.I[2] = np.clip(self.I[2] + k_u * ev * dt, -0.15, 0.15)
            th_ref = th_tab + float(np.clip(-((1 - x) * 0.02 + x * 0.012) * eu, -0.2, 0.15)) + self.I[1]
            ph_ref = ph_tab + float(np.clip(((1 - x) * 0.03 + x * 0.02) * ev, -0.25, 0.25)) + self.I[2]
            r_max = lv.yaw_rate_dps if kind == "turn" else 8.0
            r_des = float(np.clip(1.0 * e["psi"], -r_max, r_max))
            self.I[3] = np.clip(self.I[3] - 0.003 * e["psi"] * dt, -0.3, 0.3)
            ped = ff[3] + 0.014 * (math.degrees(s["r"]) - r_des) + self.I[3]
        kth, kq = (1 - x) * 3.0 + x * 1.2, (1 - x) * 1.6 + x * 0.6
        lon = ff[1] + kth * (s["theta"] - th_ref) + kq * s["q"]
        lat = ff[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"]

        # ---------- yerde (iniş sonrası / kalkış öncesi) ----------
        if kind == "land" and s["wow"] > 0:
            if self._coll_ground is None:
                self._coll_ground = float(env.ctrl[0])
            self._coll_ground = max(cfg.coll_flat, self._coll_ground - 0.3 * dt)
            coll = self._coll_ground
            self.I[0] = 0.0
        else:
            self._coll_ground = None
        return np.clip(np.array([coll, lon, lat, ped]), CTRL_LO, CTRL_HI)

    def __call__(self, obs=None) -> np.ndarray:
        env = self.env
        c = self.controls()
        tgt = np.clip(expo_inv(np.clip((c - env.trim) / env.rng_ctrl, -1.0, 1.0), env.cfg.expo), -1.0, 1.0)
        a = (tgt - (1.0 - env.cfg.action_filter_alpha) * env.filt) / env.cfg.action_filter_alpha
        return np.clip(a, -1.0, 1.0).astype(np.float32)
