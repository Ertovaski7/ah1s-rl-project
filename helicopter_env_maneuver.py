from __future__ import annotations

"""
HELICOPTER ENV MANEUVER — Δ komut + süre hedefi, 0–100 kt, attitude command
============================================================================

Komut env'inin (helicopter_env_command.py) manevra sürümü. Arayüz aynı
(Δheading / Δhız / Δirtifa); her komutun bir süre hedefi T'si var
(maneuver_curriculum.time_target). Hedef bandına T·(1+pay) içinde girip
`success_hold_s` kalmak başarı sayılır.

Komut env'inden farklar ve sebepleri (ölçüm: 2026-09-23 probları)
-----------------------------------------------------------------
* Eski env'de ajanın kumanda yetkisi ±5° komutlar için dardı (pedal ±0.20 →
  en fazla ~8°/s dönüş, elevator ±0.05 → ~2° yunuslama) ve AFCS yatış /
  yunuslamayı trim açısında tutuyordu. Sonuç: dönüş ve tırmanışta açılar
  neredeyse değişmiyordu.
* Burada ajan **attitude komutu** verir (ACAH — attitude command / attitude
  hold, ADS-33'teki cevap tipi): action[2] → yatış komutu (trim ± 60°),
  action[1] → yunuslama komutu (trim ∓ 30°; + = burun aşağı, eski elevator
  işaretiyle aynı). Komutu her JSBSim adımında (133 Hz) çalışan PI-D iç döngü
  cyclic'e çevirir (uçuş kontrol sisteminin parçası; action önermez, teacher
  değildir). Ölçülen: 45° yatışa ~2.4 s, 20° yunuslamaya ~2.5 s.
  action[0] collective (trim ± 0.25, ~±30 ft/s), action[3] pedal (trim ± 0.7,
  hover'da ~25°/s). JSBSim AFCS yalnızca sönümleme (SAS) yapar.
* Trim hıza göre çizelgelenir: JSBSim AH-1S `steady_flight_data.xml`
  tabloları (0–140 kt) + reset'te ölçülen sabit düzeltme. 0–100 kt'ın her
  hızında a = 0 ≈ düz uçuş.
* Reset: IC ile istenen hız / irtifa / heading'e "teleport" + reset'e özel
  kademeli oto-pilot (hız → pitch, yanal hız → roll, irtifa → collective,
  heading → pedal; SFD ileri beslemeli). Yanal hızı da sıfırlar (eski
  başlangıçta ~2.5 ft/s yana kayma vardı). 0–100 kt'ta 16–70 s sim'de oturur.

Observation (24): hatalar (6, eski env gibi) + takvim gecikmesi (3; hata,
"T'de sıfıra inen doğrusal takvimden" ne kadar geride) + zaman (τ = geçen /
T, T) + uçuş durumu (dikey hız, u, v, φ, θ, p, q, r, rpm) + filtrelenmiş
action (4). Mutlak irtifa / heading yok (eski env gibi).

Reward: takip çekirdekleri + ilerleme (eski env) − takvim gecikmesi cezası −
süre aşımı cezası − kuplaj (komut verilmeyen eksen) − yana kayma (v) − aşırı
açı (60° / 35° üstü) − kumanda hızı. Başarı: son giriş (oturma) süresi ≤
deadline ve bantta `success_hold_s` + kuplaj sınırı + güvenlik ihlali yok.
"""

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from command_curriculum import AXES
from helicopter_env_command import (CONTROL_DT, JSBSIM_DT, PHYSICS_STEPS, AfcsMode, CommandEnvConfig,
                                    HelicopterEnvCommand, wrap_deg)
from maneuver_curriculum import DEFAULT_MANEUVER_LEVELS, ManeuverLevel, find_maneuver_level

OBS_DIM_M = 24
SFD_KEYS = ("collective", "pitch", "roll", "yaw")        # fcs/automatic/<k>-trim-cmd-norm


@dataclass
class ManeuverEnvConfig(CommandEnvConfig):
    # --- attitude command (ACAH) iç döngüsü ------------------------------------
    roll_cmd_deg: float = 60.0
    pitch_cmd_deg: float = 30.0
    kphi: float = 4.0
    kp: float = 0.9
    kiphi: float = 2.0
    kth: float = 3.0
    kq: float = 1.6
    kith: float = 1.5
    i_lim: float = 0.3
    coll_scale: float = 0.25
    pedal_scale: float = 0.7
    coll_limits: tuple = (0.20, 1.0)
    action_filter_alpha: float = 0.5
    # --- reset ------------------------------------------------------------------
    settle_max_s: float = 110.0
    settle_min_s: float = 12.0
    settle_hold_s: float = 4.0
    cmd_min_alt_ft: float = 250.0          # hedef irtifa bunun altına inerse Δh'nin işareti çevrilir
    # --- başarı -------------------------------------------------------------------
    tol_heading_deg: float = 2.0
    tol_speed_fps: float = 2.0
    tol_alt_ft: float = 10.0
    success_hold_s: float = 5.0
    deadline_grace_frac: float = 0.25      # deadline = max(T·(1+pay), T + pay_min)
    deadline_grace_min_s: float = 1.0
    coupling_limits: tuple | None = (8.0, 8.0, 40.0)
    window_margin_s: float = 3.0
    first_command_s: tuple = (2.0, 4.0)
    max_episode_s: float = 240.0
    # --- reward -----------------------------------------------------------------
    kernel_heading: tuple = (2.0, 15.0)
    kernel_speed: tuple = (1.5, 8.0)
    kernel_alt: tuple = (6.0, 40.0)
    pen_sched: float = 1.0
    sched_scale: tuple = (20.0, 10.0, 30.0)   # heading deg, speed ft/s, alt ft
    pen_late: float = 0.5
    pen_side: float = 0.3                     # · (v / 10 ft/s)²
    pen_att: float = 1.0                      # · ((|φ|−60°)/15°)² + ((|θ|−35°)/10°)²
    pen_roll: float = 0.0
    pen_rate: float = 0.0
    pen_jerk: float = 0.0
    pen_ctrl: float = 0.0
    pen_dctrl: float = 1.0
    pen_sat: float = 0.5
    sat_threshold: float = 0.9
    # --- güvenlik -------------------------------------------------------------------
    max_roll_deg: float = 75.0
    max_pitch_deg: float = 45.0
    max_yaw_rate_dps: float = 90.0
    max_roll_rate_dps: float = 150.0
    min_agl_ft: float = 80.0
    alt_margin_ft: float = 150.0
    speed_margin_fps: float = 30.0
    heading_margin_deg: float = 45.0
    speed_limits_fps: tuple = (-40.0, 210.0)


class HelicopterEnvManeuver(HelicopterEnvCommand):
    metadata = {"render_modes": []}

    def __init__(self, level=0, levels=None, config: ManeuverEnvConfig | None = None):
        levels = list(levels or DEFAULT_MANEUVER_LEVELS)
        super().__init__(level=0, levels=levels, config=config or ManeuverEnvConfig())
        self.level_index = find_maneuver_level(level, self.levels)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM_M,), dtype=np.float32)
        self.resid = np.zeros(4)
        self.att_resid = np.zeros(2)
        self.att_cmd = np.zeros(2)                # (φ_cmd, θ_cmd) rad
        self.i_att = np.zeros(2)
        self.pending: list[dict] = []
        self.next_issue_t = 0.0
        self.fixed_schedule = False
        self.episode_end_t = np.inf
        self._sfd_cache = np.zeros(4)

    def set_level(self, level) -> int:
        self.level_index = find_maneuver_level(level, self.levels)
        return self.level_index

    @property
    def level(self) -> ManeuverLevel:
        return self.levels[self.level_index]

    # =================================================================
    # JSBSim yardımcıları
    # =================================================================

    def _sfd(self) -> np.ndarray:
        f = self.fdm
        return np.array([float(f[f"fcs/automatic/{k}-trim-cmd-norm"]) for k in SFD_KEYS])

    def _sfd_att(self) -> np.ndarray:
        f = self.fdm
        return np.array([float(f["ap/afcs/automatic/phi-trim-rad"]), float(f["ap/afcs/automatic/theta-trim-rad"])])

    def _set_sas(self, psi_deg: float):
        """AFCS: roll/pitch yalnızca oran sönümleme, yaw sönümleme; heading-hold kapalı."""
        f = self.fdm
        self._set_afcs(AfcsMode(1.0, 1.0, 0.99, False, 0.2), psi_deg)
        f["ap/afcs/adj/roll-err-ctrl-gain"] = 0.0
        f["ap/afcs/adj/pitch-err-ctrl-gain"] = 0.0
        f["fcs/automatic/steady-flight-data-enable"] = 0.0      # trim ileri beslemesini kendimiz ekliyoruz

    def _teleport_m(self, h_ft: float, u_fps: float, psi_deg: float):
        f = self.fdm
        f["fcs/automatic/steady-flight-data-enable"] = 0.0
        for k, v in (("phi-rad", 0.0), ("theta-rad", 0.0), ("psi-true-rad", math.radians(psi_deg % 360.0)),
                     ("h-agl-ft", float(h_ft)), ("u-fps", float(u_fps)), ("v-fps", 0.0), ("w-fps", 0.0),
                     ("p-rad_sec", 0.0), ("q-rad_sec", 0.0), ("r-rad_sec", 0.0)):
            f[f"ic/{k}"] = v
        f["fcs/rpm-governor-active-norm"] = 1.0
        ff = self._sfd()
        self._write_controls(np.array([np.clip(ff[0], *self.cfg.coll_limits), *np.clip(ff[1:], -1, 1)]))
        if not f.run_ic():
            raise RuntimeError("run_ic() başarısız (teleport)")
        s = self._read_state()
        if s["rpm"] < 300.0:
            raise RuntimeError(f"teleport sonrası rotor devri düştü: {s['rpm']:.1f} rpm")
        if abs(s["h"] - h_ft) > 20.0:
            raise RuntimeError(f"teleport irtifası tutmadı: {s['h']:.1f} ft")

    def _settle_m(self, h0: float, u0: float, psi0: float) -> tuple[bool, dict]:
        """Yalnızca reset: SFD ileri beslemeli kademeli oto-pilot (teacher değil)."""
        cfg = self.cfg
        self._set_sas(psi0)
        integ = np.zeros(4)
        t, hold = 0.0, 0.0
        c = np.zeros(4)
        while t < cfg.settle_max_s:
            s = self._read_state()
            ff = self._sfd()
            eu, ev, eh = u0 - s["u"], -s["v"], h0 - s["h"]
            eps = wrap_deg(psi0 - s["psi_deg"])
            integ[0] = np.clip(integ[0] + 0.004 * eh * CONTROL_DT, -0.15, 0.15)
            integ[1] = np.clip(integ[1] - 0.002 * eu * CONTROL_DT, -0.3, 0.3)
            integ[2] = np.clip(integ[2] + 0.002 * ev * CONTROL_DT, -0.3, 0.3)
            integ[3] = np.clip(integ[3] - 0.003 * eps * CONTROL_DT, -0.3, 0.3)
            th_ref = np.clip(-0.012 * eu, -0.25, 0.25) + integ[1]
            ph_ref = np.clip(0.02 * ev, -0.25, 0.25) + integ[2]
            c = np.array([
                np.clip(ff[0] + 0.004 * eh - 0.03 * s["vs"] + integ[0], *cfg.coll_limits),
                np.clip(ff[1] + 1.2 * (s["theta"] - th_ref) + 0.6 * s["q"], -1.0, 1.0),
                np.clip(ff[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"], -1.0, 1.0),
                np.clip(ff[3] - 0.02 * eps + 0.8 * s["r"] + integ[3], -1.0, 1.0)])
            self._write_controls(c)
            if not self._run_physics_plain():
                return False, dict(reason="jsbsim_stopped")
            t += CONTROL_DT
            ok = (abs(eh) < 3.0 and abs(s["vs"]) < 0.5 and abs(eu) < 1.0 and abs(s["v"]) < 1.0 and abs(eps) < 1.0
                  and abs(s["p"]) < 0.02 and abs(s["q"]) < 0.02 and abs(s["r"]) < 0.02)
            hold = hold + CONTROL_DT if ok else 0.0
            if hold >= cfg.settle_hold_s and t >= cfg.settle_min_s:
                s = self._read_state()
                return True, dict(trim=c.copy(), sfd=self._sfd(), att=(s["phi"], s["theta"]),
                                  sfd_att=self._sfd_att(), settle_s=t)
        s = self._read_state()
        return False, dict(reason=f"not settled in {cfg.settle_max_s:.0f}s (h={s['h']:.1f}, u={s['u']:.1f}, "
                                  f"v={s['v']:.2f}, psi_err={wrap_deg(psi0 - s['psi_deg']):+.1f})")

    def _run_physics_plain(self) -> bool:
        for _ in range(PHYSICS_STEPS):
            if not self.fdm.run():
                return False
        return True

    def _run_physics(self) -> bool:
        """PHYSICS_STEPS JSBSim adımı; her adımda ACAH iç döngüsü cyclic'i günceller."""
        cfg, f = self.cfg, self.fdm
        ff = self._sfd_cache
        phi_c, th_c = self.att_cmd
        base_ail, base_elev = ff[2] + self.resid[2], ff[1] + self.resid[1]
        for _ in range(PHYSICS_STEPS):
            ph, th = float(f["attitude/phi-rad"]), float(f["attitude/theta-rad"])
            p, q = float(f["velocities/p-rad_sec"]), float(f["velocities/q-rad_sec"])
            self.i_att[0] = np.clip(self.i_att[0] + cfg.kiphi * (phi_c - ph) * JSBSIM_DT, -cfg.i_lim, cfg.i_lim)
            self.i_att[1] = np.clip(self.i_att[1] + cfg.kith * (th - th_c) * JSBSIM_DT, -cfg.i_lim, cfg.i_lim)
            ail = base_ail + cfg.kphi * (phi_c - ph) - cfg.kp * p + self.i_att[0]
            elev = base_elev + cfg.kth * (th - th_c) + cfg.kq * q + self.i_att[1]
            f["fcs/aileron-cmd-norm"] = float(min(1.0, max(-1.0, ail)))
            f["fcs/elevator-cmd-norm"] = float(min(1.0, max(-1.0, elev)))
            if not f.run():
                return False
        return True

    # =================================================================
    # RESET
    # =================================================================

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if "level" in options:
            self.set_level(options["level"])
        lv, rng, cfg = self.level, self.np_random, self.cfg
        h0 = float(options.get("start_alt_ft", rng.uniform(*lv.start_alt_ft)))
        u0 = float(options.get("start_speed_fps", rng.uniform(*lv.start_speed_fps)))
        psi0 = float(options["start_heading_deg"]) % 360.0 if "start_heading_deg" in options else float(rng.uniform(0, 360))
        errors, info_setup = [], None
        for attempt in range(3):
            fresh = (attempt > 0 or self.fdm is None or self._fdm_needs_refresh or not cfg.reuse_fdm
                     or self._episodes_on_fdm >= cfg.fdm_refresh_episodes)
            try:
                if fresh:
                    self._create_fdm()
                    self._warmup_rotor()
                    self._episodes_on_fdm = 0
                self._teleport_m(h0, u0, psi0)
                ok, info_setup = self._settle_m(h0, u0, psi0)
                if ok:
                    info_setup.update(fresh_fdm=fresh, attempt=attempt)
                    break
                errors.append(info_setup["reason"])
            except Exception as exc:                          # noqa: BLE001
                errors.append(str(exc))
            self._fdm_needs_refresh = True
        else:
            raise RuntimeError("başlangıç koşulu kurulamadı: " + " | ".join(errors))
        self._fdm_needs_refresh = False
        self._episodes_on_fdm += 1
        self.setup_info = dict(info_setup, errors=errors, h0=h0, u0=u0, psi0=psi0, mode="teleport")
        return self._handover_m(lv, options)

    def _handover_m(self, lv: ManeuverLevel, options: dict):
        cfg, rng = self.cfg, self.np_random
        si = self.setup_info
        self.trim = np.asarray(si["trim"], dtype=np.float64)
        self.resid = self.trim - np.asarray(si["sfd"], dtype=np.float64)
        self.att_resid = np.asarray(si["att"], dtype=np.float64) - np.asarray(si["sfd_att"], dtype=np.float64)
        self.i_att = np.zeros(2)
        self.filt = np.zeros(4)
        self._dfilt = np.zeros(4)
        self.prev_action = np.zeros(4)
        s = self._read_state()
        self._set_sas(s["psi_deg"])
        self._sfd_cache = self._sfd()
        self.att_cmd = self._sfd_att() + self.att_resid
        self._write_controls(np.array([np.clip(self.trim[0], *cfg.coll_limits), *np.clip(self.trim[1:], -1, 1)]))

        self.psi_unwrap = s["psi_deg"]
        self._psi_prev_meas = s["psi_deg"]
        self.ref = {"heading": self.psi_unwrap, "speed": s["u"], "altitude": s["h"]}
        self.cmd_start = dict(self.ref)
        self.cmd_err0 = {a: 0.0 for a in AXES}
        self.last_delta = {a: 0.0 for a in AXES}
        self.steps = 0
        self.windows, self.failure = [], None
        # komutlar: sabit takvim (değerlendirme / canlı) ya da seviyeden dinamik takvim
        if "commands" in options:
            self.fixed_schedule = True
            self.pending = []
            for item in options["commands"]:
                t_c, c = float(item[0]), item[1]
                d = {a: float(c.get(a, 0.0)) for a in AXES}
                if len(item) > 2 and item[2]:
                    d["T"] = float(item[2])
                elif c.get("T"):
                    d["T"] = float(c["T"])
                d["_t"] = t_c
                self.pending.append(d)
            self.pending.sort(key=lambda d: d["_t"])
            self.next_issue_t = self.pending[0]["_t"] if self.pending else np.inf
            episode_s = float(options.get("episode_s", cfg.max_episode_s))
            self.max_steps = int(round(episode_s / CONTROL_DT))
            self.episode_end_t = np.inf
        else:
            self.fixed_schedule = False
            self.pending = [lv.sample_command(rng) for _ in range(lv.n_commands)]
            self.next_issue_t = float(rng.uniform(*cfg.first_command_s))
            self.max_steps = int(round(cfg.max_episode_s / CONTROL_DT))
            self.episode_end_t = np.inf
        self.schedule = []                             # uyumluluk (komut env'i API'si)
        self.next_cmd = 0
        errs = self._errors(s)
        self.prev_abs_err = {a: abs(errs[a]) for a in AXES}
        self._last_err = dict(errs)
        obs = self._obs(s, errs)
        info = dict(level=lv.name, level_index=self.level_index, setup={k: v for k, v in self.setup_info.items()
                                                                          if k not in ("trim", "sfd", "sfd_att")},
                    **self._info_state(s, errs))
        return obs, info

    # =================================================================
    # KOMUTLAR
    # =================================================================

    def queue_command(self, delta: dict, T: float | None = None) -> float:
        """Canlı kullanım: Δ komutu (+ isteğe bağlı süre hedefi T) bir sonraki adımda uygula."""
        t_cmd = self.steps * CONTROL_DT
        d = {a: float(delta.get(a, 0.0)) for a in AXES}
        if T:
            d["T"] = float(T)
        d["_t"] = t_cmd
        self.fixed_schedule = True
        self.pending.append(d)
        self.next_issue_t = min(self.next_issue_t, t_cmd) if self.pending[:-1] else t_cmd
        return t_cmd

    def _issue_due_commands(self, t: float, s: dict):
        cfg, lv = self.cfg, self.level
        while self.pending and t >= self.next_issue_t - 1e-9:
            d = self.pending.pop(0)
            if self.windows:
                self._close_window(self.windows[-1])
            cmd = {a: float(d.get(a, 0.0)) for a in AXES}
            meas = {"heading": self.psi_unwrap, "speed": s["u"], "altitude": s["h"]}
            if cmd["speed"]:
                lo, hi = lv.speed_bounds_fps
                if not lo <= meas["speed"] + cmd["speed"] <= hi:
                    cmd["speed"] = -cmd["speed"]
                cmd["speed"] = float(np.clip(meas["speed"] + cmd["speed"], lo, hi) - meas["speed"])
            if cmd["altitude"] and meas["altitude"] + cmd["altitude"] < cfg.cmd_min_alt_ft:
                cmd["altitude"] = -cmd["altitude"]
            for a in AXES:
                if cmd[a] != 0.0:
                    self.ref[a] = meas[a] + cmd[a]
                    self.cmd_start[a] = meas[a]
                    self.last_delta[a] = cmd[a]
            T = float(d["T"]) if d.get("T") else lv.time_target(cmd, s["u"])
            deadline = max(T * (1.0 + cfg.deadline_grace_frac), T + cfg.deadline_grace_min_s)
            errs = self._errors(s)
            for a in AXES:
                self.cmd_err0[a] = errs[a]
                self.prev_abs_err[a] = abs(errs[a])
            self.windows.append(dict(t=t, cmd=cmd, T=T, deadline=deadline, e0=dict(errs), streak=0.0, entry_t=None,
                                     success=None, max_abs_err={a: 0.0 for a in AXES}))
            if self.pending:
                self.next_issue_t = (self.pending[0]["_t"] if self.fixed_schedule
                                     else t + deadline + cfg.success_hold_s + cfg.window_margin_s)
            else:
                self.next_issue_t = np.inf
                if not self.fixed_schedule:
                    self.episode_end_t = t + deadline + cfg.success_hold_s + cfg.window_margin_s

    def _close_window(self, w: dict):
        if w["success"] is None:
            cfg = self.cfg
            settle = (w["entry_t"] - w["t"]) if w["entry_t"] is not None else np.inf
            on_time = settle <= w["deadline"] + 1e-9
            held = w["streak"] >= cfg.success_hold_s - 1e-9
            w["coupling_ok"] = True
            if cfg.coupling_limits is not None:
                for axis, limit in zip(AXES, cfg.coupling_limits):
                    if w["cmd"][axis] == 0.0 and w["max_abs_err"][axis] > limit:
                        w["coupling_ok"] = False
            w["settle_s"] = float(settle)
            w["on_time"] = bool(on_time)
            w["success"] = bool(self.failure is None and on_time and held and w["coupling_ok"])
            w["final_err"] = dict(self._last_err)

    def _schedule_lag(self, e: dict) -> tuple[np.ndarray, float, float]:
        """Takvim gecikmesi (eksen başına, işaretli, ölçekli), τ = geçen/T, T."""
        if not self.windows:
            return np.zeros(3), 0.0, 0.0
        w = self.windows[-1]
        t = self.steps * CONTROL_DT
        tau = (t - w["t"]) / max(w["T"], 1e-6)
        lag = np.zeros(3)
        for k, a in enumerate(AXES):
            if w["cmd"][a] == 0.0:
                continue
            sched = abs(w["e0"][a]) * max(0.0, 1.0 - tau)
            lag[k] = math.copysign(max(0.0, abs(e[a]) - sched), e[a]) / self.cfg.sched_scale[k]
        return lag, float(tau), float(w["T"])

    # =================================================================
    # OBS
    # =================================================================

    def _obs(self, s: dict, e: dict) -> np.ndarray:
        lag, tau, T = self._schedule_lag(e)
        o = np.array([
            e["heading"] / 10.0, e["heading"] / 180.0,
            e["altitude"] / 20.0, e["altitude"] / 200.0,
            e["speed"] / 4.0, e["speed"] / 30.0,
            *np.clip(lag, -1.0, 1.0),
            min(tau, 2.0) / 2.0, T / 30.0,
            s["vs"] / 20.0, s["u"] / 100.0, s["v"] / 20.0,
            s["phi"] / 1.0, s["theta"] / 0.5,
            s["p"] / 1.0, s["q"] / 0.5, s["r"] / 0.7,
            (s["rpm"] - 320.0) / 30.0,
            *self.filt,
        ], dtype=np.float64)
        o = np.nan_to_num(o, nan=0.0, posinf=5.0, neginf=-5.0)
        o[[0, 2, 4]] = np.clip(o[[0, 2, 4]], -3.0, 3.0)
        return np.clip(o, -5.0, 5.0).astype(np.float32)

    # =================================================================
    # STEP
    # =================================================================

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
        t_now = self.steps * CONTROL_DT
        s_pre = self._read_state()
        self._issue_due_commands(t_now, s_pre)

        self._dfilt = cfg.action_filter_alpha * (a - self.filt)
        self.filt += self._dfilt
        self._sfd_cache = self._sfd()
        att_n = self._sfd_att() + self.att_resid
        self.att_cmd = np.array([att_n[0] + math.radians(cfg.roll_cmd_deg) * self.filt[2],
                                 att_n[1] - math.radians(cfg.pitch_cmd_deg) * self.filt[1]])
        coll = float(np.clip(self._sfd_cache[0] + self.resid[0] + cfg.coll_scale * self.filt[0], *cfg.coll_limits))
        rud = float(np.clip(self._sfd_cache[3] + self.resid[3] + cfg.pedal_scale * self.filt[3], -1.0, 1.0))
        self.fdm["fcs/collective-cmd-norm"] = coll
        self.fdm["fcs/rudder-cmd-norm"] = rud

        ok = self._run_physics()
        self.steps += 1
        s = self._read_state()
        finite = all(np.isfinite(list(s.values())))
        if finite:
            self.psi_unwrap += wrap_deg(s["psi_deg"] - self._psi_prev_meas)
            self._psi_prev_meas = s["psi_deg"]
        e = self._errors(s) if finite else dict(self._last_err)
        if finite:
            self._last_err = dict(e)

        reward, parts = self._reward_m(s, e, a) if finite else (0.0, {})
        self.failure = self._safety(s, e, ok, finite)
        terminated = self.failure is not None
        if terminated:
            reward -= cfg.fail_penalty
        reward *= cfg.reward_scale

        t_after = self.steps * CONTROL_DT
        if self.windows and not terminated:
            w = self.windows[-1]
            inside = (abs(e["heading"]) <= cfg.tol_heading_deg and abs(e["speed"]) <= cfg.tol_speed_fps
                      and abs(e["altitude"]) <= cfg.tol_alt_ft)
            if inside:
                if w["streak"] == 0.0:
                    w["entry_t"] = t_after
                w["streak"] += CONTROL_DT
            else:
                w["streak"] = 0.0
                w["entry_t"] = None
            for k in AXES:
                w["max_abs_err"][k] = max(w["max_abs_err"][k], abs(e[k]))

        truncated = (not terminated) and (self.steps >= self.max_steps or t_after >= self.episode_end_t - 1e-9)
        controls = [float(self.fdm[f"fcs/{k}-cmd-norm"]) for k in ("collective", "elevator", "aileron", "rudder")]
        info = dict(level=self.level.name, level_index=self.level_index, reward_parts=parts, controls=controls,
                    att_cmd_deg=[math.degrees(x) for x in self.att_cmd], failure=self.failure,
                    **self._info_state(s, e))
        if self.windows:
            w = self.windows[-1]
            info.update(cmd_T=w["T"], cmd_deadline=w["deadline"], cmd_elapsed=t_after - w["t"])
        if terminated or truncated:
            for w in self.windows:
                self._close_window(w)
            n_ok = sum(int(w["success"]) for w in self.windows)
            info["episode_success"] = bool(not terminated and self.windows and n_ok == len(self.windows))
            info["commands_ok"] = n_ok
            info["commands_total"] = len(self.windows)
            info["command_results"] = [dict(t=w["t"], cmd=w["cmd"], T=w["T"], deadline=w["deadline"], success=w["success"],
                                            settle_s=w["settle_s"], on_time=w["on_time"], final_streak_s=w["streak"],
                                            max_abs_err=w["max_abs_err"], coupling_ok=w.get("coupling_ok", True),
                                            final_err=w["final_err"]) for w in self.windows]
            info["termination"] = self.failure or "time_limit"
            if terminated:
                self._fdm_needs_refresh = True
        self.prev_action = a
        obs = self._obs(s, e) if finite else np.zeros(OBS_DIM_M, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # =================================================================
    # REWARD
    # =================================================================

    def _reward_m(self, s: dict, e: dict, a: np.ndarray):
        cfg = self.cfg
        track = (cfg.w_heading * self._kernel(e["heading"], cfg.kernel_heading)
                 + cfg.w_speed * self._kernel(e["speed"], cfg.kernel_speed)
                 + cfg.w_alt * self._kernel(e["altitude"], cfg.kernel_alt))
        progress = 0.0
        for a_name, min_scale in zip(AXES, cfg.progress_min_scale):
            scale = max(min_scale, abs(self.last_delta[a_name]))
            cur = abs(e[a_name])
            progress += (self.prev_abs_err[a_name] - cur) / scale
            self.prev_abs_err[a_name] = cur
        progress *= cfg.progress_weight
        lag, tau, T = self._schedule_lag(e)
        sched = cfg.pen_sched * float(np.sum(np.minimum(1.0, np.abs(lag)) ** 2))
        late = 0.0
        if self.windows:
            w = self.windows[-1]
            t = self.steps * CONTROL_DT
            inside = (abs(e["heading"]) <= cfg.tol_heading_deg and abs(e["speed"]) <= cfg.tol_speed_fps
                      and abs(e["altitude"]) <= cfg.tol_alt_ft)
            if t - w["t"] > w["deadline"] and not inside:
                late = cfg.pen_late
        side = cfg.pen_side * (s["v"] / 10.0) ** 2
        att = cfg.pen_att * ((max(0.0, abs(math.degrees(s["phi"])) - 60.0) / 15.0) ** 2
                             + (max(0.0, abs(math.degrees(s["theta"])) - 35.0) / 10.0) ** 2)
        smooth = (cfg.pen_dctrl * float(np.mean(self._dfilt ** 2))
                  + cfg.pen_sat * float(np.mean(np.maximum(np.abs(a) - cfg.sat_threshold, 0.0))))
        couple = 0.0
        if cfg.coupling_limits is not None and cfg.pen_couple > 0.0:
            cmd = self.windows[-1]["cmd"] if self.windows else None
            for axis, lim in zip(AXES, cfg.coupling_limits):
                if cmd is None or cmd[axis] == 0.0:
                    x = min(1.0, max(0.0, abs(e[axis]) - cfg.couple_soft_frac * lim) / lim)
                    couple += cfg.pen_couple * x * x
        r = track + progress - sched - late - side - att - smooth - couple
        return float(r), dict(track=track, progress=progress, schedule=sched, late=late, side=side, attitude=att,
                              smooth=smooth, coupling=couple)


# =====================================================================
# HIZLI TEST:  python helicopter_env_maneuver.py
# =====================================================================

if __name__ == "__main__":
    import time
    for lvl, u0 in (("M1", 20.0), ("M3", 100.0), ("M4", 160.0)):
        env = HelicopterEnvManeuver(level=lvl)
        t0 = time.time()
        obs, info = env.reset(seed=0, options=dict(start_speed_fps=u0))
        print(f"{lvl} reset {time.time() - t0:.2f}s settle={info['setup'].get('settle_s', 0):.1f}s sim "
              f"u={info['speed']:.1f} h={info['altitude']:.1f} v={info['lateral_speed']:+.2f} obs={obs.shape}")
        t0 = time.time()
        done, ret, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, info = env.step(np.zeros(4))
            ret += r
            n += 1
            done = term or trunc
        print(f"   a=0: {n} adım ({n / (time.time() - t0):.0f} adım/s) return={ret:.1f} bitiş={info['termination']} "
              f"sonuç={[(round(c['T'], 1), c['success'], round(c['settle_s'], 1) if np.isfinite(c['settle_s']) else None) for c in info['command_results']]}")
