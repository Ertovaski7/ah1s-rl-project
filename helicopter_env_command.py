from __future__ import annotations

"""
HELICOPTER ENV COMMAND — Δheading / Δhız / Δirtifa komut takibi (PPO, sıfırdan)
===============================================================================

Mentorun planı (2026-09-22): teacher/student yok. Helikopter belirli bir
irtifada (300 ft, 15 ft/s) başlar; kısa episode'larda küçük komutlar verilir
(önce ±5° heading, öğrenince ±10° ...; sonra hız). Rüzgâr yok.

Bu env'in eski env'lerden (Stage 1/2, Turn) farkı
------------------------------------------------
1. Hedefler SABİT DEĞİL: observation mutlak irtifa / mutlak heading içermez;
   yalnızca komuta göre HATALAR (h_ref − h, v_ref − u, ψ_ref − ψ) + uçuş durumu.
   → "hangi irtifa / heading'den başlarsam başlayayım, verdiğim Δ'yı uygula".
2. Tek ağ, tek observation: heading, hız ve irtifa hataları baştan observation'da.
   Bir seviyede (±5°) eğitilen ağırlıklar sonraki seviyeye (±10°, hız, irtifa)
   aynen devredilir.
3. Başlangıç: rotor yerde ısınır → JSBSim initial condition (IC) ile istenen
   irtifa / hız / heading'e "teleport" → reset-only stabilizasyon kontrolcüsü
   birkaç saniyede oturtur → PPO devralır. (Stabilizasyon kontrolcüsü teacher
   DEĞİLDİR: yalnızca başlangıç koşulunu kurar; PPO'ya action önermez, verisi
   eğitimde kullanılmaz.) Teleport başarısız olursa eski güvenli yol: yerden
   kalkış + referans tırmanış.
4. AFCS yalnızca stabilizasyon (SAS): heading'i AFCS tutmaz. psi-trim her
   adımda mevcut heading'e eşitlenir ("follow"), yani heading'i değiştiren /
   tutan PPO'dur. (Eski sistemde psi-trim sabitti → heading'i AFCS tutuyordu.)

Observation (19, hepsi ölçekli, [-5, 5] kırpılır)
-------------------------------------------------
  0  ψ hatası (ince)   = e_ψ / 10°          6  dikey hız / 10 ft/s
  1  ψ hatası (kaba)   = e_ψ / 180°         7  u (ileri hava hızı) / 30 ft/s
  2  irtifa hatası (i) = e_h / 20 ft        8  v (yanal hız) / 10 ft/s
  3  irtifa hatası (k) = e_h / 200 ft       9  roll / 0.35 rad     10 pitch / 0.35 rad
  4  hız hatası (ince) = e_v / 4 ft/s      11  p / 0.5   12 q / 0.5   13 r / 0.25 rad/s
  5  hız hatası (kaba) = e_v / 30 ft/s     14  (rotor rpm − 320) / 30
 15–18 filtrelenmiş action (= kumandanın trimden sapması, [-1, 1])
  e_ψ = unwrap edilmiş kalan dönüş (±180°'den büyük komutlarda da yönü korur).
  u mutlak hızdır: hover ile ileri uçuşun dinamiği farklı, ağın rejimi bilmesi gerekir.

Action (4, [-1, 1]) — handover anındaki trim etrafında residual
---------------------------------------------------------------
  kumanda = trim + ölçek · f,   f ← f + α (a − f)   (birinci dereceden filtre)
  a0 collective, a1 elevator (boyuna cyclic), a2 aileron (yanal cyclic), a3 rudder (pedal)
  a = 0 → trim (başlangıçta helikopter düz uçar).

Episode
-------
  Seviyeye göre 90–120 s. Komutlar seviyenin `command_windows` aralıklarında
  (varsayılan t≈3–8 s ve t≈40–50 s) verilir. Δ komutu, komut anındaki ÖLÇÜLEN
  değere göre uygulanır (h_ref = h + Δh ...). Komut verilmeyen eksen önceki
  referansını korur (düz uç / hızı ve irtifayı koru).

Reward (v2; v1 için CommandEnvConfig.v1())
------------------------------------------
  Üç eksende takip çekirdeği (ince Gauss + kaba Laplace), potansiyel tabanlı
  ilerleme terimi, roll / açısal hız / yanal kayma cezaları, kumanda hızı +
  doygunluk cezası (v1'de son model elevator'ı uçtan uca oynatıyordu) ve komut
  verilmeyen eksen için kuplaj cezası (başarı kriteriyle hizalı).

Başarı (curriculum için; reward'dan ayrı)
-----------------------------------------
  Her komut penceresinin SON `success_hold_s` (10 s) boyunca aynı anda:
  |e_ψ| ≤ 1.5°, |e_v| ≤ 1.5 ft/s, |e_h| ≤ 10 ft  ve güvenlik ihlali yok.
  v2: ayrıca komut VERİLMEYEN eksen pencere boyunca sınır içinde kalmalı
  (heading 5°, hız 4 ft/s, irtifa 25 ft) — dönüşte hızı/irtifayı kaçırma.
  Episode başarılı = tüm pencereler başarılı.

Birimler: ft, ft/s, derece (repo ile aynı). Kontrol adımı 0.075 s (13.3 Hz).
"""

import math
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import jsbsim
import numpy as np
from gymnasium import spaces

from command_curriculum import AXES, DEFAULT_LEVELS, Level, find_level


JSBSIM_DT = 0.0075
PHYSICS_STEPS = 10
CONTROL_DT = JSBSIM_DT * PHYSICS_STEPS

# Resmi AH-1S trim / kalibrasyon değerleri (helicopter_env_v2.py, _stage2.py)
AILERON_TRIM = 0.19095
RUDDER_TRIM = 0.39
PHI_TRIM_RAD = -0.049254
THETA_TRIM_RAD = -0.006428
ELEV_CAL_V = np.array([0.0, 8.0, 16.0, 22.0])
ELEV_CAL_E = np.array([-0.18, -0.1558, -0.13, -0.11])
ELEV_EXTRAP_SLOPE = 0.02 / 6.0          # 22 ft/s üstü doğrusal uzatma (ölçülmedi)
CRUISE_COLLECTIVE = 0.604               # Stage 2 doğrulama izi, ~10–15 ft/s @ 300 ft
PEDAL_GAIN = 0.18                       # rotor_control.xml: fcs/adj/pedal-gain (rad / birim rudder)

OBS_DIM = 19

try:                                    # her yeni FDM'de JSBSim başlık yazısını basma
    jsbsim.FGJSBBase().debug_lvl = 0
except Exception:                       # noqa: BLE001
    pass


def wrap_deg(x: float) -> float:
    return float((float(x) + 180.0) % 360.0 - 180.0)


def elevator_ff(v: float) -> float:
    right = ELEV_CAL_E[-1] + (v - ELEV_CAL_V[-1]) * ELEV_EXTRAP_SLOPE
    return float(np.interp(v, ELEV_CAL_V, ELEV_CAL_E, right=right))


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class AfcsMode:
    """AH-1S AFCS ayarı (JSBSim systems/afcs.xml).

    AFCS kanalları (her biri `*-channel-active-norm` ile 0..1 ölçeklenir):
      roll : −0.133·(φ − φ_trim) − 0.096·p  (+ pilot girişini 0.33 ile güçlendirme)
             → yatış açısı tutma + roll sönümleme
      pitch: +0.281·(θ − θ_trim) + 0.727·q  (+ 0.475 güçlendirme)
             → yunuslama açısı tutma + pitch sönümleme
      yaw  : heading_hold · PI(sin(ψ − ψ_trim)) + yaw_rate_gain · r
             → heading TUTMA (PI) ve/veya yaw sönümleme (rate damper)
    Bu env'de eğitim modunda heading-hold KAPALI: heading'i PPO tutar/değiştirir;
    AFCS yalnızca roll/pitch'i stabilize eder ve yaw'ı sönümler (SAS).
    Yaw PID'nin integratörü channel-active > 0.999 iken çalışır; eğitimde
    0.99 kullanıyoruz → integratör sıfırda kalır (kapalı heading-hold kurulmaz).
    Pozitif rudder (pedal) → burun SOLA (heading azalır), pozitif aileron → sağa yatış.
    """
    pitch: float = 0.5
    roll: float = 1.0
    yaw: float = 1.0
    heading_hold: bool = True
    yaw_rate_gain: float = 0.0          # JSBSim varsayılanı 0 (sönümleme yok); >0 → sönümleme


@dataclass
class CommandEnvConfig:
    # --- başlangıç -----------------------------------------------------
    start_mode: str = "teleport"            # teleport | climb
    reuse_fdm: bool = True                  # başarılı episode sonrası aynı FDM'i yeniden teleport et
    fdm_refresh_episodes: int = 25          # bu kadar episode'da bir FDM'i sıfırdan kur
    random_start_heading: bool = True
    start_heading_deg: float = 180.0        # random_start_heading=False ise
    settle_max_s: float = 120.0
    settle_min_s: float = 25.0
    settle_hold_s: float = 5.0              # sıkı denge kriteri bu kadar süre sağlanmalı
    trim_avg_s: float = 3.0                 # trim = son bu kadar saniyenin kumanda ortalaması
    climb_max_s: float = 180.0
    # reset (stabilizasyon): heading-hold açık (psi-trim = başlangıç heading'i) + yaw sönümleme
    afcs_setup: AfcsMode = field(default_factory=lambda: AfcsMode(0.5, 1.0, 1.0, True, 0.2))
    # eğitim: heading-hold KAPALI, yalnızca yaw sönümleme (SAS)
    afcs_train: AfcsMode = field(default_factory=lambda: AfcsMode(0.5, 1.0, 0.99, False, 0.2))

    # --- action --------------------------------------------------------
    # (collective, elevator, aileron, rudder) — trim etrafında en büyük sapma.
    # AFCS açıkken yaklaşık etkiler (rotor_control.xml + afcs.xml'den):
    #   collective 0.06 → ±0.013 rad kolektif  (~±10 ft/s dikey hız)
    #   elevator   0.05 → ±3° pitch             (düşük hızda ~±15 ft/s)
    #   aileron    0.30 → ±8.6° yatış (roll AFCS 1.0 ile)
    #   rudder     0.20 → ±0.036 rad kuyruk rotoru (yaw damper 0.2 ile ~±10°/s)
    # İLK TAHMİN: diagnose_command_env.py ile gerçek JSBSim'de doğrulanmalı.
    action_scale: tuple = (0.06, 0.05, 0.30, 0.20)
    action_filter_alpha: float = 0.4        # 1.0 → filtre yok
    control_limits: tuple = ((0.40, 0.80), (-0.35, 0.10), (-1.0, 1.0), (-1.0, 1.0))

    # --- komut sınırları -------------------------------------------------
    cmd_speed_range: tuple = (3.0, 60.0)    # v_ref bu aralığın dışına çıkarsa Δv'nin işareti çevrilir
    cmd_min_alt_ft: float = 150.0           # h_ref bunun altına inerse Δh'nin işareti çevrilir

    # --- reward (v2) ---------------------------------------------------
    # v1 (ilk koşu) ile fark: çekirdeğin ince kısmı Gauss (sıfır civarında düz → küçük
    # hatayı kovalamak için kumandayı uçtan uca oynatmaya teşvik yok), hız/irtifa
    # ağırlığı 0.5 → 0.7, kumanda hızı (pen_dctrl) ve doygunluk (pen_sat) cezası.
    # v1 ayarları: CommandEnvConfig.v1()
    w_heading: float = 1.0
    w_speed: float = 0.7
    w_alt: float = 0.7
    kernel_heading: tuple = (1.5, 10.0)     # derece  (ince, kaba) çekirdek genişlikleri
    kernel_speed: tuple = (1.0, 5.0)        # ft/s
    kernel_alt: tuple = (5.0, 30.0)         # ft
    fine_kernel: str = "gauss"              # "gauss" (v2) | "laplace" (v1)
    progress_weight: float = 10.0           # bir komutu tamamen kapatmanın toplam ilerleme ödülü (eksen başına)
    progress_min_scale: tuple = (5.0, 2.0, 10.0)   # heading deg, speed ft/s, alt ft
    pen_roll: float = 0.10                  # · (φ / 0.35)²
    pen_rate: float = 0.05                  # · Σ (rate / ölçek)²
    pen_side: float = 0.02                  # · |v| [ft/s]
    pen_jerk: float = 0.05                  # · mean((a − a_prev)²)
    pen_ctrl: float = 0.01                  # · mean(f²)
    pen_dctrl: float = 5.0                  # · mean((f_t − f_{t−1})²)   kumanda hızı (gerçek kumanda değişimi)
    pen_sat: float = 2.0                    # · mean(max(0, |a| − sat_threshold))   doygunluk / bang-bang
    sat_threshold: float = 0.8
    # Kuplaj cezası (başarı kriteriyle hizalı): komut VERİLMEYEN eksenin hatası
    # sınırın yarısını aşınca karesel ceza, x = (|e| − 0.5·sınır)/sınır ≤ 1 ile kırpılı:
    # sınırda 0.25·pen, en çok pen (eksen başına). Kırpma → adım ödülü çoğunlukla
    # pozitif kalır, ajan "erken düşüp cezadan kaçma"yı öğrenmez.
    # (Yalnızca takip çekirdeğiyle ajan hızlı dönmek için hızı 5–10 ft/s kaçırmayı
    # "ucuz" buluyordu → kuplajsız v2 denemesinde H5'te başarı %1'e düştü.)
    pen_couple: float = 2.0
    couple_soft_frac: float = 0.5
    fail_penalty: float = 50.0
    # Toplam ödül bu katsayıyla çarpılır (adım başı ≤ ~0.25). PPO'nun value loss'u
    # küçük kalsın; max_grad_norm kırpması policy gradyanını ezmesin diye.
    reward_scale: float = 0.1

    # --- başarı (curriculum) --------------------------------------------
    tol_heading_deg: float = 1.5
    tol_speed_fps: float = 1.5
    tol_alt_ft: float = 10.0
    success_hold_s: float = 10.0
    # v2: komut VERİLMEYEN eksen pencere boyunca bu sınırlar içinde kalmalı
    # (ör. dönüş sırasında hız ±4 ft/s, irtifa ±25 ft). None → kontrol yok (v1).
    coupling_limits: tuple | None = (5.0, 4.0, 25.0)   # heading deg, speed ft/s, alt ft

    @classmethod
    def v1(cls, **kw):
        """İlk koşunun (2026-09-22, cmd_v1) reward / başarı ayarları."""
        base = dict(w_speed=0.5, w_alt=0.5, fine_kernel="laplace", pen_dctrl=0.0, pen_sat=0.0,
                    pen_couple=0.0, coupling_limits=None)
        base.update(kw)
        return cls(**base)

    # --- güvenlik (episode'u bitirir) -------------------------------------
    max_roll_deg: float = 40.0
    max_pitch_deg: float = 30.0
    max_yaw_rate_dps: float = 45.0
    max_roll_rate_dps: float = 90.0
    min_agl_ft: float = 50.0
    alt_margin_ft: float = 75.0             # [min(ref, komut anı), max(...)] ± marj dışına çıkma
    speed_margin_fps: float = 12.0
    heading_margin_deg: float = 30.0        # ters yöne ya da hedefin ötesine bu kadar gitme
    speed_limits_fps: tuple = (-5.0, 150.0)
    rpm_limits: tuple = (280.0, 380.0)


# =====================================================================
# SETUP CONTROLLER GAINS (yalnızca reset; probe / teacher testlerinde 25 ft/s @ 300 ft kurdu)
# =====================================================================

KH = 0.08            # irtifa hatası → istenen dikey hız [1/s]
VS_UP, VS_DN = 5.0, 4.0
COLL_BASE = 0.6085
K_FF_VS, KP_VS, KI_VS = 0.0058, 0.004, 0.0012
KP_U, KI_U = 0.0025, 0.0003
I_LIM = 60.0
SETUP_COLL_RANGE = (0.50, 0.70)
SETUP_ELEV_RANGE = (-0.20, -0.06)


class HelicopterEnvCommand(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level=0, levels=None, config: CommandEnvConfig | None = None):
        super().__init__()
        self.levels: list[Level] = list(levels or DEFAULT_LEVELS)
        self.cfg = config or CommandEnvConfig()
        self.level_index = find_level(level, self.levels)

        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.fdm = None
        self._fdm_needs_refresh = True
        self._episodes_on_fdm = 0
        self._scale = np.asarray(self.cfg.action_scale, dtype=np.float64)
        self._lims = np.asarray(self.cfg.control_limits, dtype=np.float64)

        # episode state (reset'te doldurulur)
        self.steps = 0
        self.max_steps = 1
        self.trim = np.zeros(4)
        self.filt = np.zeros(4)
        self._dfilt = np.zeros(4)
        self.prev_action = np.zeros(4)
        self.ref = {a: 0.0 for a in AXES}
        self.cmd_start = {a: 0.0 for a in AXES}
        self.cmd_err0 = {a: 0.0 for a in AXES}
        self.last_delta = {a: 0.0 for a in AXES}
        self.prev_abs_err = {a: 0.0 for a in AXES}
        self._last_err = {a: 0.0 for a in AXES}
        self.psi_unwrap = 0.0
        self._psi_prev_meas = 0.0
        self.schedule: list[tuple[float, dict]] = []
        self.next_cmd = 0
        self.windows: list[dict] = []
        self.failure: str | None = None
        self.setup_info: dict = {}

    # =================================================================
    # CURRICULUM API (VecEnv.env_method ile çağrılır)
    # =================================================================

    def set_level(self, level) -> int:
        self.level_index = find_level(level, self.levels)
        return self.level_index

    def get_level(self) -> int:
        return self.level_index

    @property
    def level(self) -> Level:
        return self.levels[self.level_index]

    # =================================================================
    # JSBSIM
    # =================================================================

    def _create_fdm(self):
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        if not fdm.load_model("ah1s"):
            raise RuntimeError("AH-1S modeli yüklenemedi.")
        if not fdm.load_ic("reset00.xml", True):
            raise RuntimeError("reset00.xml yüklenemedi.")
        fdm.set_dt(JSBSIM_DT)
        # Resmi AH-1S kurulumu (helicopter_env_v2.py ile birebir)
        fdm["ap/afcs/psi-trim-rad"] = np.pi
        fdm["propulsion/tank[0]/contents-lbs"] = 0.0
        fdm["propulsion/tank[1]/contents-lbs"] = 0.0
        fdm["aero/setup/downwash-enable"] = 1.0
        fdm["aero/setup/Nr_limiter"] = 0.05
        fdm["fcs/adj/collective-profile"] = 0.0
        fdm["fcs/adj/center-sensitivity"] = 1.0
        for k in ("collective", "elevator", "aileron", "rudder"):
            fdm[f"fcs/{k}-cmd-norm"] = 0.0
        fdm["fcs/rpm-governor-active-norm"] = 0.0
        for ch in ("yaw", "pitch", "roll"):
            fdm[f"ap/afcs/{ch}-channel-active-norm"] = 0.0
        fdm.run_ic()
        self.fdm = fdm

    def _warmup_rotor(self):
        fdm = self.fdm
        t0 = float(fdm["simulation/sim-time-sec"])
        while True:
            elapsed = float(fdm["simulation/sim-time-sec"]) - t0
            fdm["fcs/rpm-governor-active-norm"] = float(np.clip(elapsed / 5.0, 0.0, 1.0))
            fdm.run()
            rpm = float(fdm["propulsion/engine/rotor-rpm"])
            if elapsed >= 7.0 and rpm >= 320.0:
                break
            if elapsed > 12.0:
                raise RuntimeError(f"Rotor warmup başarısız. RPM={rpm:.1f}")
        fdm["fcs/rpm-governor-active-norm"] = 1.0
        fdm["ap/afcs/manual/phi-trim-rad"] = PHI_TRIM_RAD
        fdm["ap/afcs/manual/theta-trim-rad"] = THETA_TRIM_RAD

    def _teleport(self, h_ft: float, u_fps: float, psi_deg: float):
        """Dönen rotorla birlikte helikopteri havaya taşı (IC + run_ic).

        run_ic() konum / hız / attitude'u IC'den yeniden kurar; rotor devri
        (FGRotor iç durumu) korunur. Kontrol eder, korunmadıysa hata verir.
        """
        fdm = self.fdm
        fdm["ic/phi-rad"] = PHI_TRIM_RAD
        fdm["ic/theta-rad"] = 0.0
        fdm["ic/psi-true-rad"] = math.radians(psi_deg % 360.0)
        fdm["ic/h-agl-ft"] = float(h_ft)
        fdm["ic/u-fps"] = float(u_fps)
        fdm["ic/v-fps"] = 0.0
        fdm["ic/w-fps"] = 0.0
        fdm["ic/p-rad_sec"] = 0.0
        fdm["ic/q-rad_sec"] = 0.0
        fdm["ic/r-rad_sec"] = 0.0
        # Havada rotor inmesin diye kumandalar hemen seyir trimine
        fdm["fcs/collective-cmd-norm"] = CRUISE_COLLECTIVE
        fdm["fcs/elevator-cmd-norm"] = elevator_ff(u_fps)
        fdm["fcs/aileron-cmd-norm"] = AILERON_TRIM
        fdm["fcs/rudder-cmd-norm"] = RUDDER_TRIM
        fdm["fcs/rpm-governor-active-norm"] = 1.0
        if not fdm.run_ic():
            raise RuntimeError("run_ic() başarısız (teleport)")
        s = self._read_state()
        if s["rpm"] < 300.0:
            raise RuntimeError(f"teleport sonrası rotor devri düştü: {s['rpm']:.1f} rpm")
        if abs(s["h"] - h_ft) > 20.0:
            raise RuntimeError(f"teleport irtifası tutmadı: {s['h']:.1f} ft (istenen {h_ft:.1f})")

    def _read_state(self) -> dict:
        f = self.fdm
        return dict(
            h=float(f["position/h-agl-ft"]),
            vs=float(f["velocities/h-dot-fps"]),
            u=float(f["velocities/u-aero-fps"]),
            v=float(f["velocities/v-aero-fps"]),
            phi=float(f["attitude/phi-rad"]),
            theta=float(f["attitude/theta-rad"]),
            psi_deg=math.degrees(float(f["attitude/psi-rad"])) % 360.0,
            p=float(f["velocities/p-rad_sec"]),
            q=float(f["velocities/q-rad_sec"]),
            r=float(f["velocities/r-rad_sec"]),
            rpm=float(f["propulsion/engine/rotor-rpm"]),
        )

    def _set_afcs(self, mode: AfcsMode, psi_ref_deg: float):
        f = self.fdm
        f["ap/afcs/pitch-channel-active-norm"] = float(mode.pitch)
        f["ap/afcs/roll-channel-active-norm"] = float(mode.roll)
        f["ap/afcs/yaw-channel-active-norm"] = float(mode.yaw)
        f["ap/afcs/heading-hold-enable"] = 1.0 if mode.heading_hold else 0.0
        f["ap/afcs/adj/yaw-rate-ctrl-gain"] = float(mode.yaw_rate_gain)
        f["ap/afcs/psi-trim-rad"] = math.radians(psi_ref_deg % 360.0)

    def _reset_yaw_integrator(self, psi_ref_deg: float):
        """Yaw PID integratörü yalnızca channel-active > 0.999 iken çalışır; bir
        adım 0.99'a çekmek onu sıfırlar (FDM yeniden kullanıldığında birikmesin)."""
        f = self.fdm
        f["ap/afcs/psi-trim-rad"] = math.radians(psi_ref_deg % 360.0)
        f["ap/afcs/yaw-channel-active-norm"] = 0.99
        self.fdm.run()

    def _write_controls(self, c):
        f = self.fdm
        f["fcs/collective-cmd-norm"] = float(c[0])
        f["fcs/elevator-cmd-norm"] = float(c[1])
        f["fcs/aileron-cmd-norm"] = float(c[2])
        f["fcs/rudder-cmd-norm"] = float(c[3])

    def _run_physics(self) -> bool:
        for _ in range(PHYSICS_STEPS):
            if not self.fdm.run():
                return False
        return True

    # =================================================================
    # RESET-ONLY STABILIZATION CONTROLLER (teacher DEĞİL)
    # =================================================================

    def _settle(self, h0: float, u0: float, psi0: float, climb: bool) -> tuple[bool, dict]:
        """Klasik PI: collective ← irtifa/dikey hız, elevator ← hız; heading'i AFCS
        (psi-trim = psi0) tutar. Oturunca True döner; kumanda trimlerini kaydeder."""
        cfg = self.cfg
        dt = CONTROL_DT
        max_s = cfg.climb_max_s if climb else cfg.settle_max_s
        i_vs = (CRUISE_COLLECTIVE - COLL_BASE) / KI_VS      # seyir collective'inden başla
        i_u = 0.0
        hist = deque(maxlen=max(1, int(round(cfg.trim_avg_s / dt))))
        stable_t, t = 0.0, 0.0
        self._reset_yaw_integrator(psi0)
        s = self._read_state()
        was_low = climb
        base = cfg.afcs_setup
        while t < max_s:
            low = climb and s["h"] < h0 - 20.0
            u_tgt = 0.0 if low else u0
            mode = AfcsMode(1.0 if low else base.pitch, base.roll, base.yaw, base.heading_hold, base.yaw_rate_gain)
            self._set_afcs(mode, psi0)

            if low:
                # Yerden tırmanış: Stage2Refine._reference_collective ile aynı (doğrulanmış) yasa
                vs_des = float(np.clip(0.05 * (h0 - s["h"]), -3.0, 6.0))
                coll = float(np.clip(COLL_BASE + 0.0058 * vs_des + 0.0025 * (vs_des - s["vs"]), 0.59, 0.65))
            else:
                if was_low:                                   # PI'ya kesintisiz geçiş
                    i_vs = (float(self.fdm["fcs/collective-cmd-norm"]) - COLL_BASE) / KI_VS
                    was_low = False
                vs_des = float(np.clip(KH * (h0 - s["h"]), -VS_DN, VS_UP))
                evs = vs_des - s["vs"]
                cu = COLL_BASE + K_FF_VS * vs_des + KP_VS * evs + KI_VS * i_vs
                coll = float(np.clip(cu, *SETUP_COLL_RANGE))
                if not ((cu > SETUP_COLL_RANGE[1] and evs > 0) or (cu < SETUP_COLL_RANGE[0] and evs < 0)):
                    i_vs = float(np.clip(i_vs + evs * dt, -I_LIM, I_LIM))

            if low:
                elev = -0.1558                                # resmi hover trimi (Stage 1/2 tırmanışı)
            else:
                eu = u_tgt - s["u"]
                eu_ = elevator_ff(u_tgt) + KP_U * eu + KI_U * i_u
                elev = float(np.clip(eu_, *SETUP_ELEV_RANGE))
                if not ((eu_ > SETUP_ELEV_RANGE[1] and eu > 0) or (eu_ < SETUP_ELEV_RANGE[0] and eu < 0)):
                    i_u = float(np.clip(i_u + eu * dt, -I_LIM, I_LIM))

            ctrl = (coll, elev, AILERON_TRIM, RUDDER_TRIM)
            self._write_controls(ctrl)
            if not self._run_physics():
                return False, dict(reason="JSBSim stopped during setup", t=t)
            t += dt
            s = self._read_state()
            hist.append(ctrl)
            if not all(np.isfinite(list(s.values()))):
                return False, dict(reason="non-finite state during setup", t=t)
            if abs(s["phi"]) > 0.6 or abs(s["theta"]) > 0.6 or (not climb and s["h"] < 30.0):
                return False, dict(reason=f"unsafe during setup (h={s['h']:.0f}, "
                                          f"phi={math.degrees(s['phi']):.0f}°)", t=t)
            # sıkı denge: trim gerçekten a = 0'da düz uçuş versin
            ok = (abs(s["h"] - h0) <= 2.0 and abs(s["vs"]) <= 0.15 and abs(s["u"] - u0) <= 0.5
                  and abs(wrap_deg(s["psi_deg"] - psi0)) <= 1.0
                  and max(abs(s["p"]), abs(s["q"])) <= math.radians(1.0) and abs(s["r"]) <= math.radians(0.5))
            stable_t = stable_t + dt if ok else 0.0
            if t >= cfg.settle_min_s and stable_t >= cfg.settle_hold_s and len(hist) == hist.maxlen:
                h_arr = np.asarray(hist)
                # kumandalar da oturmuş olmalı (PI integratörü hâlâ kayıyorsa trim yanlış olur)
                if np.ptp(h_arr[:, 0]) <= 6e-4 and np.ptp(h_arr[:, 1]) <= 1e-3:
                    return True, dict(reason="", t=t, trim=h_arr.mean(axis=0), rpm=s["rpm"])
        return False, dict(reason=f"not settled in {max_s:.0f}s (h={s['h']:.1f}, u={s['u']:.1f}, "
                                  f"psi_err={wrap_deg(s['psi_deg'] - psi0):+.1f})", t=t)

    # =================================================================
    # RESET
    # =================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        if "level" in options:
            self.set_level(options["level"])
        lv = self.level
        rng = self.np_random
        cfg = self.cfg

        h0 = float(options.get("start_alt_ft", rng.uniform(*lv.start_alt_ft)))
        u0 = float(options.get("start_speed_fps", rng.uniform(*lv.start_speed_fps)))
        if "start_heading_deg" in options:
            psi0 = float(options["start_heading_deg"]) % 360.0
        else:
            psi0 = float(rng.uniform(0.0, 360.0)) if cfg.random_start_heading else cfg.start_heading_deg

        errors = []
        info_setup = None
        for attempt in range(3):
            mode = cfg.start_mode if attempt < 2 else "climb"
            fresh = (attempt > 0 or mode == "climb" or self.fdm is None or self._fdm_needs_refresh
                     or not cfg.reuse_fdm or self._episodes_on_fdm >= cfg.fdm_refresh_episodes)
            try:
                if fresh:
                    self._create_fdm()
                    self._warmup_rotor()
                    self._episodes_on_fdm = 0
                if mode == "teleport":
                    self._teleport(h0, u0, psi0)
                else:
                    psi0 = self._read_state()["psi_deg"]     # yerde başlangıç heading'i
                ok, info_setup = self._settle(h0, u0, psi0, climb=(mode == "climb"))
                if ok:
                    info_setup.update(mode=mode, fresh_fdm=fresh, attempt=attempt)
                    break
                errors.append(f"[{mode}] {info_setup['reason']}")
            except Exception as exc:                          # noqa: BLE001
                errors.append(f"[{mode}] {exc}")
            self._fdm_needs_refresh = True
        else:
            raise RuntimeError("başlangıç koşulu kurulamadı: " + " | ".join(errors))

        self._fdm_needs_refresh = False
        self._episodes_on_fdm += 1
        self.setup_info = dict(info_setup, errors=errors, h0=h0, u0=u0, psi0=psi0)
        return self._handover(lv, options)

    def _handover(self, lv: Level, options: dict):
        cfg = self.cfg
        rng = self.np_random
        self.trim = np.asarray(self.setup_info["trim"], dtype=np.float64)
        # Stabilizasyonda heading-hold PI'ı pedala sabit bir katkı veriyordu
        # (ap/rudder-cmd, rad). Eğitimde heading-hold kapanacağı için bu katkıyı
        # rudder trimine taşı (pedal-gain 0.18 rad / birim) → a = 0 yine denge.
        yaw_ap = float(self.fdm["ap/rudder-cmd"])
        self.trim[3] += yaw_ap / PEDAL_GAIN
        self.setup_info["rudder_trim_from_afcs"] = yaw_ap / PEDAL_GAIN
        self.filt = np.zeros(4)
        self._dfilt = np.zeros(4)
        self.prev_action = np.zeros(4)
        s = self._read_state()
        self._set_afcs(cfg.afcs_train, s["psi_deg"])
        self._write_controls(np.clip(self.trim, self._lims[:, 0], self._lims[:, 1]))

        self.psi_unwrap = s["psi_deg"]
        self._psi_prev_meas = s["psi_deg"]
        self.ref = {"heading": self.psi_unwrap, "speed": s["u"], "altitude": s["h"]}
        self.cmd_start = dict(self.ref)
        self.cmd_err0 = {a: 0.0 for a in AXES}
        self.last_delta = {a: 0.0 for a in AXES}

        episode_s = float(options.get("episode_s", lv.episode_s))
        self.steps = 0
        self.max_steps = int(round(episode_s / CONTROL_DT))
        if "commands" in options:                 # değerlendirme: [(t, {"heading": .., ...}), ...]
            sched = [(float(t), {a: float(c.get(a, 0.0)) for a in AXES}) for t, c in options["commands"]]
        else:
            sched = [(float(rng.uniform(*w)), lv.sample_command(rng)) for w in lv.command_windows]
            sched = [x for x in sched if x[0] <= episode_s - cfg.success_hold_s - 5.0]
        self.schedule = sorted(sched, key=lambda x: x[0])
        self.next_cmd = 0
        self.windows = []
        self.failure = None
        errs = self._errors(s)
        self.prev_abs_err = {a: abs(errs[a]) for a in AXES}
        self._last_err = dict(errs)
        obs = self._obs(s, errs)
        info = dict(level=lv.name, level_index=self.level_index, setup=self.setup_info,
                    schedule=[(t, dict(c)) for t, c in self.schedule], **self._info_state(s, errs))
        return obs, info

    # =================================================================
    # COMMANDS / ERRORS / OBS
    # =================================================================

    def _issue_due_commands(self, t: float, s: dict):
        cfg = self.cfg
        while self.next_cmd < len(self.schedule) and t >= self.schedule[self.next_cmd][0] - 1e-9:
            t_cmd, cmd = self.schedule[self.next_cmd]
            if self.windows:
                self._close_window(self.windows[-1])
            cmd = dict(cmd)
            meas = {"heading": self.psi_unwrap, "speed": s["u"], "altitude": s["h"]}
            # güvenli referans aralıkları: dışına taşan Δ'nın işaretini çevir (gerekirse kırp)
            if cmd["speed"]:
                lo, hi = cfg.cmd_speed_range
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
            errs = self._errors(s)
            for a in AXES:
                # güvenlik marjı ve ilerleme ödülü bu andaki hataya göre ölçülür
                self.cmd_err0[a] = errs[a]
                self.prev_abs_err[a] = abs(errs[a])
            self.windows.append(dict(t=t, cmd=cmd, streak=0.0, success=None,
                                     max_abs_err={a: 0.0 for a in AXES}))
            self.next_cmd += 1

    def _close_window(self, w: dict):
        if w["success"] is None:
            ok = self.failure is None and w["streak"] >= self.cfg.success_hold_s - 1e-9
            lim = self.cfg.coupling_limits
            w["coupling_ok"] = True
            if lim is not None:
                # komut verilmeyen eksen pencere boyunca sınır içinde kalmalı (kuplaj)
                for axis, limit in zip(AXES, lim):
                    if w["cmd"][axis] == 0.0 and w["max_abs_err"][axis] > limit:
                        w["coupling_ok"] = False
            w["success"] = bool(ok and w["coupling_ok"])
            w["final_err"] = dict(self._last_err)

    def _errors(self, s: dict) -> dict:
        return {"heading": self.ref["heading"] - self.psi_unwrap,
                "speed": self.ref["speed"] - s["u"],
                "altitude": self.ref["altitude"] - s["h"]}

    def _obs(self, s: dict, e: dict) -> np.ndarray:
        o = np.array([
            e["heading"] / 10.0, e["heading"] / 180.0,
            e["altitude"] / 20.0, e["altitude"] / 200.0,
            e["speed"] / 4.0, e["speed"] / 30.0,
            s["vs"] / 10.0, s["u"] / 30.0, s["v"] / 10.0,
            s["phi"] / 0.35, s["theta"] / 0.35,
            s["p"] / 0.5, s["q"] / 0.5, s["r"] / 0.25,
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

        # action → kumanda (trim + ölçek · filtrelenmiş action)
        self._dfilt = cfg.action_filter_alpha * (a - self.filt)
        self.filt += self._dfilt
        ctrl = np.clip(self.trim + self._scale * self.filt, self._lims[:, 0], self._lims[:, 1])
        self._write_controls(ctrl)

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

        reward, parts = self._reward(s, e, a) if finite else (0.0, {})
        self.failure = self._safety(s, e, ok, finite)
        terminated = self.failure is not None
        if terminated:
            reward -= cfg.fail_penalty
        reward *= cfg.reward_scale

        # başarı sayacı (aktif komut penceresi)
        if self.windows and not terminated:
            w = self.windows[-1]
            inside = (abs(e["heading"]) <= cfg.tol_heading_deg and abs(e["speed"]) <= cfg.tol_speed_fps
                      and abs(e["altitude"]) <= cfg.tol_alt_ft)
            w["streak"] = w["streak"] + CONTROL_DT if inside else 0.0
            for k in AXES:
                w["max_abs_err"][k] = max(w["max_abs_err"][k], abs(e[k]))

        truncated = (not terminated) and self.steps >= self.max_steps
        info = dict(level=self.level.name, level_index=self.level_index, reward_parts=parts,
                    controls=ctrl.tolist(), failure=self.failure, **self._info_state(s, e))
        if terminated or truncated:
            for w in self.windows:
                self._close_window(w)
            n_ok = sum(int(w["success"]) for w in self.windows)
            info["episode_success"] = bool(not terminated and self.windows and n_ok == len(self.windows))
            info["commands_ok"] = n_ok
            info["commands_total"] = len(self.windows)
            info["command_results"] = [dict(t=w["t"], cmd=w["cmd"], success=w["success"],
                                            final_streak_s=w["streak"], max_abs_err=w["max_abs_err"],
                                            coupling_ok=w.get("coupling_ok", True), final_err=w["final_err"])
                                       for w in self.windows]
            info["termination"] = self.failure or "time_limit"
            if terminated:
                self._fdm_needs_refresh = True     # güvenlik ihlali → sonraki reset'te temiz FDM
        self.prev_action = a
        obs = self._obs(s, e) if finite else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # =================================================================
    # REWARD / SAFETY
    # =================================================================

    def _kernel(self, err: float, widths) -> float:
        a, b = widths
        fine = math.exp(-(err / a) ** 2) if self.cfg.fine_kernel == "gauss" else math.exp(-abs(err) / a)
        return 0.5 * fine + 0.5 * math.exp(-abs(err) / b)

    def _reward(self, s: dict, e: dict, a: np.ndarray):
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
        pen = (cfg.pen_roll * (s["phi"] / 0.35) ** 2
               + cfg.pen_rate * ((s["p"] / 0.5) ** 2 + (s["q"] / 0.5) ** 2 + (s["r"] / 0.25) ** 2)
               + cfg.pen_side * abs(s["v"])
               + cfg.pen_jerk * float(np.mean((a - self.prev_action) ** 2))
               + cfg.pen_ctrl * float(np.mean(self.filt ** 2))
               + cfg.pen_dctrl * float(np.mean(self._dfilt ** 2))
               + cfg.pen_sat * float(np.mean(np.maximum(np.abs(a) - cfg.sat_threshold, 0.0))))
        couple = 0.0
        if cfg.coupling_limits is not None and cfg.pen_couple > 0.0:
            cmd = self.windows[-1]["cmd"] if self.windows else None
            for axis, lim in zip(AXES, cfg.coupling_limits):
                if cmd is None or cmd[axis] == 0.0:          # bu pencerede komut verilmeyen eksen
                    x = min(1.0, max(0.0, abs(e[axis]) - cfg.couple_soft_frac * lim) / lim)
                    couple += cfg.pen_couple * x * x
        r = track + progress - pen - couple
        return float(r), dict(track=track, progress=progress, penalty=pen, coupling=couple)

    def _safety(self, s: dict, e: dict, ok: bool, finite: bool) -> str | None:
        cfg = self.cfg
        if not ok:
            return "jsbsim_stopped"
        if not finite:
            return "non_finite_state"
        if abs(math.degrees(s["phi"])) > cfg.max_roll_deg:
            return "roll_limit"
        if abs(math.degrees(s["theta"])) > cfg.max_pitch_deg:
            return "pitch_limit"
        if abs(math.degrees(s["r"])) > cfg.max_yaw_rate_dps:
            return "yaw_rate_limit"
        if abs(math.degrees(s["p"])) > cfg.max_roll_rate_dps:
            return "roll_rate_limit"
        if s["h"] < cfg.min_agl_ft:
            return "too_low"
        if not cfg.rpm_limits[0] <= s["rpm"] <= cfg.rpm_limits[1]:
            return "rotor_rpm"
        if not cfg.speed_limits_fps[0] <= s["u"] <= cfg.speed_limits_fps[1]:
            return "speed_limit"
        lo = min(self.ref["altitude"], self.cmd_start["altitude"]) - cfg.alt_margin_ft
        hi = max(self.ref["altitude"], self.cmd_start["altitude"]) + cfg.alt_margin_ft
        if not lo <= s["h"] <= hi:
            return "altitude_deviation"
        lo = min(self.ref["speed"], self.cmd_start["speed"]) - cfg.speed_margin_fps
        hi = max(self.ref["speed"], self.cmd_start["speed"]) + cfg.speed_margin_fps
        if not lo <= s["u"] <= hi:
            return "speed_deviation"
        # heading: komut anındaki hata e0'a göre ters yöne ya da hedefin ötesine > marj
        e0 = self.cmd_err0["heading"]
        eh = e["heading"]
        if abs(eh) > abs(e0) + cfg.heading_margin_deg:
            return "heading_wrong_way"
        if e0 != 0.0 and eh * math.copysign(1.0, e0) < -cfg.heading_margin_deg:
            return "heading_overshoot"
        return None

    def _info_state(self, s: dict, e: dict) -> dict:
        return dict(t=self.steps * CONTROL_DT, altitude=s["h"], vertical_speed=s["vs"], speed=s["u"],
                    lateral_speed=s["v"], roll_deg=math.degrees(s["phi"]), pitch_deg=math.degrees(s["theta"]),
                    yaw_rate_dps=math.degrees(s["r"]), heading_deg=s["psi_deg"], rotor_rpm=s["rpm"],
                    heading_unwrapped=self.psi_unwrap, ref=dict(self.ref),
                    err_heading=e["heading"], err_speed=e["speed"], err_altitude=e["altitude"])

    def close(self):
        self.fdm = None


# =====================================================================
# QUICK SELF-TEST:  python helicopter_env_command.py
# =====================================================================

if __name__ == "__main__":
    import time
    env = HelicopterEnvCommand(level="H1")
    t0 = time.time()
    obs, info = env.reset(seed=0)
    print(f"reset {time.time() - t0:.2f}s  setup={ {k: v for k, v in info['setup'].items() if k != 'trim'} }")
    print(f"trim={np.round(env.trim, 4).tolist()}  schedule={info['schedule']}")
    ret, t0 = 0.0, time.time()
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(4))
        ret += r
        if env.steps % 200 == 0 or term or trunc:
            print(f"t={info['t']:5.1f} h={info['altitude']:6.1f} u={info['speed']:5.1f} "
                  f"hdg={info['heading_deg']:6.1f} eψ={info['err_heading']:+6.2f} "
                  f"eh={info['err_altitude']:+6.1f} ev={info['err_speed']:+5.2f} "
                  f"roll={info['roll_deg']:+5.1f} r={r:+.3f}")
        if term or trunc:
            break
    print(f"return={ret:.1f} steps={env.steps} wall={time.time() - t0:.1f}s "
          f"({env.steps / max(time.time() - t0, 1e-9):.0f} steps/s) termination={info['termination']} "
          f"success={info['episode_success']} results={info['command_results']}")
