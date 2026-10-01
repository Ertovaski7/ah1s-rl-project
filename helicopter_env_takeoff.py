from __future__ import annotations

"""
HELICOPTER ENV TAKEOFF — yerden kalkış, hover ve iniş; PPO dört kumandayı DOĞRUDAN kullanır
===========================================================================================

Mentor isteği (2026-09-24): rotor warm-up'tan sonra yerden kalkıp 1000 ft'e kadar istenen irtifada hover.
Ajan dört kumandanın dördünü de öğrenir — collective, longitudinal cyclic (elevator), lateral cyclic (aileron),
pedal (rudder); hiçbiri AFCS'ye bırakılmaz ya da yetkisi daraltılmaz. **Kullanıcı seçimi:** doğrudan kumanda,
AFCS yalnızca SAS (rate damping; gerçek AH-1S'teki SCAS gibi), attitude / heading hold yok.

Neden dördü de gerekli (probe, 2026-09-24): cyclic ve pedal hover trimde sabit, collective 0 → 0.62 rampası →
kalkıştan 12 s sonra heading +81° dönmüş (yaw 11°/s), helikopter pad'den ~220 ft sürüklenmiş. Dikey kalkışta da
tork değişimi pedalla, tail rotor itkisinin yarattığı yana kayma (translating tendency) lateral cyclic'le,
burun salınımı longitudinal cyclic'le karşılanmalı.

Action (4) — a ∈ [−1, 1]:
  a → alçak geçiren filtre (α) → expo  y = 0.2·a + 0.8·a³ (merkezde ince, uçta tam yetki)
    → kumanda = hover trim + aralık·y, kırpma (collective [0, 1], diğerleri [−1, 1])
    → kumanda hızı sınırı (collective 0.6/s, cyclic / pedal 3/s).
  a = 0 → OGE hover trimi (8500 lbs, probe): collective 0.603, elevator −0.151, aileron 0.192, rudder 0.410.
  a = ±1 → kumandanın bütün hareket aralığı. Yerde başlangıçta collective 0 (flat pitch), cyclic / pedal trimde.

Observation (29): hedef konum hatası burun eksenine göre (ileri, sağa; ince + kaba), irtifa hatası (ince + kaba),
heading hatası (ince + kaba), takvim gecikmesi (konum, irtifa, heading), τ = geçen / T, T, yer hızı (ileri, yana),
dikey hız, φ, θ, p, q, r, kızak yüksekliği (log), rotor devri, kızaklardaki ağırlık oranı, iniş bayrağı,
kumandaların o anki konumu (a uzayında, 4).

Görevler (`takeoff_curriculum.py`): kalkış, hover tut, yerinde dönüş, yer değiştirme (reposition), bob-up / down,
hedef değişikliği (tırmanırken yeni irtifa, ölçülen duruma göre), iniş, bozucu sonrası toparlanma. Her görevin süre
hedefi T; başarı = son sınırdan önce hover bandına girip tutma süresi boyunca kalmak + kuplaj + güvenlik.
Hover bandı: konum ±6…12 ft, irtifa ±3…10 ft (hedef irtifaya göre), heading ±3°, yatay hız ≤ 2 ft/s, dikey ≤ 2.
İniş bandı: dört kızak noktası yerde, ağırlığın ≥ %70'i kızaklarda (collective indirilmiş), yer hızı ≤ 1 ft/s,
hedefe ≤ 8 ft, heading ±5°, 3 s; penceredeki en sert temas ≤ 4 ft/s.

Başlangıç: yerde (warm-up sonrası), havada (reset'e özel PID ile oturtulmuş hover) ve eğitimde iniş için hedefe yakın
başlangıçlar (reverse curriculum): "touch" (yerde, collective kısmen kalkık) ve "low" (kızaklar 1.5–4 ft).

Reward (ölçek 0.1): hedef çekirdekleri (konum, irtifa, heading) + yönlendirme çekirdekleri (hedefe doğru istenen
yatay / dikey / yaw hızı) + ilerleme − takvim gecikmesi − süre aşımı − aşırı açı / oran − kumanda hızı ve doygunluk −
kuplaj (komut verilmeyen eksen, sınırın %25'inden) − sert temas − yere yakınken fazla alçalma hızı + görev başarı
ödülü. İniş penceresinde: alçalma profili aşımı ve pad'e hizalanma cezası; temas yumuşaksa (≤ 3 ft/s tam, ≥ 5 ft/s
yok) dört nokta temas + kızaklardaki ağırlık + iniş bandında olma + collective'i flat pitch'e indirme (action
uzayında) ödülleri; yerdeyken konum çekimi yok. Tasarımın nasıl bulunduğu: README bölüm 30.4.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from helicopter_env_command import CONTROL_DT, AfcsMode, HelicopterEnvCommand, wrap_deg
# tork göstergesi / güç tavanı / tork cezası ortak fizik katmanında (2026-09-29; ifadeler aynı, sonuçlar birebir)
from physics_ext import PhysicsExt, psi_to_throttle, torque_penalty, torque_psi  # noqa: F401  (dışarıya da açık)
from takeoff_curriculum import (DEFAULT_TAKEOFF_LEVELS, GROUND_H_FT, LAND_PROFILE_K, LAND_PROFILE_MIN_FPS, MAX_HOVER_H_FT,
                                MAX_TARGET_H_FT, depart_time_target,
                                MIN_HOVER_H_FT, TakeoffLevel, deadline_of, find_takeoff_level)

OBS_DIM_T = 29                                    # + 1 (tork göstergesi) TakeoffEnvConfig.torque_obs ile
R_EARTH_FT = 20_902_231.0
# Repo'daki uçak kopyası (2026-09-28): kalibre yer etkisi (Systems/ground_effect.xml) + güç tavanı (Systems/rpm_governor.xml).
REPO_AIRCRAFT_DIR = str(Path(__file__).resolve().parent / "aircraft")
GE_MODELS = {"calibrated": 1.0, "stock": 0.0}     # "off": ge/enable = 0
ELECTRIC_HP = 1500.0                              # Engines/electric_1500hp.xml (T53 yerine)
KT_FPS = 1.68781
NOMINAL_RPM = 324.0
HOVER_TRIM = (0.603, -0.151, 0.192, 0.410)       # OGE hover, 8500 lbs, SAS (probe 2026-09-24)
CTRL_LO = np.array([0.0, -1.0, -1.0, -1.0])
CTRL_HI = np.array([1.0, 1.0, 1.0, 1.0])
ACTIVE_AXES = {                                   # komut verilen eksenler (kuplaj diğerlerine bakar)
    "hold": ("xy", "h", "psi"), "recover": ("xy", "h", "psi"),
    "takeoff": ("h",), "climb_to": ("h",), "bob": ("h",), "turn": ("psi",), "move": ("xy",), "land": ("h",),
    "depart": ("xy", "h", "psi")}


def expo(x, e: float):
    return (1.0 - e) * x + e * x ** 3


def expo_inv(y, e: float):
    """y = (1−e)·x + e·x³ → x (tek gerçek kök, Cardano)."""
    y = np.asarray(y, dtype=np.float64)
    if e <= 0.0:
        return y
    p, q = (1.0 - e) / e, -y / e
    d = np.sqrt(q * q / 4.0 + p ** 3 / 27.0)
    return np.cbrt(-q / 2.0 + d) + np.cbrt(-q / 2.0 - d)


@dataclass
class TakeoffEnvConfig:
    # --- kumanda ---------------------------------------------------------------------
    hover_trim: tuple = HOVER_TRIM
    ctrl_range: tuple = (0.6, 1.0, 1.0, 1.0)      # a = ±1 → trim ± aralık (sonra kırpma → tam hareket aralığı)
    expo: float = 0.8
    action_filter_alpha: float = 0.5
    rate_limit: tuple = (0.6, 3.0, 3.0, 3.0)      # kumanda birimi / s
    # --- başarı ----------------------------------------------------------------------------
    tol_h: tuple = (3.0, 0.01, 10.0)              # clip(a + b·h*, a, c): 3 ft (alçak) … 10 ft (1000 ft)
    tol_xy: tuple = (6.0, 0.007, 12.0)
    tol_psi_deg: float = 3.0
    tol_v_fps: float = 2.0
    tol_vs_fps: float = 2.0
    land_tol_xy_ft: float = 8.0
    land_tol_psi_deg: float = 5.0
    land_weight_frac: float = 0.7                 # ≈ collective ≤ 0.2 (yerde ölçüldü: 0.3 → %51, 0.2 → %70, 0 → %98)
    land_hold_s: float = 3.0
    touchdown_ok_fps: float = 4.0
    coupling_limits: tuple = (20.0, 15.0, 10.0)   # komut verilmeyen eksen: konum ft, irtifa ft, heading °
    gap_s: tuple = (0.5, 1.5)                     # görev bitince bir sonrakine kadar
    end_margin_s: float = 1.0
    max_episode_s: float = 600.0                  # K9: 1000 ft kalkış + manevralar + 1000 ft iniş (profil ~210 s) sığsın
    # --- reward ---------------------------------------------------------------------------
    w_xy: float = 1.0
    w_h: float = 1.0
    w_psi: float = 0.7
    w_v: float = 0.5
    w_vs: float = 0.5
    w_r: float = 0.3
    kernel_xy: tuple = (3.0, 25.0)
    kernel_h: tuple = (3.0, 30.0)
    kernel_psi: tuple = (2.0, 15.0)
    kernel_v: tuple = (1.5, 6.0)
    kernel_vs: tuple = (1.5, 6.0)
    kernel_r: tuple = (3.0, 15.0)
    guide_k_xy: float = 0.25                      # istenen yatay hız = k · konum hatası (1/s)
    guide_k_h: float = 0.3
    guide_k_psi: float = 0.6
    hold_v_max: float = 3.0                       # konum tutarken istenen yatay hızın üst sınırı
    progress_weight: float = 10.0
    progress_min_scale: tuple = (10.0, 10.0, 10.0)
    pen_sched: float = 1.0
    sched_scale: tuple = (20.0, 20.0, 20.0)
    pen_late: float = 0.5
    pen_att: float = 1.0                          # · ((|φ|−20°)/10°)² + ((|θ|−15°)/10°)²
    pen_rate: float = 0.05
    pen_dctrl: float = 2.0                        # · mean((Δkumanda / aralık / (hız sınırı·dt))²)
    pen_sat: float = 0.5
    sat_threshold: float = 0.95
    # komut verilmeyen eksen: sınırın %25'inden itibaren (konum 5 ft, irtifa 3.75 ft, heading 2.5°) karesel ceza.
    # (0.5 / 2.0 ile uzun dikey geçişlerde konum ~0.5 ft/s sürükleniyordu: 700 ft alçalmada 20–25 ft → kuplaj hatası)
    pen_couple: float = 3.0
    couple_soft_frac: float = 0.25
    pen_touchdown: float = 25.0                   # temas: · ((|ḣ| − 2) / 3)²  (3 ft/s → 2.8, 4 → 11, 5 → 25, 6 → 44)
    touchdown_free_fps: float = 2.0
    # görev penceresi başarıyla bitince (süre hedefi + tutma + kuplaj + iniş temas hızı) tek seferlik ödül:
    # başarı ölçütünü doğrudan ödüllendirir. (to_v6: yerdeki adım başına ödül yüzünden ajan son 1–2 ft'i
    # 4–5 ft/s ile "düşüyordu"; ölçüt ≤ 4 ft/s.)
    w_task_success: float = 30.0
    # yere yakınken alçalma hızı sınırı (flare): izin = 1.5 + 0.2·kızak yüksekliği ft/s (40 ft'te 9.5, 10 ft'te 3.5)
    pen_sink: float = 3.0                         # · ((−ḣ − izin) / 3)², kızak yüksekliği < 60 ft iken
    sink_base_fps: float = 1.5
    sink_slope: float = 0.2
    # iniş penceresinde alçalma profili: istenen alçalma hızını (≤ seviyenin descent_fps'i) 2 ft/s'den fazla aşma cezası.
    # (to_v11 K8: ajan 200+ ft'ten ~15 ft/s dalıp geç frenliyordu; ağır helikopterde fren yetmiyor → 5 ft/s temas.
    # İndirimli getiri (γ = 0.995) yere erken varmayı — başarı ödülü, yerdeki ödül — kârlı kılıyor.)
    pen_land_sink: float = 3.0                    # · ((−ḣ − (|ḣ_istenen| + 2)) / 3)²
    # iniş penceresinde havadayken pad'e hizalanma: · ((konum hatası − 2 ft) / 4)²  (8 ft'te 2.25). Yavaş alçalmada
    # (to_v12) sürüklenme ~11 ft'e çıkıp temas pad'in 8 ft toleransı dışında oluyordu.
    pen_land_xy: float = 0.5
    w_land: float = 2.5                           # iniş penceresinde: 0.3·(dört nokta yerde) + 0.4·(kızaklardaki ağırlık) + 0.3·(bantta)
    soft_full_fps: float = 3.0                    # iniş ödülleri (w_land, w_coll_down) × yumuşaklık: en sert temas ≤ 3 ft/s → 1,
    soft_zero_fps: float = 5.0                    # ≥ 5 ft/s → 0 (başarı ölçütü ≤ 4 ft/s)
    # iniş penceresinde, kızaklar yere değmişken: collective'i IGE hover triminin altına indirme ödülü — action
    # uzayında (a) doğrusal. Neden: to_v2–v4 K7a'da ajan yumuşak temas edip (−2 ft/s) sekti ve 1–2 ft'te IGE hover'da
    # kaldı; yerde hafif yüklü başlasa bile collective'i kaldırıp havalandı. Expo (y = 0.2a + 0.8a³) yüzünden trim
    # çevresinde collective çok az değişiyor; kızaklarda %70 ağırlık (collective ~0.2) a ≈ −0.9 ister, keşif gürültüsü
    # (σ ≈ 0.2) oraya ulaşmıyor. Kumandanın a-uzayındaki konumuna doğrusal ödül sabit bir eğim verir.
    w_coll_down: float = 1.5                      # · clip((a_IGE − a_coll) / (a_IGE − a_flat), 0, 1)
    # hedef: collective tam aşağı (flat pitch, gerçek iniş prosedürü). to_v13'te hedef "IGE trim − 0.35" idi; ağır
    # helikopterde %70 ağırlık için bu yetmiyor, ajan collective'i 20 s boyunca yavaşça indirip son sınırı kaçırıyordu.
    coll_flat: float = 0.05
    pen_slide: float = 0.2                        # yerdeyken yatay hız (ft/s)
    # yerinde dönüşte yaw hızı sınırı (2026-10-01): |r| > turn_rate_cap_frac · 1.25 · seviyenin yaw_rate_dps'i (F: 15 → 28 °/s)
    # üstü · pen_turn_rate · ((|r| − sınır) / 10)². Neden: dönüş ödülü (indirimli ilerleme + heading çekirdeği) hızlı dönmeyi
    # kârlı kılıyordu; ajan 180°'yi ~5 s'de 40–44 °/s ile dönüp konumu 15–25 ft kaydırıyordu (ADS-33 istenen 180°/10 s,
    # ±3 ft). 0 → kapalı (eski modeller).
    pen_turn_rate: float = 0.0
    turn_rate_cap_frac: float = 1.5
    fail_penalty: float = 50.0
    reward_scale: float = 0.1
    # --- güvenlik ------------------------------------------------------------------------------
    max_roll_deg: float = 45.0
    max_pitch_deg: float = 40.0
    max_rate_dps: float = 100.0
    max_yaw_rate_dps: float = 90.0
    ground_max_att_deg: float = 15.0              # yerdeyken (devrilme / dynamic rollover)
    crash_vs_fps: float = 10.0                    # temas anında bundan hızlı iniş → kaza
    flyaway_ft: float = 150.0
    alt_over_ft: float = 200.0
    max_speed_fps: float = 60.0
    rpm_limits: tuple = (280.0, 380.0)
    # --- reset --------------------------------------------------------------------------------
    settle_max_s: float = 60.0
    settle_hold_s: float = 2.0
    ground_settle_s: float = 1.0
    # --- uçak / fizik (2026-09-28) --------------------------------------------------------------------
    # "stock": JSBSim paketindeki ah1s (eski modellerin eğitildiği fizik, birebir). "repo": aircraft/ah1s kopyası.
    aircraft: str = "stock"
    ground_effect: str = "calibrated"             # yalnızca "repo": "calibrated" (Hayden + C-B) | "stock" | "off"
    power_cap_psi: float = 0.0                    # yalnızca "repo": motor gücü tavanı, bu torkun nominal devirdeki
                                                  # gücü (0 → tavan yok = 1500 hp). Aşılmak istenirse rotor devri düşer.
    # --- tork (modelin kendi göstergesi) --------------------------------------------------------------
    torque_obs: bool = False                      # obs'un sonuna (psi − 50) / 10 → OBS_DIM_T + 1
    torque_cont_psi: float = 50.0                 # el kitabı: sürekli
    torque_max_psi: float = 56.0                  # el kitabı: %100 (30 dk / kalkış gücü)
    pen_torque_cont: float = 0.0                  # 50–56 psi: · ((psi − 50) / 6)²   (kalkış gücü bölgesi, hafif)
    pen_torque_over: float = 0.0                  # 56 psi üstü: · min(((psi − 56) / 3)², 9)
    rpm_low_warn: float = 314.0                   # %97: güç tavanında collective fazla çekilirse rotor devri düşer
    pen_rpm_low: float = 0.0                      # · ((314 − rpm) / 10)²
    # tırmanış süre hedefi ve yönlendirmesi ağırlığa göre (güç payı): v_max ≈ 0.8 · (56 − psi_hover(W)) / 0.62 ft/s,
    # psi_hover(W) = 44.6 + 7.5 · (W − 8500) / 1000 (OGE hover, ölçüldü 2026-09-28; +0.62 psi / (ft/s) tırmanış)
    torque_aware_climb: bool = False
    # 2026-10-01 (F10 ölçümü: 1000 → 1500 ft hover tırmanışında 56 psi üstü 25 s; hover gücü irtifa / sıcaklıkla artıyor,
    # yönlendirme 1.25 × %80 pay = payın %100'ünü istiyordu):
    #   torque_density_climb: tırmanış hızı tahmininde hover torku hava yoğunluğuyla düzeltilir (+15 psi · (σ_ref − σ)/σ_ref;
    #     σ_ref = Edwards standart günü 0.935; ölçüm: 9700 lbs'de yoğunluk irtifası 2290 → 5373 ft için +1.3 psi)
    #   torque_feedback_climb: istenen tırmanış hızı ölçülen torkun (1 s süzülmüş) payıyla da sınırlı:
    #     ḣ_istenen ≤ ḣ + (56 − 1.5 − psi_süzülmüş)/0.62 — denge 54.5 psi; tork sınıra yaklaşınca yönlendirme tırmanışı
    #     azaltmayı ister (ilk denemede 0.8·(56 − psi) ile zayıftı: 56.5 psi'de istenen yalnızca 0.65 ft/s daha az)
    torque_density_climb: bool = False
    torque_feedback_climb: bool = False
    # --- hareketli hedef (ileri kalkış "depart", 2026-09-28) ------------------------------------------------------
    track_obs: bool = False                       # obs'un sonuna hedefin hızı (burun ekseninde ileri / sağa, /20) → +2
    track_tol_xy: float = 25.0                    # depart bandı: hareketli hedef noktasına uzaklık
    track_tol_v: float = 4.0                      # depart bandı: yer hızı vektörü − hedef hızı (ft/s)
    max_speed_track_fps: float = 130.0            # depart sırasında hız güvenlik sınırı (~77 kt)
    flyaway_track_ft: float = 300.0               # depart sırasında hareketli hedeften en büyük uzaklık (yakalama payı)
    # hedef noktası helikopterin rota üzerindeki izdüşümünden en fazla bu kadar önde / arkada (profil noktasına doğru):
    # hata eğitimdeki "move" görevleri kadar kalır (dağılım dışına çıkmaz); hızı hedef hız gözlemi ve rehberlik söyler.
    # Başarı yine profil noktasına ≤ track_tol_xy ister (geride kalan yakalamalı).
    track_lead_ft: float = 60.0
    kernel_v_track: tuple = (3.0, 20.0)           # depart: hız rehberliği çekirdeği (geniş: 20–30 ft/s farkta da eğim)
    w_v_track: float = 1.0                        # depart: hız rehberliği ağırlığı (w_v yerine)
    w_track_lag: float = 5.0                      # depart: profil noktasına göre gerideliğin azalması / artması (ft / 10)
    # --- ortak fizik katmanı (2026-09-29, physics_ext.py): yakıt tüketimi, rüzgâr / gust / türbülans ------------------
    # None → kapalı (eski modeller birebir). Ör. {"fuel": {"enable": True}, "wind": {"enable": True, "speed_kt": [0, 15]},
    # "turb": {"enable": True, "levels": ["none", "light"]}} — PhysicsExtConfig.from_dict; modelle birlikte kaydedilir.
    physics: dict | None = None
    fuel_exhausted_ends: bool = True              # yakıt bitip motor ayrılınca bölüm biter (kesme, başarısızlık cezası yok)


class HelicopterEnvTakeoff(HelicopterEnvCommand):
    """Yerde (warm-up sonrası) ya da havada başlayan kalkış / hover / iniş env'i; 4 kumanda doğrudan."""
    metadata = {"render_modes": []}

    def __init__(self, level=0, levels=None, config: TakeoffEnvConfig | None = None, rehearsal: bool = False):
        gym.Env.__init__(self)
        self.levels: list[TakeoffLevel] = list(levels or DEFAULT_TAKEOFF_LEVELS)
        self.cfg = config or TakeoffEnvConfig()
        self.level_index = find_takeoff_level(level, self.levels)
        # tekrar (rehearsal): eğitimde seviyenin p_rehearse kadarı eski seviyelerden — unutmayı önler
        # (to_v1 / to_v2: yalnızca iniş seviyesinde eğitince K5 %100 → %65, 300 ft kalkış yavaşladı)
        self.rehearsal = bool(rehearsal)
        if self.cfg.aircraft not in ("stock", "repo"):
            raise ValueError(f"aircraft: 'stock' ya da 'repo' ({self.cfg.aircraft})")
        if self.cfg.aircraft == "repo" and self.cfg.ground_effect not in (*GE_MODELS, "off"):
            raise ValueError(f"ground_effect: {sorted(GE_MODELS)} ya da 'off' ({self.cfg.ground_effect})")
        if self.cfg.aircraft == "stock" and self.cfg.power_cap_psi > 0.0:
            raise ValueError("power_cap_psi yalnızca aircraft='repo' ile (governor gaz tavanı repo kopyasında)")
        self.obs_dim = OBS_DIM_T + int(bool(self.cfg.torque_obs)) + 2 * int(bool(self.cfg.track_obs))
        self.ext = PhysicsExt(self.cfg.physics, CONTROL_DT)     # kapalıyken hiçbir şey yapmaz (RNG'ye de dokunmaz)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.fdm = None
        self._fdm_needs_refresh = True
        self._episodes_on_fdm = 0
        self.trim = np.asarray(self.cfg.hover_trim, dtype=np.float64)
        self.rng_ctrl = np.asarray(self.cfg.ctrl_range, dtype=np.float64)
        self.steps, self.max_steps = 0, 1
        self.filt = np.zeros(4)
        self.ctrl = self.trim.copy()
        self._dctrl = np.zeros(4)
        self.prev_action = np.zeros(4)
        self.windows: list[dict] = []
        self.pending: list[dict] = []
        self.fixed_schedule = False
        self.next_issue_t = 0.0
        self.episode_end_t = np.inf
        self.failure: str | None = None
        self.setup_info: dict = {}
        self.auto_end = True
        self.target = dict(n=0.0, e=0.0, h=GROUND_H_FT, psi=0.0)
        self.kick = None
        self.track = None                         # depart: hareketli hedef (başlangıç, rota, hız, ivme, t0)
        self.psi_unwrap = 0.0
        self._psi_prev_meas = 0.0
        self._lat0 = self._lon0 = 0.0
        self._prev_wow = 4
        self._prev_vs = 0.0
        self._airborne_once = False
        self.prev_abs_err = {"xy": 0.0, "h": 0.0, "psi": 0.0}
        self._last_err = None
        self._ep_level = self.level_index          # episode'un başladığı seviye (seviye atlaması episode ortasında olursa)

    # =================================================================
    # CURRICULUM API
    # =================================================================

    def set_level(self, level) -> int:
        self.level_index = find_takeoff_level(level, self.levels)
        return self.level_index

    def get_level(self) -> int:
        return self.level_index

    @property
    def level(self) -> TakeoffLevel:
        return self.levels[self.level_index]

    @property
    def ep_level(self) -> TakeoffLevel:
        """Episode'un başladığı seviye (görevler ve süre hedefleri bununla)."""
        return self.levels[self._ep_level]

    def climb_fps(self, lv: TakeoffLevel | None = None) -> float:
        """Tırmanış hızı (süre hedefi / yönlendirme): seviyeninki; torque_aware_climb ile ağırlığın güç payına göre."""
        lv = lv or self.ep_level
        if not self.cfg.torque_aware_climb:
            return lv.climb_fps
        w = float(self.setup_info.get("weight", 8500.0)) if self.setup_info else 8500.0
        psi_hover = 44.6 + 7.5 * (w - 8500.0) / 1000.0 + self._density_psi()
        return float(np.clip(0.8 * (self.cfg.torque_max_psi - psi_hover) / 0.62, 1.5, lv.climb_fps))

    def _density_psi(self) -> float:
        """Hover torkunun hava yoğunluğu düzeltmesi (psi; torque_density_climb kapalıyken 0)."""
        if not self.cfg.torque_density_climb or self.fdm is None:
            return 0.0
        sigma = float(self.fdm["atmosphere/rho-slugs_ft3"]) / 0.0023769
        return 15.0 * (0.935 - sigma) / 0.935

    def _update_psi_f(self, s: dict):
        a = 1.0 - math.exp(-CONTROL_DT / 1.0)
        psi_f = getattr(self, "_psi_f", None)
        self._psi_f = s["torque_psi"] if psi_f is None else psi_f + a * (s["torque_psi"] - psi_f)

    def _climb_feedback_cap(self, s: dict, up: float) -> float:
        """torque_feedback_climb: istenen tırmanışın üst sınırı, ölçülen torkun (1 s süzülmüş) payıyla."""
        if not self.cfg.torque_feedback_climb:
            return up
        psi_f = getattr(self, "_psi_f", None)
        psi_f = s["torque_psi"] if psi_f is None else psi_f
        # denge: süzülmüş tork sınırın 1.5 psi altında (56 → 54.5); üstündeyse mevcut tırmanıştan daha azını ister
        return min(up, max(1.5, s["vs"] + (self.cfg.torque_max_psi - 1.5 - psi_f) / 0.62))

    # =================================================================
    # JSBSim yardımcıları
    # =================================================================

    def _aircraft_dir(self) -> str | None:
        return REPO_AIRCRAFT_DIR if self.cfg.aircraft == "repo" else None

    def _configure_aircraft(self, fdm):
        cfg = self.cfg
        if cfg.aircraft != "repo":
            return
        fdm["ge/model"] = GE_MODELS.get(cfg.ground_effect, 1.0)
        fdm["ge/enable"] = 0.0 if cfg.ground_effect == "off" else 1.0
        fdm["fcs/throttle-max-norm"] = float(np.clip(psi_to_throttle(cfg.power_cap_psi), 0.05, 1.0)) \
            if cfg.power_cap_psi > 0.0 else 1.0

    def _ige_coll_offset(self) -> float:
        """Yere çok yakın (kızak 1–2 ft) hover triminin stok fiziğe göre farkı: kalibre yer etkisi daha güçlü →
        aynı ağırlık için ~0.04 daha az collective (ölçüldü 2026-09-28, 8500 ve 10280 lbs). 'touch' başlangıcının
        collective'i ve iniş ödülündeki IGE trimi buna göre kaydırılır."""
        if self.cfg.aircraft != "repo":
            return 0.0
        return {"calibrated": -0.04, "stock": 0.0, "off": 0.045}.get(self.cfg.ground_effect, 0.0)

    def _ige_coll(self, weight: float) -> float:
        """Yere çok yakın hover'ın collective'i (iniş ödülünde collective'i bunun altına indirme): OGE hover trimi − 0.04,
        ağırlıkla +0.057 / 1000 lbs (probe), kalibre yer etkisi farkı. (Alt sınıf trimi hava hızı / ağırlık tablosundan
        alıyorsa bunu ezer.)"""
        return self.trim[0] - 0.04 + 0.057 * (weight - 8500.0) / 1000.0 + self._ige_coll_offset()

    def _set_sas(self, psi_deg: float):
        """AFCS: roll / pitch yalnızca oran sönümleme, yaw sönümleme; attitude ve heading hold kapalı."""
        f = self.fdm
        self._set_afcs(AfcsMode(1.0, 1.0, 0.99, False, 0.2), psi_deg)
        f["ap/afcs/adj/roll-err-ctrl-gain"] = 0.0
        f["ap/afcs/adj/pitch-err-ctrl-gain"] = 0.0
        f["fcs/automatic/steady-flight-data-enable"] = 0.0

    def _run_plain(self) -> bool:
        self.ext.before_step(self.fdm)                         # rüzgâr (run_ic siliyor) / türbülans / gust
        return self._run_physics()

    def _nav(self) -> dict:
        f = self.fdm
        lat, lon = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        n = (lat - self._lat0) * R_EARTH_FT
        e = (lon - self._lon0) * R_EARTH_FT * math.cos(self._lat0)
        vn, ve = float(f["velocities/v-north-fps"]), float(f["velocities/v-east-fps"])
        return dict(n=n, e=e, vn=vn, ve=ve)

    def _state(self) -> dict:
        f = self.fdm
        s = self._read_state()
        s.update(self._nav())
        ps = math.radians(s["psi_deg"])
        s["ug"] = s["vn"] * math.cos(ps) + s["ve"] * math.sin(ps)
        s["vg"] = -s["vn"] * math.sin(ps) + s["ve"] * math.cos(ps)
        s["vh"] = math.hypot(s["vn"], s["ve"])
        s["wow"] = int(sum(float(f[f"gear/unit[{i}]/WOW"]) > 0.5 for i in range(4)))
        s["tail"] = float(f["contact/unit[4]/WOW"]) > 0.5
        w = max(1.0, float(f["inertia/weight-lbs"]))
        s["weight"] = w
        s["wfrac"] = float(np.clip(-float(f["forces/fbz-gear-lbs"]) / w, 0.0, 1.5))
        s["hs"] = s["h"] - GROUND_H_FT
        s["torque_psi"] = float(torque_psi(float(f["propulsion/engine/torque-lbsft"])))
        return s

    def _controls_now(self) -> np.ndarray:
        f = self.fdm
        return np.array([float(f[f"fcs/{k}-cmd-norm"]) for k in ("collective", "elevator", "aileron", "rudder")])

    def _set_fuel(self, fuel):
        f = self.fdm
        f["propulsion/tank[0]/contents-lbs"] = float(fuel[0])
        f["propulsion/tank[1]/contents-lbs"] = float(fuel[1])

    def _set_ic_from_state(self, dn=0.0, de=0.0, dw=0.0, dphi=0.0, dtheta=0.0, h=None, psi_deg=None, still=False):
        """Dönen rotorla birlikte konum / hız / attitude'u IC'den yeniden kur (teleport ya da bozucu itki).
        dn, de: yer hızı eklemesi (kuzey / doğu, ft/s); dw: gövde dikey hız; dphi / dtheta: derece."""
        f = self.fdm
        psi = math.radians(psi_deg) if psi_deg is not None else float(f["attitude/psi-rad"])
        u, v, w = (0.0, 0.0, 0.0) if still else (float(f["velocities/u-fps"]), float(f["velocities/v-fps"]),
                                                 float(f["velocities/w-fps"]))
        u += dn * math.cos(psi) + de * math.sin(psi)
        v += -dn * math.sin(psi) + de * math.cos(psi)
        w += dw
        vals = {"ic/lat-geod-rad": float(f["position/lat-geod-rad"]), "ic/long-gc-rad": float(f["position/long-gc-rad"]),
                "ic/h-agl-ft": float(h if h is not None else f["position/h-agl-ft"]),
                "ic/phi-rad": (0.0 if still else float(f["attitude/phi-rad"])) + math.radians(dphi),
                "ic/theta-rad": (0.0 if still else float(f["attitude/theta-rad"])) + math.radians(dtheta),
                "ic/psi-true-rad": psi % (2.0 * math.pi),
                "ic/u-fps": u, "ic/v-fps": v, "ic/w-fps": w,
                "ic/p-rad_sec": 0.0 if still else float(f["velocities/p-rad_sec"]),
                "ic/q-rad_sec": 0.0 if still else float(f["velocities/q-rad_sec"]),
                "ic/r-rad_sec": 0.0 if still else float(f["velocities/r-rad_sec"])}
        for k, x in vals.items():
            f[k] = x
        if not f.run_ic():
            raise RuntimeError("run_ic() başarısız")

    def _teleport_ground(self, psi_deg: float):
        c = np.r_[0.0, self.trim[1:]]                  # collective flat pitch, cyclic / pedal trimde (pilot nötr)
        self._write_controls(c)
        self._set_ic_from_state(h=GROUND_H_FT, psi_deg=psi_deg, still=True)
        self._set_sas(psi_deg)
        t = 0.0
        while t < self.cfg.ground_settle_s:
            self._write_controls(c)
            if not self._run_plain():
                raise RuntimeError("JSBSim yerde durdu")
            t += CONTROL_DT
        s = self._state()
        if s["rpm"] < 300.0 or s["wow"] < 4:
            raise RuntimeError(f"yerde başlangıç kurulamadı (rpm {s['rpm']:.0f}, kızak {s['wow']})")

    def _raise_collective(self, c_target: float):
        """'touch' başlangıcı: yerdeyken collective'i 1 s'de c_target'a kaldır, 0.5 s bekle (kızaklar hafif yüklü;
        inişin son aşaması gibi). Cyclic / pedal hover triminde."""
        c = np.r_[0.0, self.trim[1:]]
        t = 0.0
        while t < 1.5 - 1e-9:
            c[0] = float(np.clip(c_target * min(1.0, t / 1.0), 0.0, 1.0))
            self._write_controls(c)
            if not self._run_plain():
                raise RuntimeError("JSBSim yerde durdu (touch)")
            t += CONTROL_DT
        s = self._state()
        if s["wow"] < 2 or s["vh"] > 2.0:
            raise RuntimeError(f"touch başlangıcı kurulamadı (kızak {s['wow']}, yer hızı {s['vh']:.1f})")

    def _settle_hover(self, h0: float, psi0: float, fuel) -> tuple[bool, dict]:
        """Yalnızca reset: havada başlangıç için konum / irtifa / heading tutan PID (teacher değil; policy'ye action
        önermez, verisi eğitimde kullanılmaz). Kararlı olunca kumanda konumunu döndürür."""
        cfg = self.cfg
        coll0 = self.trim[0] + 0.057 * (sum(fuel) / 1000.0)            # ağırlığa göre ileri besleme (probe)
        c = np.array([coll0, *self.trim[1:]])
        self._write_controls(c)
        self._set_ic_from_state(h=h0, psi_deg=psi0, still=True)
        self._set_sas(psi0)
        integ = np.zeros(4)
        t, hold = 0.0, 0.0
        # 2. evre (2026-10-01; yalnızca 1. evre başarısızsa, yani eskiden kurulamayan başlangıçlarda): güçlü yan rüzgârda
        # pedal integratörü ±0.3'te doyuyor (25 kt'ta 2.7–3.5° kalıcı heading hatası) ya da yanal döngü ~1 ft/s'lik yavaş
        # salınımda kalıyor (15 kt soldan) → integratör ±0.6, ölçütler rüzgârla gevşer, 30 s daha. 1. evre birebir eski.
        wind = float(self.ext.params.get("wind_kt", 0.0)) if self.ext.armed else 0.0
        t_end, relaxed = cfg.settle_max_s, False
        while t < t_end or (not relaxed and wind > 0.0):
            if t >= t_end:
                relaxed, t_end, hold = True, t_end + 30.0, 0.0
            s = self._state()
            ps = math.radians(s["psi_deg"])
            dn, de = -s["n"], -s["e"]
            xf, yr = dn * math.cos(ps) + de * math.sin(ps), -dn * math.sin(ps) + de * math.cos(ps)
            eu = float(np.clip(0.12 * xf, -6, 6)) - s["ug"]
            ev = float(np.clip(0.12 * yr, -6, 6)) - s["vg"]
            evs = float(np.clip(0.4 * (h0 - s["h"]), -6, 6)) - s["vs"]
            eps = wrap_deg(psi0 - s["psi_deg"])
            integ[0] = np.clip(integ[0] + 0.01 * evs * CONTROL_DT, -0.3, 0.3)
            integ[1] = np.clip(integ[1] - 0.004 * eu * CONTROL_DT, -0.3, 0.3)
            integ[2] = np.clip(integ[2] + 0.004 * ev * CONTROL_DT, -0.3, 0.3)
            integ[3] = np.clip(integ[3] - 0.003 * eps * CONTROL_DT, -(0.6 if relaxed else 0.3), 0.6 if relaxed else 0.3)
            th_ref = float(np.clip(-0.02 * eu, -0.25, 0.25)) + integ[1]
            ph_ref = float(np.clip(0.03 * ev, -0.25, 0.25)) + integ[2]
            c = np.array([coll0 + 0.03 * evs + integ[0],
                          self.trim[1] + 3.0 * (s["theta"] - th_ref) + 1.6 * s["q"],
                          self.trim[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"],
                          self.trim[3] - 0.02 * eps + 0.8 * s["r"] + integ[3]])
            c = np.clip(c, CTRL_LO, CTRL_HI)
            self._write_controls(c)
            if not self._run_plain():
                return False, dict(reason="jsbsim_stopped")
            t += CONTROL_DT
            v_tol, p_tol, psi_tol = (0.5 + 0.04 * wind, 2.0 + 0.1 * wind, 2.0) if relaxed else (0.5, 2.0, 1.0)
            ok = (abs(h0 - s["h"]) < 1.5 and abs(s["vs"]) < 0.5 and s["vh"] < v_tol and math.hypot(s["n"], s["e"]) < p_tol
                  and abs(eps) < psi_tol and max(abs(s["p"]), abs(s["q"]), abs(s["r"])) < 0.02 and s["wow"] == 0)
            hold = hold + CONTROL_DT if ok else 0.0
            if hold >= cfg.settle_hold_s:
                return True, dict(settle_s=t, ctrl=c.copy(), relaxed=relaxed)
        s = self._state()
        return False, dict(reason=f"hover kurulamadı ({t:.0f} s): h={s['h']:.1f} v={s['vh']:.1f}")

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
            ep_idx = find_takeoff_level(lv0.rehearse[int(rng.integers(len(lv0.rehearse)))], self.levels)
        lv = self.levels[ep_idx]
        st = lv.sample_start(rng)
        start = options.get("start", st["start"])
        if start not in ("ground", "hover", "touch", "low"):
            raise ValueError(f"bilinmeyen başlangıç: {start}")
        air = start in ("hover", "low")
        h0 = float(options.get("start_alt_ft", st["h"] if air else GROUND_H_FT))
        if air and "start_alt_ft" not in options and st["start"] != start:
            h0 = (float(rng.uniform(*lv.hover_start_alt_ft)) if start == "hover"
                  else GROUND_H_FT + float(rng.uniform(*lv.low_hover_hs_ft)))
        psi0 = float(options["start_heading_deg"]) % 360.0 if "start_heading_deg" in options else float(rng.uniform(0, 360))
        fuel = tuple(options["fuel"]) if options.get("fuel") is not None else lv.sample_fuel(rng)
        c_touch = None
        if start == "touch":                                    # yerde kısmen kalkık collective (ağırlığa göre)
            c_touch = (float(options.get("touch_coll", rng.uniform(*lv.touch_coll))) + 0.057 * sum(fuel) / 1000.0
                       + self._ige_coll_offset())
        self.ext.begin_episode(rng, psi0, options.get("physics"))
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
                               weight=float(self.fdm["inertia/weight-lbs"]), cg_x_in=float(self.fdm["inertia/cg-x-in"]))
        # havada başlangıç bozukluğu (K1): küçük hız / attitude sapması
        pert = float(options.get("start_perturb", lv.start_perturb)) if start == "hover" else 0.0
        if pert > 0.0:
            a = float(rng.uniform(0, 2 * math.pi))
            self._set_ic_from_state(dn=3.0 * pert * math.cos(a), de=3.0 * pert * math.sin(a),
                                    dw=float(rng.uniform(-1.5, 1.5)) * pert,
                                    dphi=float(rng.uniform(-3, 3)) * pert, dtheta=float(rng.uniform(-3, 3)) * pert)
        # reverse curriculum (ileri kalkış): havada başlayıp burun yönünde ileri giderken (hover attitude'unda)
        v0_kt = float(options["start_speed_kt"]) if "start_speed_kt" in options else (
            float(rng.uniform(*lv.start_speed_kt)) if start == "hover" and lv.start_speed_kt[1] > 0.0 else 0.0)
        if start == "hover" and v0_kt > 0.0:
            ps0 = math.radians(psi0)
            self._set_ic_from_state(dn=v0_kt * KT_FPS * math.cos(ps0), de=v0_kt * KT_FPS * math.sin(ps0))
        if start == "hover" and lv.depart_from_speed and "tasks" not in options:
            options["tasks"] = [dict(kind="depart", h=float(rng.uniform(*lv.depart_alt_ft)),
                                     v_kt=float(np.clip(v0_kt + rng.uniform(-8.0, 12.0), *lv.depart_kt)),
                                     accel=float(rng.uniform(*lv.depart_accel)), v0_kt=v0_kt)]
        self.setup_info["start_speed_kt"] = v0_kt
        return self._handover(lv, options, start, ep_idx)

    def _handover(self, lv: TakeoffLevel, options: dict, start: str, ep_idx: int | None = None):
        cfg, rng = self.cfg, self.np_random
        self._ep_level = self.level_index if ep_idx is None else int(ep_idx)
        self.ctrl = self._controls_now()
        self.filt = np.clip(expo_inv(np.clip((self.ctrl - self.trim) / self.rng_ctrl, -1.0, 1.0), cfg.expo), -1.0, 1.0)
        self._dctrl = np.zeros(4)
        self.prev_action = self.filt.copy()
        self.kick = None
        self.track = None
        s = self._state()
        self._set_sas(s["psi_deg"])
        self.psi_unwrap = s["psi_deg"]
        self._psi_prev_meas = s["psi_deg"]
        self._prev_wow = s["wow"]
        self._prev_vs = s["vs"]
        self._airborne_once = s["wow"] == 0
        self.target = dict(n=s["n"], e=s["e"], h=s["h"] if start in ("hover", "low") else GROUND_H_FT, psi=self.psi_unwrap)
        self.steps = 0
        self.windows, self.failure = [], None
        self.ext.start_disturbances(self.fdm)                   # fizik katmanı: türbülans / gust / yakıt sayacı başlar
        if self.ext.armed:
            self.setup_info["physics"] = dict(self.ext.params)
        if "tasks" in options:                                   # sabit görev listesi (değerlendirme / canlı)
            self.fixed_schedule = any("_t" in d for d in options["tasks"])
            self.pending = [dict(d) for d in options["tasks"]]
        else:
            self.fixed_schedule = False
            self.pending = lv.sample_schedule(rng, start)
        self.next_issue_t = float(self.pending[0].get("_t", 0.0)) if self.pending else np.inf
        self.max_steps = int(round(float(options.get("episode_s", cfg.max_episode_s)) / CONTROL_DT))
        self.episode_end_t = np.inf
        self.auto_end = not bool(options.get("live", False))      # canlı / kayıt: görevler bitince episode bitmesin
        e = self._errors(s)
        self.prev_abs_err = {"xy": e["xy"], "h": abs(e["h"]), "psi": abs(e["psi"])}
        self._last_err = dict(e)
        self._psi_f = None                                       # torque_feedback_climb süzgeci
        self._issue_due_tasks(0.0, s)
        e = self._errors(s)
        self.prev_abs_err = {"xy": e["xy"], "h": abs(e["h"]), "psi": abs(e["psi"])}
        obs = self._obs(s, e)
        info = dict(level=lv.name, level_index=self._ep_level, setup=dict(self.setup_info), **self._info_state(s, e))
        return obs, info

    # =================================================================
    # GÖREVLER
    # =================================================================

    def queue_task(self, task: dict) -> float:
        """Canlı kullanım: görevi bir sonraki adımda ver (önceki pencere 'kesildi' sayılır)."""
        t = self.steps * CONTROL_DT
        d = dict(task)
        d["_t"] = t
        self.fixed_schedule = True
        self.pending.append(d)
        self.next_issue_t = min(self.next_issue_t, t)
        return t

    def _tol(self, h_target: float) -> tuple[float, float]:
        a, b, c = self.cfg.tol_h
        x, y, z = self.cfg.tol_xy
        hh = max(0.0, h_target - GROUND_H_FT)
        return float(np.clip(a + b * hh, a, c)), float(np.clip(x + y * hh, x, z))

    def _issue_due_tasks(self, t: float, s: dict):
        while self.pending and t >= self.next_issue_t - 1e-9:
            d = self.pending.pop(0)
            if self.windows and not self.windows[-1]["closed"]:
                self._close_window(self.windows[-1], interrupt_t=t)
            w = self._new_window(d, t, s)
            T = w["T"]
            self.windows.append(w)
            self._init_progress(s)
            if self.pending:
                nxt = self.pending[0]
                if "_t" in nxt:
                    self.next_issue_t = float(nxt["_t"])
                elif "_frac" in nxt:
                    self.next_issue_t = t + float(nxt["_frac"]) * T
                else:
                    self.next_issue_t = np.inf                  # pencere bitince (başarı ya da süre aşımı)
            else:
                self.next_issue_t = np.inf

    def _init_progress(self, s: dict):
        """İlerleme ödülünün başlangıç hataları (yeni görev penceresi açılınca)."""
        e2 = self._errors(s)
        self.prev_abs_err = {"xy": e2["xy"], "h": abs(e2["h"]), "psi": abs(e2["psi"])}

    def _new_window(self, d: dict, t: float, s: dict) -> dict:
        """Görevi hedefe çevirir ve görev penceresini (süre hedefi, bant, kuplaj) kurar."""
        cfg, lv = self.cfg, self.ep_level
        kind = d["kind"]
        tg = self.target
        ps = math.radians(s["psi_deg"])
        allow = 0.0
        if kind != "recover":
            self.track = None                               # yeni görev hareketli hedefi bitirir
        if kind == "hold":
            if not d.get("keep"):                           # keep: hedefte kal (canlı: görev bitince otomatik)
                tg.update(n=s["n"], e=s["e"], h=float(np.clip(s["h"], MIN_HOVER_H_FT, MAX_TARGET_H_FT)),
                          psi=self.psi_unwrap)
            amount = 0.0
        elif kind in ("takeoff", "climb_to"):
            tg["h"] = float(np.clip(d["h"], MIN_HOVER_H_FT, MAX_TARGET_H_FT))
            amount = tg["h"] - s["h"]
            if kind == "climb_to" and s["vs"] * amount < 0.0 and abs(s["vs"]) > 2.0:
                allow = abs(s["vs"]) / 10.0 + s["vs"] ** 2 / 20.0 / max(1.0, lv.climb_fps)   # önce dur
        elif kind == "turn":
            tg["psi"] = self.psi_unwrap + float(d["dpsi"])
            amount = float(d["dpsi"])
        elif kind == "move":
            dx, dy = float(d.get("dx", 0.0)), float(d.get("dy", 0.0))
            tg["n"] = s["n"] + dx * math.cos(ps) - dy * math.sin(ps)
            tg["e"] = s["e"] + dx * math.sin(ps) + dy * math.cos(ps)
            amount = math.hypot(dx, dy)
        elif kind == "bob":
            dh = float(d["dh"])
            # eğitimde örneklenen görev (_flip) zarf dışına düşerse ters yöne (eski davranış birebir); canlı / arayüz
            # komutu ters ÇEVRİLMEZ, doğal sınıra kırpılır (2026-10-01: 30 ft'te −25 ft komutu 55 ft'e çıkarıyordu)
            if d.get("_flip") and not MIN_HOVER_H_FT <= s["h"] + dh <= MAX_HOVER_H_FT:
                dh = -dh
            tg["h"] = float(np.clip(s["h"] + dh, MIN_HOVER_H_FT, MAX_HOVER_H_FT if d.get("_flip") else MAX_TARGET_H_FT))
            amount = tg["h"] - s["h"]
        elif kind == "land":
            tg["h"] = GROUND_H_FT
            amount = max(0.0, s["h"] - GROUND_H_FT)
        elif kind == "recover":
            self._apply_disturbance(d, t)
            amount = 0.0
        elif kind == "depart":
            # hedef noktası rota (şimdiki heading) boyunca 0'dan v'ye sabit ivmeyle ilerler; irtifa hedefi h
            v = float(d.get("v_kt", 40.0)) * KT_FPS
            acc = float(d.get("accel", 2.5))
            tg["h"] = float(np.clip(d["h"], MIN_HOVER_H_FT, MAX_TARGET_H_FT))
            tg.update(n=s["n"], e=s["e"], psi=self.psi_unwrap)
            v0 = float(d.get("v0_kt", 0.0)) * KT_FPS
            self.track = dict(n0=s["n"], e0=s["e"], c=ps, v=v, v0=v0, a=acc, t0=t, lag_ft=0.0, prev_lag=0.0)
            amount = tg["h"] - s["h"]
        else:
            raise ValueError(f"bilinmeyen görev: {kind}")
        if d.get("T"):
            T = float(d["T"])
        elif kind == "depart":
            T = depart_time_target(amount, self.track["v"], self.track["a"], lv, self.track["v0"])
        elif kind in ("takeoff", "climb_to", "bob") and amount > 0.0 and cfg.torque_aware_climb:
            T = lv.lag_s + amount / self.climb_fps(lv) + allow
        else:
            T = lv.time_target(kind, amount) + allow
        first = not self.windows
        hold = cfg.land_hold_s if kind == "land" else (lv.hold_first_s if first else lv.hold_s)
        e = self._errors(s)
        w = dict(t=t, kind=kind, task={k: v for k, v in d.items() if not k.startswith("_")}, T=T, h_start=s["h"],
                 deadline=deadline_of(T), allow_s=allow, hold_s=hold, active=ACTIVE_AXES[kind],
                 target=dict(tg), e0={"xy": e["xy"], "h": abs(e["h"]), "psi": abs(e["psi"])},
                 streak=0.0, entry_t=None, settle_s=float("nan"), on_time=False, success=False, closed=False,
                 complete=False, interrupted=False, coupling_ok=True, max_err={"xy": 0.0, "h": 0.0, "psi": 0.0},
                 touchdown_vs=0.0 if kind == "land" and s["wow"] > 0 else None,    # zaten yerde → temas yok
                 final_err=None, tol=self._tol(tg["h"]))
        return w

    def _track_ref(self, t: float) -> tuple[float, float, float, float]:
        """Hareketli hedef (n, e, v_n, v_e): 0'dan v'ye a ivmesiyle, sonra sabit hız."""
        tr = self.track
        tt = max(0.0, t - tr["t0"])
        v0, v = tr.get("v0", 0.0), tr["v"]
        sgn = 1.0 if v >= v0 else -1.0
        t_acc = abs(v - v0) / tr["a"]
        if tt < t_acc:
            dist, vel = v0 * tt + 0.5 * sgn * tr["a"] * tt * tt, v0 + sgn * tr["a"] * tt
        else:
            dist, vel = v0 * t_acc + 0.5 * sgn * tr["a"] * t_acc * t_acc + v * (tt - t_acc), v
        c = tr["c"]
        return tr["n0"] + dist * math.cos(c), tr["e0"] + dist * math.sin(c), vel * math.cos(c), vel * math.sin(c)

    def _update_track(self, t: float, s: dict | None = None):
        if self.track is None:
            return
        n, e, _, _ = self._track_ref(t)
        tr = self.track
        c = tr["c"]
        along_ref = (n - tr["n0"]) * math.cos(c) + (e - tr["e0"]) * math.sin(c)
        if s is not None:
            along_h = (s["n"] - tr["n0"]) * math.cos(c) + (s["e"] - tr["e0"]) * math.sin(c)
            L = self.cfg.track_lead_ft
            along = along_h + float(np.clip(along_ref - along_h, -L, L))
        else:
            along = along_ref
        tr["lag_ft"] = along_ref - along                        # profile göre geride kalma (+) — bilgi için
        tr["pn"], tr["pe"] = n, e                               # profil noktası (başarı bandı buna göre)
        self.target["n"], self.target["e"] = tr["n0"] + along * math.cos(c), tr["e0"] + along * math.sin(c)

    def _yaw_ff(self) -> float:
        """Hedef heading'in değişim hızı (°/s): yönlendirmede istenen yaw hızına eklenir (sabit hedefte 0)."""
        return 0.0

    def _target_vel_body(self, s: dict) -> tuple[float, float]:
        """Hareketli hedefin hızı burun ekseninde (ileri, sağa) ft/s; hedef sabitse (0, 0)."""
        if self.track is None:
            return 0.0, 0.0
        _, _, vn, ve = self._track_ref(self.steps * CONTROL_DT)
        ps = math.radians(s["psi_deg"])
        return vn * math.cos(ps) + ve * math.sin(ps), -vn * math.sin(ps) + ve * math.cos(ps)

    def _apply_disturbance(self, d: dict, t: float):
        kind = d.get("dist")
        if kind == "kick":
            self.kick = (int(d["axis"]), float(d["mag"]), t + float(d["dur"]))
        elif kind == "push":
            self._set_ic_from_state(dn=float(d["dn"]), de=float(d["de"]), dw=float(d.get("dw", 0.0)))
        elif kind == "tilt":
            self._set_ic_from_state(dphi=float(d.get("dphi", 0.0)), dtheta=float(d.get("dtheta", 0.0)))

    def _inside(self, s: dict, e: dict, w: dict) -> bool:
        cfg = self.cfg
        if w["kind"] == "depart":
            tol_h, _ = w["tol"]
            vf, vr = self._target_vel_body(s)
            dv = math.hypot(s["ug"] - vf, s["vg"] - vr)
            tr = self.track or {}
            d_prof = math.hypot(s["n"] - tr.get("pn", s["n"]), s["e"] - tr.get("pe", s["e"]))
            return (s["wow"] == 0 and d_prof <= cfg.track_tol_xy and abs(e["h"]) <= tol_h
                    and abs(e["psi"]) <= cfg.tol_psi_deg and dv <= cfg.track_tol_v and abs(s["vs"]) <= cfg.tol_vs_fps)
        if w["kind"] == "land":
            return (s["wow"] == 4 and s["wfrac"] >= cfg.land_weight_frac and s["vh"] <= 1.0
                    and e["xy"] <= cfg.land_tol_xy_ft and abs(e["psi"]) <= cfg.land_tol_psi_deg)
        tol_h, tol_xy = w["tol"]
        return (s["wow"] == 0 and e["xy"] <= tol_xy and abs(e["h"]) <= tol_h and abs(e["psi"]) <= cfg.tol_psi_deg
                and s["vh"] <= cfg.tol_v_fps and abs(s["vs"]) <= cfg.tol_vs_fps)

    def _close_window(self, w: dict, interrupt_t: float | None = None):
        if w["closed"]:
            return
        w["closed"] = True
        ok_fail = self.failure is None
        if interrupt_t is not None and not w["complete"]:
            w["interrupted"] = True
            w["success"] = bool(w["coupling_ok"] and ok_fail)
        else:
            w["success"] = bool(w["on_time"] and w["streak"] >= w["hold_s"] - 1e-9 and w["coupling_ok"] and ok_fail
                                and (w["kind"] != "land" or (w["touchdown_vs"] is not None
                                                             and w["touchdown_vs"] >= -self.cfg.touchdown_ok_fps)))
        w["final_err"] = dict(self._last_err or {})

    def _update_window(self, s: dict, e: dict, t: float):
        cfg = self.cfg
        w = self.windows[-1]
        if w["closed"]:
            return
        for k, lim in zip(("xy", "h", "psi"), w.get("coupling_lim") or cfg.coupling_limits):
            val = e["xy"] if k == "xy" else abs(e[k])
            w["max_err"][k] = max(w["max_err"][k], val)
            if k not in w["active"] and val > lim:
                w["coupling_ok"] = False
        if w["kind"] == "land" and s["wow"] > 0 and self._prev_wow == 0 and self._airborne_once:
            vs_td = float(self._prev_vs)                       # en sert temas (sekip yeniden değerse o da sayılır)
            w["touchdown_vs"] = vs_td if w["touchdown_vs"] is None else min(w["touchdown_vs"], vs_td)
        if w["kind"] == "land" and w["touchdown_vs"] is None and s["wow"] > 0 and not self._airborne_once:
            w["touchdown_vs"] = 0.0                            # hiç kalkmadan (yerde) — temas yok sayılır
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
                w["complete"] = True                            # başarıyla bitti → kısa aradan sonra sıradaki
                self._close_window(w)
                self._schedule_next(t)
            elif elapsed > w["deadline"] and not w["on_time"]:
                # son sınır geçti, bantta değil → başarısız; ajan yine de hedefe varmaya çalışsın diye sıradaki görev
                # (ya da episode sonu) son sınır + tutma süresi kadar sonra
                w["complete"] = True
                self._close_window(w)
                self._schedule_next(w["t"] + w["deadline"] + w["hold_s"])

    def _schedule_next(self, t_next: float):
        cfg = self.cfg
        if not self.pending:
            if self.auto_end:
                self.episode_end_t = t_next + cfg.end_margin_s
            elif self.windows and self.windows[-1]["kind"] != "land" and self._prev_wow == 0:
                # canlı / değerlendirme (episode kendiliğinden bitmiyor): görev bitince aynı hedefte "hover tut"
                # penceresi — eğitimde son görevden sonra episode bittiği için ajan kapanmış pencereyle (τ ≫ 1) uzun süre
                # uçmadı; öyle kalınca hover'da yavaşça sürükleniyordu (ADS-33 hover MTE'sinde 30 s'de ~15 ft)
                self.pending.append(dict(kind="hold", keep=True, auto=True))
                self.next_issue_t = t_next + float(self.np_random.uniform(*cfg.gap_s))
            return
        nxt = self.pending[0]
        if "_t" in nxt:
            return
        gap = float(self.np_random.uniform(*cfg.gap_s))
        if "_frac" in nxt:
            self.next_issue_t = min(self.next_issue_t, t_next + gap)
        else:
            self.next_issue_t = t_next + gap

    # =================================================================
    # HATALAR / OBS
    # =================================================================

    def _errors(self, s: dict) -> dict:
        tg = self.target
        dn, de = tg["n"] - s["n"], tg["e"] - s["e"]
        ps = math.radians(s["psi_deg"])
        return dict(fwd=dn * math.cos(ps) + de * math.sin(ps), right=-dn * math.sin(ps) + de * math.cos(ps),
                    xy=math.hypot(dn, de), h=tg["h"] - s["h"], psi=tg["psi"] - self.psi_unwrap)

    def _schedule_lag(self, e: dict) -> tuple[np.ndarray, float, float]:
        if not self.windows:
            return np.zeros(3), 0.0, 0.0
        w = self.windows[-1]
        t = self.steps * CONTROL_DT
        tau = (t - w["t"]) / max(w["T"], 1e-6)
        lag = np.zeros(3)
        for k, a in enumerate(("xy", "h", "psi")):
            if a not in w["active"] or w["kind"] in ("hold", "recover", "pirouette"):
                continue
            cur = e["xy"] if a == "xy" else e[a]
            sched = w["e0"][a] * max(0.0, 1.0 - max(0.0, tau))
            lag[k] = math.copysign(max(0.0, abs(cur) - sched), cur if a != "xy" else 1.0) / self.cfg.sched_scale[k]
        return lag, float(tau), float(w["T"])

    def _obs(self, s: dict, e: dict) -> np.ndarray:
        lag, tau, T = self._schedule_lag(e)
        # iniş bayrağı: iniş penceresi açıkken; pencere kapandıktan sonra da kızaklar yerdeyken (canlı uçuşta iniş
        # bitince yeni görev gelene kadar helikopter yerde kalsın — bayrak düşünce ajan kalkış başlangıcı sanıyordu)
        wl = self.windows[-1] if self.windows else None
        land = 1.0 if wl is not None and wl["kind"] == "land" and (not wl["closed"] or s["wow"] > 0) else 0.0
        o = np.array([
            e["fwd"] / 10.0, e["fwd"] / 100.0, e["right"] / 10.0, e["right"] / 100.0,
            e["h"] / 10.0, e["h"] / 200.0, e["psi"] / 10.0, e["psi"] / 180.0,
            *np.clip(lag, -1.0, 1.0), min(max(tau, 0.0), 2.0) / 2.0, T / 60.0,
            s["ug"] / 20.0, s["vg"] / 20.0, s["vs"] / 20.0,
            s["phi"] / 0.5, s["theta"] / 0.5, s["p"] / 1.0, s["q"] / 1.0, s["r"] / 1.0,
            math.log1p(max(0.0, s["hs"]) / 5.0) / 3.0, (s["rpm"] - 320.0) / 30.0, s["wfrac"], land,
            *self.filt,
            *([(s["torque_psi"] - self.cfg.torque_cont_psi) / 10.0] if self.cfg.torque_obs else []),
            *([v / 20.0 for v in self._target_vel_body(s)] if self.cfg.track_obs else []),
        ], dtype=np.float64)
        o = np.nan_to_num(o, nan=0.0, posinf=5.0, neginf=-5.0)
        o[[0, 2, 4, 6]] = np.clip(o[[0, 2, 4, 6]], -3.0, 3.0)
        return np.clip(o, -5.0, 5.0).astype(np.float32)

    # =================================================================
    # STEP
    # =================================================================

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
        t_now = self.steps * CONTROL_DT
        s_pre = self._state()
        self._issue_due_tasks(t_now, s_pre)

        # a → filtre → expo → kumanda → hız sınırı; filtre durumu gerçek kumandaya göre güncellenir
        filt = self.filt + cfg.action_filter_alpha * (a - self.filt)
        target_c = np.clip(self.trim + self.rng_ctrl * expo(filt, cfg.expo), CTRL_LO, CTRL_HI)
        step_max = np.asarray(cfg.rate_limit) * CONTROL_DT
        new_c = self.ctrl + np.clip(target_c - self.ctrl, -step_max, step_max)
        self._dctrl = (new_c - self.ctrl) / (step_max + 1e-9)
        self.ctrl = new_c
        self.filt = np.clip(expo_inv(np.clip((self.ctrl - self.trim) / self.rng_ctrl, -1.0, 1.0), cfg.expo), -1.0, 1.0)
        out = self.ctrl.copy()
        if self.kick is not None:                               # bozucu: kumanda darbesi (ajan göremez)
            ax, mag, t_end = self.kick
            if t_now < t_end:
                out[ax] += mag
            else:
                self.kick = None
        self._write_controls(np.clip(out, CTRL_LO, CTRL_HI))

        ok = self._run_plain()
        self.steps += 1
        ext_events = self.ext.after_step(self.fdm)             # yakıt / motor / tork istatistiği (kapalıyken [])
        t_after = self.steps * CONTROL_DT
        s = self._state()
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

        if finite and self.cfg.torque_feedback_climb:            # tırmanış yönlendirmesi için süzülmüş tork (1 s)
            self._update_psi_f(s)
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
                reward += cfg.w_task_success * cfg.reward_scale         # görev başarıyla tamamlandı
        self._prev_wow, self._prev_vs = s["wow"], s["vs"]

        fuel_out = self.ext.engine_is_out and cfg.fuel_exhausted_ends
        truncated = (not terminated) and (self.steps >= self.max_steps or t_after >= self.episode_end_t - 1e-9
                                          or fuel_out)
        # seviye = episode'un başladığı seviye: atlama episode ortasında olursa eski seviyenin görevleri yeni seviyeye
        # sayılmasın (curriculum kapıları görev türüne göre bakıyor)
        info = dict(level=self.levels[self._ep_level].name, level_index=self._ep_level, reward_parts=parts,
                    controls=[float(x) for x in out], failure=self.failure, **self._info_state(s, e))
        if self.ext.armed:
            info["physics"] = self.ext.info(self.fdm)
            if ext_events:
                info["physics_events"] = ext_events
        if self.windows:
            w = self.windows[-1]
            info.update(task_kind=w["kind"], cmd_T=w["T"], cmd_deadline=w["deadline"], cmd_elapsed=t_after - w["t"])
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

    def _result(self, w: dict) -> dict:
        fe = w.get("final_err") or {}
        return dict(t=w["t"], kind=w["kind"], cmd=dict(w["task"], kind=w["kind"]), T=w["T"], deadline=w["deadline"],
                    success=w["success"], settle_s=w["settle_s"], on_time=w["on_time"], final_streak_s=w["streak"],
                    hold_s=w["hold_s"], max_abs_err=dict(w["max_err"]), coupling_ok=w["coupling_ok"],
                    interrupted=w["interrupted"], touchdown_vs=w["touchdown_vs"], target=dict(w["target"]),
                    category="interrupted" if w["interrupted"] else w["kind"], allow_s=w["allow_s"],
                    final_err=dict(heading=float(fe.get("psi", 0.0)), xy=float(fe.get("xy", 0.0)),
                                   altitude=float(fe.get("h", 0.0))))

    # =================================================================
    # REWARD / GÜVENLİK
    # =================================================================

    def _reward(self, s: dict, e: dict, a: np.ndarray):
        cfg, lv = self.cfg, self.ep_level
        w = self.windows[-1] if self.windows else None
        kind = w["kind"] if w is not None and not w["closed"] else "hold"
        K = self._kernel2
        # inişte yere değdikten sonra: iniş toleransı içinde konum çekimi yok, istenen yatay hız 0 (kızaklar yerde
        # kayamaz; to_v10'da ajan pad merkezine "gitmek" için cyclic'i ileri itip burnu 6° aşağıda, iki kızak noktası
        # üstünde oturuyordu → dört nokta temas yok → iniş bandı yok)
        ground_land = kind == "land" and s["wow"] > 0
        e_xy_tr = max(0.0, e["xy"] - cfg.land_tol_xy_ft) if ground_land else e["xy"]
        track = (cfg.w_xy * K(e_xy_tr, cfg.kernel_xy) + cfg.w_h * K(e["h"], cfg.kernel_h)
                 + cfg.w_psi * K(e["psi"], cfg.kernel_psi))
        # yönlendirme: hedefe doğru istenen hızlar (yaklaştıkça azalan)
        v_max = lv.move_fps * 1.25 if kind in ("move", "depart", "pirouette") else cfg.hold_v_max
        vx, vy = cfg.guide_k_xy * e["fwd"], cfg.guide_k_xy * e["right"]
        vn = math.hypot(vx, vy)
        if vn > v_max:
            vx, vy = vx * v_max / vn, vy * v_max / vn
        if self.track is not None:                              # hareketli hedef: hedefin hızı + yakalama
            vf, vr = self._target_vel_body(s)
            vx, vy = vx + vf, vy + vr
        if ground_land:
            vx = vy = 0.0
        if kind == "land":
            vs_des = 0.0 if s["wow"] > 0 else -float(np.clip(LAND_PROFILE_K * max(0.0, s["hs"]), LAND_PROFILE_MIN_FPS,
                                                             lv.descent_fps))
        else:
            up = self.climb_fps(lv) * 1.25 if kind in ("takeoff", "climb_to", "bob") else 3.0
            up = self._climb_feedback_cap(s, up)
            dn = lv.descent_fps * 1.25 if kind in ("climb_to", "bob") else 3.0
            vs_des = float(np.clip(cfg.guide_k_h * e["h"], -dn, up))
        r_max = lv.yaw_rate_dps * 1.25 if kind == "turn" else 5.0
        r_des = float(np.clip(cfg.guide_k_psi * e["psi"], -r_max, r_max)) + self._yaw_ff()
        guide = ((cfg.w_v_track if self.track is not None else cfg.w_v)
                 * K(math.hypot(s["ug"] - vx, s["vg"] - vy), cfg.kernel_v_track if self.track is not None else cfg.kernel_v)
                 + cfg.w_vs * K(s["vs"] - vs_des, cfg.kernel_vs)
                 + cfg.w_r * K(math.degrees(s["r"]) - r_des, cfg.kernel_r))
        progress = 0.0
        for k, mn in zip(("xy", "h", "psi"), cfg.progress_min_scale):
            cur = e["xy"] if k == "xy" else abs(e[k])
            scale = max(mn, w["e0"][k] if w is not None else 0.0)
            if not (kind == "land" and k == "h"):             # inişte irtifa ilerlemesi yok: hızlı alçalmayı ödüllendirmesin
                progress += (self.prev_abs_err[k] - cur) / scale
            self.prev_abs_err[k] = cur
        progress *= cfg.progress_weight
        if self.track is not None:                              # depart: profil noktasına yetişme (geride kalma artarsa −)
            lag = max(0.0, float(self.track.get("lag_ft", 0.0)))
            progress += cfg.w_track_lag * (self.track.get("prev_lag", 0.0) - lag) / 10.0
            self.track["prev_lag"] = lag
        lag, tau, T = self._schedule_lag(e)
        sched = cfg.pen_sched * float(np.sum(np.minimum(1.0, np.abs(lag)) ** 2))
        late = 0.0
        if w is not None and not w["closed"]:
            t = self.steps * CONTROL_DT
            if t - w["t"] > w["deadline"] and not self._inside(s, e, w):
                late = cfg.pen_late
        phi, th = abs(math.degrees(s["phi"])), abs(math.degrees(s["theta"]))
        att = cfg.pen_att * ((max(0.0, phi - 20.0) / 10.0) ** 2 + (max(0.0, th - 15.0) / 10.0) ** 2)
        rate = cfg.pen_rate * ((s["p"] / 0.5) ** 2 + (s["q"] / 0.5) ** 2 + (s["r"] / 1.0) ** 2)
        smooth = (cfg.pen_dctrl * float(np.mean(self._dctrl ** 2)) * 0.1
                  + cfg.pen_sat * float(np.mean(np.maximum(np.abs(a) - cfg.sat_threshold, 0.0))))
        couple = 0.0
        if w is not None and not w["closed"]:
            for k, lim in zip(("xy", "h", "psi"), w.get("coupling_lim") or cfg.coupling_limits):
                if k in w["active"]:
                    continue
                val = e["xy"] if k == "xy" else abs(e[k])
                x = min(1.0, max(0.0, val - cfg.couple_soft_frac * lim) / lim)
                couple += cfg.pen_couple * x * x
        sink = 0.0
        if 0.0 < s["hs"] < 60.0 and s["wow"] == 0:
            allow = cfg.sink_base_fps + cfg.sink_slope * s["hs"]
            sink = cfg.pen_sink * (max(0.0, -s["vs"] - allow) / 3.0) ** 2
        if kind == "turn" and cfg.pen_turn_rate > 0.0:            # yerinde dönüşte aşırı yaw hızı
            r_cap = cfg.turn_rate_cap_frac * lv.yaw_rate_dps * 1.25
            rate += cfg.pen_turn_rate * (max(0.0, abs(math.degrees(s["r"])) - r_cap) / 10.0) ** 2
        if kind == "land" and s["wow"] == 0:
            sink += cfg.pen_land_sink * (max(0.0, -s["vs"] - (abs(vs_des) + 2.0)) / 3.0) ** 2
            couple += cfg.pen_land_xy * (max(0.0, e["xy"] - 2.0) / 4.0) ** 2
        ground = 0.0
        contact_now = s["wow"] > 0 and self._prev_wow == 0 and self._airborne_once
        if s["wow"] > 0:
            ground += cfg.pen_slide * s["vh"]
            if contact_now:                                     # temas anı
                ground += cfg.pen_touchdown * max(0.0, (abs(min(0.0, self._prev_vs)) - cfg.touchdown_free_fps) / 3.0) ** 2
        # iniş ödülleri yalnızca yumuşak temasla: bu penceredeki en sert temas ≤ 3 ft/s → tam, 4 → yarı, ≥ 5 → yok.
        # (to_v8: yerdeki adım başına ödül, son 2–3 ft'i collective'i kesip 4–5 ft/s ile "düşmeyi" kârlı yapıyordu.)
        soft = 1.0
        if kind == "land":
            td = w["touchdown_vs"]
            if contact_now:
                td = self._prev_vs if td is None else min(td, self._prev_vs)
            if td is not None:
                soft = float(np.clip((cfg.soft_zero_fps + td) / (cfg.soft_zero_fps - cfg.soft_full_fps), 0.0, 1.0))
        if kind == "land" and s["wow"] > 0 and e["xy"] <= 10.0:
            # dört nokta temas + kızaklardaki ağırlık + iniş bandının içinde olmak (ölçütün kendisi; to_v17'de ağır
            # helikopterde collective ~0.3'te kalıyordu → ağırlık %60, bant %70 istiyor)
            inside = float(self._inside(s, e, w))
            ground -= soft * cfg.w_land * (0.3 * float(s["wow"] == 4) + 0.4 * min(1.0, s["wfrac"]) + 0.3 * inside)
        cdown = 0.0
        if kind == "land" and s["wow"] > 0 and e["xy"] <= 10.0:
            c_ige = self._ige_coll(s["weight"])                                        # IGE hover trimi (probe)
            a_ige, a_set = expo_inv(np.clip((np.array([c_ige, cfg.coll_flat]) - self.trim[0]) / self.rng_ctrl[0],
                                            -1.0, 1.0), cfg.expo)
            cdown = soft * cfg.w_coll_down * float(np.clip((a_ige - self.filt[0]) / max(1e-3, a_ige - a_set), 0.0, 1.0))
        # tork (modelin kendi göstergesi): 50–56 psi kalkış gücü bölgesi hafif, 56 psi (%100) üstü güçlü ceza
        torque = torque_penalty(s["torque_psi"], s["rpm"], cfg.pen_torque_cont, cfg.pen_torque_over, cfg.pen_rpm_low,
                                cfg.torque_cont_psi, cfg.torque_max_psi, cfg.rpm_low_warn)
        r = track + guide + progress - sched - late - att - rate - smooth - couple - ground - sink + cdown - torque
        return float(r), dict(track=track, guide=guide, progress=progress, schedule=sched, late=late, attitude=att,
                              rate=rate, smooth=smooth, coupling=couple, ground=-ground, sink=-sink, coll_down=cdown,
                              torque=-torque)

    def _kernel2(self, err: float, widths) -> float:
        a, b = widths
        return 0.5 * math.exp(-(err / a) ** 2) + 0.5 * math.exp(-abs(err) / b)

    def _safety(self, s: dict, e: dict, ok: bool, finite: bool) -> str | None:
        cfg = self.cfg
        if not ok:
            return "jsbsim_stopped"
        if not finite:
            return "non_finite_state"
        if s["tail"]:
            return "tail_strike"
        phi, th = abs(math.degrees(s["phi"])), abs(math.degrees(s["theta"]))
        if s["wow"] > 0:
            if max(phi, th) > cfg.ground_max_att_deg:
                return "rollover"
            if self._prev_wow == 0 and self._airborne_once and self._prev_vs < -cfg.crash_vs_fps:
                return "hard_landing"
        if phi > cfg.max_roll_deg:
            return "roll_limit"
        if th > cfg.max_pitch_deg:
            return "pitch_limit"
        if max(abs(math.degrees(s["p"])), abs(math.degrees(s["q"]))) > cfg.max_rate_dps:
            return "rate_limit"
        if abs(math.degrees(s["r"])) > cfg.max_yaw_rate_dps:
            return "yaw_rate_limit"
        if not cfg.rpm_limits[0] <= s["rpm"] <= cfg.rpm_limits[1]:
            return "rotor_rpm"
        if e["xy"] > (cfg.flyaway_track_ft if self.track is not None else cfg.flyaway_ft):
            return "position_deviation"
        top = max(self.target["h"], self.windows[-1]["h_start"] if self.windows else self.target["h"])
        if s["h"] > top + cfg.alt_over_ft:
            return "altitude_deviation"
        if s["vh"] > (cfg.max_speed_track_fps if self.track is not None else cfg.max_speed_fps):
            return "speed_limit"
        return None

    def _info_state(self, s: dict, e: dict) -> dict:
        return dict(t=self.steps * CONTROL_DT, altitude=s["h"], skid_height=s["hs"], vertical_speed=s["vs"],
                    speed=s["ug"], lateral_speed=s["vg"], ground_speed=s["vh"], north=s["n"], east=s["e"],
                    roll_deg=math.degrees(s["phi"]), pitch_deg=math.degrees(s["theta"]),
                    yaw_rate_dps=math.degrees(s["r"]), heading_deg=s["psi_deg"], heading_unwrapped=self.psi_unwrap,
                    rotor_rpm=s["rpm"], wow=s["wow"], weight_on_skids=s["wfrac"], weight_lbs=s["weight"],
                    torque_psi=s["torque_psi"], tracking=self.track is not None,
                    target=dict(self.target), err_fwd=e["fwd"], err_right=e["right"], err_xy=e["xy"],
                    err_altitude=e["h"], err_heading=e["psi"], filt=[float(x) for x in self.filt])


# =====================================================================
# HIZLI TEST:  python helicopter_env_takeoff.py
# =====================================================================

if __name__ == "__main__":
    import time
    for lvl in ("K1", "K2", "K5", "K7", "K8"):
        env = HelicopterEnvTakeoff(level=lvl)
        t0 = time.time()
        obs, info = env.reset(seed=0)
        print(f"{lvl} reset {time.time() - t0:.2f}s start={info['setup']['start']} h={info['altitude']:.1f} "
              f"W={info['setup']['weight']:.0f} obs={obs.shape} görevler={[w['kind'] for w in env.windows]} + "
              f"{[d['kind'] for d in env.pending]}")
        t0 = time.time()
        done, ret, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, info = env.step(np.zeros(4))
            ret += r
            n += 1
            done = term or trunc
        print(f"   a=0: {n} adım ({n / (time.time() - t0):.0f} adım/s) return={ret:.1f} bitiş={info['termination']} "
              f"h={info['altitude']:.1f} konum hatası={info['err_xy']:.1f} ft "
              f"sonuç={[(c['kind'], c['success'], round(c['T'], 1)) for c in info['command_results']]}")
