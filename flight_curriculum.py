from __future__ import annotations

"""
FLIGHT CURRICULUM — tek ajan: kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δ komutları → duruş → iniş
===========================================================================================================

`helicopter_env_flight.py` ile kullanılır (dört kumanda doğrudan, AFCS yalnızca SAS, güç tavanı %100 tork, tork cezası
ve yakıt tüketimi BAŞTAN açık). İki eksenli curriculum:

  görev zorluğu : hover temeli → kalkış / iniş → ileri uçuş (tek eksen) → ileri uçuş (birleşik) → geçişler
                  (hover → ileri uçuş hızlanması, ileri uçuştan duruş) → sürekli zincir
  çevre zorluğu : E0 sakin → E1 hafif rüzgâr → E2 orta rüzgâr + hafif türbülans → E3 rüzgâr + hafif / orta türbülans
                  (türbülans: physics_ext `dryden_agl`; kullanıcı seçimi 2026-09-29: yok → hafif → orta)

Her seviye bir görev aşaması ve bir çevre karışımı taşır; seviyeler önce görevi sakin havada öğretir, sonra aynı
görevleri rüzgâr / türbülansla tekrarlatır (tablo: DEFAULT_FLIGHT_LEVELS'ın altındaki açıklama).

Görev türleri (takeoff_curriculum'dakilere ek):
  cruise : ileri uçuşta Δ komut — hava hızı (gövde ekseninde ileri, kt), heading, irtifa; tek eksen ya da birleşik.
           Yeni Δ ÖLÇÜLEN duruma göre; komut verilmeyen eksen önceki hedefine devam eder (manevra env'i kuralı).
           Hover'dan verilen hız komutu = hızlanma (ileri uçuşa geçiş; "accel" kategorisi).
  stop   : ileri uçuştan duruş — referans noktası yer izi boyunca mevcut yer hızından sabit yavaşlamayla durur; sonra
           o noktada hover (depart'ın hareketli hedef mekanizması, tersine).
Süre hedefleri: `flight_time_target`.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from takeoff_curriculum import (GROUND_H_FT, MAX_HOVER_H_FT, MIN_HOVER_H_FT, TANK_CAPACITY_LBS, TakeoffLevel,
                                deadline_of, find_takeoff_level, task_time_target)

KT = 1.6878099
CRUISE_MIN_KT = 30.0             # ileri uçuş komutlarının hız zarfı (hava hızı)
CRUISE_MAX_KT = 110.0
CRUISE_MIN_ALT_FT = 100.0        # CG AGL
CRUISE_MAX_ALT_FT = 1000.0
# Doğal komut zarfı (2026-10-01; canlı / arayüz komutları — eğitim örneklemesi seviyenin zarfıyla, yukarıdaki sabitler
# F1–F9 için birebir). Hız: AH-1S en yüksek düz uçuş hızı ~128 kt (Vertipedia) / ~130 KTAS @ 10,000 lbs 8 TOW (DTIC
# ADA025476, YAH-1S 1975); bu modelde 130 kt ~38–40 psi (gerçekte TOW rampalarıyla ~%100). 10 kt altı → duruş (hover).
# 2026-10-02: üst sınır 130 → 120 kt — trim tablosu (probe e) 120 kt'a kadar ölçüldü; 120 kt üstünde çizelge kırpılıp
# ileri besleme yanlış kalıyordu (held-out d_130kt: 128 kt'a çıkıp bantta oturamadı). Kullanıcı kararı: zarf = tablo.
# İrtifa: CG AGL; ileri uçuşta en az 50 ft, hover 12 ft (kızaklar ~6 ft). Vne (güvenlik): 170 kt (TOW ya da > 9500 lbs;
# aircav.com).
CMD_MIN_KT = 10.0
CMD_MAX_KT = 120.0
CMD_MIN_ALT_FT = 50.0
CMD_MAX_ALT_FT = 1500.0
VNE_KT = 170.0

# çevre aşamaları (physics_ext seçenekleri olarak env'e verilir; rüzgâr yönü başlangıç heading'ine göre rastgele)
ENV_STAGES = {
    "E0": dict(wind_kt=(0.0, 0.0), turb=("none",), turb_p=(1.0,), gust_rate=0.0),
    "E1": dict(wind_kt=(0.0, 10.0), turb=("none",), turb_p=(1.0,), gust_rate=0.0),
    "E2": dict(wind_kt=(0.0, 15.0), turb=("none", "light"), turb_p=(0.4, 0.6), gust_rate=0.5, gust_kt=(3.0, 8.0)),
    "E3": dict(wind_kt=(5.0, 25.0), turb=("light", "moderate"), turb_p=(0.5, 0.5), gust_rate=1.0, gust_kt=(5.0, 12.0)),
}


@dataclass
class FlightLevel(TakeoffLevel):
    # --- çevre karışımı ------------------------------------------------------------
    env_probs: dict = field(default_factory=lambda: {"E0": 1.0})
    # --- başlangıç (TakeoffLevel'inkilere ek) -----------------------------------------
    p_cruise_start: float = 0.0            # ileri uçuşta başla (teleport + reset oto-pilotu)
    cruise_start_kt: tuple = (40.0, 100.0)  # hava hızı
    cruise_start_alt_ft: tuple = (150.0, 800.0)
    # --- ileri uçuş komutları ------------------------------------------------------------
    n_cruise: tuple = (0, 0)               # ileri uçuş Δ komut sayısı (cruise başlangıcında / zincirde)
    cruise_probs: dict = field(default_factory=lambda: {"u": 0.25, "psi": 0.3, "h": 0.25, "mix": 0.2})
    du_kt: tuple = (10.0, 30.0)            # |Δhız|
    dpsi_deg: tuple = (20.0, 90.0)         # |Δheading|
    dh_ft: tuple = (50.0, 250.0)           # |Δirtifa|
    p_cruise_interrupt: float = 0.0        # sonraki komut öncekinin süre hedefinin %30–80'inde (kesen komut)
    # --- geçişler -------------------------------------------------------------------------
    p_accel: float = 0.0                   # hover başlangıcında: hızlanıp ileri uçuşa geç (cruise görevi, hover'dan)
    accel_kt: tuple = (40.0, 80.0)         # hedef hava hızı
    accel_climb_ft: tuple = (0.0, 200.0)   # hızlanırken tırmanış (Δh)
    p_stop: float = 0.0                    # ileri uçuştan sonra duruş (hover'a geçiş)
    stop_decel: tuple = (2.0, 3.0)         # ft/s² (referans noktasının yavaşlaması)
    p_chain: float = 0.0                   # yerden: kalkış → hızlanma → Δ komutlar → duruş → (iniş)
    chain_takeoff_ft: tuple = (20.0, 60.0)
    p_land_after_stop: float = 0.0
    # --- ileri uçuş çevikliği (süre hedefi) --------------------------------------------------
    cruise_accel_fps2: float = 2.5
    cruise_decel_fps2: float = 2.5
    cruise_bank_deg: float = 20.0          # dönüş hızı = min(g·tanφ / V, cruise_yaw_max)
    cruise_yaw_max_dps: float = 8.0
    cruise_climb_fps: float = 10.0
    cruise_descent_fps: float = 10.0
    cruise_lag_s: float = 3.0
    cruise_hold_s: float = 5.0
    # --- örnekleme zarfı (_flip'li görevlerde Δ bunun dışına düşerse ters çevrilir; F1–F9: eski sabitler) ----------
    cruise_kt_env: tuple = (CRUISE_MIN_KT, CRUISE_MAX_KT)
    cruise_alt_env: tuple = (CRUISE_MIN_ALT_FT, CRUISE_MAX_ALT_FT)
    # --- hover hassasiyeti: hover görevlerinde kuplaj sınırları ve hover bandının konum toleransı bu katsayıyla ------
    # (türbülans ölçeğinin üstüne; 0.5 → yerinde dönüşte konum kayması ≤ 10 ft, hover bandı 3–6 ft; ADS-33 ±3 / ±6 ft)
    hover_precision: float = 1.0
    # --- hava sıcaklığı (standart günden sapma, °C; (0, 0) → kapalı, RNG'ye dokunmaz) ---------------------------------
    delta_T_C: tuple = (0.0, 0.0)

    # ------------------------------------------------------------------------------------------
    def sample_env(self, rng) -> tuple[str, dict]:
        """Çevre aşaması + physics_ext seçenekleri (rüzgâr hızı / yönü, türbülans seviyesi, gust hızı)."""
        keys = list(self.env_probs)
        p = np.array([self.env_probs[k] for k in keys], dtype=float)
        name = keys[int(rng.choice(len(keys), p=p / p.sum()))]
        st = ENV_STAGES[name]
        opts = dict(wind_kt=float(rng.uniform(*st["wind_kt"])), wind_dir_deg=float(rng.uniform(0.0, 360.0)),
                    wind_dir_relative=True)
        tp = np.array(st["turb_p"], dtype=float)
        opts["turb_level"] = st["turb"][int(rng.choice(len(st["turb"]), p=tp / tp.sum()))]
        if st.get("gust_rate", 0.0) > 0.0:
            opts["gust_rate_per_min"] = float(st["gust_rate"])
            opts["gust_kt"] = tuple(st["gust_kt"])
        if tuple(self.delta_T_C) != (0.0, 0.0):                    # (eski seviyelerde ek rastgele sayı yok)
            opts["delta_T_C"] = float(rng.uniform(*self.delta_T_C))
        return name, opts

    def sample_start(self, rng) -> dict:
        if self.p_cruise_start > 0.0 and rng.random() < self.p_cruise_start:
            return dict(start="cruise", h=float(rng.uniform(*self.cruise_start_alt_ft)),
                        u_kt=float(rng.uniform(*self.cruise_start_kt)))
        return TakeoffLevel.sample_start(self, rng)

    # ------------------------------------------------------------------------------------------
    def sample_cruise(self, rng, kind: str | None = None) -> dict:
        kind = kind or self._pick(rng, self.cruise_probs)
        d = dict(kind="cruise")
        if kind in ("u", "mix"):
            d["du_kt"] = float(rng.uniform(*self.du_kt) * rng.choice((-1.0, 1.0)))
        if kind in ("psi", "mix"):
            d["dpsi"] = float(rng.uniform(*self.dpsi_deg) * rng.choice((-1.0, 1.0)))
        if kind in ("h", "mix"):
            d["dh"] = float(rng.uniform(*self.dh_ft) * rng.choice((-1.0, 1.0)))
        if kind == "mix" and rng.random() < 0.5:                  # bazen yalnızca iki eksen
            drop = ("du_kt", "dpsi", "dh")[int(rng.integers(3))]
            d.pop(drop, None)
            if len([k for k in ("du_kt", "dpsi", "dh") if k in d]) == 0:
                d["dpsi"] = float(rng.uniform(*self.dpsi_deg))
        d["_flip"] = True              # örneklenen görev: seviyenin zarfı + ters çevirme (canlı komut kırpılır)
        return d

    def _cruise_block(self, rng) -> list[dict]:
        out = []
        for i in range(int(rng.integers(self.n_cruise[0], self.n_cruise[1] + 1))):
            d = self.sample_cruise(rng)
            if i > 0 and self.p_cruise_interrupt > 0.0 and rng.random() < self.p_cruise_interrupt:
                d["_frac"] = float(rng.uniform(0.3, 0.8))
            out.append(d)
        return out

    def sample_schedule(self, rng, start: str) -> list[dict]:
        if start == "cruise":
            tasks = [dict(kind="cruise", hold=True)]                  # önce mevcut hız / heading / irtifayı tut
            tasks += self._cruise_block(rng)
            if self.p_stop > 0.0 and rng.random() < self.p_stop:
                tasks.append(dict(kind="stop", decel=float(rng.uniform(*self.stop_decel))))
                if rng.random() < self.p_land_after_stop:
                    tasks.append(dict(kind="land"))
            return tasks
        if start == "ground" and self.p_chain > 0.0 and rng.random() < self.p_chain:
            tasks = [dict(kind="takeoff", h=float(rng.uniform(*self.chain_takeoff_ft)))]
            tasks.append(dict(kind="cruise", u_kt=float(rng.uniform(*self.accel_kt)),
                              dh=float(rng.uniform(*self.accel_climb_ft)), accel=True, _flip=True))
            tasks += self._cruise_block(rng)
            tasks.append(dict(kind="stop", decel=float(rng.uniform(*self.stop_decel))))
            if rng.random() < self.p_land_after_stop:
                tasks.append(dict(kind="land"))
            return tasks
        if start == "hover" and self.p_accel > 0.0 and rng.random() < self.p_accel:
            tasks = [dict(kind="hold")]
            tasks.append(dict(kind="cruise", u_kt=float(rng.uniform(*self.accel_kt)),
                              dh=float(rng.uniform(*self.accel_climb_ft)), accel=True, _flip=True))
            tasks += self._cruise_block(rng)
            if self.p_stop > 0.0 and rng.random() < self.p_stop:
                tasks.append(dict(kind="stop", decel=float(rng.uniform(*self.stop_decel))))
            return tasks
        return TakeoffLevel.sample_schedule(self, rng, start)

    def time_target(self, kind: str, amount) -> float:
        if kind in ("cruise", "stop"):
            raise ValueError("cruise / stop süre hedefi flight_time_target ile")
        return task_time_target(kind, amount, self)


def cruise_yaw_rate_dps(lv: FlightLevel, v_fps: float) -> float:
    """Koordineli dönüş hızı: g·tanφ / V (düşük hızda üst sınır)."""
    v = max(20.0, abs(v_fps))
    return float(min(lv.cruise_yaw_max_dps, math.degrees(32.174 * math.tan(math.radians(lv.cruise_bank_deg)) / v)))


def flight_time_target(kind: str, lv: FlightLevel, du_fps: float = 0.0, dpsi_deg: float = 0.0, dh_ft: float = 0.0,
                       v_fps: float = 0.0, v0_fps: float = 0.0, decel: float = 2.5) -> float:
    """cruise: tepki + max(|Δu| / ivme, |Δψ| / dönüş hızı, |Δh| / tırmanış) — eksenler aynı anda;
    stop: tepki + v0 / yavaşlama + 5 s oturma."""
    if kind == "stop":
        return lv.cruise_lag_s + abs(v0_fps) / max(0.5, decel) + 5.0
    acc = lv.cruise_accel_fps2 if du_fps >= 0.0 else lv.cruise_decel_fps2
    rate_h = lv.cruise_climb_fps if dh_ft >= 0.0 else lv.cruise_descent_fps
    v_turn = max(abs(v_fps), abs(v_fps + du_fps))
    parts = [abs(du_fps) / acc, abs(dpsi_deg) / cruise_yaw_rate_dps(lv, v_turn), abs(dh_ft) / rate_h]
    return lv.cruise_lag_s + max(parts)


def find_flight_level(level, levels=None) -> int:
    return find_takeoff_level(level, levels or DEFAULT_FLIGHT_LEVELS)


# ---------------------------------------------------------------------------------
# Seviyeler (görev × çevre)
# ---------------------------------------------------------------------------------
#
#   seviye | görev                                                        | çevre
#   F1     | tut: %50 hover (havada), %50 ileri uçuş (40–100 kt)            | E0
#   F2     | %50 hover (kalkış + manevralar), %50 ileri uçuş tek eksen Δ      | E0
#   F3     | %60 ileri uçuş birleşik / büyük Δ + kesen, %40 hover manevraları | E0 / E1
#   F4     | geçişler: hover → hızlanma, ileri uçuş → duruş                  | E0 / E1
#   F5     | zincir: yerden kalkış → hızlanma → Δ → duruş                    | E0 / E1
#   F6a    | oturma okulu: yerde hafif yüklü / çok alçak hover → otur         | E0
#   F6     | iniş (alçak hover / hafif yüklü başlangıçlar)                   | E0 / E1
#   F7     | karma (tüm görevler)                                            | E1 / E2
#   F8     | karma + zincir, rüzgâr + türbülans                              | E1 / E2 / E3
#   F9     | cila: F8 + daha çok hover manevrası, sakin hover manevrası tekrarı | E1 / E2 / E3
#   F10    | doğal zarf (10–120 kt, 50–1500 ft, Δψ ≤ 270°), pirouette, hover   | E0–E3, hava sıcaklığı
#          | hassasiyeti ×0.5, hover dönüşü ≤ 360° (2026-10-01)                | standart −10…+30 °C
#   F11    | hassasiyet okulu: dönüş / pirouette / kayma / bob, %50 F10 tekrarı | E0–E2, −10…+30 °C
#   F12    | denge: F10 karışımı + %45 tekrar (F11 ×3, F6, F7, F8, F9)          | F10'un çevresi
#   F13    | denge + iniş: F10 karışımı + %50 tekrar (F6a ×2, F6 ×2, F11 ×2, F8, F9) | F10'un çevresi
#
# F11 / F12 (2026-10-01): F11'de 3.5 M adım hassasiyeti getirdi (yerinde dönüşte kayma 18 → 6 ft) ama genel becerileri
# aşındırdı (seçim takımında tüm görevler 18/21 → 8/21: inişte collective ~0.2'de kalıp kızaklarda ağırlık < %70,
# ileri uçuşta irtifa bandına oturamama, hızlanmada daha uzun 56 psi aşımı). F12 F10 karışımına dönüp hassasiyeti
# F11 tekrarıyla korur.
#
# Hover ve ileri uçuş F1'den itibaren birlikte (2026-09-29, fl_v1 / fl_v2): hover'da eğitilmiş ağ ileri uçuşa
# geçince ilk seviyede ileri uçuş %0'da kaldı ve hover başarısı (deterministik F2) %100 → %25'e düştü (ağın ileri uçuş
# gözlemlerine tepkisi — büyük hava hızı / tork girdileri — hover için hiç eğitilmemişti). Sıfırdan yalnızca ileri
# uçuşta ise "tut" %97'ye hemen çıktı. İki rejim baştan birlikte öğretiliyor.
# Rehearsal: F2'den itibaren episode'ların %30–45'i eski seviyelerden (unutmaya karşı; kalkış ajanında tek seviyede
# eğitmek K5'i %100 → %65'e düşürmüştü, depart denemesi hover'ı %97 → %64'e).
# F6a (2026-09-29, fl_v3): F5'ten doğrudan F6'ya geçince 0.23 M adımda iniş %0 — ajan kızaklar ~2 ft'teyken hover
# ediyor, yerde hafif yüklü başlasa bile collective'i kaldırıp havalanıyor (kalkış ajanının sıfırdan tekrarında görülen
# yerel optimum, README 30.7); %2.5 kuyruk çarpması. Aynı sürede deterministik F3 10/12 → 3/12. Önerilen düzeltme:
# yalnızca "touch" / çok alçak hover başlangıçlı ara seviye (oturma okulu).

_HOVER_MIX = {"turn": 1 / 3, "move": 1 / 3, "bob": 1 / 3}
_FUEL = (150.0, 600.0)          # tank başına → 8800–9700 lbs (OGE hover ≤ ~53 psi, güç tavanı 56); en az 300 lbs yakıt

DEFAULT_FLIGHT_LEVELS: list[FlightLevel] = [
    FlightLevel(
        name="F1", description="Tut: %50 hover (havada başla, 15–300 ft, küçük bozukluk; 10 s), %50 ileri uçuş "
                               "(40–100 kt, 150–800 ft; hız / irtifa / heading'i 10 s tut)",
        p_cruise_start=0.5, n_cruise=(0, 0), cruise_start_kt=(40.0, 100.0), cruise_start_alt_ft=(150.0, 800.0),
        cruise_hold_s=10.0, p_hover_start=1.0, hover_start_alt_ft=(15.0, 300.0), start_perturb=0.3, hold_first_s=10.0,
        hold_T_s=10.0, fuel_lbs=_FUEL, promote_threshold=0.8),
    FlightLevel(
        name="F2", description="%50 hover: kalkış 10–300 ft ya da havada başla + 0–2 hover manevrası; %50 ileri uçuş: "
                               "1–2 tek eksenli Δ (hız ±10–20 kt / heading ±15–60° / irtifa ±50–150 ft, yumuşak süre hedefi)",
        p_cruise_start=0.5, n_cruise=(1, 2), cruise_probs={"u": 1 / 3, "psi": 1 / 3, "h": 1 / 3},
        du_kt=(10.0, 20.0), dpsi_deg=(15.0, 60.0), dh_ft=(50.0, 150.0), cruise_accel_fps2=2.0, cruise_decel_fps2=2.0,
        cruise_bank_deg=15.0, cruise_climb_fps=8.0, cruise_descent_fps=8.0, cruise_lag_s=4.0,
        p_hover_start=0.4, hover_start_alt_ft=(15.0, 300.0), takeoff_alt_ft=(10.0, 300.0), n_tasks=(0, 2),
        task_probs=_HOVER_MIX, turn_deg=(30.0, 120.0), move_ft=(15.0, 50.0), bob_ft=(15.0, 40.0), climb_fps=8.0,
        lag_s=5.0, fuel_lbs=_FUEL, promote_threshold=0.75, rehearse=("F1",), p_rehearse=0.3),
    FlightLevel(
        name="F3", description="%60 ileri uçuş: 2–4 Δ, birleşik (hız + heading + irtifa), büyük (±35 kt / ±150° / ±300 ft), "
                               "%30 kesen komut; %40 hover manevraları (1–3); %40 hafif rüzgâr",
        p_cruise_start=0.6, n_cruise=(2, 4), cruise_probs={"u": 0.2, "psi": 0.25, "h": 0.2, "mix": 0.35},
        du_kt=(10.0, 35.0), dpsi_deg=(20.0, 150.0), dh_ft=(50.0, 300.0), p_cruise_interrupt=0.3,
        p_hover_start=0.5, hover_start_alt_ft=(15.0, 300.0), takeoff_alt_ft=(10.0, 300.0), n_tasks=(1, 3),
        task_probs=_HOVER_MIX, climb_fps=8.0, lag_s=4.0, fuel_lbs=_FUEL, env_probs={"E0": 0.6, "E1": 0.4},
        promote_threshold=0.7, rehearse=("F1", "F2"), p_rehearse=0.35),
    FlightLevel(
        name="F4", description="Geçişler: hover → hızlanma (40–80 kt, +0–200 ft) + 0–1 Δ → duruş; ileri uçuş → duruş (hover)",
        p_hover_start=1.0, hover_start_alt_ft=(30.0, 300.0), p_cruise_start=0.5, cruise_start_kt=(30.0, 80.0),
        cruise_start_alt_ft=(100.0, 500.0), p_accel=1.0, accel_kt=(40.0, 80.0), accel_climb_ft=(0.0, 200.0),
        n_cruise=(0, 1), p_stop=1.0, stop_decel=(2.0, 3.0), fuel_lbs=_FUEL, env_probs={"E0": 0.6, "E1": 0.4},
        promote_threshold=0.7, rehearse=("F2", "F3"), p_rehearse=0.4),
    FlightLevel(
        name="F5", description="Zincir: yerden kalkış (20–60 ft) → hızlanma (40–80 kt) → 1–3 Δ komut → duruş",
        p_chain=1.0, chain_takeoff_ft=(20.0, 60.0), accel_kt=(40.0, 80.0), accel_climb_ft=(50.0, 300.0), n_cruise=(1, 3),
        cruise_probs={"u": 0.25, "psi": 0.3, "h": 0.2, "mix": 0.25}, du_kt=(10.0, 30.0), dpsi_deg=(20.0, 120.0),
        dh_ft=(50.0, 250.0), stop_decel=(2.0, 3.0), climb_fps=8.0, lag_s=5.0, fuel_lbs=_FUEL,
        env_probs={"E0": 0.6, "E1": 0.4}, promote_threshold=0.65, rehearse=("F2", "F3", "F4"), p_rehearse=0.4),
    FlightLevel(
        name="F6a", description="Oturma okulu (iniş, ters curriculum): %60 yerde hafif yüklü başla (collective'i indir, "
                                "otur), %40 çok alçak hover'dan (kızaklar 1.5–4 ft: yere değ, otur); sakin hava",
        p_touch_start=0.6, touch_coll=(0.05, 0.52), p_low_hover_start=0.4, p_land=1.0, n_tasks=(0, 0),
        descent_fps=5.0, lag_s=4.0, fuel_lbs=_FUEL, promote_threshold=0.8, rehearse=("F3", "F4", "F5"), p_rehearse=0.2),
    FlightLevel(
        name="F6", description="İniş: alçak hover / kısa kalkıştan pad'e iniş (%15 yerde hafif yüklü, %15 çok alçak "
                               "hover'dan); duruştan sonra iniş F7'den itibaren",
        p_hover_start=0.5, hover_start_alt_ft=(12.0, 60.0), takeoff_alt_ft=(12.0, 60.0), hold_first_s=5.0,
        n_tasks=(0, 1), move_ft=(15.0, 40.0), bob_ft=(10.0, 30.0), p_land=1.0, climb_fps=6.0, descent_fps=5.0,
        lag_s=4.0, p_touch_start=0.15, p_low_hover_start=0.15, fuel_lbs=_FUEL, env_probs={"E0": 0.7, "E1": 0.3},
        promote_threshold=0.7, rehearse=("F3", "F4", "F5"), p_rehearse=0.4),
    FlightLevel(
        name="F7", description="Karma: hover / kalkış / iniş / ileri uçuş / geçişler / zincir; hafif–orta rüzgâr, %30 "
                               "hafif türbülans",
        p_hover_start=0.3, hover_start_alt_ft=(15.0, 400.0), takeoff_alt_ft=(10.0, 600.0), n_tasks=(0, 3),
        p_target_change=0.2, p_land=0.3, p_cruise_start=0.35, n_cruise=(1, 3), p_cruise_interrupt=0.2,
        du_kt=(10.0, 30.0), dpsi_deg=(20.0, 150.0), dh_ft=(50.0, 250.0), p_accel=0.5, p_stop=0.6, p_chain=0.4,
        p_land_after_stop=0.3, p_touch_start=0.05, p_low_hover_start=0.05, climb_fps=10.0, descent_fps=5.0, lag_s=4.0,
        fuel_lbs=_FUEL, env_probs={"E1": 0.5, "E2": 0.5}, promote_threshold=0.65,
        rehearse=("F3", "F4", "F5", "F6"), p_rehearse=0.3),
    FlightLevel(
        name="F8", description="Son seviye: karma + zincir; rüzgâr 0–25 kt, türbülans yok / hafif / orta, gust'lar",
        p_hover_start=0.25, hover_start_alt_ft=(15.0, 400.0), takeoff_alt_ft=(10.0, 600.0), n_tasks=(0, 3),
        p_target_change=0.2, p_land=0.3, p_cruise_start=0.3, n_cruise=(1, 4), p_cruise_interrupt=0.25,
        du_kt=(10.0, 35.0), dpsi_deg=(20.0, 180.0), dh_ft=(50.0, 300.0), p_accel=0.5, p_stop=0.6, p_chain=0.5,
        p_land_after_stop=0.3, p_touch_start=0.05, p_low_hover_start=0.05, climb_fps=10.0, descent_fps=5.0, lag_s=4.0,
        fuel_lbs=_FUEL, env_probs={"E1": 0.3, "E2": 0.4, "E3": 0.3}, promote_threshold=0.6,
        rehearse=("F3", "F4", "F5", "F6", "F7"), p_rehearse=0.3),
    FlightLevel(
        name="F9", description="Cila (F8'in çevresi): F8 karışımı + daha çok hover manevrası (havada başlangıç %40, kalkış / hover'dan sonra 1–3 manevra); tekrar F2 (sakin hover manevraları) / F3 / F5 / F6 / F7 / F8",
        p_hover_start=0.4, hover_start_alt_ft=(15.0, 400.0), takeoff_alt_ft=(10.0, 600.0), n_tasks=(1, 3),
        p_target_change=0.2, p_land=0.3, p_cruise_start=0.3, n_cruise=(1, 4), p_cruise_interrupt=0.25,
        du_kt=(10.0, 35.0), dpsi_deg=(20.0, 180.0), dh_ft=(50.0, 300.0), p_accel=0.4, p_stop=0.6, p_chain=0.5,
        p_land_after_stop=0.3, p_touch_start=0.05, p_low_hover_start=0.05, climb_fps=10.0, descent_fps=5.0, lag_s=4.0,
        fuel_lbs=_FUEL, env_probs={"E1": 0.3, "E2": 0.4, "E3": 0.3}, promote_threshold=0.6,
        rehearse=("F2", "F3", "F5", "F6", "F7", "F8"), p_rehearse=0.35),
    FlightLevel(
        name="F10", description="Doğal zarf + hassasiyet + sıcak gün (2026-10-01): F9 karışımı; ileri uçuş 10–120 kt (2026-10-02; önce 130), "
                                "50–1500 ft, Δhız ≤ 50 kt, Δψ ≤ 270°, Δh ≤ 600 ft; hover dönüşü ≤ 360°, pirouette; hover "
                                "hassasiyeti ×0.5; hava sıcaklığı standart −10…+30 °C",
        p_hover_start=0.4, hover_start_alt_ft=(15.0, 800.0), takeoff_alt_ft=(10.0, 1000.0), n_tasks=(1, 3),
        task_probs={"turn": 0.3, "move": 0.2, "bob": 0.2, "pirouette": 0.3}, turn_deg=(30.0, 360.0),
        pirouette_radius_ft=(80.0, 120.0), pirouette_s=(40.0, 60.0),
        p_target_change=0.2, p_land=0.3, p_cruise_start=0.3, cruise_start_kt=(15.0, 115.0),
        cruise_start_alt_ft=(60.0, 1450.0), n_cruise=(1, 4), p_cruise_interrupt=0.25, du_kt=(10.0, 50.0),
        dpsi_deg=(20.0, 270.0), dh_ft=(50.0, 600.0), cruise_kt_env=(CMD_MIN_KT, CMD_MAX_KT),
        cruise_alt_env=(CMD_MIN_ALT_FT, CMD_MAX_ALT_FT), p_accel=0.4, accel_kt=(15.0, 110.0), accel_climb_ft=(0.0, 300.0),
        p_stop=0.6, p_chain=0.5, p_land_after_stop=0.3, p_touch_start=0.05, p_low_hover_start=0.05, climb_fps=10.0,
        descent_fps=5.0, lag_s=4.0, fuel_lbs=_FUEL, env_probs={"E0": 0.15, "E1": 0.25, "E2": 0.35, "E3": 0.25},
        hover_precision=0.5, delta_T_C=(-10.0, 30.0), promote_threshold=0.6,
        rehearse=("F2", "F3", "F5", "F6", "F7", "F8", "F9"), p_rehearse=0.3),
    FlightLevel(
        name="F11", description="Hassasiyet okulu (2026-10-01): hover'da 2–4 görev — yerinde dönüş 30–360° (%40), pirouette "
                                "(%30), kayma, bob; hover hassasiyeti ×0.5; sakin / hafif rüzgâr; −10…+30 °C; %50 F10 tekrarı",
        p_hover_start=0.7, hover_start_alt_ft=(12.0, 150.0), takeoff_alt_ft=(10.0, 100.0), n_tasks=(2, 4),
        task_probs={"turn": 0.4, "pirouette": 0.3, "move": 0.15, "bob": 0.15}, turn_deg=(30.0, 360.0),
        pirouette_radius_ft=(80.0, 120.0), pirouette_s=(40.0, 60.0), climb_fps=8.0, descent_fps=5.0, lag_s=4.0,
        p_land=0.2, fuel_lbs=_FUEL, env_probs={"E0": 0.4, "E1": 0.4, "E2": 0.2}, hover_precision=0.5,
        delta_T_C=(-10.0, 30.0), promote_threshold=0.6, rehearse=("F10",), p_rehearse=0.5),
]
# F12 (2026-10-01): F10'un kendisi (görevler, doğal zarf, çevre, hassasiyet ×0.5), tekrar F11 ağırlıklı — bkz. yukarıdaki not
DEFAULT_FLIGHT_LEVELS.append(replace(
    DEFAULT_FLIGHT_LEVELS[[lv.name for lv in DEFAULT_FLIGHT_LEVELS].index("F10")], name="F12",
    description="Denge (2026-10-01): F10 karışımı; tekrar %45 — F11 (×3, hassasiyet), F6 (iniş), F7, F8, F9",
    rehearse=("F11", "F11", "F11", "F6", "F7", "F8", "F9"), p_rehearse=0.45))
# F13 (2026-10-01): F12 + iniş tekrarı. F11 soyunda ajan yerde collective'i ~0.2'de tutuyor (action −0.85; fl10'da −1.0 →
# collective 0.02, kızaklarda ağırlık %96) → kızaklarda ağırlık %61–64 < %70 → F6a / F6'da iniş 0/8. Tekrar %50: F6a ×2
# (oturma okulu), F6 ×2, F11 ×2, F8, F9.
DEFAULT_FLIGHT_LEVELS.append(replace(
    DEFAULT_FLIGHT_LEVELS[[lv.name for lv in DEFAULT_FLIGHT_LEVELS].index("F12")], name="F13",
    description="Denge + iniş (2026-10-01): F10 karışımı; tekrar %50 — F6a ×2, F6 ×2, F11 ×2, F8, F9",
    rehearse=("F6a", "F6a", "F6", "F6", "F11", "F11", "F8", "F9"), p_rehearse=0.5))


__all__ = ["FlightLevel", "DEFAULT_FLIGHT_LEVELS", "ENV_STAGES", "find_flight_level", "flight_time_target",
           "cruise_yaw_rate_dps", "deadline_of", "CRUISE_MIN_KT", "CRUISE_MAX_KT", "CRUISE_MIN_ALT_FT",
           "CRUISE_MAX_ALT_FT", "GROUND_H_FT", "MIN_HOVER_H_FT", "MAX_HOVER_H_FT", "TANK_CAPACITY_LBS", "KT",
           "CMD_MIN_KT", "CMD_MAX_KT", "CMD_MIN_ALT_FT", "CMD_MAX_ALT_FT", "VNE_KT"]
