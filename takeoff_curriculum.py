from __future__ import annotations

"""
TAKEOFF CURRICULUM — yerden kalkış, hover, hover manevraları, hedef değişikliği, iniş, ağırlık ve bozucular
==========================================================================================================

`helicopter_env_takeoff.py` ile kullanılır. Ajan dört kumandayı (collective, longitudinal cyclic, lateral
cyclic, pedal) doğrudan kullanır; bu dosya yalnızca görevleri ve süre hedeflerini tanımlar.

Görev türleri (bir episode bir görev dizisidir):
  hold      : olduğun yerde hover'ı tut (havada başlayan episode'un ilk görevi)
  takeoff   : yerden kalk, pad'in üstünde hedef irtifada (CG AGL, ft) hover
  climb_to  : hedef değişikliği — kalkış sürerken yeni irtifa (mutlak)
  turn      : yerinde dönüş (pedal), Δψ
  move      : yer değiştirme (reposition / sidestep): burun eksenine göre ileri / sağa Δx, Δy
  bob       : bob-up / bob-down, Δh
  land      : bulunduğun hedef noktaya yumuşak iniş, kızaklar yerde, collective indirilmiş
  recover   : bozucu (kumanda darbesi, itki, attitude) sonrası hover'a dönüş

Süre hedefi T = tepki + (yol / hız) (ör. kalkış: Δh / tırmanış hızı); son sınır = max(1.25·T, T + 2 s).
Seviyeler K1 → K9: önce havada hover'ı tutmak, sonra kalkış (alçak → 1000 ft), hover manevraları (ADS-33
hover MTE'leri: hovering turn, lateral reposition, bob-up / bob-down), hedef değişikliği, iniş (önce alçak
hover'dan: K7a), ağırlık / CG ve bozucular, en sonda karma.
"""

import math
from dataclasses import dataclass, field

import numpy as np

GROUND_H_FT = 6.3              # yerde CG yüksekliği (reset00.xml altitudeAGL); kızak yüksekliği = h − 6.3
# iniş alçalma profili (env'in yönlendirme hedefi ve iniş süre hedefi aynı profili kullanır):
#   ḣ_istenen = −clip(LAND_PROFILE_K · kızak yüksekliği, LAND_PROFILE_MIN_FPS, descent_fps)
LAND_PROFILE_K = 0.25          # 1/s
LAND_PROFILE_MIN_FPS = 1.0     # son ~4 ft: 1 ft/s
LAND_SETTLE_S = 3.0            # temas → collective aşağı, dört nokta, oturma
MIN_HOVER_H_FT = 12.0          # hover hedefi bunun altına inmez (kızaklar ~6 ft yukarıda)
MAX_HOVER_H_FT = 1000.0
TANK_CAPACITY_LBS = 890.0      # iki tank; boş ağırlık 8500 lbs → 8500–10280 lbs, CG 169.5–175.2 in


@dataclass
class TakeoffLevel:
    name: str
    description: str
    # --- başlangıç ---------------------------------------------------------------
    p_hover_start: float = 0.0             # 0 → yerde (rotor warm-up sonrası), 1 → havada (teleport + reset oto-pilotu)
    hover_start_alt_ft: tuple = (15.0, 150.0)
    # iniş için geriye doğru curriculum (reverse curriculum): hedefe yakın başlangıçlar, görev yalnızca "land"
    p_touch_start: float = 0.0             # yerde, collective kısmen kalkık (kızaklar hafif yüklü): collective'i indir
    touch_coll: tuple = (0.30, 0.52)       # 8500 lbs için; ağırlıkla +0.057 / 1000 lbs
    p_low_hover_start: float = 0.0         # çok alçak hover (kızaklar 1.5–4 ft): yere değ ve otur
    low_hover_hs_ft: tuple = (1.5, 4.0)
    start_perturb: float = 0.0             # havada başlangıç bozukluğu ölçeği: hız ±3·x ft/s, attitude ±3·x°
    # --- görevler ----------------------------------------------------------------
    takeoff_alt_ft: tuple = (10.0, 25.0)   # CG AGL
    hold_first_s: float = 10.0             # ilk görev (kalkış / hover tut) için hover'da kalma süresi
    hold_s: float = 5.0                    # diğer görevler
    n_tasks: tuple = (0, 0)                # kalkıştan (ya da hover tut'tan) sonra hover manevrası sayısı
    task_probs: dict = field(default_factory=lambda: {"turn": 1 / 3, "move": 1 / 3, "bob": 1 / 3})
    turn_deg: tuple = (30.0, 180.0)
    move_ft: tuple = (15.0, 60.0)
    bob_ft: tuple = (15.0, 50.0)
    p_target_change: float = 0.0           # kalkış sürerken yeni irtifa (bir ya da iki kez)
    n_target_changes: tuple = (1, 2)
    change_frac: tuple = (0.25, 0.75)      # önceki süre hedefinin bu kesrinde
    p_land: float = 0.0                    # son görev iniş
    # --- çeviklik (süre hedefi) ----------------------------------------------------
    climb_fps: float = 8.0
    descent_fps: float = 6.0
    move_fps: float = 8.0
    accel_fps2: float = 4.0
    yaw_rate_dps: float = 15.0
    lag_s: float = 3.0
    hold_T_s: float = 8.0                  # "hover tut" görevinin süre hedefi (başlangıç bozukluğunu toparlama)
    recover_s: float = 10.0                # bozucu sonrası toparlanma süre hedefi
    # --- ağırlık / bozucular ----------------------------------------------------------
    fuel_lbs: tuple = (0.0, 0.0)           # her tank için [min, max]
    n_disturb: tuple = (0, 0)              # hover'da bozucu sayısı
    disturb_probs: dict = field(default_factory=lambda: {"kick": 0.4, "push": 0.35, "tilt": 0.25})
    promote_threshold: float | None = None
    # --- tekrar (rehearsal): eğitimde episode'ların bu kadarı eski seviyelerden (unutmayı önler) ---------------
    rehearse: tuple = ()
    p_rehearse: float = 0.0

    # ------------------------------------------------------------------------------
    def sample_start(self, rng) -> dict:
        if self.p_touch_start + self.p_low_hover_start > 0.0:    # (yalnızca bu seviyelerde ek rastgele sayı)
            u = rng.random()
            if u < self.p_touch_start:
                return dict(start="touch", h=GROUND_H_FT)
            if u < self.p_touch_start + self.p_low_hover_start:
                return dict(start="low", h=GROUND_H_FT + float(rng.uniform(*self.low_hover_hs_ft)))
        if rng.random() < self.p_hover_start:
            return dict(start="hover", h=float(rng.uniform(*self.hover_start_alt_ft)))
        return dict(start="ground", h=GROUND_H_FT)

    def sample_fuel(self, rng) -> tuple:
        lo, hi = self.fuel_lbs
        return float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))

    def _pick(self, rng, probs: dict) -> str:
        keys = list(probs)
        p = np.array([probs[k] for k in keys], dtype=float)
        return keys[int(rng.choice(len(keys), p=p / p.sum()))]

    def sample_disturbance(self, rng) -> dict:
        kind = self._pick(rng, self.disturb_probs)
        if kind == "kick":                 # kumanda darbesi: bir eksende 0.4–1.0 s ek sapma
            axis = int(rng.choice(4, p=[0.2, 0.25, 0.3, 0.25]))
            mag = float(rng.uniform(*((0.05, 0.10), (0.15, 0.35), (0.15, 0.35), (0.2, 0.4))[axis]))
            return dict(kind="recover", dist="kick", axis=axis, mag=mag * rng.choice((-1.0, 1.0)),
                        dur=float(rng.uniform(0.4, 1.0)))
        if kind == "push":                 # itki: yatay hız 5–12 ft/s (rastgele yön) + dikey ±4 ft/s
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            mag = float(rng.uniform(5.0, 12.0))
            return dict(kind="recover", dist="push", dn=mag * math.cos(ang), de=mag * math.sin(ang),
                        dw=float(rng.uniform(-4.0, 4.0)))
        return dict(kind="recover", dist="tilt", dphi=float(rng.uniform(5.0, 12.0) * rng.choice((-1.0, 1.0))),
                    dtheta=float(rng.uniform(4.0, 10.0) * rng.choice((-1.0, 1.0))) * float(rng.random() < 0.5))

    def sample_task(self, rng) -> dict:
        kind = self._pick(rng, self.task_probs)
        if kind == "turn":
            return dict(kind="turn", dpsi=float(rng.uniform(*self.turn_deg) * rng.choice((-1.0, 1.0))))
        if kind == "move":
            d = float(rng.uniform(*self.move_ft))
            ang = float(rng.choice((0.0, 90.0, 180.0, 270.0)) + rng.uniform(-20.0, 20.0) * float(rng.random() < 0.3))
            return dict(kind="move", dx=d * math.cos(math.radians(ang)), dy=d * math.sin(math.radians(ang)))
        return dict(kind="bob", dh=float(rng.uniform(*self.bob_ft) * rng.choice((-1.0, 1.0))))

    def sample_schedule(self, rng, start: str) -> list[dict]:
        if start in ("touch", "low"):                               # iniş son aşaması: yalnızca otur
            return [dict(kind="land")]
        tasks: list[dict] = []
        if start == "ground":
            h1 = float(rng.uniform(*self.takeoff_alt_ft))
            tasks.append(dict(kind="takeoff", h=h1))
            if rng.random() < self.p_target_change:
                h = h1
                for _ in range(int(rng.integers(self.n_target_changes[0], self.n_target_changes[1] + 1))):
                    h = _changed_altitude(rng, h)
                    tasks.append(dict(kind="climb_to", h=h, _frac=float(rng.uniform(*self.change_frac))))
        else:
            tasks.append(dict(kind="hold"))
        n = int(rng.integers(self.n_tasks[0], self.n_tasks[1] + 1))
        body = [self.sample_task(rng) for _ in range(n)]
        for _ in range(int(rng.integers(self.n_disturb[0], self.n_disturb[1] + 1))):
            body.insert(int(rng.integers(0, len(body) + 1)), self.sample_disturbance(rng))
        tasks += body
        if rng.random() < self.p_land:
            tasks.append(dict(kind="land"))
        return tasks

    def time_target(self, kind: str, amount: float) -> float:
        return task_time_target(kind, amount, self)


def _changed_altitude(rng, h: float) -> float:
    """Kalkış sürerken yeni hedef: daha aşağıda dur, daha yükseğe çık ya da alçak hover'a dön."""
    opts = []
    if h > 40.0:
        opts.append(("lower", max(MIN_HOVER_H_FT + 3.0, 0.3 * h), 0.7 * h))
    if h < 700.0:
        opts.append(("higher", 1.3 * h + 10.0, min(MAX_HOVER_H_FT, 2.5 * h + 50.0)))
    if h > 60.0:
        opts.append(("low", MIN_HOVER_H_FT + 3.0, 30.0))
    _, lo, hi = opts[int(rng.integers(len(opts)))]
    return float(rng.uniform(lo, max(lo, hi)))


def task_time_target(kind: str, amount: float, lv: TakeoffLevel) -> float:
    """amount: kalkış / climb_to / bob → Δh (ft, işaretli); turn → Δψ (°); move → mesafe (ft);
    land → kızak yüksekliği (ft); hold / recover → kullanılmaz."""
    if kind in ("takeoff", "climb_to", "bob"):
        rate = lv.climb_fps if amount >= 0.0 else lv.descent_fps
        return lv.lag_s + abs(amount) / rate
    if kind == "turn":
        return lv.lag_s + abs(amount) / lv.yaw_rate_dps
    if kind == "move":
        return lv.lag_s + abs(amount) / lv.move_fps + lv.move_fps / lv.accel_fps2
    if kind == "land":
        return lv.lag_s + land_profile_time(amount, lv.descent_fps) + LAND_SETTLE_S
    if kind == "recover":
        return lv.recover_s
    return lv.hold_T_s                                                  # hold


def land_profile_time(hs: float, descent_fps: float) -> float:
    """Alçalma profilini izleyerek kızak yüksekliği hs'den temasa süre: descent_fps ile sabit alçalma, sonra
    ḣ = −k·hs (üstel yavaşlama), en sonda 1 ft/s. (Eski süre hedefi hs / descent + 3 s idi; profil yavaşlamasını
    saymıyordu: 50 ft'ten iniş profille ~23 s sürerken son sınır 20 s'ydi.)"""
    hs = max(0.0, float(hs))
    k, v_min = LAND_PROFILE_K, LAND_PROFILE_MIN_FPS
    t = 0.0
    h_fast = descent_fps / k                      # bunun üstünde hız sınırı descent_fps
    if hs > h_fast:
        t += (hs - h_fast) / descent_fps
        hs = h_fast
    h_slow = v_min / k                            # bunun altında 1 ft/s
    if hs > h_slow:
        t += math.log(hs / h_slow) / k
        hs = h_slow
    return t + hs / v_min


def deadline_of(T: float) -> float:
    return max(1.25 * T, T + 2.0)


# ---------------------------------------------------------------------------------
# Seviyeler
# ---------------------------------------------------------------------------------

_MIX = {"turn": 1 / 3, "move": 1 / 3, "bob": 1 / 3}

DEFAULT_TAKEOFF_LEVELS: list[TakeoffLevel] = [
    TakeoffLevel(
        name="K1", description="Hover'ı tut: havada başla (15–150 ft, küçük bozukluk), dört kumandayla 15 s sabit kal",
        p_hover_start=1.0, hover_start_alt_ft=(15.0, 150.0), start_perturb=0.3, hold_first_s=15.0, hold_T_s=10.0),
    TakeoffLevel(
        name="K2", description="Yerden kalkış → alçak hover (10–25 ft), pad'in üstünde 10 s",
        takeoff_alt_ft=(10.0, 25.0), climb_fps=5.0, lag_s=6.0),
    TakeoffLevel(
        name="K3", description="Kalkış → 25–200 ft hover (tırmanış 8 ft/s)",
        takeoff_alt_ft=(25.0, 200.0), climb_fps=8.0, lag_s=5.0),
    TakeoffLevel(
        name="K4", description="Kalkış → 200–1000 ft hover (tırmanış 12 ft/s)",
        takeoff_alt_ft=(200.0, 1000.0), climb_fps=12.0, lag_s=5.0),
    TakeoffLevel(
        name="K5", description="Hover manevraları: yerinde dönüş, ileri / geri / yana kayma, bob-up / bob-down",
        p_hover_start=0.5, hover_start_alt_ft=(20.0, 300.0), takeoff_alt_ft=(15.0, 300.0), n_tasks=(2, 4),
        task_probs=_MIX, turn_deg=(30.0, 180.0), move_ft=(15.0, 60.0), bob_ft=(15.0, 50.0), climb_fps=8.0,
        lag_s=4.0, promote_threshold=0.7),
    TakeoffLevel(
        name="K6", description="Hedef değişikliği: kalkış sürerken yeni irtifa (aşağıda dur / yükseğe çık / alçak hover)",
        takeoff_alt_ft=(60.0, 1000.0), p_target_change=0.9, n_target_changes=(1, 2), n_tasks=(0, 1), climb_fps=10.0,
        lag_s=5.0, promote_threshold=0.7),
    TakeoffLevel(
        name="K7a", description="İniş (alçak irtifadan): 12–60 ft hover ya da kısa kalkış (+0–1 manevra) → pad'e yumuşak iniş; "
                                "%35 yerde hafif yüklü, %35 çok alçak hover'dan (reverse curriculum)",
        p_hover_start=0.5, hover_start_alt_ft=(12.0, 60.0), takeoff_alt_ft=(12.0, 60.0), hold_first_s=5.0, n_tasks=(0, 1),
        move_ft=(15.0, 40.0), bob_ft=(10.0, 30.0), p_land=1.0, climb_fps=6.0, descent_fps=5.0, lag_s=4.0,
        p_touch_start=0.35, p_low_hover_start=0.35, rehearse=("K5", "K6"), p_rehearse=0.15),
    TakeoffLevel(
        name="K7", description="İniş: (kalkış / hover, 0–2 manevra) → pad'e yumuşak iniş, kızaklar yerde",
        p_hover_start=0.5, hover_start_alt_ft=(15.0, 200.0), takeoff_alt_ft=(15.0, 200.0), n_tasks=(0, 2), p_land=1.0,
        climb_fps=8.0, descent_fps=5.0, lag_s=4.0, promote_threshold=0.7, rehearse=("K5", "K6", "K7a"),
        p_rehearse=0.2),
    TakeoffLevel(
        name="K8", description="Ağırlık / CG (8500–10280 lbs) ve hover'da bozucular (kumanda darbesi, itki, attitude); "
                               "%20 ağırlıklı iniş son aşaması (yerde hafif yüklü / çok alçak hover)",
        p_hover_start=0.3, hover_start_alt_ft=(20.0, 300.0), takeoff_alt_ft=(15.0, 600.0), n_tasks=(1, 3),
        p_target_change=0.3, p_land=0.4, fuel_lbs=(0.0, TANK_CAPACITY_LBS), n_disturb=(1, 2), climb_fps=10.0,
        descent_fps=5.0, lag_s=4.0, promote_threshold=0.65, rehearse=("K4", "K5", "K6", "K7"), p_rehearse=0.1,
        p_touch_start=0.1, p_low_hover_start=0.1),
    TakeoffLevel(
        name="K9", description="Karma (uzun ince ayar): tüm görevler, rastgele ağırlık, bozucular",
        p_hover_start=0.3, hover_start_alt_ft=(15.0, 400.0), takeoff_alt_ft=(10.0, 1000.0), n_tasks=(0, 3),
        p_target_change=0.3, p_land=0.5, fuel_lbs=(0.0, TANK_CAPACITY_LBS), n_disturb=(0, 2), climb_fps=10.0,
        descent_fps=5.0, lag_s=4.0, promote_threshold=0.65, rehearse=("K5", "K6", "K7"), p_rehearse=0.08,
        p_touch_start=0.05, p_low_hover_start=0.05),
]


def find_takeoff_level(level, levels=None) -> int:
    levels = levels or DEFAULT_TAKEOFF_LEVELS
    if isinstance(level, (int, np.integer)):
        return int(np.clip(level, 0, len(levels) - 1))
    if str(level).isdigit():
        return int(np.clip(int(level), 0, len(levels) - 1))
    for i, lv in enumerate(levels):
        if lv.name.lower() == str(level).lower():
            return i
    raise KeyError(f"seviye yok: {level}")


__all__ = ["TakeoffLevel", "DEFAULT_TAKEOFF_LEVELS", "find_takeoff_level", "task_time_target", "deadline_of",
           "land_profile_time", "GROUND_H_FT", "MIN_HOVER_H_FT", "MAX_HOVER_H_FT", "TANK_CAPACITY_LBS",
           "LAND_PROFILE_K", "LAND_PROFILE_MIN_FPS", "LAND_SETTLE_S"]
