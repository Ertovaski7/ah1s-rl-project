from __future__ import annotations

"""
MANEUVER CURRICULUM — Δ komutları + süre hedefi, 0–100 kt
=========================================================

Komut curriculum'unun (command_curriculum.py) devamı. Arayüz aynı: Δheading /
Δhız / Δirtifa. Yeni olan: her komutun bir **süre hedefi** T var. Ajan hedef
bandına T (+ pay) içinde girip orada kalmalı. Süre kısaldıkça helikopterin tek
yolu açıları kullanmak: ileri uçuşta yatışlı dönüş, hızlanmada burun aşağı,
yavaşlamada burun yukarı, tırmanışta collective + yunuslama.

Süre hedefi seviyenin "çeviklik" değerlerinden hesaplanır (komut anındaki hıza
göre; `time_target`):

    T = gecikme + max(|Δψ| / ψ̇_hedef(V), |Δv| / ivme_hedef, |Δh| / tırmanış_hedef)

    ψ̇_hedef(V) = min(pedal_ψ̇, g·tan(yatış_hedef) / V)     (V: ileri hız)

Düşük hızda dönüş pedalla (pedal_ψ̇), ileri uçuşta yatışla sınırlıdır: 60 kt'ta
45° yatış ≈ 18°/s, 100 kt'ta ≈ 11°/s. Arayüz T'yi doğrudan da verebilir.

Referans (ADS-33, askerî helikopter uçuş niteliği standardı): hover / düşük hız
büyük açı değişimlerinde "orta" çeviklik ≈ pitch 13°/s, yaw 22°/s; "agresif"
≈ pitch 30°/s, yaw 50°/s. Bu modelde (JSBSim AH-1S) ölçülen tavanlar: yatışa
geçiş ~22–27°/s, pitch ~15°/s, pedal dönüşü ~25°/s, collective ile ~30 ft/s.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from command_curriculum import AXES, AxisRange, Level

G_FPS2 = 32.174
KT = 1.6878                                   # ft/s / kt


@dataclass(frozen=True)
class ManeuverLevel(Level):
    """Level + çeviklik hedefleri. start_speed_fps: başlangıç hızı aralığı (ft/s)."""
    pedal_rate_dps: float = 12.0              # düşük hızda dönüş hızı hedefi
    bank_deg: float = 20.0                    # ileri uçuşta dönüş için yatış hedefi
    accel_fps2: float = 2.0                   # hız değişimi
    climb_fps: float = 6.0                    # irtifa değişimi
    lag_s: float = 2.0                        # tepki gecikmesi payı
    n_commands: int = 3                       # episode başına komut sayısı
    speed_bounds_fps: tuple = (0.0, 170.0)    # hedef hız bu aralıkta kalır (Δv işareti buna göre seçilir)

    def time_target(self, cmd: dict, speed_fps: float) -> float:
        return time_target(cmd, speed_fps, self.pedal_rate_dps, self.bank_deg, self.accel_fps2, self.climb_fps,
                           self.lag_s)


def heading_rate_target(speed_fps: float, pedal_rate_dps: float, bank_deg: float) -> float:
    v = max(abs(speed_fps), 1.0)
    bank_rate = math.degrees(G_FPS2 * math.tan(math.radians(bank_deg)) / v)
    return float(min(pedal_rate_dps, bank_rate))


def time_target(cmd: dict, speed_fps: float, pedal_rate_dps: float, bank_deg: float, accel_fps2: float,
                climb_fps: float, lag_s: float) -> float:
    need = [0.0]
    if cmd.get("heading"):
        need.append(abs(cmd["heading"]) / heading_rate_target(speed_fps, pedal_rate_dps, bank_deg))
    if cmd.get("speed"):
        need.append(abs(cmd["speed"]) / accel_fps2)
    if cmd.get("altitude"):
        need.append(abs(cmd["altitude"]) / climb_fps)
    return float(lag_s + max(need))


# ---------------------------------------------------------------------------
# Seviyeler
# ---------------------------------------------------------------------------

_H = lambda mx, prev=0.0, mn=5.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)     # noqa: E731
_V = lambda mx, prev=0.0, mn=3.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)     # noqa: E731
_A = lambda mx, prev=0.0, mn=15.0: AxisRange(max_abs=mx, prev_max=prev, min_abs=mn)    # noqa: E731

_MIX3 = {"heading": 0.45, "speed": 0.30, "altitude": 0.25}

DEFAULT_MANEUVER_LEVELS: list[ManeuverLevel] = [
    ManeuverLevel(
        name="M1", description="Temel kontrol, düşük hız (0–30 kt): küçük komutlar, rahat süre",
        heading=_H(60.0), speed=_V(15.0), altitude=_A(60.0), axis_probs=_MIX3,
        start_alt_ft=(500.0, 900.0), start_speed_fps=(0.0, 50.0),
        pedal_rate_dps=10.0, bank_deg=15.0, accel_fps2=2.0, climb_fps=6.0, lag_s=3.0),
    ManeuverLevel(
        name="M2", description="Düşük hızda çeviklik: hızlı pedal dönüşü, ivmelenme/yavaşlama, bob-up/down",
        heading=_H(180.0, 60.0), speed=_V(30.0, 15.0), altitude=_A(100.0, 60.0), axis_probs=_MIX3,
        start_alt_ft=(500.0, 1000.0), start_speed_fps=(0.0, 50.0),
        pedal_rate_dps=18.0, bank_deg=25.0, accel_fps2=4.0, climb_fps=12.0, lag_s=2.5),
    ManeuverLevel(
        name="M3", description="İleri uçuş (40–80 kt): yatışlı dönüş (30°), hız ve irtifa değişimi",
        heading=_H(120.0, 0.0, 10.0), speed=_V(30.0, 0.0, 5.0), altitude=_A(120.0, 0.0, 20.0), axis_probs=_MIX3,
        start_alt_ft=(600.0, 1200.0), start_speed_fps=(40.0 * KT, 80.0 * KT),
        pedal_rate_dps=18.0, bank_deg=30.0, accel_fps2=4.0, climb_fps=12.0, lag_s=2.5,
        mix_single=0.5, single_ranges={"heading": _H(180.0), "speed": _V(30.0), "altitude": _A(100.0)}),
    ManeuverLevel(
        name="M4", description="Tüm zarf (0–100 kt): 45° yatış, 22°/s pedal, 5 ft/s² ivme, 16 ft/s tırmanış",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 30.0, 5.0), altitude=_A(150.0, 100.0, 20.0), axis_probs=_MIX3,
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT),
        pedal_rate_dps=22.0, bank_deg=45.0, accel_fps2=5.0, climb_fps=16.0, lag_s=2.2),
    ManeuverLevel(
        name="M5", description="Agresif + birleşik: 55° yatış, 25°/s pedal, 6 ft/s² ivme, 20 ft/s tırmanış",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.5,
        single_ranges={"heading": _H(180.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT),
        pedal_rate_dps=25.0, bank_deg=55.0, accel_fps2=6.0, climb_fps=20.0, lag_s=2.0),
]


# ---------------------------------------------------------------------------
# Zorlayıcı episode'lar (dayanıklılık / robustness): S1, S2
# ---------------------------------------------------------------------------
# Manevra bitmeden yeni komut gelir (kesilen komut). Yeni Δ o anki ÖLÇÜLEN duruma göre
# uygulanır (pilot gibi "buradan itibaren"); komut verilmeyen eksen önceki hedefine devam
# eder (dönüş ortasında Δh → dönüş sürer, tırmanış eklenir). Süre hedefi o andaki hareketi
# hesaba katar (dynamic_time_target): ters yöne dönen bir yaw hızını / dikey hızı / ivmeyi
# önce durdurmak gerekir.

_STRESS_RANGE = {"heading": 180.0, "speed": 50.0, "altitude": 150.0}


def main_axis(cmd: dict) -> str | None:
    """Komutun ana ekseni: aralığına göre en büyük |Δ|."""
    best, val = None, 0.0
    for a in AXES:
        x = abs(cmd.get(a, 0.0)) / _STRESS_RANGE[a]
        if x > val:
            best, val = a, x
    return best


def dynamic_time_target(eff: dict, speed_fps: float, yaw_rate_dps: float, vs_fps: float, u_dot_fps2: float,
                        lv: "ManeuverLevel", yaw_decel_dps2: float = 12.0, vs_decel_fps2: float = 10.0,
                        accel_jerk_fps3: float = 4.0) -> tuple[float, float]:
    """Süre hedefi + o anki hareketin payı. eff: eksen başına kalan Δ (komut + devam eden hata).

    Ters yöndeki hareket önce durdurulur: durma süresi |x|/a ve durma yolu x²/(2a) eklenir.
    Hareketsiz başlangıçta (düz uçuş) sonuç time_target ile aynıdır. Döndürür: (T, pay)."""
    need, need0 = [0.0], [0.0]

    def axis(delta, rate, cur, decel):
        base = abs(delta) / rate
        if delta and cur * delta < 0.0 and abs(cur) > 1e-6:
            t_stop, d_stop = abs(cur) / decel, cur * cur / (2.0 * decel)
            return base, t_stop + (abs(delta) + d_stop) / rate
        return base, base

    if eff.get("heading"):
        b, x = axis(eff["heading"], heading_rate_target(speed_fps, lv.pedal_rate_dps, lv.bank_deg),
                    yaw_rate_dps if abs(yaw_rate_dps) > 2.0 else 0.0, yaw_decel_dps2)
        need0.append(b), need.append(x)
    if eff.get("speed"):
        b, x = axis(eff["speed"], lv.accel_fps2, u_dot_fps2 if abs(u_dot_fps2) > 1.0 else 0.0, accel_jerk_fps3)
        need0.append(b), need.append(x)
    if eff.get("altitude"):
        b, x = axis(eff["altitude"], lv.climb_fps, vs_fps if abs(vs_fps) > 2.0 else 0.0, vs_decel_fps2)
        need0.append(b), need.append(x)
    T0, T = lv.lag_s + max(need0), lv.lag_s + max(need)
    return float(T), float(T - T0)


@dataclass(frozen=True)
class StressLevel(ManeuverLevel):
    """Zorlayıcı episode: kesilen / ters komutlar (+ isteğe bağlı zarf sınırları) ve düz M5 tekrarı.

    Episode'un ilk komutu rastgele; sonrakiler `p_interrupt` olasılıkla öncekinin süre hedefinin
    `interrupt_frac` kesrinde gelir (manevra bitmeden). Kesen komutun türü:
      reverse: aynı eksen, ters işaret (slalomda ani ters dönüş, tırmanırken ani dalış, ani fren)
      cross:   başka eksen; önceki eksen devam eder (dönüş ortasında ani tırmanma)
      extend:  aynı eksen, aynı yön ("biraz daha")
      random:  M5 gibi rastgele (tek ya da birleşik)
    """
    n_commands_range: tuple = (4, 6)
    p_interrupt: float = 0.7
    interrupt_frac: tuple = (0.2, 0.8)
    kind_probs: dict = field(default_factory=lambda: {"reverse": 0.4, "cross": 0.3, "extend": 0.1, "random": 0.2})
    p_standard: float = 0.25                 # bu oranda episode düz M5 (3 komut, tam süre) — unutmasın
    edge: bool = False                       # zarf sınırları: kenar başlangıçları + sınıra giden komutlar
    p_edge_start: float = 0.6
    p_edge_cmd: float = 0.3
    low_alt_ft: tuple = (300.0, 550.0)
    # Seviye atlama eşiği (episode başarısı). Episode'da 4–8 komut var ve hepsi başarılı olmalı; komut başına
    # ~%92 başarı ≈ episode'da %65. Komut türü kapıları (kesen / kesilen / tek eksen / birleşik) yine %80.
    promote_threshold: float = 0.65
    hover_speed_fps: tuple = (0.0, 8.0 * KT)
    fast_speed_fps: tuple = (85.0 * KT, 100.0 * KT)

    def sample_start(self, rng: np.random.Generator) -> tuple[float, float]:
        h = float(rng.uniform(*self.start_alt_ft))
        u = float(rng.uniform(*self.start_speed_fps))
        if self.edge and rng.random() < self.p_edge_start:
            kind = rng.choice(["hover", "fast", "low"])
            if kind == "hover":
                u = float(rng.uniform(*self.hover_speed_fps))
            elif kind == "fast":
                u = float(rng.uniform(*self.fast_speed_fps))
            else:
                h = float(rng.uniform(*self.low_alt_ft))
        return h, u

    def _mag(self, rng, axis, lo=0.4, hi=1.0) -> float:
        mn = {"heading": 10.0, "speed": 5.0, "altitude": 20.0}[axis]
        return float(max(mn, rng.uniform(lo, hi) * _STRESS_RANGE[axis]))

    def _edge_item(self, rng) -> dict:
        kind = rng.choice(["speed_max", "speed_min", "alt_floor"])
        return {"heading": 0.0, "speed": 0.0, "altitude": 0.0, "_edge": str(kind)}

    def sample_schedule(self, rng: np.random.Generator) -> list[dict]:
        """Episode'un komut listesi: {heading, speed, altitude, _frac (None = önceki bitince), _cat, _edge}."""
        if rng.random() < self.p_standard:
            return [dict(self.sample_command(rng), _frac=None, _cat=None) for _ in range(3)]
        n = int(rng.integers(self.n_commands_range[0], self.n_commands_range[1] + 1))
        items = [dict(self.sample_command(rng), _frac=None, _cat=None)]
        kinds, p = list(self.kind_probs), np.array(list(self.kind_probs.values()), dtype=float)
        for _ in range(n - 1):
            prev = items[-1]
            if self.edge and rng.random() < self.p_edge_cmd:
                it = self._edge_item(rng)
            elif rng.random() < self.p_interrupt:
                kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
                ax = main_axis(prev) or "heading"
                it = {a: 0.0 for a in AXES}
                if kind == "reverse":
                    it[ax] = -math.copysign(self._mag(rng, ax, 0.5, 1.0), prev.get(ax, 0.0) or 1.0)
                elif kind == "cross":
                    others = [a for a in AXES if a != ax]
                    w = np.array([self.axis_probs.get(a, 1.0) for a in others], dtype=float)
                    ax2 = others[int(rng.choice(len(others), p=w / w.sum()))]
                    it[ax2] = self._mag(rng, ax2) * (1.0 if rng.random() < 0.5 else -1.0)
                elif kind == "extend":
                    it[ax] = math.copysign(self._mag(rng, ax, 0.25, 0.6), prev.get(ax, 0.0) or 1.0)
                else:
                    it = self.sample_command(rng)
                it["_frac"] = float(rng.uniform(*self.interrupt_frac))
                it["_cat"] = "interrupt"
                items.append(it)
                continue
            else:
                it = self.sample_command(rng)
            it.setdefault("_frac", None)
            it.setdefault("_cat", None)
            items.append(it)
        return items


_M5_AGILITY = dict(pedal_rate_dps=25.0, bank_deg=55.0, accel_fps2=6.0, climb_fps=20.0, lag_s=2.0)

STRESS_LEVELS: list[ManeuverLevel] = [
    StressLevel(
        name="S1", description="Dayanıklılık: manevra bitmeden gelen / ters yöne çeviren komutlar (0–100 kt)",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.5,
        single_ranges={"heading": _H(180.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT), **_M5_AGILITY),
    StressLevel(
        name="S2", description="Dayanıklılık + zarf sınırları: hover / 100 kt / alçak irtifa, sınıra giden komutlar",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.5,
        single_ranges={"heading": _H(180.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT), edge=True, **_M5_AGILITY),
    StressLevel(
        name="S3", description="Sert: art arda kesmeler, testere (±180° ters dönüş), birleşik komutlar, zarf sınırları",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.4,
        single_ranges={"heading": _H(180.0, 90.0, 10.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT), edge=True, p_edge_cmd=0.2,
        n_commands_range=(5, 8), p_interrupt=0.9, interrupt_frac=(0.15, 0.6), p_standard=0.2, promote_threshold=0.55,
        kind_probs={"reverse": 0.5, "cross": 0.2, "extend": 0.1, "random": 0.2}, **_M5_AGILITY),
    StressLevel(
        name="S4", description="Karma (uzun ince ayar için): %50 düz M5, kalan kesilen / ters komutlar ve zarf sınırları",
        heading=_H(180.0, 0.0, 10.0), speed=_V(50.0, 0.0, 5.0), altitude=_A(150.0, 0.0, 20.0), axis_probs=_MIX3,
        combined=True, mix_single=0.5,
        single_ranges={"heading": _H(180.0), "speed": _V(50.0), "altitude": _A(150.0)},
        start_alt_ft=(600.0, 1500.0), start_speed_fps=(0.0, 100.0 * KT), edge=True, p_edge_start=0.5, p_edge_cmd=0.2,
        n_commands_range=(3, 6), p_interrupt=0.7, interrupt_frac=(0.2, 0.8), p_standard=0.5, promote_threshold=0.6,
        kind_probs={"reverse": 0.45, "cross": 0.25, "extend": 0.1, "random": 0.2}, **_M5_AGILITY),
]

DEFAULT_MANEUVER_LEVELS = DEFAULT_MANEUVER_LEVELS + STRESS_LEVELS


def find_maneuver_level(level, levels=None) -> int:
    levels = levels or DEFAULT_MANEUVER_LEVELS
    if isinstance(level, (int, np.integer)):
        return int(np.clip(level, 0, len(levels) - 1))
    for i, lv in enumerate(levels):
        if lv.name.lower() == str(level).lower():
            return i
    raise KeyError(f"seviye yok: {level}")


def maneuver_level_names(levels=None) -> list[str]:
    return [lv.name for lv in (levels or DEFAULT_MANEUVER_LEVELS)]


__all__ = ["AXES", "ManeuverLevel", "StressLevel", "DEFAULT_MANEUVER_LEVELS", "STRESS_LEVELS", "find_maneuver_level",
           "maneuver_level_names", "time_target", "dynamic_time_target", "heading_rate_target", "main_axis",
           "KT", "G_FPS2"]
