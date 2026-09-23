from __future__ import annotations

"""
COMMAND CURRICULUM — Δheading / Δhız / Δirtifa seviyeleri
=========================================================

`HelicopterEnvCommand` (helicopter_env_command.py) her episode'da bir seviyenin
(Level) kurallarına göre komut üretir. Eğitim scripti
(train_command_curriculum.py) başarı oranına bakıp bir sonraki seviyeye geçer.

Mentorun planı (2026-09-22):
  * Teacher / student yok; PPO sıfırdan (rastgele ağırlıklarla) başlar.
  * Helikopter belirli bir irtifada (300 ft, 15 ft/s) başlar.
  * Önce küçük heading komutları (±5°) → öğrenince ±10° → ... ; sonra hız için
    aynısı. Rüzgâr yok.

Bir seviye = "bu episode'da hangi komutlar, hangi büyüklükte gelebilir".
Önemli: observation / action her seviyede AYNI. Bu yüzden bir seviyede eğitilen
ağırlıklar bir sonrakine olduğu gibi devredilir (ağ sıfırlanmaz).

Büyüklük seçimi (AxisRange):
  * `new_band_prob` olasılıkla YENİ bant: prev_max < |Δ| <= max_abs
  * aksi halde tüm aralık:            min_abs <= |Δ| <= max_abs  (tekrar /
    rehearsal — önceki seviyeleri unutmasın diye)
  * işaret (+/−) her zaman rastgele. + heading = sağa (saat yönü), projedeki
    konvansiyonla aynı.

Birimler: derece, ft/s, ft (repo ile aynı). 1 kt ≈ 1.688 ft/s.
"""

from dataclasses import dataclass, field, replace

import numpy as np


AXES = ("heading", "speed", "altitude")


@dataclass(frozen=True)
class AxisRange:
    max_abs: float
    prev_max: float = 0.0
    min_abs: float = 0.0
    new_band_prob: float = 0.75

    def sample(self, rng: np.random.Generator) -> float:
        lo_new = max(self.prev_max, self.min_abs)
        if self.prev_max > 0.0 and rng.random() < self.new_band_prob and self.max_abs > lo_new:
            mag = rng.uniform(lo_new, self.max_abs)
        else:
            mag = rng.uniform(self.min_abs, self.max_abs)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return float(sign * mag)


@dataclass(frozen=True)
class Level:
    name: str
    description: str
    heading: AxisRange | None = None
    speed: AxisRange | None = None
    altitude: AxisRange | None = None
    # Bir komutta hangi eksen değişsin (combined=False iken).  Olasılıklar
    # normalize edilir; aralığı None olan eksen seçilmez.
    axis_probs: dict = field(default_factory=lambda: {"heading": 1.0})
    # True → her komutta aralığı tanımlı TÜM eksenler birlikte değişir.
    combined: bool = False
    # Birleşik seviyede tekrar: bu olasılıkla tek eksenli bir komut (single_ranges'ten)
    # verilir → "dönüşte hızı / irtifayı koru" becerisi unutulmasın.
    mix_single: float = 0.0
    single_ranges: dict = field(default_factory=dict)
    # Episode yapısı
    episode_s: float = 90.0
    # Her komutun verileceği zaman aralığı (s, handover'dan itibaren).
    command_windows: tuple = ((3.0, 8.0), (40.0, 50.0))
    # Başlangıç koşulu (aralık; ikisi eşitse sabit)
    start_alt_ft: tuple = (300.0, 300.0)
    start_speed_fps: tuple = (15.0, 15.0)

    def ranges(self) -> dict:
        return {a: getattr(self, a) for a in AXES if getattr(self, a) is not None}

    def sample_command(self, rng: np.random.Generator) -> dict:
        """Tek bir komut: {'heading': Δψ [deg], 'speed': Δv [ft/s], 'altitude': Δh [ft]}."""
        cmd = {a: 0.0 for a in AXES}
        ranges = self.ranges()
        if not ranges:
            return cmd
        if self.single_ranges and self.mix_single > 0.0 and rng.random() < self.mix_single:
            names = list(self.single_ranges)
            p = np.array([self.axis_probs.get(a, 1.0) for a in names], dtype=float)
            axis = names[int(rng.choice(len(names), p=p / p.sum()))]
            cmd[axis] = self.single_ranges[axis].sample(rng)
            return cmd
        if self.combined:
            for a, r in ranges.items():
                cmd[a] = r.sample(rng)
            return cmd
        names = [a for a in AXES if a in ranges and self.axis_probs.get(a, 0.0) > 0.0]
        if not names:
            names = list(ranges)
            p = np.ones(len(names)) / len(names)
        else:
            p = np.array([self.axis_probs[a] for a in names], dtype=float)
            p = p / p.sum()
        axis = names[int(rng.choice(len(names), p=p))]
        cmd[axis] = ranges[axis].sample(rng)
        return cmd


# =====================================================================
# VARSAYILAN MÜFREDAT
# =====================================================================
# Sırayı / aralıkları mentorla birlikte değiştirmek için yalnızca bu listeyi
# düzenlemek yeterli.  Eğitim scripti seviyeleri bu sırayla dolaşır.

_H180 = AxisRange(max_abs=180.0, min_abs=1.0)           # heading tekrarı (tüm aralık)
_V10 = AxisRange(max_abs=10.0, min_abs=1.0)             # hız tekrarı
_A100 = AxisRange(max_abs=100.0, min_abs=3.0)           # irtifa tekrarı
_SINGLE = {"heading": _H180, "speed": _V10, "altitude": _A100}


DEFAULT_LEVELS: list[Level] = [
    # --- 1) HEADING --------------------------------------------------
    Level("H1", "heading ±5°", heading=AxisRange(5.0, 0.0, 1.0)),
    Level("H2", "heading ±10°", heading=AxisRange(10.0, 5.0, 1.0)),
    Level("H3", "heading ±20°", heading=AxisRange(20.0, 10.0, 1.0)),
    Level("H4", "heading ±45°", heading=AxisRange(45.0, 20.0, 1.0)),
    Level("H5", "heading ±90°", heading=AxisRange(90.0, 45.0, 1.0),
          episode_s=110.0, command_windows=((3.0, 8.0), (50.0, 60.0))),
    Level("H6", "heading ±180°", heading=AxisRange(180.0, 90.0, 1.0),
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    # --- 2) HIZ (heading tekrarıyla karışık) -------------------------
    Level("V1", "hız ±3 ft/s (+ heading tekrarı)", heading=_H180,
          speed=AxisRange(3.0, 0.0, 1.0), axis_probs={"heading": 0.4, "speed": 0.6},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("V2", "hız ±6 ft/s", heading=_H180, speed=AxisRange(6.0, 3.0, 1.0),
          axis_probs={"heading": 0.4, "speed": 0.6},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("V3", "hız ±10 ft/s", heading=_H180, speed=AxisRange(10.0, 6.0, 1.0),
          axis_probs={"heading": 0.4, "speed": 0.6},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    # --- 3) İRTİFA (heading / hız tekrarıyla) -------------------------
    # Not: tekrar oranı düşükken (%20) ajan büyük dönüşleri UNUTTU (A1'de ±130°
    # dönüşler ters yöne gitti) ama seviyeyi yine %80 ile geçti. Bu yüzden tekrar
    # %30–40'a çıkarıldı ve seviye atlama artık HER eksen için ayrı başarı istiyor
    # (train_command_curriculum.py: --axis-gate).
    Level("A1", "irtifa ±10 ft", heading=_H180, speed=_V10, altitude=AxisRange(10.0, 0.0, 3.0),
          axis_probs={"heading": 0.3, "speed": 0.2, "altitude": 0.5},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("A2", "irtifa ±25 ft", heading=_H180, speed=_V10, altitude=AxisRange(25.0, 10.0, 3.0),
          axis_probs={"heading": 0.3, "speed": 0.2, "altitude": 0.5},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("A3", "irtifa ±50 ft", heading=_H180, speed=_V10, altitude=AxisRange(50.0, 25.0, 3.0),
          axis_probs={"heading": 0.3, "speed": 0.2, "altitude": 0.5},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("A4", "irtifa ±100 ft", heading=_H180, speed=_V10, altitude=AxisRange(100.0, 50.0, 3.0),
          axis_probs={"heading": 0.3, "speed": 0.2, "altitude": 0.5},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    # --- 4) BİRLEŞİK + farklı başlangıç koşulları ---------------------
    # Birleşik seviyelerde komutların yarısı tek eksenli tekrar (tüm aralıklar).
    Level("C1", "birleşik: Δψ ±45°, Δv ±6, Δh ±50 aynı anda (+ tek eksen tekrarı)",
          heading=AxisRange(45.0, 0.0, 1.0), speed=AxisRange(6.0, 0.0, 1.0),
          altitude=AxisRange(50.0, 0.0, 3.0), combined=True, mix_single=0.5, single_ranges=_SINGLE,
          axis_probs={"heading": 0.4, "speed": 0.3, "altitude": 0.3},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0))),
    Level("R1", "birleşik + rastgele başlangıç (200–1000 ft, 10–25 ft/s)",
          heading=AxisRange(90.0, 0.0, 1.0), speed=AxisRange(8.0, 0.0, 1.0),
          altitude=AxisRange(100.0, 0.0, 3.0), combined=True, mix_single=0.5, single_ranges=_SINGLE,
          axis_probs={"heading": 0.4, "speed": 0.3, "altitude": 0.3},
          episode_s=120.0, command_windows=((3.0, 8.0), (55.0, 65.0)),
          start_alt_ft=(200.0, 1000.0), start_speed_fps=(10.0, 25.0)),
]


def command_axis(cmd: dict) -> str:
    """Bir komutun türü: tek eksen adı ya da birden çok eksen değişiyorsa 'combined'."""
    active = [a for a in AXES if cmd.get(a, 0.0) != 0.0]
    if not active:
        return "none"
    return active[0] if len(active) == 1 else "combined"


def level_names(levels=None) -> list[str]:
    return [lv.name for lv in (levels or DEFAULT_LEVELS)]


def find_level(name_or_index, levels=None) -> int:
    levels = levels or DEFAULT_LEVELS
    if isinstance(name_or_index, int) or str(name_or_index).isdigit():
        i = int(name_or_index)
        if not 0 <= i < len(levels):
            raise ValueError(f"level index {i} out of range 0..{len(levels) - 1}")
        return i
    names = level_names(levels)
    if name_or_index not in names:
        raise ValueError(f"unknown level {name_or_index!r}; known: {names}")
    return names.index(name_or_index)


def with_episode(level: Level, episode_s: float | None = None, windows=None) -> Level:
    """Bir seviyenin episode süresini / komut zamanlarını değiştir (deney için)."""
    kw = {}
    if episode_s is not None:
        kw["episode_s"] = float(episode_s)
    if windows is not None:
        kw["command_windows"] = tuple(tuple(map(float, w)) for w in windows)
    return replace(level, **kw)
