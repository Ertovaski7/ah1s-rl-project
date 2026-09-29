from __future__ import annotations

"""
PHYSICS EXT — ortak fizik katmanı: tork göstergesi / cezası, yakıt tüketimi, rüzgâr / gust / türbülans
=======================================================================================================

Env'den bağımsız; kalkış (`helicopter_env_takeoff.py`) ve manevra (`helicopter_env_maneuver.py`) env'leri ve yeni
tek ajan env'i aynı modülü kullanır. Her özellik config ile açılır; **varsayılan: hepsi kapalı** → eski modeller
(`takeoff_final.zip`, `maneuver_robust_final.zip`, ...) birebir aynı davranır. Env ayarı olarak modelin zip'inde
taşınır (`ah1s_env_overrides = {"physics": {...}}`, `PhysicsExtConfig.to_dict / from_dict`).

Kullanım (env içinde):
    ext = PhysicsExt(cfg.physics)                     # None / {} → pasif
    ext.reset(fdm, rng, heading_deg)                  # FDM kurulduktan hemen sonra: bölüm parametreleri + sabit rüzgâr
    ext.start_disturbances(fdm)                       # handover'da: türbülans ve gust'lar başlar
    ext.before_step(fdm)                              # her kontrol adımından önce (türbülans enjeksiyonu, kesme)
    events = ext.after_step(fdm)                      # her kontrol adımından sonra (yakıt, motor, istatistik)
    pen, parts = ext.penalty(fdm)                     # tork cezası (açıksa)
    ext.summary()                                     # bölüm özeti (tepe psi, sınır üstü süre, yakıt, olaylar)

1) Tork (psi)
   `ah1s.xml` "bell instruments": psi = 0.00416·Q_ana_rotor − 7.33 (model yazarı ±%20 hata olabilir diyor); el
   kitabı sınırları 50 psi sürekli, 56 psi 30 dk (= %100 tork). Hover kalibrasyonu el kitabının hover şeklini
   tutuyor (TM 55-1520-234-10 şekil 7-5, 8500 lbs, ~2600 ft yoğunluk irtifası: el kitabı ~44 psi, model 44.9 psi).
   Ceza fonksiyonu `torque_penalty` (50–56 psi rampası, 56 üstü karesel, düşük rotor devri).
   Fiziksel güç tavanı (`fcs/throttle-max-norm`) `ground-effect-torque` branch'indeki uçak kopyasında; bu modül onu
   yalnızca property varsa ayarlar.

2) Yakıt
   JSBSim AH-1S motoru `electric_1500hp` → yakıt yakmıyor. Burada her kontrol adımında W_f·dt tanklardan düşülür.
   model = "chart" (önerilen): W_f(psi, basınç irtifası) = a(h) + b(h)·psi·(N_r/324) — AH-1S operatör el kitabı
     TM 55-1520-234-10 şekil 7-8 sayfa 7 (seyir, T53-L-703, 4 TOW, FAT +15 °C, JP-4, ECU kapalı, 324 rotor rpm):
     tork ve yakıt akışı ölçekleri aynı grafikte karşılıklı; 400 dpi taramadan eksen çentikleriyle okundu
     (`docs/physics_ext/fuel_chart_digitized.csv`). Eşdeğer SFC (1290 shp = 56 psi ile): 50 psi 0.64, 30 psi 0.81
     lb/shp/h; doğru 1800 shp'ye uzatılınca 0.56 → yayımlanmış kalkış SFC'si 0.568 ile tutarlı.
   model = "sfc": W_f = sfc · P, sfc sabit (varsayılan 0.568 lb/shp/h: T53-L-703 kalkış gücünde, Purdue AAE
     propulsion veritabanı). P = "gauge" (psi → shp: 1290 shp @ 56 psi, aircav) ya da "rotor" (modelin fiziksel
     gücü: ana + kuyruk rotoru Q·Ω).
   Tank çekimi: el kitabı iki hücre (ön / arka) ve her birinde yakıt pompası tarif ediyor; ağırlık-denge grafiği
     yakıt momentini tek bir doğru olarak veriyor (kol ~200–203 in, yakıt azaldıkça neredeyse sabit) → iki hücre
     birlikte boşalıyor. Çekim sırası açıkça yazılmıyor → VARSAYIM: iki tanktan eşit çekim ("equal"; biri
     boşalırsa kalan diğerinden). Tanklar boşalınca motor ayrılır: `fcs/rpm-governor-active-norm = 0` (governor gazı
     0'a çeker, FGTransmission serbest tekerlek) ve "fuel_exhausted" olayı kaydedilir.

3) Rüzgâr / gust / türbülans
   Sabit rüzgâr: `atmosphere/wind-{north,east}-fps` (yön "nereden esiyor", derece; mutlak ya da başlangıç
   heading'ine göre). İsteğe bağlı yer yakını kesme (MIL-F-8785C log profili, W20 = 20 ft'teki rüzgâr).
   Gust: JSBSim 1−cos gust'ı (`atmosphere/cosine-gust/*`, yerel NED çerçevesi), Poisson zamanlı ya da sabit liste.
   Türbülans:
     backend "dryden_agl" (önerilen): MIL-F-8785C alçak irtifa Dryden modeli, yükseklik = AGL (kızak değil CG),
       σ_w = 0.1·W20, L_w = h, σ_u = σ_v = σ_w/(0.177 + 0.000823h)^0.4, L_u = L_v = h/(0.177 + 0.000823h)^1.2;
       bileşenler ortalama rüzgâr eksenlerinde; kesin (üstel) ayrıklaştırma, kontrol adımında güncellenir;
       `atmosphere/gust-{north,east,down}-fps`'e yazılır (öteleme; açısal türbülans yok). Donmuş alan hızı
       V = max(hava hızı, v_min).
     backend "jsbsim_milspec" / "jsbsim_tustin": JSBSim'in kendi modeli (turb-type 3 / 4). DİKKAT: JSBSim
       yüksekliği MSL alıyor (FGWinds::Run → Turbulence(in.AltitudeASL)); Edwards zemini 2283 ft MSL → her zaman
       "orta / yüksek irtifa" dalı: L = 1750 ft, σ yalnızca şiddet tablosundan (W20 etkisiz). Probe (c) karşılaştırır.
   MIL-F-8785C şiddet referansı (alçak irtifa): hafif W20 = 15 kt, orta 30 kt, şiddetli 45 kt.
"""

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

# =====================================================================================================================
# Sabitler (kaynaklar modül başlığında ve docs/physics_ext/README.md'de)
# =====================================================================================================================
KT_TO_FPS = 1.6878099
PSI_PROPERTY = "propulsion/engine/bell-torque-sensor-psi"
PSI_CONT = 50.0
PSI_LIMIT = 56.0
NOMINAL_RPM = 324.0
SHP_AT_100 = 1290.0                   # aircav.com T53-L-703: "limited to 1290 shp at 100% torque"
PSI_AT_100 = 56.0                     # aircav.com: 35 psi = %62.5 → %100 = 56 psi
T53_703_TO_SFC = 0.568                # lb/shp/h @ 1800 shp kalkış (Purdue AAE propulsion DB, T53 sayfası)

# TM 55-1520-234-10 şekil 7-8 sayfa 7: yakıt akışı W_f = a + b·psi (lb/h), basınç irtifasına göre (ft)
FUEL_CHART_ALT_FT = (0.0, 2000.0, 4000.0, 6000.0)
FUEL_CHART_A = (292.6, 273.9, 251.7, 238.3)
FUEL_CHART_B = (9.446, 9.416, 9.501, 9.376)

TANK_X_IN = (146.0, 206.0)            # ah1s.xml: tank 0 önde, tank 1 arkada (z = 38 in), her biri 890 lbs
TANK_CAPACITY_LBS = 890.0

# MIL-F-8785C alçak irtifa türbülans şiddeti → W20 (20 ft'te rüzgâr hızı, kt)
TURB_W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}
# JSBSim milspec/tustin şiddet indeksi (olasılık eğrisi, MIL-F-8785C şekil 7): hafif 3, orta 4, şiddetli 6
TURB_JSBSIM_SEVERITY = {"none": 0, "light": 3, "moderate": 4, "severe": 6}


# =====================================================================================================================
# CONFIG
# =====================================================================================================================

@dataclass
class TorqueExtConfig:
    enable: bool = False                  # psi izlenir, ceza hesaplanır (env ödüle ekler)
    obs: bool = False                     # env gözleme psi ekler (env'in kararı; burada yalnızca bayrak)
    psi_cont: float = PSI_CONT
    psi_limit: float = PSI_LIMIT
    pen_cont: float = 0.3                 # 50–56 psi arasında doğrusal rampa, 56'da pen_cont
    pen_over: float = 2.0                 # 56 üstü: pen_over · min(((psi − 56)/over_scale)², over_cap)
    over_scale: float = 3.0
    over_cap: float = 9.0
    pen_rpm_low: float = 1.0              # rotor devri rpm_low altında: pen_rpm_low · min(((rpm_low − rpm)/10)², 4)
    rpm_low: float = 314.0
    power_cap_psi: float | None = None    # fiziksel tavan (yalnızca uçak kopyasında fcs/throttle-max-norm varsa)


@dataclass
class FuelExtConfig:
    enable: bool = False
    model: str = "chart"                  # "chart" (TM 55-1520-234-10) | "sfc" (sabit SFC × güç)
    sfc: float = T53_703_TO_SFC           # model="sfc"
    power_source: str = "gauge"           # model="sfc": "gauge" (psi → shp) | "rotor" (Q·Ω ana + kuyruk)
    shp_at_100: float = SHP_AT_100
    psi_at_100: float = PSI_AT_100
    draw: str = "equal"                   # "equal" | "proportional" | "fwd_first" | "aft_first"
    unusable_lbs: float = 0.0             # toplam bunun altına inince motor ayrılır
    engine_out_on_empty: bool = True
    burn_scale: float = 1.0               # 1 = gerçek; >1 yalnızca test / hızlandırılmış senaryo için


@dataclass
class WindExtConfig:
    enable: bool = False
    speed_kt: tuple = (0.0, 0.0)          # bölüm başında düzgün dağılımdan (kesme açıksa W20)
    dir_deg: tuple = (0.0, 360.0)         # rüzgârın GELDİĞİ yön
    dir_relative: bool = True             # True: başlangıç heading'ine göre (0 = karşıdan, 90 = sağdan)
    shear: bool = False                   # MIL-F-8785C: W(h) = W20·ln(h/z0)/ln(20/z0), h ≤ 1000 ft (CG AGL)
    z0_ft: float = 0.15
    shear_min_frac: float = 0.3           # yerde (h → z0) sıfıra inmesin


@dataclass
class GustExtConfig:
    enable: bool = False
    rate_per_min: float = 0.0             # Poisson; 0 → yalnızca `schedule`
    magnitude_kt: tuple = (5.0, 15.0)
    rise_s: tuple = (1.0, 3.0)
    hold_s: tuple = (0.0, 2.0)
    decay_s: tuple = (1.0, 3.0)
    vertical_frac: tuple = (0.0, 0.0)     # dikey bileşen / toplam (işaret rastgele)
    along_wind_prob: float = 0.5          # ortalama rüzgâr yönünde (diğerleri rastgele yatay yön)
    first_after_s: float = 5.0
    schedule: tuple = ()                  # sabit gust listesi: ({"t": s, "mag_kt":, "dir_deg": (gidiş, mutlak),
                                          #   "vert": frac, "rise":, "hold":, "decay":}, ...)


@dataclass
class TurbExtConfig:
    enable: bool = False
    backend: str = "dryden_agl"           # "dryden_agl" | "jsbsim_milspec" | "jsbsim_tustin"
    levels: tuple = ("light",)            # bölüm başında seçilir (none / light / moderate / severe)
    level_probs: tuple = ()               # boş → eşit olasılık
    scale: float = 1.0                    # dryden_agl: σ çarpanı (curriculum için)
    v_min_fps: float = 15.0               # dryden_agl: donmuş alan hızı alt sınırı (sakin hover)
    h_min_ft: float = 10.0                # MIL / JSBSim alt sınırı
    seed_jsbsim: bool = True              # jsbsim_*: atmosphere/randomseed bölüm RNG'sinden


@dataclass
class PhysicsExtConfig:
    torque: TorqueExtConfig = field(default_factory=TorqueExtConfig)
    fuel: FuelExtConfig = field(default_factory=FuelExtConfig)
    wind: WindExtConfig = field(default_factory=WindExtConfig)
    gust: GustExtConfig = field(default_factory=GustExtConfig)
    turb: TurbExtConfig = field(default_factory=TurbExtConfig)

    @property
    def active(self) -> bool:
        return any(getattr(self, k).enable for k in ("torque", "fuel", "wind", "gust", "turb"))

    def to_dict(self, only_changes: bool = True) -> dict:
        """JSON'a uygun dict (model zip'indeki ah1s_env_overrides["physics"] için); varsayılandan farklı olanlar."""
        out = {}
        for f in fields(self):
            sub = getattr(self, f.name)
            d = asdict(sub)
            if only_changes:
                ref = asdict(type(sub)())
                d = {k: v for k, v in d.items() if v != ref[k]}
            if d:
                out[f.name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        return out

    @classmethod
    def from_dict(cls, d: dict | None) -> "PhysicsExtConfig":
        cfg = cls()
        for name, sub in (d or {}).items():
            if name not in {f.name for f in fields(cls)}:
                raise ValueError(f"bilinmeyen physics bölümü: {name}")
            obj = getattr(cfg, name)
            known = {f.name for f in fields(obj)}
            for k, v in (sub or {}).items():
                if k not in known:
                    raise ValueError(f"bilinmeyen physics.{name} ayarı: {k}")
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
        return cfg


def as_physics_config(x) -> PhysicsExtConfig:
    if x is None:
        return PhysicsExtConfig()
    if isinstance(x, PhysicsExtConfig):
        return x
    if isinstance(x, dict):
        return PhysicsExtConfig.from_dict(x)
    raise TypeError(f"physics config: {type(x)}")


# =====================================================================================================================
# SAF FONKSİYONLAR (probe'lar ve env'ler doğrudan kullanabilir)
# =====================================================================================================================

def torque_psi(fdm) -> float:
    return float(fdm[PSI_PROPERTY])


def torque_penalty(psi: float, rpm: float, cfg: TorqueExtConfig) -> tuple[float, dict]:
    """50–56 psi doğrusal rampa (sürekli sınır aşımı), 56 üstü karesel (kırpılı), düşük rotor devri."""
    cont = cfg.pen_cont * float(np.clip((psi - cfg.psi_cont) / max(1e-6, cfg.psi_limit - cfg.psi_cont), 0.0, 1.0))
    over = cfg.pen_over * min(max(0.0, (psi - cfg.psi_limit) / cfg.over_scale) ** 2, cfg.over_cap)
    rpm_low = cfg.pen_rpm_low * min(max(0.0, (cfg.rpm_low - rpm) / 10.0) ** 2, 4.0)
    return cont + over + rpm_low, dict(torque_cont=cont, torque_over=over, rpm_low=rpm_low)


def fuel_flow_chart_lbh(psi: float, rpm: float, pressure_alt_ft: float) -> float:
    """TM 55-1520-234-10 şekil 7-8 (sayfa 7) doğruları; psi, rotor devriyle güce eşdeğerlenir (psi·N/324)."""
    h = float(np.clip(pressure_alt_ft, FUEL_CHART_ALT_FT[0], FUEL_CHART_ALT_FT[-1]))
    a = float(np.interp(h, FUEL_CHART_ALT_FT, FUEL_CHART_A))
    b = float(np.interp(h, FUEL_CHART_ALT_FT, FUEL_CHART_B))
    return a + b * max(0.0, psi) * max(0.0, rpm) / NOMINAL_RPM


def shaft_power_gauge_shp(psi: float, rpm: float, shp_at_100: float = SHP_AT_100, psi_at_100: float = PSI_AT_100):
    return max(0.0, psi) * shp_at_100 / psi_at_100 * max(0.0, rpm) / NOMINAL_RPM


def shaft_power_rotor_hp(fdm) -> float:
    """Modelin fiziksel gücü: ana + kuyruk rotoru aerodinamik torku × açısal hız (hp)."""
    p = 0.0
    for i in (0, 1):
        q = float(fdm[f"propulsion/engine[{i}]/torque-lbsft"])
        w = float(fdm[f"propulsion/engine[{i}]/rotor-rpm"]) * 2.0 * math.pi / 60.0
        p += q * w / 550.0
    return max(0.0, p)


def wind_ned_fps(speed_fps: float, from_deg: float) -> tuple[float, float]:
    """Meteorolojik yön (rüzgârın geldiği yön) → hava kütlesinin hızı (kuzey, doğu), ft/s."""
    a = math.radians(from_deg)
    return -speed_fps * math.cos(a), -speed_fps * math.sin(a)


def shear_factor(h_ft: float, z0_ft: float = 0.15, min_frac: float = 0.3) -> float:
    """MIL-F-8785C: W(h)/W20 = ln(h/z0)/ln(20/z0), 1000 ft üstü sabit."""
    h = float(np.clip(h_ft, z0_ft * 1.01, 1000.0))
    return max(min_frac, math.log(h / z0_ft) / math.log(20.0 / z0_ft))


def dryden_low_alt_params(h_ft: float, w20_fps: float) -> dict:
    """MIL-F-8785C alçak irtifa (h < 1000 ft) ölçek uzunlukları (ft) ve şiddetleri (ft/s)."""
    h = float(np.clip(h_ft, 10.0, 1000.0))
    k = 0.177 + 0.000823 * h
    sig_w = 0.1 * w20_fps
    return dict(L_u=h / k ** 1.2, L_v=h / k ** 1.2, L_w=h, sig_u=sig_w / k ** 0.4, sig_v=sig_w / k ** 0.4, sig_w=sig_w)


def draw_from_tanks(contents, amount: float, mode: str = "equal") -> np.ndarray:
    """contents (lbs, tank 0 ön / tank 1 arka) − amount; boşalan tanktan eksik kalan diğerinden."""
    c = np.maximum(np.asarray(contents, dtype=np.float64), 0.0)
    amount = min(max(0.0, float(amount)), float(c.sum()))
    if amount <= 0.0:
        return c
    if mode == "proportional":
        return c - amount * c / c.sum()
    if mode in ("fwd_first", "aft_first"):
        order = (0, 1) if mode == "fwd_first" else (1, 0)
        rest = amount
        for i in order:
            d = min(c[i], rest)
            c[i] -= d
            rest -= d
        return c
    if mode != "equal":
        raise ValueError(f"bilinmeyen tank çekim sırası: {mode}")
    rest = amount
    for _ in range(2):                                 # eşit pay; biri yetmezse kalan diğerinden
        live = [i for i in (0, 1) if c[i] > 1e-12]
        if not live or rest <= 1e-12:
            break
        share = rest / len(live)
        for i in live:
            d = min(c[i], share)
            c[i] -= d
            rest -= d
    return np.maximum(c, 0.0)


# =====================================================================================================================
# DURUMLU KATMAN
# =====================================================================================================================

class PhysicsExt:
    """Bir env örneğine bağlı fizik katmanı. Varsayılan config → hiçbir şey yapmaz (eski modeller birebir)."""

    def __init__(self, cfg=None, control_dt: float = 0.075):
        self.cfg = as_physics_config(cfg)
        self.dt = float(control_dt)
        self.rng = np.random.default_rng(0)
        self.t = 0.0
        self.params: dict = {}
        self.events: list[dict] = []
        self.stats: dict = {}
        self._turb_on = False
        self._xi = np.zeros(3)
        self._gust_next_t = np.inf
        self._gust_queue: list[dict] = []
        self._engine_out = False
        self._wind_ned = (0.0, 0.0)
        self._armed = False                          # reset() çağrılana kadar katman hiçbir şey yapmaz
        self.stats = self._new_stats()
        self._last = dict(psi=0.0, rpm=NOMINAL_RPM, fuel_flow_lbh=0.0, shaft_shp=0.0, pen=0.0)

    @staticmethod
    def _new_stats() -> dict:
        return dict(t=0.0, peak_psi=-np.inf, t_over_cont=0.0, t_over_limit=0.0, min_rpm=np.inf, fuel_used_lbs=0.0,
                    fuel0_lbs=None, engine_out_t=None, gusts=0)

    @property
    def active(self) -> bool:
        return self.cfg.active

    # -----------------------------------------------------------------------------------------------------------
    # bölüm başı
    # -----------------------------------------------------------------------------------------------------------
    def reset(self, fdm, rng=None, heading_deg: float = 0.0, options: dict | None = None) -> dict:
        """FDM kurulduktan sonra (tercihen reset PID'inden ÖNCE: helikopter rüzgârda otursun). Bölüm parametrelerini
        seçer, sabit rüzgârı ve güç tavanını uygular. options: {"wind_kt":, "wind_dir_deg":, "turb_level":,
        "gusts": [...]} → config'teki rastgele seçimi ezer (değerlendirme / canlı)."""
        cfg, o = self.cfg, dict(options or {})
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0.0
        self.events, self._engine_out, self._turb_on = [], False, False
        self._xi = np.zeros(3)
        self.stats = self._new_stats()
        self._armed = self.active or bool(o)
        p = {}
        if cfg.wind.enable or "wind_kt" in o:
            w = cfg.wind
            spd = float(o.get("wind_kt", self.rng.uniform(*w.speed_kt)))
            d = float(o.get("wind_dir_deg", self.rng.uniform(*w.dir_deg)))
            rel = bool(o.get("wind_dir_relative", w.dir_relative))
            d_abs = (heading_deg + d) % 360.0 if rel else d % 360.0
            p.update(wind_kt=spd, wind_dir_deg=d, wind_dir_relative=rel, wind_from_deg=d_abs)
            self._wind_ned = wind_ned_fps(spd * KT_TO_FPS, d_abs)
        else:
            self._wind_ned = (0.0, 0.0)
        if cfg.turb.enable or "turb_level" in o:
            tb = cfg.turb
            if "turb_level" in o:
                lvl = o["turb_level"]
            else:
                probs = np.asarray(tb.level_probs, dtype=np.float64) if tb.level_probs else None
                lvl = tb.levels[int(self.rng.choice(len(tb.levels), p=probs / probs.sum() if probs is not None else None))]
            p.update(turb_level=lvl, turb_backend=o.get("turb_backend", tb.backend),
                     turb_w20_kt=float(o.get("turb_w20_kt", TURB_W20_KT[lvl])))
        self._gust_queue = []
        if cfg.gust.enable or "gusts" in o:
            self._gust_queue = [dict(g) for g in (o.get("gusts", cfg.gust.schedule) or ())]
        self.params = p
        self.apply_wind(fdm)
        self._apply_power_cap(fdm)
        if cfg.fuel.enable:
            self.stats["fuel0_lbs"] = self.fuel_total(fdm)
        return dict(p)

    def _apply_power_cap(self, fdm):
        cap = self.cfg.torque.power_cap_psi
        if cap is None:
            return
        pm = fdm.get_property_manager()
        if not pm.hasNode("fcs/throttle-max-norm"):
            raise RuntimeError("power_cap_psi için uçak kopyasında fcs/throttle-max-norm yok (ground-effect-torque)")
        raise NotImplementedError("güç tavanı ground-effect-torque branch'indeki uygulamayla bağlanacak")

    def apply_wind(self, fdm, h_agl_ft: float | None = None):
        vn, ve = self._wind_ned
        if self.cfg.wind.shear and (vn or ve):
            h = float(fdm["position/h-agl-ft"]) if h_agl_ft is None else h_agl_ft
            k = shear_factor(h, self.cfg.wind.z0_ft, self.cfg.wind.shear_min_frac)
            vn, ve = vn * k, ve * k
        if self.cfg.wind.enable or vn or ve:
            fdm["atmosphere/wind-north-fps"] = vn
            fdm["atmosphere/wind-east-fps"] = ve
            fdm["atmosphere/wind-down-fps"] = 0.0

    def start_disturbances(self, fdm):
        """Handover'da: türbülans ve gust zamanlaması başlar (reset PID'i sakin havada / sabit rüzgârda oturur)."""
        p = self.params
        if "turb_level" in p and p["turb_level"] != "none":
            self._turb_on = True
            be = p["turb_backend"]
            if be.startswith("jsbsim"):
                fdm["atmosphere/turb-type"] = 3 if be == "jsbsim_milspec" else 4
                fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = p["turb_w20_kt"] * KT_TO_FPS
                fdm["atmosphere/turbulence/milspec/severity"] = TURB_JSBSIM_SEVERITY[p["turb_level"]]
                if self.cfg.turb.seed_jsbsim:
                    fdm["atmosphere/randomseed"] = int(self.rng.integers(1, 2 ** 31 - 1))
            elif be != "dryden_agl":
                raise ValueError(f"bilinmeyen türbülans backend'i: {be}")
        g = self.cfg.gust
        if g.enable and g.rate_per_min > 0.0:
            self._gust_next_t = self.t + g.first_after_s + float(self.rng.exponential(60.0 / g.rate_per_min))
        else:
            self._gust_next_t = np.inf

    # -----------------------------------------------------------------------------------------------------------
    # adım
    # -----------------------------------------------------------------------------------------------------------
    def before_step(self, fdm):
        """Her kontrol adımından (ve reset'teki her fizik koşusundan) önce. Sabit rüzgâr HER adımda yeniden yazılır:
        JSBSim `run_ic()` rüzgârı IC'deki rüzgârla (0) eziyor (FGFDMExec::Initialize → Winds->SetWindNED), env'ler
        teleport / bozucu için run_ic çağırıyor."""
        if not self._armed:
            return
        self.apply_wind(fdm)
        if self._turb_on and self.params.get("turb_backend") == "dryden_agl":
            self._dryden_step(fdm)
        self._gust_step(fdm)

    def _dryden_step(self, fdm):
        tb, p = self.cfg.turb, self.params
        h = max(tb.h_min_ft, float(fdm["position/h-agl-ft"]))
        w20 = p["turb_w20_kt"] * KT_TO_FPS * tb.scale
        q = dryden_low_alt_params(h, w20)
        V = max(tb.v_min_fps, float(fdm["velocities/vt-fps"]))
        # MIL-HDBK-1797 / Yeager: u ~ τ_u = L_u/V, v ~ τ_u/2, w ~ τ_w/2 (birinci derece yaklaşık), kesin ayrıklaştırma
        taus = np.array([q["L_u"] / V, 0.5 * q["L_v"] / V, 0.5 * q["L_w"] / V])
        sig = np.array([q["sig_u"], q["sig_v"], q["sig_w"]])
        a = np.exp(-self.dt / taus)
        self._xi = a * self._xi + sig * np.sqrt(1.0 - a * a) * self.rng.standard_normal(3)
        vn, ve = self._wind_ned
        psiw = math.atan2(ve, vn) if (vn or ve) else math.radians(float(fdm["attitude/psi-deg"]))
        c, s = math.cos(psiw), math.sin(psiw)
        u, v, w = self._xi
        fdm["atmosphere/gust-north-fps"] = c * u - s * v
        fdm["atmosphere/gust-east-fps"] = s * u + c * v
        fdm["atmosphere/gust-down-fps"] = w

    def _gust_step(self, fdm):
        g = self.cfg.gust
        due = None
        if self._gust_queue and self.t >= float(self._gust_queue[0]["t"]) - 1e-9:
            due = self._gust_queue.pop(0)
        elif self.t >= self._gust_next_t:
            due = self._random_gust()
            self._gust_next_t = self.t + float(self.rng.exponential(60.0 / g.rate_per_min))
        if due is None:
            return
        mag = float(due["mag_kt"]) * KT_TO_FPS
        d = math.radians(float(due["dir_deg"]))                   # gust'ın GİTTİĞİ yön (mutlak)
        vf = float(due.get("vert", 0.0))
        hvec = math.sqrt(max(0.0, 1.0 - vf * vf))
        fdm["atmosphere/cosine-gust/startup-duration-sec"] = float(due.get("rise", 2.0))
        fdm["atmosphere/cosine-gust/steady-duration-sec"] = float(due.get("hold", 1.0))
        fdm["atmosphere/cosine-gust/end-duration-sec"] = float(due.get("decay", 2.0))
        fdm["atmosphere/cosine-gust/magnitude-ft_sec"] = mag
        fdm["atmosphere/cosine-gust/frame"] = 3                    # yerel NED
        fdm["atmosphere/cosine-gust/X-velocity-ft_sec"] = hvec * math.cos(d)
        fdm["atmosphere/cosine-gust/Y-velocity-ft_sec"] = hvec * math.sin(d)
        fdm["atmosphere/cosine-gust/Z-velocity-ft_sec"] = vf
        fdm["atmosphere/cosine-gust/start"] = 1
        self.stats["gusts"] += 1
        self.events.append(dict(t=self.t, kind="gust", **{k: v for k, v in due.items() if k != "t"}))

    def _random_gust(self) -> dict:
        g, rng = self.cfg.gust, self.rng
        vn, ve = self._wind_ned
        if (vn or ve) and rng.random() < g.along_wind_prob:
            d = math.degrees(math.atan2(ve, vn))
        else:
            d = float(rng.uniform(0.0, 360.0))
        vf = float(rng.uniform(*g.vertical_frac)) * (1.0 if rng.random() < 0.5 else -1.0)
        return dict(t=self.t, mag_kt=float(rng.uniform(*g.magnitude_kt)), dir_deg=d % 360.0, vert=vf,
                    rise=float(rng.uniform(*g.rise_s)), hold=float(rng.uniform(*g.hold_s)),
                    decay=float(rng.uniform(*g.decay_s)))

    def after_step(self, fdm, dt: float | None = None) -> list[dict]:
        """Kontrol adımından sonra: tork istatistiği, yakıt yakma, yakıt bitince motor ayrılması. Yeni olayları döndürür."""
        dt = self.dt if dt is None else float(dt)
        n0 = len(self.events)
        if not self._armed:
            return []
        self.t += dt
        psi, rpm = torque_psi(fdm), float(fdm["propulsion/engine/rotor-rpm"])
        st = self.stats
        st["t"] = self.t
        st["peak_psi"] = max(st["peak_psi"], psi)
        st["min_rpm"] = min(st["min_rpm"], rpm)
        if psi > self.cfg.torque.psi_cont:
            st["t_over_cont"] += dt
        if psi > self.cfg.torque.psi_limit:
            st["t_over_limit"] += dt
        self._last.update(psi=psi, rpm=rpm)
        if self.cfg.fuel.enable:
            self._burn(fdm, psi, rpm, dt)
        return self.events[n0:]

    def _burn(self, fdm, psi: float, rpm: float, dt: float):
        fc = self.cfg.fuel
        if self._engine_out:
            self._last.update(fuel_flow_lbh=0.0, shaft_shp=0.0)
            return
        if fc.model == "chart":
            ff = fuel_flow_chart_lbh(psi, rpm, float(fdm["atmosphere/pressure-altitude"]))
            shp = shaft_power_gauge_shp(psi, rpm, fc.shp_at_100, fc.psi_at_100)
        elif fc.model == "sfc":
            shp = (shaft_power_gauge_shp(psi, rpm, fc.shp_at_100, fc.psi_at_100) if fc.power_source == "gauge"
                   else shaft_power_rotor_hp(fdm))
            ff = fc.sfc * shp
        else:
            raise ValueError(f"bilinmeyen yakıt modeli: {fc.model}")
        # governor gazı kesmişse (motor boşta / kapalı) yakıt yok sayılır
        if float(fdm["fcs/rpm-governor-active-norm"]) <= 0.0:
            ff = 0.0
        burn = ff * fc.burn_scale * dt / 3600.0
        c = np.array([float(fdm["propulsion/tank[0]/contents-lbs"]), float(fdm["propulsion/tank[1]/contents-lbs"])])
        usable = max(0.0, c.sum() - fc.unusable_lbs)
        burn = min(burn, usable)
        c2 = draw_from_tanks(c, burn, fc.draw)
        fdm["propulsion/tank[0]/contents-lbs"] = float(c2[0])
        fdm["propulsion/tank[1]/contents-lbs"] = float(c2[1])
        self.stats["fuel_used_lbs"] += burn
        self._last.update(fuel_flow_lbh=ff, shaft_shp=shp)
        if fc.engine_out_on_empty and c2.sum() - fc.unusable_lbs <= 1e-9 and ff > 0.0:
            self.engine_out(fdm, reason="fuel_exhausted")

    def engine_out(self, fdm, reason: str = "engine_failure"):
        """Motor ayrılır (28 Eylül motor arızası mekanizması): governor kapalı → gaz 0 → serbest tekerlek."""
        if self._engine_out:
            return
        fdm["fcs/rpm-governor-active-norm"] = 0.0
        self._engine_out = True
        self.stats["engine_out_t"] = self.t
        self.events.append(dict(t=self.t, kind=reason))

    @property
    def engine_is_out(self) -> bool:
        return self._engine_out

    # -----------------------------------------------------------------------------------------------------------
    # ödül / gözlem / özet
    # -----------------------------------------------------------------------------------------------------------
    def penalty(self, fdm=None) -> tuple[float, dict]:
        if not self.cfg.torque.enable:
            return 0.0, {}
        psi = self._last["psi"] if fdm is None else torque_psi(fdm)
        rpm = self._last["rpm"] if fdm is None else float(fdm["propulsion/engine/rotor-rpm"])
        pen, parts = torque_penalty(psi, rpm, self.cfg.torque)
        self._last["pen"] = pen
        return pen, parts

    @staticmethod
    def fuel_total(fdm) -> float:
        return float(fdm["propulsion/tank[0]/contents-lbs"]) + float(fdm["propulsion/tank[1]/contents-lbs"])

    def fuel_fraction(self, fdm) -> float:
        return self.fuel_total(fdm) / (2.0 * TANK_CAPACITY_LBS)

    @staticmethod
    def air_ground_velocities(fdm) -> dict:
        """Gövde ekseninde hava hızı (u, v, w: hava kütlesine göre) ve yer hızı (ileri, yana), ft/s."""
        psi = float(fdm["attitude/psi-rad"])
        vn, ve = float(fdm["velocities/v-north-fps"]), float(fdm["velocities/v-east-fps"])
        return dict(u_air=float(fdm["velocities/u-aero-fps"]), v_air=float(fdm["velocities/v-aero-fps"]),
                    w_air=float(fdm["velocities/w-aero-fps"]),
                    u_gnd=vn * math.cos(psi) + ve * math.sin(psi), v_gnd=-vn * math.sin(psi) + ve * math.cos(psi))

    def info(self) -> dict:
        d = dict(self._last)
        d["engine_out"] = self._engine_out
        return d

    def summary(self) -> dict:
        s = dict(self.stats)
        s["params"] = dict(self.params)
        s["events"] = list(self.events)
        if s["peak_psi"] == -np.inf:
            s["peak_psi"] = float("nan")
        if s["min_rpm"] == np.inf:
            s["min_rpm"] = float("nan")
        return s


# =====================================================================================================================
# HIZLI TEST:  python physics_ext.py
# =====================================================================================================================

if __name__ == "__main__":
    # 1) varsayılan config pasif
    cfg = PhysicsExtConfig()
    assert not cfg.active and cfg.to_dict() == {}
    # 2) dict ↔ config gidiş-dönüş (model zip'indeki ah1s_env_overrides["physics"])
    d = {"fuel": {"enable": True}, "wind": {"enable": True, "speed_kt": [0, 20]}, "turb": {"enable": True}}
    c2 = PhysicsExtConfig.from_dict(d)
    assert c2.active and c2.wind.speed_kt == (0, 20) and PhysicsExtConfig.from_dict(c2.to_dict()).to_dict() == c2.to_dict()
    # 3) tank çekimi
    assert np.allclose(draw_from_tanks((890, 890), 100, "equal"), (840, 840))
    assert np.allclose(draw_from_tanks((10, 890), 100, "equal"), (0, 800))
    assert np.allclose(draw_from_tanks((300, 900), 120, "proportional"), (270, 810))
    assert np.allclose(draw_from_tanks((890, 890), 1000, "fwd_first"), (0, 780))
    assert np.allclose(draw_from_tanks((890, 890), 5000, "aft_first"), (0, 0))
    # 4) yakıt akışı: el kitabı doğruları (2000 ft, 43 psi ≈ 678 lb/h @ +15 °C; el kitabı örneği −30 °C'de 648)
    ff = fuel_flow_chart_lbh(43.0, 324.0, 2000.0)
    assert 670 < ff < 690, ff
    # eşdeğer SFC 1800 shp'ye uzatılınca yayımlanmış 0.568'e yakın mı?
    psi_1800 = 1800.0 / (SHP_AT_100 / PSI_AT_100)
    sfc_1800 = fuel_flow_chart_lbh(psi_1800, 324.0, 0.0) / 1800.0
    assert abs(sfc_1800 - T53_703_TO_SFC) < 0.03, sfc_1800
    # 5) rüzgâr yönü: kuzeyden 10 kt → hava kütlesi güneye
    vn, ve = wind_ned_fps(10 * KT_TO_FPS, 0.0)
    assert vn < 0 and abs(ve) < 1e-9
    # 6) MIL-F-8785C: 100 ft, W20 = 15 kt → σ_w = 2.53, σ_u = 4.35 ft/s
    q = dryden_low_alt_params(100.0, 15 * KT_TO_FPS)
    assert abs(q["sig_w"] - 2.53) < 0.01 and abs(q["sig_u"] - 4.35) < 0.02
    # 7) tork cezası
    assert torque_penalty(49.0, 324.0, TorqueExtConfig())[0] == 0.0
    assert abs(torque_penalty(53.0, 324.0, TorqueExtConfig())[0] - 0.15) < 1e-9
    print(f"physics_ext: tamam (43 psi / 2000 ft → {ff:.0f} lb/h; 1800 shp'de eşdeğer SFC {sfc_1800:.3f})")
