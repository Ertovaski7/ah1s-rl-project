# Physics-ext — ortak fizik katmanı: tork, yakıt, rüzgâr / türbülans (Aşama 1, eğitim yok)

Kalkış ve manevra env'lerinin (ve Aşama 2'deki tek ajan env'inin) ortak kullanacağı fizik katmanı `physics_ext.py` ve
beş probe (a–e). **Her özellik config ile açılır, varsayılan kapalıdır** → eski modeller aynı davranır. Model zip'inde
env ayarı olarak taşınır: `ah1s_env_overrides = {"physics": PhysicsExtConfig(...).to_dict()}`.

> Durum (2026-09-29): `physics-ext` branch'i `ground-effect-torque`'un (kullanıcının bundle'ı) üstüne taşındı. Tork
> göstergesi / güç tavanı / tork cezası branch'teki uygulamanın aynısı (fonksiyonlar `physics_ext`'e taşındı, kalkış
> env'i oradan alıyor). Katman kalkış ve manevra env'lerine bağlandı (`physics` config alanı, varsayılan None); eski
> modellerin değerlendirmeleri birebir aynı (kalkış 28 senaryo × 2 model, manevra dayanıklılık 49 koşu: JSON'larda 0
> fark). Probe'lar iki uçakla koşuldu: **stok JSBSim 1.3.1 AH-1S** (dosya adında ek yok; aşağıdaki bölüm 4) ve
> **repo uçağı + 56 psi güç tavanı** (`--aircraft repo --power-cap 56`, dosya eki `_repo_cap56`; bölüm 4b — Aşama 2'nin
> tek ajan env'i bu fizikle ve `trim_table_airspeed_repo_cap56.json` ile çalışıyor).

## 1. Modül: `physics_ext.py`

```python
# env'de: TakeoffEnvConfig(physics={...}) / ManeuverEnvConfig(physics={...}); FlightEnvConfig'te yakıt varsayılan açık
from physics_ext import PhysicsExt, torque_penalty, torque_psi
ext = PhysicsExt({"fuel": {"enable": True},
                  "wind": {"enable": True, "speed_kt": [0, 20], "dir_deg": [0, 360]},
                  "turb": {"enable": True, "levels": ["none", "light", "moderate"]}}, control_dt=0.075)
ext.begin_episode(env.np_random, heading_deg, options)   # reset başı: bölüm parametreleri (kapalıyken RNG'ye dokunmaz)
ext.attach(fdm)                      # her FDM kurulumunda: önceki bölümün rüzgâr / gust / türbülansını temizle, rüzgârı yaz
ext.start_disturbances(fdm)          # handover: türbülans + gust + yakıt / tork sayacı başlar
ext.before_step(fdm)                 # her kontrol adımından önce (env._run_plain): rüzgâr (run_ic siliyor), türbülans, gust
events = ext.after_step(fdm)         # sonra: tork istatistiği, yakıt yakma, yakıt bitince motor ayrılması
ext.info(fdm); ext.summary()         # adım bilgisi · bölüm özeti (tepe psi, 50 / 56 psi üstü süre, yakıt, olaylar)
# options (değerlendirme / canlı, config'i ezer): wind_kt, wind_dir_deg, wind_dir_relative, turb_level, gusts, gust_rate_per_min, gust_kt
```

| bölüm | ne yapar | varsayılan |
|---|---|---|
| tork (fonksiyonlar) | psi = 0.00416·Q − 7.33 (ah1s.xml'deki gösterge); `torque_penalty`: 50–56 psi `pen_cont·((psi−50)/6)²`, 56 üstü `pen_over·min(((psi−56)/3)², 9)`, rotor < 314 rpm `pen_rpm_low·((314−rpm)/10)²` (katsayılar env config'inde; kalkış env'inin 2026-09-28 uygulamasıyla aynı); güç tavanı `psi_to_throttle` (repo uçağının governor'ında); istatistik katmanda | env'de 0 |
| `fuel` | her kontrol adımında W_f·dt tanklardan; tanklar bitince motor ayrılır (`fcs/rpm-governor-active-norm = 0`) + `fuel_exhausted` olayı | kapalı |
| `wind` | sabit rüzgâr (hız / yön aralığı, heading'e göre ya da mutlak), isteğe bağlı MIL-F-8785C log kesmesi | kapalı |
| `gust` | JSBSim 1−cos gust'ı (yerel NED), Poisson ya da sabit liste | kapalı |
| `turb` | `dryden_agl` (bizim; önerilen) ya da `jsbsim_tustin` / `jsbsim_milspec` | kapalı |

Test: `python physics_ext.py` (varsayılan pasif, dict gidiş-dönüş, tank çekimi, yakıt doğruları, rüzgâr yönü, MIL
parametreleri, ceza).

## 2. Kaynaklar

- **Yakıt akışı:** TM 55-1520-234-10, *Operator's Manual AH-1S* (1976, Change 30'a kadar),
  [archive.org](https://archive.org/details/TM55152023410AH1S): şekil 7-8 sayfa 7 (seyir, T53-L-703, 4 TOW,
  324 rotor / 6600 motor rpm, JP-4, ECU kapalı, FAT +15 °C). Tork ve yakıt akışı ölçekleri aynı grafikte
  karşılıklı ("torque pressure may be converted directly to fuel flow"). 400 dpi taramada eksen çentikleri
  pikselle okundu (`digitize_fuel_chart.py`, `fuel_chart_digitized.csv`): her irtifada **W_f = a + b·psi**,
  a = 292.6 / 273.9 / 251.7 / 238.3 lb/h (0 / 2000 / 4000 / 6000 ft), b ≈ 9.4 lb/h/psi, artık ≤ 4 lb/h.
  Aynı el kitabının şekil 7-5'i (hover torku) ve şekil 7-11'i (flat pitch 324 rpm: ~360 lb/h, 2–3 bin ft) çapraz
  kontrol için okundu.
- **SFC:** T53-L-703 kalkış gücü 1800 shp, SFC **0.568 lb/shp/h**
  ([Purdue AAE propulsion, T53](https://engineering.purdue.edu/~propulsi/propulsion/jets/tprops/t53.html)).
  El kitabı doğrusu 1800 shp'ye uzatılınca 0.573 (deniz seviyesi) → iki kaynak tutarlı.
- **Tork ↔ güç:** "703 … limited to 1290 shp at 100% torque because of power train limitations", "35 psi = %62.5"
  → %100 = 56 psi ([aircav T53-L-703](https://www.aircav.com/cobra/T53-703.html)); 23.0 shp/psi (× N_r/324).
- **Yakıt sistemi:** el kitabı bölüm 2 (ön / arka yakıt hücresi, her birinde pompa; ön pompa arızasında <320 lbs'de
  burun aşağı > 15° yakıt kesilebilir → hücreler bağlantılı), bölüm 6 yakıt moment grafiği (tek doğru, kol
  ~200–203 in → yakıt CG'si boşalırken neredeyse sabit). Çekim sırası açıkça yazılmıyor.
- **Türbülans:** MIL-F-8785C (alçak irtifa Dryden: L_w = h, σ_w = 0.1·W20; hafif / orta / şiddetli W20 = 15 / 30 /
  45 kt); JSBSim kaynak kodu v1.3.1 (`FGWinds.cpp`, `FGFDMExec.cpp`).

## 3. Varsayımlar (doğrulanamayanlar işaretli)

1. **Yakıt akışı el kitabının +15 °C seyir grafiğinden** (JSBSim standart günü ~2600 ft'te ~10 °C; fark ~%0.5). ECU
   kapalı (açıkken +%4). JP-4 lb/h, modelin tankları JP-5 (lb başına ısıl değer hemen hemen aynı).
2. **psi → güç** yalnızca raporlanan eşdeğer SFC ve devir düzeltmesi için (1290 shp @ 56 psi; model yazarı psi
   dönüşümünde ±%20 hata olabilir diyor).
3. **Tank çekimi: iki tanktan eşit** (el kitabı sırayı yazmıyor; moment grafiği birlikte boşalmayla tutarlı). Biri
   boşalırsa kalan diğerinden. Seçenekler: `proportional`, `fwd_first`, `aft_first`.
4. Motor ayrılması = governor kapalı (gaz 0 → serbest tekerlek), 28 Eylül mekanizması. Kullanılamayan yakıt 0.
5. `dryden_agl`: yalnızca öteleme türbülansı (açısal yok), donmuş alan hızı V = max(hava hızı, 15 ft/s), kontrol
   adımında (13.3 Hz) güncelleme, bileşenler ortalama rüzgâr eksenlerinde (rüzgâr yoksa heading).

## 4. Probe'lar

### (a) Tam depoyla yakıt bitme süresi — `probe_a_fuel_endurance.py`, `fig_a_fuel_endurance.png`

300 ft AGL, 2 × 890 lbs (10,280 lbs), eşit çekim, tutucu (ölçüm PID'i) hover'da konumu / 60 kt'ta hava hızını tutar;
tanklar bitince motor ayrılır.

| koşul | chart (el kitabı) | sfc 0.568 × psi→shp | sfc 0.568 × rotor gücü | el kitabı ile hesap |
|---|---|---|---|---|
| OGE hover | **2.38 h** | 2.69 h | 3.42 h | 2.40 h (aynı yakıt) · 2.34 h (10,000 lbs, 1700 lbs JP-4) |
| 60 kt | **3.48 h** | 5.28 h | 6.16 h | 3.44 h · 3.33 h |

Gerçek dayanıklılık ~2–2.5 saat (kullanıcı) → hover ve tipik seyir için chart modeli tutuyor; 60 kt en az güç
(maks. dayanıklılık) hızına yakın, ~3.4 h makul. Sabit SFC kısmi güçte yakıtı %15–35 eksik veriyor (turboşaftta SFC
kısmi güçte artar: el kitabı eşdeğeri 50 psi 0.64, 30 psi 0.81 lb/shp/h). Modelin fiziksel rotor gücü psi'den
~%21 az (hover 8500 lbs: ana 766 + kuyruk 48 hp; psi'den 1030 shp) — psi kalibrasyonu kuyruk rotoru / kayıpları da
içeriyor.
Tam depoyla OGE hover 56 psi'yi ilk **16 dk**, 50 psi'yi ilk **78 dk** aşıyor (10,280 lbs OGE 57.7 psi).

**Modelin tork gereksinimi el kitabıyla örtüşüyor** (bu probe'un yan ürünü): hover 8600 lbs 45.1 ↔ 44.9 psi,
9000 48.0 ↔ 47.7, 9500 51.7 ↔ 51.1, 10,200 57.0 ↔ 55.9; 60 kt 8600 22.9 ↔ 24.0, 10,200 29.1 ↔ 29.3. Seyirde 60–100
kt ±1–2 psi, 120 kt'ta model ~%10–15 düşük (el kitabı 4 TOW rampalı yapı → fazladan sürükleme).

### (b) Rüzgârın rotor modeline yansıması — `probe_b_wind_trim.py`, `fig_b_wind_trim.png`

8500 lbs, 300 ft, heading 0. Rüzgârda yerinde hover ↔ rüzgârsız aynı hava hızı vektöründe uçuş:
**karşı 10/20/30, sağdan 10/20, arkadan 10/20 kt: trim farkı ≤ 0.0002 (kumanda), rotor açıları ≤ 0.03 mrad,
θ/φ ≤ 0.01°** → JSBSim rotoru, downwash ve gövde aerodinamiği hava hızını tutarlı kullanıyor (Galile değişmez);
"deneysel" downwash rüzgâra özgü bir hata üretmiyor.

- Soldan rüzgâr (= sola yana uçuş) 10–30 kt: iki durumda da aynı sınır çevrimi (limit cycle) → tutucu sıkı ölçüte
  oturmuyor; ortalamalar arasındaki fark ≤ 0.014. Ayrı test: `aero/setup/downwash-enable = 0` ile tam oturuyor →
  **downwash modelinin sola yana uçuşta ürettiği küçük salınım** (elevator σ ≈ 0.015, periyot 7–12 s), rüzgârla
  ilgisiz. Sağa yana uçuş temiz.
- Sağdan 30 kt (pedal 0.84) ve arkadan 30 kt: basit tutucu stabilize edemiyor (iki durumda da) → zarf kenarı.
- **SFD tablosu (yer hızıyla) hover-in-wind'de ciddi yanlış:** 20 kt karşı rüzgârda gerçek trim − SFD =
  coll −0.058, lon +0.159, lat −0.112, ped −0.107; 20 kt sağdan: ped +0.236.

### (c) Türbülans şiddeti — `probe_c_turbulence.py`, `probe_c_turbulence.txt`, `fig_c_turbulence.png`

H0 hover 100 ft sakin, H15 hover 100 ft 15 kt karşı rüzgâr, F60 60 kt 300 ft; 6 tohum. `dryden_agl`:

| seviye (W20) | σ yatay / dikey ft/s | açık döngü 10 s: en büyük p·q·r °/s, kayma | PID: konum / irtifa / heading en büyük (hover) | PID bantta (hover / F60) |
|---|---|---|---|---|
| hafif (15 kt) | ~3.4 / 2.4 | ≤ 2.1 °/s, 15–18 ft | 3.3 ft / 1.4 ft / 3° | %99–100 / %42 |
| orta (30 kt) | ~6.8 / 4.8 | ≤ 4.2 °/s, 30–35 ft | 6.5 ft / 2.7 ft / 6° | %83–91 / %14 |
| şiddetli (45 kt) | ~10 / 7.2 | ≤ 6.3 °/s, 45–53 ft | 10 ft / 4.2 ft / 10° | %55–67 / %6 |

- **Öğrenilebilir görünen: hafif ve orta.** Basit PID bile hover bandında (±6 ft / ±3 ft / ±3°) hafifte %100, ortada
  ~%85 kalıyor; açısal bozucu küçük. Şiddetli ancak gevşek toleranslarla anlamlı.
- F60'ta ±2 ft/s hava hızı toleransı türbülansta tanım gereği tutulamaz (hava hızı ölçümü bozucunun kendisi, σ_u ≈ 3
  ft/s) → Aşama 2'de başarı ölçütü süzülmüş hava hızı ya da yer hızı ile olmalı.
- **JSBSim'in kendi türbülansı bu helikopterde kullanılamaz:** hafif seviyede bile açık döngüde 10 s'de p/q/r
  6–900 °/s, PID konumu 100–630 ft kaçırıyor; sakin hover'da orta / şiddetli **NaN** üretiyor. Sebepler (kaynak
  kodu): (1) yükseklik MSL (`Turbulence(in.AltitudeASL)`) → Edwards'ta (2283 ft) hep orta irtifa dalı (L = 1750 ft,
  σ şiddet tablosundan, W20 etkisiz); (2) açısal türbülans p/q/r kanat açıklığıyla ölçekleniyor (AH-1S "kanadı"
  10.75 ft) ve rotora `AeroPQR` üzerinden giriyor; (3) hava hızı ~0'da Tustin ayrıklaştırması sayısal olarak ıraksıyor.

### (d) Yakıt / CG → hover trimi — `probe_d_fuel_cg_trim.py`, `fig_d_fuel_cg_trim.png`

OGE hover, yakıt %100 → %0, üç çekim sırası. Doğrusal uydurma (artık ≤ 0.001):

    elevator = −0.1513 + 0.0109·(CGx − 172 in) − 0.0038·ΔW/1000     θ = −0.38° + 0.116°·(CGx − 172 in)
    collective = 0.6058 + 0.0557·ΔW/1000     pedal = 0.4113 + 0.0019·(CGx − 172) + 0.0402·ΔW/1000

- **Eşit çekimde CG x yalnızca 172.69 → 172.00 in kayıyor** (yakıt CG'si 176 in, boş CG'ye yakın) → uzunlamasına cyclic
  trimi **0.001** değişiyor, θ sabit. Değişen collective (−0.10) ve pedal (−0.073, tork azaldıkça).
- Sıralı çekim olsaydı (önce ön / arka tank) CG ±3.2 in → elevator ±0.032, θ ±0.37°.
- CG z 68.6 → 75.0 in yükseliyor; hover trimine etkisi yok.

### (e) Hava hızına dayalı trim tablosu — `probe_e_trim_table.py`, `trim_table_airspeed.json`, `fig_e_trim_table.png`

Düz, dengeli uçuş (yana hava hızı 0, heading sabit), 300 ft AGL, u_air −20…120 kt × 8500 / 9094 / 9686 / 10,280 lbs;
tablonun 76 noktası ve 1000 ft'teki 14 nokta oturdu (yana uçuş ızgarasında 6 nokta — v = ±20 kt ve sola 10 kt — sıkı ölçüte oturmadı; ortalamalar yine anlamlı). `trim_table.py` (`AirspeedTrimTable`: çift doğrusal ara değer, isteğe bağlı irtifa
düzeltmesi). u_air = 0'da sabit hover trimine eşit (8500 lbs: 0.605 / −0.151 / 0.193 / 0.412).

- SFD tablosunun collective / pedal sütunları tablo kırılma noktalarında (0, 20, 40 … kt) ölçümle **birebir aynı**
  (0.000) — SFD bu modelden üretilmiş; aradaki doğrusal ara değer 0.014'e kadar sapıyor. Uzunlamasına / yanal cyclic
  sütunları ise −0.10…+0.11 farklı (SFD tam AFCS varsayıyor; bizde yalnızca SAS).
- Ağırlık: collective +0.057 / 1000 lbs (hover), pedal +0.04, yüksek hızda elevator +0.027 / 1000 lbs.
- İrtifa (1000 − 300 ft AGL): collective +0.010, elevator / aileron +0.003, pedal +0.002.
- Yana hava hızı ±20 kt (hover yakını): aileron ±0.15, pedal ±0.24…0.50, collective −0.05 → tablo 1B (ileri hız)
  tutuldu; yana trimi ajan gözlemdeki v_air ile karşılamalı (gerekirse 2B tablo).

## 4b. Repo uçağı (kalibre yer etkisi) + 56 psi güç tavanı — `*_repo_cap56.*`

Aynı probe'lar `--aircraft repo --power-cap 56` ile (300 ft AGL'de yer etkisi ~0; fark güç tavanından):

- **(a)** Tam depo (10,280 lbs) OGE hover **2.38 h**, 60 kt **3.48 h** (stokla aynı; chart modeli psi'yi okuyor). Hover
  ilk 16.9 dk 56 psi'nin, 78 dk 50 psi'nin üstünde; tepe 64.7 psi — tavanda devir düşüyor, gösterge Q = P/Ω ile
  artıyor. Model ↔ el kitabı hover torku 9000–10,200 lbs'de 0.3–2.3 psi.
- **(b)** Rüzgârda hover ↔ rüzgârsız aynı hava hızı: iki tutucunun oturduğu çiftlerde en büyük fark ≤ 0.0002 (stokla
  aynı sonuç). Sağdan 30, arkadan 30 ve soldan 10–30 kt'ta tutucu oturmuyor (zarf kenarı / downwash sınır çevrimi).
- **(c)** Türbülans: stokla aynı tablo (hafif: hover bandında %99–100, orta: %79–89, şiddetli: %55–68; F60'ta ±2 ft/s
  hız bandı tutulamıyor).
- **(d)** Hover trimi: `elevator = −0.1507 + 0.0109·(CGx − 172) − 0.0049·ΔW/1000`, `collective = 0.5927 + 0.0774·ΔW/1000`
  (stokta 0.0557 — tavana yaklaştıkça devir düşüyor, aynı taşıma için daha çok collective); eşit çekimde elevator yine
  yalnızca 0.003 değişiyor.
- **(e)** Trim tablosu: 8500–9686 lbs'de stokla ±0.003; **10,280 lbs OGE hover 59.6 psi (tavanın üstü), collective
  0.772** (10 kt'ta 0.691) → tam yakıtla OGE hover güç tavanının ötesinde. Aşama 2 curriculum'u bu yüzden 8800–9700 lbs
  kullanıyor (OGE hover ≤ ~53 psi); ağır kalkış yer etkisi + ETL gerektirir (kalkış env'inin `depart` görevi).

## 5. Beklenmedik bulgular

1. **`run_ic()` rüzgârı siliyor:** `FGFDMExec::Initialize` → `Winds->SetWindNED(IC rüzgârı)`; env'ler teleport /
   bozucu için run_ic çağırıyor → `PhysicsExt.before_step` sabit rüzgârı her adımda yeniden yazıyor.
2. JSBSim milspec / tustin türbülansı bu uçakta ıraksıyor (yukarıda) → `dryden_agl` yazıldı.
3. SFD cyclic sütunları bizim AFCS ayarında ~0.1 yanlış; collective / pedal kırılma noktalarında doğru.
4. Sola yana uçuşta downwash kaynaklı küçük sınır çevrimi.
5. Governor'un integrali yavaş: hover'da rotor devri ~100 s'de 321 → 323.5 rpm, collective trimi ±0.003.
6. Modelin psi'si el kitabının hover ve seyir grafikleriyle ~1 psi içinde örtüşüyor (10,280 lbs hover'da model
   +1…2.5 psi yüksek; 10,280 lbs gerçek azami ağırlığın üstünde).

## 6. Durum / açık

- Yapıldı: tork fonksiyonlarının branch'teki uygulamayla birleştirilmesi, env entegrasyonu (kalkış + manevra; `physics`
  varsayılan None), eski modellerin birebir aynı sonuç verdiğinin testi (evaluate_takeoff 28 senaryo × takeoff_final /
  takeoff_torque, evaluate_robustness 49 koşu: 0 fark), probe'ların repo uçağı + güç tavanıyla tekrarı (4b).
- Manevra env'inde yakıt katmanı desteklenmiyor (FDM her bölümde yeniden kullanılıyor; `physics.fuel` → hata).
- Yakıt akışı el kitabının seyir grafiğinden; hover'da (yer etkisi / ağır) doğrusal uzatma — doğrulanmadı.
- Motor gücü sıcaklık / irtifayla düşmüyor (elektrik motoru; 56 psi tavanı sabit).
