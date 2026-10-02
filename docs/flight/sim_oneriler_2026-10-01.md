# Simülasyon: JSBSim'de olup kullanmadıklarımız, düzeltilmesi gereken parametreler (2026-10-01)

Bu notun her maddesi bu repo'daki AH-1S modeliyle ölçüldü ya da JSBSim 1.3.1'de denendi; ölçülmeyenler ayrıca
belirtildi. Uygulananlar `natural-limits` branch'inde, diğerleri öneri olarak duruyor.

## 1. Bu branch'te uygulananlar (sonuç modeli `models_flight/flight_v2.zip`)

| özellik | JSBSim karşılığı | durum |
|---|---|---|
| Hava sıcaklığı / yoğunluk irtifası | `atmosphere/delta-T` (run_ic'den sonra da kalıyor, ölçüldü) | `physics_ext` `atmo` bölümü ve `delta_T_C` seçeneği; F10 −10…+30 °C ile eğitiyor; gözleme yoğunluk (`density_obs`, obs 43) |
| Gust'ların üst üste binmesi | `atmosphere/cosine-gust/*` | JSBSim, gust sürerken gelen yeni gust'ın yönünü güncellemiyor, genliği basamakla değiştiriyor, süreyi yeniden başlatmıyor → yeni gust öncekinin bitişine erteleniyor |
| Yer yakını rüzgâr kesmesi | (JSBSim'de yok; bizim MIL-F-8785C log profili) | seçenek: `wind_shear` (eğitimde kapalı) |
| Türbülans şiddeti rüzgâra bağlı | (bizim Dryden'ımız) | seçenek: `turb_from_wind` → W20 = ortalama rüzgâr (eğitimde kapalı; şimdi E3'te 5 kt rüzgârla "orta" türbülans gelebiliyor) |
| Reset hover PID'i güçlü yan rüzgârda | — | 20–25 kt yan rüzgârda hover kurulamıyordu (pedal integratörü ±0.3'te doyuyor → 2.7–3.5° kalıcı heading hatası; soldan rüzgârda ~1 ft/s yavaş salınım). İkinci evre: integratör ±0.6, ölçütler rüzgârla gevşek. Birinci evre birebir eski |

## 2. JSBSim'de olup kullanmadıklarımız (öneri, öncelik sırasıyla)

1. **Turboşaft motor (FGTurboProp) — en büyük açık.** Model `electric_1500hp` ile uçuyor (`ah1s.xml`: "should become
   avco_lycoming_t53 turbo-shaft engine"). JSBSim paketindeki `engine/avco_lycoming_t53.xml` aeromatic'in itki tabanlı
   (turbojet tipi, `milthrust 3320 lbf`) taslağı; olduğu gibi kullanılamaz. **Fizibilite denendi:** `turboprop_engine`
   (FGTurboProp; N1 dinamiği, ITT, PSFC, tanklardan yakıt) ana rotoru sürebiliyor — kaba bir tanımla rotor devire çıktı,
   governor devri tuttu, collective kaldırınca helikopter kalktı, yakıt tanklardan JSBSim'in kendisi tarafından çekildi
   (40 s'de 8 lbs). Kalibre edilmedi (güç tabloları, governor kazançları, yakıt akışı).
   Kazanç: (a) güç mevcut sıcaklık / irtifayla düşer — şimdi düşmüyor (aşağıda 3.1); (b) motor tepki gecikmesi (N1) ve
   collective çekilince rotor devri düşmesi; (c) yakıt ve motor arızası JSBSim içinde. Maliyet: T53-L-703 verisiyle tablo
   (TM 55-1520-234-10 performans grafikleri), doğrulama, sonra yeniden eğitim (ince ayar). Birkaç gün.
   Ara adım (ucuz): mevcut güç tavanını (`fcs/throttle-max-norm`) yoğunluk irtifası / sıcaklıkla düşür.
2. **Harici yükler ve nokta kütleler** (`<pointmass>`, `inertia/pointmass-weight-lbs[i]`, yük sürüklemesi aero fonksiyonu).
   Modelde hiç nokta kütle yok (8500 lbs tek parça). TOW rampaları / roket kovanları / mühimmat eklenirse: (a) yüksek
   hızda tork gerçekçi olur (aşağıda 3.2); (b) atış = ani ağırlık / CG değişimi bozucusu olarak eğitilebilir.
3. **Sensör ve aktüatör dinamikleri** (FCS `<sensor>`: gürültü, gecikme, bias, drift, kuantalama; `<actuator>`: gecikme,
   hız sınırı, ölü bölge, histerezis). Şimdi gözlem kusursuz (düşük hızda hava hızı dahil — gerçekte pitot ~20–30 kt altında
   güvenilmez) ve kumanda hız sınırı Python'da. Sağlamlık için (domain randomization) en ucuz yol Python'da gözlem gürültüsü;
   JSBSim XML'iyle yapmak daha fiziksel.
4. **Pist / arazi yüksekliği** (`ic/terrain-elevation-ft`; şimdi hep Edwards, 2283 ft MSL). Yüksek irtifa pisti ve
   (`position/terrain-elevation-asl-ft` her adımda yazılarak) yükseltilmiş platform / arazi takibi. Eğimli zemin Python'dan
   yapılamıyor (C++ ground callback gerekiyor).
5. **Nem** (`atmosphere/RH`): yoğunluğa küçük etki; tek satır.
6. **Mikroburst** (`atmosphere/updownburst/*`): JSBSim 1.3.1 yalnızca `number-of-cells`'i açıyor, hücre konumu / gücü
   property olarak ayarlanamıyor → kullanmak için kendi modelimiz (halka girdap) gerekir; iniş / kalkış tehlikesi eğitimi.
7. JSBSim'in kendi türbülansı (Milspec / Tustin) — denendi, bu helikopterde kullanılamaz (MSL irtifa, kanat açıklığıyla
   ölçeklenen açısal türbülans → NaN); bizim Dryden'ımız kalmalı.

## 3. Düzeltilmesi gereken parametreler / bilinen model hataları (ölçüldü)

1. **Sıcak / yüksek performans fazla iyi.** 9700 lbs OGE hover torku: standart gün (yoğunluk irtifası 2290 ft) 53.6 psi,
   +15 °C (3875 ft) 54.2, +30 °C (5373 ft) 54.9 psi — neredeyse değişmiyor, çünkü elektrik motorunun gücü düşmüyor.
   Gerçek: YAH-1S 4000 ft basınç irtifası / 95 °F'ta (yoğunluk irtifası ~7300 ft) OGE hover ancak 9175 lbs'de
   (DTIC ADA025476). Çözüm: madde 2.1.
2. **Yüksek hızda tork düşük.** Modelde düz uçuş: 9700 lbs 130 kt 39.6 psi, 150 kt 47.0 psi. Gerçek AH-1S en yüksek düz uçuş
   hızı ~128–130 kt (TOW'lu, ~%100 tork). Seyir grafiğiyle 60–100 kt ±1–2 psi tutuyor, 120 kt'ta model %10–15 düşük
   (docs/physics_ext). Çözüm: madde 2.2 (TOW rampası sürüklemesi) ya da gövde düz levha alanının kalibrasyonu.
3. **Kuyruk rotorunun gücü motordan çekilmiyor.** Pedal basamağında kuyruk rotoru 58 → 86 hp, ana motor gücü 895 → 895 hp
   (değişmedi); hover'da ~60 hp, yerinde dönüşte ~110 hp "bedava". Tork göstergesi yalnızca ana rotor torku. Gerçekte motor
   çıkışı kuyruk rotorunu da sürer. Çözüm: kuyruk rotoru gücünü güç tavanından düş (ya da göstergeye ekle). Not: kuyruk
   rotoru YAH-1S sınırını (187 shp sürekli, 260 shp 4 s) ölçtüğümüz manevralarda aşmadı (en fazla 114 hp).
4. **Sıcak günde ajan zayıftı (eğitimsiz).** flight_final kısa zincir: standart gün 4/4, +15 °C 2/4, +30 °C 1/4; inişte
   temas hızı −1.3 → −3.6 / −6.5 ft/s (sınır 4). F10–F13 −10…+30 °C ile eğitti, gözlemde hava yoğunluğu var: held-out
   sıcak gün grubunda görev 19/31 (flight_final) → 27/31 (flight_f10) → **30/31 (flight_v2)**, hepsi güvenli
   (docs/flight/README.md bölüm 8.6). Not: bu, elektrik motorunun gücü düşmeyen modelinde — madde 3.1 düzelince sıcak /
   yüksek performans gerçekte daha zor olacak.
5. **FGRotor iç dt'si** 1/120 s (set_dt load_model'den sonra çağrılıyor; sim dt 0.0075 s). Yer etkisi bu dt için kalibre;
   değiştirilirse kalibrasyon ve ince ayar gerekir (README 31).
6. **Hava hızı gözlemi düşük hızda kusursuz** (yana hava hızı dahil) — gerçek göstergede değil; sim-to-real hedeflenirse
   gürültü / bozulma eklenmeli (madde 2.3).
7. **Trim tablosu tek atmosfer** (standart gün, 300 ft): sıcaklık değişince collective trimi kayıyor; ajan artık yoğunluğu
   görüyor (obs 43). İsteğe bağlı: tabloya yoğunluk ekseni.

## 4. Kaynaklar
- Hız / tork / ağırlık: Vertipedia AH-1S (en yüksek düz uçuş 128 kt, Vne 170 kt, 10,000 lbs); aircav.com Cobra (Vne 190 kt
  temiz, 170 kt TOW ya da > 9500 lbs; yana 35 kt, geriye 30 kt; tork %0–88 sürekli, %88–100 30 dk, %100 en fazla);
  DTIC ADA025476 (YAH-1S 1975: 10,000 lbs'de ~130 KTAS düz uçuş, 4000 ft / 95 °F'ta OGE hover 9175 lbs, kuyruk rotoru
  187 shp sürekli / 260 shp 4 s).
- JSBSim 1.3.1 property kataloğu ve paket dosyaları (`engine/avco_lycoming_t53.xml`, `engine/engtm601.xml`).
