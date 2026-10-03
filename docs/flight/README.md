# Tek ajanlı sürekli uçuş (Aşama 2) — `helicopter_env_flight.py`, `flight_curriculum.py`

Tek bir PPO ajanı (Stable-Baselines3, sıfırdan, öğretmen–öğrenci yok), tek ve sürekli bir episode'da: rotor warm-up
(scriptli) → kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δhız / Δheading / Δirtifa → duruş (hover) → isteğe bağlı
iniş. Dört kumanda doğrudan (collective, boylamsal / yanal cyclic, pedal); AFCS yalnızca SAS. Fizik baştan açık: repo
uçağı (kalibre yer etkisi), güç tavanı %100 tork (56 psi), tork gözlemi ve cezası, yakıt tüketimi (el kitabı grafiği;
bitince motor ayrılır). Rüzgâr / gust / türbülans seviyenin çevre aşamasından.

> Durum (2026-09-29): **eğitim bitti** — sonuç modeli `models_flight/flight_final.zip` (fl_v9 4.5 M). Sonuç tablosu ve
> şekiller bölüm 6'da.
>
> **Güncelleme (2026-10-01): yeni sonuç modeli `models_flight/flight_v2.zip`** — doğal komut zarfı + komut
> yönlendirici, sıcak gün (hava yoğunluğu gözlemi), hover hassasiyeti; held-out test takımı. Bölüm 8.

## 1. Env

- Kalkış env'inin (`helicopter_env_takeoff.py`) alt sınıfı; aynı action hattı (a → filtre → expo → trim ± aralık → hız
  sınırı), aynı hover / kalkış / iniş görevleri ve ödülü (to_v18'in son tasarımı).
- **Trim çizelgesi** (`trim_table_airspeed_repo_cap56.json`, probe e): gövde ekseninde ileri hava hızı × ağırlık. Tabloya
  **görevin referans hızı** girer (hover 0; ileri uçuşta hedef hıza 2.5 ft/s²'lik rampa; duruşta 0'a yavaşlama oranıyla).
  Ölçülen hava hızıyla çizelgelemek (ilk tasarım) doğal hız kararlılığını sıfırlıyordu (bölüm 5, fl_v0).
- **Yeni görevler:** `cruise` (hava hızı u*, heading ψ*, irtifa h*; Δ'lar ölçülen duruma göre, komut verilmeyen eksen
  önceki hedefine devam eder; hover'dan hız komutu = geçiş / "accel"), `stop` (referans noktası yer izi boyunca mevcut
  yer hızından sabit yavaşlamayla durur; sonra o noktada hover bandı).
- **Gözlem 42:** kalkış env'inin 29'u + tork + 12 (hava hızı ileri / yana, yer hızı ileri / yana, ileri uçuş bayrağı, hız
  hatası ince / kaba, hedef hava hızı, hareketli hedefin hızı ileri / yana, ağırlık, referans hızı). Kalkış env'inin hız
  girdileri **hız hatası** olarak verilir (hover'da yer hızı, duruşta hareketli hedefe göre, ileri uçuşta referans hava
  hızına göre). Rüzgârın kendisi verilmez (hava / yer hızı farkından dolaylı).
- **Başarı (ileri uçuş):** hava hızı ±4 ft/s (1 s süzülmüş), irtifa ±12 ft, heading ±3°, dikey hız ≤ 4 ft/s, yana hava
  hızı ≤ 8 ft/s, 5 s; komut verilmeyen eksen hız 15 ft/s / heading 10° / irtifa 40 ft. Süre hedefi: tepki (3 s) +
  max(|Δu| / 2.5 ft/s², |Δψ| / koordineli dönüş hızı (20° yatış, ≤ 8°/s), |Δh| / 10 ft/s); son sınır max(1.25·T, T + 2).
  **Türbülansta bantlar genişler:** hafif ×1.25, orta ×2 (ADS-33'ün desired / adequate ayrımı gibi; kural tabanlı pilot
  orta türbülansta ±4 ft/s / ±3° bandında 5 s kalamıyor). Güvenlik sınırları ve iniş temas hızı değişmez.
- **Güvenlik (ileri uçuş / duruş pencerelerinde):** hava hızı ≤ 130 kt, yatış ≤ 60° (30 kt üstü), hızlıyken yere değme
  (yer hızı > 15 ft/s) ya da 40 kt üstünde kızak < 8 ft, hareketli / izleyen hedeften 300 ft. Hover görevlerinde kalkış
  env'inin sınırları (yer hızı ≤ 60 ft/s — rüzgârda hover'da hava hızı büyük olabilir).

## 2. Curriculum (görev × çevre) — `flight_curriculum.py`

| seviye | görev | çevre | eşik | rehearsal |
|---|---|---|---|---|
| F1 | tut: %50 hover (havada başla, 15–300 ft, küçük bozukluk; 10 s) · %50 ileri uçuş (40–100 kt, 150–800 ft; hız / irtifa / heading'i 10 s tut) | E0 | %80 | — |
| F2 | %50 hover: kalkış 10–300 ft (%60) ya da havada başla (%40) + 0–2 manevra (dönüş 30–120°, kayma 15–50 ft, dikey 15–40 ft) · %50 ileri uçuş: 1–2 tek eksenli Δ (±10–20 kt / ±15–60° / ±50–150 ft; yumuşak süre hedefi: 2 ft/s², 15° yatış, 8 ft/s, tepki 4 s) | E0 | %75 | F1 %30 |
| F3 | %60 ileri uçuş: 2–4 Δ, tek eksen ya da birleşik (±10–35 kt / ±20–150° / ±50–300 ft), %30 kesen komut · %40 hover: kalkış ya da havada başla + 1–3 manevra | E0 %60 · E1 %40 | %70 | F1, F2 %35 |
| F4 | geçişler: hover (30–300 ft) → hızlanma 40–80 kt (+0–200 ft) + 0–1 Δ → duruş (2–3 ft/s²) · %50 ileri uçuşta başla (30–80 kt, 100–500 ft) + 0–1 Δ → duruş | E0 %60 · E1 %40 | %70 | F2, F3 %40 |
| F5 | zincir: yerden kalkış (20–60 ft) → hızlanma 40–80 kt (+50–300 ft) → 1–3 Δ (±10–30 kt / ±20–120° / ±50–250 ft) → duruş | E0 %60 · E1 %40 | %65 | F2–F4 %40 |
| F6 | iniş: alçak hover (12–60 ft) ya da kısa kalkış + 0–1 manevra → pad'e iniş; %30 yerde hafif yüklü, %30 çok alçak hover'dan başla | E0 %70 · E1 %30 | %70 | F3–F5 %40 |
| F7 | karma: hover / kalkış (10–600 ft) / iniş / ileri uçuş (1–3 Δ, %20 kesen) / geçişler / zincir; duruştan sonra %30 iniş | E1 %50 · E2 %50 | %65 | F3–F6 %30 |
| F8 | son: F7 + daha büyük Δ (1–4 Δ, ±10–35 kt / ±20–180° / ±50–300 ft, %25 kesen), %50 zincir | E1 %30 · E2 %40 · E3 %30 | %60 | F3–F7 %30 |

Çevre aşamaları: E0 sakin · E1 rüzgâr 0–10 kt · E2 0–15 kt + %60 hafif türbülans + gust (dakikada 0.5, 3–8 kt) ·
E3 5–25 kt + hafif / orta türbülans (%50 / %50) + gust (dakikada 1, 5–12 kt). Rüzgâr yönü rastgele (başlangıç heading'ine
göre). Tork cezası ve yakıt her seviyede açık; ağırlık 8800–9700 lbs (tank başına 150–600 lbs; OGE hover ≤ ~53 psi).

**Hover ve ileri uçuş F1'den itibaren birlikte** (bölüm 5, fl_v2): hover'da eğitilmiş ağ ileri uçuşa geçince ileri uçuş
%0'da kaldı ve hover'ı unuttu. Seviye atlama: son 100 episode başarısı ≥ seviye eşiği **ve** görev türü başına kapı
(seviyedeki her komut türü, tekrar edilenler dahil, ≥ %80, en az 30 komut) — ya da deterministik değerlendirmede mevcut
seviyenin başarısı ≥ seviye eşiği (`--promote-on-eval`, 12 episode, sabit seed'ler).

## 3. Yapılabilirlik ve ödül kontrolü — `check_flight_scripted.py`, `scripted_pilot.py` (RL değil)

Kural tabanlı PID pilot (reset PID'lerinin genelleştirilmişi; eğitimde kullanılmaz, policy'ye action önermez) sabit
senaryoları ve seviyelerden örneklenen episode'ları uçar:

- **Tam zincir uçulabiliyor:** yerden kalkış 40 ft → 60 kt'a hızlanma (+150 ft) → +30 kt → +90° → −100 ft → duruş →
  iniş, sakin havada 7/7 görev (süre hedefleri ve bantlarla). Rüzgâr / türbülansta da güvenli (21/21 senaryo, düşme yok).
- Pilotun zayıf olduğu yerler (ajanın aşması beklenen): 80 kt'ta dönüşlerde ±3° bandına oturmadan önce aşma, birleşik
  komutlarda hız düşürürken tırmanma, hover'da konum salınımı. Görev bantları / süre hedefleri sıkı ama yapılabilir.
- **Ödül tutarlılığı:** aynı episode'larda pilotun getirisi sıfır action'ınkinden 31/31 (ilk sürüm) ve 16/16 (referans
  hızı çizelgesiyle) büyük.
- `python evaluate_flight.py --scripted` aynı pilotu değerlendirme takımında da uçurur (karşılaştırma tabanı).

## 4. Değerlendirme — `evaluate_flight.py`

Sabit takım (eğitimde kullanılmayan senaryolar, deterministik policy): kısa / uzun görev zinciri × sakin / 15 kt rüzgâr /
25 kt rüzgâr + orta türbülans + gust × hafif (8800 lbs) / ağır (9700 lbs) = 12 zincir + 9 öğe senaryosu (hover 30 s,
yandan rüzgârda hover, 60 → 100 → 40 kt, 80 kt'ta ±dönüşler, ±300 ft, hover → 100 kt, 100 kt'tan duruş, 300 ft'ten
iniş, arkadan rüzgârda hızlanma). Rapor: güvenli / tüm görevler / görev başarısı; **ADS-33 benzeri sınıf** (her pencere:
istenen = env başarısı; yeterli = bantlar 2×, son sınır 1.5×, kuplaj 2×); tutma doğruluğu; tork (en yüksek, 50 / 56 psi
üstü süre, 56 üstü en uzun kesintisiz süre, en düşük devir); yakıt (harcanan, ortalama akış).

## 5. Eğitim koşuları

Ortam: Claude'un cloud konteyneri, 2 çekirdek CPU, SB3 PPO, 2 paralel env (~640 adım/s, ~2.3 M adım/saat). Ortak
ayarlar: n_steps 4096 × 2 env, batch 512, 10 epoch, lr 3e-4 (F6a'dan itibaren 1e-4 + KL 0.02), γ 0.995, λ 0.95,
σ0 = e^−1.2, ayrı pi / vf ağları (256×256, tanh), `--promote-on-eval`, 500 bin adımda bir deterministik değerlendirme
(12'şer episode, seed 90000+; en iyisi `best.zip`). Her koşunun günlüğü, `progress.csv`, `eval.csv` ve
`curriculum_state.json`'ı: `docs/flight/runs/<koşu>/`. **Sonuç modelinin soyu:** fl_v3 (0 → 2.77 M, F5 geçilene kadar)
→ fl_v6 (0.5 M) → fl_v8 (4.0 M) → fl_v9 (4.5 M) = toplam ~11.8 M adım, ~5 saat CPU.

| koşu | başlangıç | değişiklik | sonuç |
|---|---|---|---|
| fl_v0 | sıfırdan, 256×256 | trim ölçülen hava hızıyla (τ 2 s); F1 = hover tut + 0–2 manevra | F1'de 300 bin adımda başarı %6, **episode'ların %50'si 150 ft sapmayla** bitiyordu (kalkış env'inin K1'inde %2). Rastgele policy'yle: ölçülen hıza göre trim 5/16, referans hızına göre 1/16 kaçış. Durduruldu. |
| fl_v1 | sıfırdan, 256×256 | trim görevin referans hızıyla; F1 = yalnızca hover tut | **F1 160 bin, F2 337 bin adımda geçildi** (toplam ~0.5 M, 16 dk). F3'te (ileri uçuşta başlangıç) episode'ların %80'i 5 s içinde 40° pitch ile bitti: hover policy'si 90 kt'lık yer hızını (gözlemde ±5'te kırpılmış) "fren yap" diye okuyup cyclic'i tam geri çekiyordu. Durduruldu. |
| fl_v2 | fl_v1'in F2 modeli, F3'ten | kalkış env'inin hız girdileri hız hatası (hover'da aynı; ileri uçuşta referans hava hızına göre) | pitch kaçışı bitti, ama ileri uçuş 0.76 M adımda **%0**; deterministik F2 (hover) %100 → %25 (unutma). Durduruldu. |
| diag_f3 | sıfırdan, yalnızca ileri uçuş (tanı) | — | "tut" hemen %97; heading komutları 0.5 M adımda %0 (dönüş yatışla yapılır; ajan bulamadı) → koordineli dönüş yönlendirmesi (istenen yatış = atan(r·V/g), 15→30 kt'ta devreye girer) ve ilk Δ seviyesinde yumuşak süre hedefleri |
| fl_v3 | sıfırdan, 256×256, `--fine-from F7`, seed 4 | hover + ileri uçuş F1'den birlikte; dönüş yönlendirmesi; F2'de yumuşak Δ'lar | F1 0.25 M (7.8 dk), F2 1.0 M, F3 2.0 M, F4 2.5 M, **F5 2.8 M adımda (72 dk)** geçildi; F5 anında deterministik F3 10/12, F5 7/12. F6'da (iniş) 0.23 M adımda iniş %0 — kızaklar ~2 ft'te hover, yerde başlasa havalanıyor (kalkış ajanının sıfırdan tekrarındaki yerel optimum); aynı sürede deterministik F3 10/12 → 3/12. 3.1 M adımda bir yer başlangıcı kurulamadı (kızak 3 temas noktası, 3 deneme aynı koşulla) → eğitim durdu. |
| fl_v4 | fl_v3'ün F5 modeli, yeni F6a | F6a oturma okulu: %60 yerde hafif yüklü, %40 çok alçak hover başlangıcı; seviyeden örneklenen episode kurulamazsa yeniden örnekle | 0.5 M adımda iniş %0 |
| fl_v5 | aynı | **episode sonu başarıdan bağımsız** (`end_at_deadline`): eskiden son görev başarılı olunca episode 1 s sonra bitiyordu — adım başına ödül pozitif olduğundan oturmak 43, havalanıp 2 ft'te hover etmek 60 getiri; şimdi 89 ↔ 61 | 0.55 M adımda %0 |
| fl_v6 | aynı; F6a'dan itibaren lr 1e-4 + KL 0.02 | yere yakın havadayken collective indirme ödülü (`w_coll_down_air`); F6a tekrar %20 | 0.56 M adımda %0; F3 / F5 korunuyor (det. %75 / %75) |
| fl_t7 | aynı, tekrar yok (tanı) | — | 0.2 M adımda (1000+ iniş episode'u) %0, F5 %58 → %17: sorun örnek azlığı değil |
| **fl_v8** | fl_v6'nın 0.5 M modeli | **yerde komut edilen collective'e (ham action) ödül** (`w_coll_down_act`). Neden: collective hız sınırlı (0.6/s); yerde başlangıçta hedef kumanda mevcut kumandanın çok üstündeyken action'daki küçük değişiklik kumandayı hiç değiştirmiyor → kumandaya bağlı iniş ödülleri yerel gradyan vermiyordu | 0.4 M adımda ajan yerde kalmayı öğrendi; **F6a 0.87 M, F6 1.5 M (det. %83), F7 3.5 M, F8 4.0 M adımda** geçildi (det. F7 %67, F8 %67; son seviye geçilince eğitim bitti, 103 dk). Takımda: güvenli 21/21, tüm görevler 12/21, görev 94/104; zayıf: yerinde dönüş (ADS-33 yetersiz, 180°'de 17–42 ft kayma), orta türbülansta iniş, 9700 lbs'de geçişte 7–9 s 57 psi |
| **fl_v9** (cila) | fl_v8'in F8 modeli, F9'da (`--no-promote`), 5 M adım (120 dk) | F9: F8 karışımı + havada başlangıç %40, 1–3 hover manevrası, sakin hover manevrası tekrarı (F2); 56 psi üstüne doğrusal ek ceza (`pen_torque_over_lin`) | eğitim başarısı (stokastik, son 100 episode) F9 %45–64 → %70–88; son 100 komutta yerinde dönüş %43 → %85–98, kayma %67 → %92–99, iniş %66 → %77–86 (fl_v8'in F7'sinde dönüş / kayma %20–30 idi). Takımda en iyi ara model **4.5 M** (tüm görevler 17/21, görev 102/105, iniş 13/13); 5 M'de iniş 8/13'e düştü (dalgalanma) → **sonuç modeli = fl_v9 4.5 M** |

## 6. Sonuçlar

**Sonuç modeli: `models_flight/flight_final.zip`** = fl_v9'un 4.5 M adımdaki ara modeli (soy toplamı ~11.8 M adım, ~5 saat
CPU; sha256 `models_sha256.txt`'de). Obs 42, dört kumanda, AFCS yalnızca SAS, repo uçağı + 56 psi güç tavanı + tork
cezası + yakıt tüketimi. **Seçim:** fl_v9'un 3 / 4 / 4.5 / 5 M ara modelleri ve fl_v8'in F8 modeli değerlendirme takımında
karşılaştırıldı; en iyisi 4.5 M (takım bu yüzden seçimde de kullanıldı → sayıları biraz iyimser). Bağımsız kontrol:
seviye istatistikleri (başka seed'ler) ve ADS-33 karnesi. 4 M ara modeli istatistiksel olarak eşdeğer (takımda tüm görevler
16/21 ↔ 17/21, seviyelerde ortalama %90 ↔ %86).

### 6.1. Değerlendirme takımı — `evaluate_flight.py` (deterministik; `eval_final.json`, `eval_fl_v8_F8.json`, `eval_pid.json`)

Zincirler: kısa (yerden 50 ft → 60 kt'a hızlanma +100 ft → duruş → iniş) ve uzun (30 ft → 80 kt +200 ft → +20 kt → +90° →
−100 ft → −30 kt / −60° / +50 ft → +180° → duruş → iniş) × sakin / 15 kt rüzgâr (45° sağdan) / 25 kt rüzgâr + orta
türbülans + gust × 8800 / 9700 lbs; 9 öğe senaryosu. "İstenen" = env'in başarı bandı + süre hedefi + kuplaj; "yeterli" =
bantlar 2×, son sınır 1.5× (ADS-33'ün desired / adequate ayrımı gibi; Cooper-Harper değil).

| ölçüt | **RL sonuç (fl_v9 4.5 M)** | RL fl_v8 F8 (4.0 M) | PID pilot (RL değil) |
|---|---|---|---|
| güvenli (21 senaryo; düşme / sınır aşımı yok) | **21/21** | 21/21 | 21/21 |
| tüm görevleri başarılı senaryo | **17/21** | 12/21 | 5/21 |
| görev başarısı (istenen) | **102/105** | 94/104 | 65/104 |
| görev: yeterli ya da iyi | **105/105** | 100/104 | 94/104 |
| zincir, sakin: tüm / görev | 4/4 · 26/26 | 3/4 · 25/26 | 0/4 · 20/26 |
| zincir, 15 kt rüzgâr: tüm / görev | 4/4 · 26/26 | 3/4 · 25/26 | 0/4 · 15/26 |
| zincir, 25 kt + orta türbülans + gust: tüm / görev | 2/4 · 23/26 | 0/4 · 19/26 | 0/4 · 8/26 |
| öğe senaryoları: tüm / görev | 7/9 · 27/27 | 6/9 · 25/26 | 5/9 · 22/26 |
| 56 psi (%100) aşılan senaryo | 9/21 | 11/21 | 14/21 |
| 56 psi üstü toplam süre | 43 s / 3648 s (%1.2) | 54 s / 3736 s | 37 s / 4318 s |
| 56 psi üstü en uzun kesintisiz | 8.5 s | 8.9 s | 2.8 s |
| en yüksek tork | 64.8 psi (türbülans, anlık) | 66.8 psi | 61.9 psi |
| 50 psi üstü toplam süre | 330 s | 334 s | 347 s |
| en düşük rotor devri | 311 rpm | 312 | 317 |
| yakıt: toplam, ortalama akış | 565 lbs, 568 lb/h | 574 lbs, 562 lb/h | 695 lbs, 589 lb/h |

Görev türü başına (istenen / yeterli+ / n), sonuç modeli: hızlanma 14/14/14, duruş 13/13/13, **iniş 13/13/13**, kalkış
12/12/12, hover tut 13/13/13, ileri uçuşta hız 6/8/8, heading 14/14/14, irtifa 7/8/8, birleşik 6/6/6. **56 psi
aşımları:** türbülanssız senaryolarda en fazla 56.9–58.4 psi ve çoğunlukla 9700 lbs'de (OGE hover ~53 psi, tavana 3 psi),
hover → ileri uçuş geçişinde (hızlanma penceresi, 0 → ~30 kt'ta tırmanırken; kısa zincir 15 kt rüzgârda 8.5 s
kesintisiz) ve kalkış anında ~1 s; 60 psi üstü tepeler yalnızca orta türbülansta, anlık (en fazla 64.8 psi).

### 6.2. Seviye istatistikleri (20 episode / seviye, deterministik, seed 70000+; `eval_final_levels.json`)

| seviye | episode başarısı | çevre | notlar |
|---|---|---|---|
| F3 (ileri uçuş 2–4 birleşik Δ + hover manevraları) | %70 | E0 9 · E1 11 | ileri uçuşta heading %70 (10 komut), irtifa %83 (6); hover görevleri 15/16 (kayma 3/4) |
| F5 (yerden kalkış → hızlanma → Δ → duruş) | **%100** | E0 10 · E1 10 | 97/97 görev |
| F6 (iniş) | %95 | E0 12 · E1 8 | iniş 20/20 |
| F7 (karma, 0–15 kt, hafif türbülans) | %85 | E1 9 · E2 11 | |
| F8 (karma + zincir, 0–25 kt, hafif / orta türbülans, gust) | %80 | E1 7 · E2 6 · E3 7 | iniş 4/5, yerinde dönüş 4/5 |

Düşme / güvenlik sınırı aşımı: 100 episode'un hiçbirinde (hepsi `time_limit`).

### 6.3. ADS-33 karnesi — `evaluate_ads33.py --heavy` (sakin hava; `docs/ads33/karne_flight_final.json`)

| MTE | 8800 lbs | 9700 lbs | kalkış ajanı `takeoff_torque` (8500 lbs) |
|---|---|---|---|
| hover (sağ / sol yaklaşma) | yeterli / yeterli | yeterli / yeterli | yeterli / yeterli |
| hovering turn ±180° | yetersiz / yetersiz | yetersiz / yetersiz | yeterli / yetersiz |
| vertical (+25 / −25 ft) | **istenen** | yeterli | yeterli |
| pirouette | yetersiz / yetersiz | yetersiz / yetersiz | yetersiz / yetersiz |
| landing | **istenen** | yeterli | yetersiz |
| toplam | istenen 2, yeterli 6, yetersiz 8 / 16 | | istenen 0, yeterli 4, yetersiz 4 / 8 |

Hovering turn 180°'yi 5–10 s'de tamamlıyor (ADS-33 istenen ≤ 10 s) ama dönüş sırasında konum 8–22 ft kayıyor (ADS-33
yeterli ±6 ft; eğitimdeki dönüş görevinin kuplaj sınırı 20 ft — daha gevşek). fl_v8'de 17–42 ft idi (cila öncesi).

### 6.4. Şekiller

- `fig_flight_chain.png` — uzun zincir, 15 kt rüzgâr, 9700 lbs: irtifa, hava / yer hızı ve trim çizelgesi, heading,
  tork (50 / 56 psi çizgileri), yakıt, yer izi; görev pencereleri.
- `fig_flight_training.png` — soy boyunca seviye, eğitim başarısı ve deterministik değerlendirmeler (fl_v3 → fl_v6 → fl_v8
  → fl_v9).
- `fig_flight_chain_pid.png` — aynı zincir, PID pilot (karşılaştırma).

### 6.5. Bilinen sınırlar / açık konular

- **Yerinde dönüş ve pirouette hassasiyeti** ADS-33'ün altında (konum kayması); eğitim toleransı sıkılaştırılabilir.
  (→ 2026-10-01: flight_v2, bölüm 8.)
- **Tork:** 9700 lbs'de (tavana 3 psi) rüzgârlı geçişlerde 56 psi 2–3 psi aşılıyor (en uzun 8.5 s). Doğrusal ek ceza aşımı
  fl_v8'e göre azalttı, sıfırlamadı. AH-1S'in 56 psi üstü geçici sınırı (süre) el kitabından doğrulanmadı — **varsayım
  yok, yalnızca ölçüm raporlandı.**
- **Orta türbülans + 25 kt:** güvenli, ama uzun zincirde ileri uçuşta hız / irtifa bantları (2× genişletilmiş) her zaman
  tutmuyor (2/4 zincir tüm görevler).
- **Soy tek seed ve birden çok koşu:** ödül / curriculum değişiklikleri koşular arasında yapıldı (iniş ödülleri F6a'da,
  doğrusal tork cezası F9'da). Son kodla sıfırdan tek koşu doğrulanmadı.
- **İleri uçuşta en yüksek hız 100 kt'a kadar eğitildi** (şartname 0–100 kt); 130 kt güvenlik sınırı.
- Rüzgârın kendisi gözlemde yok (hava / yer hızı farkından dolaylı); sensör gürültüsü yok.

## 7. Çalıştırma

```bash
python helicopter_env_flight.py                                    # duman testi (F1/F2/F4/F5/F8, sıfır action)
python docs/flight/check_flight_scripted.py --scenarios all --levels F3,F5,F6a,F6 --seeds 3 --zero   # PID pilot + ödül
# eğitim, son kodla (F1 → F8, sonra F9 cila). Sonuç modelinin soyu birden çok koşudan geldi (bölüm 5: fl_v3 F5'e kadar →
# fl_v6 → fl_v8 → fl_v9) ve aradaki kod değişiklikleri F6a'dan önceki seviyeleri de etkiler (end_at_deadline, doğrusal
# tork cezası) — bu iki komutla sıfırdan tek seferde yeniden üretme DOĞRULANMADI.
python train_command_curriculum.py --task flight --out runs/fl --stop-after F8 --total-steps 40000000 --n-envs 2 \
    --n-steps 4096 --batch-size 512 --net 256,256 --eval-freq 500000 --eval-episodes 12 --eval-levels F3,F5,F6 \
    --promote-on-eval --snapshot-freq 1000000 --fine-from F6a
python train_command_curriculum.py --task flight --out runs/fl_cila --init-model runs/fl/models/level_08_F8.zip \
    --level F9 --no-promote --total-steps 5000000 --n-envs 2 --n-steps 4096 --batch-size 512 --net 256,256 \
    --eval-freq 500000 --eval-episodes 12 --eval-levels F3,F6,F8 --snapshot-freq 500000 --fine-from F6a
# değerlendirme (ara modeller arasından seçim: takım + seviye istatistikleri + ADS-33)
python evaluate_flight.py --model models_flight/flight_final.zip --json docs/flight/eval_final.json
python evaluate_flight.py --model models_flight/flight_final.zip --levels F3,F5,F6,F7,F8 --episodes 20 --no-scenarios \
    --json docs/flight/eval_final_levels.json
python evaluate_flight.py --scripted --json docs/flight/eval_pid.json               # PID pilot tabanı (RL değil)
python evaluate_ads33.py --model models_flight/flight_final.zip --heavy --json docs/ads33/karne_flight_final.json
# şekiller ve demolar
python docs/flight/fig_flight.py --model models_flight/flight_final.zip \
    --runs docs/flight/runs/fl_v3:2765704 docs/flight/runs/fl_v6:500000 docs/flight/runs/fl_v8 docs/flight/runs/fl_v9:4500000
python docs/flight/fig_flight.py --scripted                                        # aynı zincir, PID pilot
python command_viz.py record                                                       # demo uçuşları (fl_* dahil)
python command_viz.py --model models_flight/flight_final.zip --start ground --wind-kt 15 --turb light   # canlı 3D
```

## 8. Doğal komut zarfı, komut yönlendirici, held-out takım, hover hassasiyeti — `flight_v2` (2026-10-01, branch `natural-limits`)

Kullanıcı isteği: Δ komutlarda yapay sınırlar kalksın, yalnızca doğal sınırlar kalsın; arayüz kolaylığı; ayrı test
takımı; hover hassasiyeti; JSBSim'in kullanmadığımız özellikleri. Ölçümler ve öneriler: `sim_oneriler_2026-10-01.md`.

**Özet.** Yeni sonuç modeli `models_flight/flight_v2.zip` (gözlem 43: + hava yoğunluğu). flight_final'e göre held-out
takımda: güvenli 35/35 (aynı), tüm görevler 26 → 28/35, sıcak gün grubu görev 19 → 30/31, 56 psi üstü 50 → 30 s;
yerinde dönüşte en büyük konum kayması 32 → 6 ft, pirouette 19 → 10 ft; ADS-33 hovering turn 4 yetersiz → 3 yeterli.
Bedeli: ileri uçuşta hız değişimleri biraz yavaş (seçim takımında görevlerin "istenen"i 102 → 98/105, hepsi yine en az
"yeterli"; held-out seviyelerde F3 / F5 / F7 / F8 1–2 episode düşük) ve ADS-33 inişinde temas noktası 3–3.6 ft yanda.
Genel görevlerde zamanlama öncelikliyse ara model `models_flight/flight_f10.zip` (F10 3 M: hassasiyeti flight_final gibi).

### 8.1. Doğal zarf

| eksen | eski (eğitim örneklemesi) | doğal komut zarfı | kaynak |
|---|---|---|---|
| hız (hava hızı) | 30–110 kt (ileri uçuş) | 0 = hover; 10–130 kt | AH-1S en yüksek düz uçuş ~128 kt (Vertipedia), YAH-1S 10,000 lbs ~130 KTAS (DTIC ADA025476) |
| irtifa (CG AGL) | ileri uçuş 100–1000, hover 12–1000 ft | ileri uçuş 50–1500, hover 12–1500 ft | kullanıcı (0–1500 ft) |
| heading | — | serbest; tek komutta ±360° | |
| güvenlik | hava hızı 130 kt | Vne 170 kt (TOW ya da > 9500 lbs) | aircav.com |

Komutlar artık **asla ters çevrilmiyor**: env'in eski "zarf dışına düşerse Δ'yı ters çevir" kuralı yalnızca müfredatın
örneklediği görevlerde (`_flip` bayrağı; F1–F9 birebir — 16 örneklenen episode'da obs / ödül / sonuç aynı, `power_aware_climb`
kapalıyken), canlı / arayüz komutu doğal sınıra kırpılıyor ve pencereye `clipped` yazılıyor. Eskiden: 40 kt'ta −20 kt → 60 kt,
100 kt'ta +30 kt → 70 kt, 900 ft'te +200 ft → 702 ft, 250 ft'te −200 ft → 451 ft, hover 30 ft'te bob −25 ft → 55 ft.

**Eğitimsiz genelleme** (flight_final, yalnızca zarf sabitleri açılarak, sakin hava): 120 kt ✓, 130 kt ✗ (128 kt'a çıktı,
bantta oturmadı), 40 → 25 / 15 / 10 kt ✓, 20 kt'ta +90° ✓, hover → 20 kt ✓, 1000 → 1500 ft hover ✓, 1500 ft'te hover /
ileri uçuş ✓, 100 → 1500 ft tırmanış ✓, 300 → 50 ft ✓, 120 kt'ta +90° ✓; 1500 ft'ten dikey iniş son sınırı kaçırdı (yüksekten
iniş için yönlendirici önce ileri uçuşta alçaltmalı — açık).

### 8.2. Komut yönlendirici — `flight_commands.py`

Arayüzün tek giriş noktası: `route_command(env, speed_kt | dspeed_kt, heading_deg | dheading_deg, alt_ft | dalt_ft)`,
`route_action(env, "takeoff" | "land" | "stop" | "pirouette")`, `apply(env, sonuç)`. Rejim: yerde (≥ 3 kızak noktası,
kızak < 1 ft) / ileri uçuş (ileri uçuş penceresi ya da hava hızı ≥ 25 kt) / hover.

| rejim | hız | heading | irtifa | iniş / hover eylemleri |
|---|---|---|---|---|
| yerde | reddedilir (önce kalkış) | — | kalkış | — |
| hover | ≥ 10 kt → (önce yerinde dönüş) + hızlanma | yerinde dönüş | climb_to | land / hold / pirouette |
| ileri uçuş | < 10 kt → duruş (+ hover komutları); 10–130 kt → cruise u | cruise Δψ (mutlak heading → en kısa yön) | cruise h | önce duruş, sonra görev |

Eskiden rejime uymayan komut episode'u bitiriyordu (80 kt'ta hover dönüşü / hold / iniş → 0.1 s'de `speed_limit`) ya da
istenmeyen hızlanma yapıyordu (hover'da cruise Δψ → 30 kt). `command_viz.py` canlı modu bütün uçuş görevlerini
yönlendiriciden geçiriyor; sayfada "Komut" satırı (sayı = mutlak, +/− ile başlarsa Δ), pirouette düğmesi, yeniden
başlatmada sıcaklık farkı; yönlendiricinin açıklaması (kırpma dahil) görev ipucunda. Kontrol: `docs/flight/check_natural_limits.py`.

### 8.3. Env / fizik değişiklikleri

- `power_aware_climb` (varsayılan açık): ileri uçuş / hızlanma pencerelerinde istenen tırmanış hızı güç payıyla sınırlı —
  0.8·(56 − psi_düz(u, W))/0.62 ft/s (trim tablosunun düz uçuş torku; 9700 lbs'de 0 kt 3.6, 20 kt 13 ft/s); hızlanmanın süre
  hedefinde tırmanış ETL'den (35 kt) sonra. 2026-09-30 bulgusu: 9700 lbs'deki 56 psi aşımlarının hepsi hızlanma
  penceresinde, 0–32 kt'ta, ödülün vs_des'i 12.5 ft/s tavandayken.
- `pirouette` görevi (ADS-33): hedef noktası çember üzerinde ilerler, hedef heading merkeze; hedefin hızı gözlemde
  (38–39) ve yönlendirmede, heading'in dönüş hızı yaw yönlendirmesine ileri besleme; tur sırasında hareketli hedeften
  uzaklık / heading farkı sınırı (`pir_lim`), tur bitince başlangıç noktasında hover bandı. **Açısal hız yamuk profilli**
  (`ramp_s`, varsayılan 4 s; tur t_c + ramp_s): ilk sürümde hedef 14 ft/s'ye anında çıkıp turun sonunda anında
  duruyordu — fl11 3 M sürekli rejimde hedefin 6–8 ft yakınında, ama başta ~13 ft geride kalıp sonda ~8 ft aşıyordu
  (en büyük uzaklık 18.6 ft). Rampayla aynı model, eğitimsiz: 10.4 ft.
- Hover hassasiyeti: seviyenin `hover_precision`'ı kuplaj sınırlarını ve hover bandının konum toleransını ölçekler (F10 /
  F11: 0.5 → yerinde dönüşte konum ≤ 10 ft, hover bandı 3–6 ft, pirouette'te hareketli hedefe ≤ 10 ft).
- Yerinde dönüşte yaw hızı cezası (`pen_turn_rate`; fl10b'de 1.0, F11 koşularında 2.0): |r| > 1.5·1.25·15 = 28 °/s
  üstü. Neden: ajan 180°'yi ~5 s'de 40–44 °/s ile dönüp konumu 15–25 ft kaydırıyordu; dönüş ödülü (indirimli ilerleme +
  heading çekirdeği) hızlı dönmeyi kârlı kılıyordu; yaw hızı yönlendirmesi (18.75 °/s) zayıf kalıyordu. ADS-33 istenen
  180°/10 s. F11 (hassasiyet okulu): hover'da 2–4 görev (dönüş %40, pirouette %30, kayma, bob), %50 F10 tekrarı.
- Tırmanışta tork geri beslemesi (`torque_feedback_climb`) ve yoğunluk düzeltmesi (`torque_density_climb`), F11
  koşularında açık: istenen tırmanış hızı, filtreli tork 54.5 psi'yi geçince düşürülür
  (vs_des ≤ vs + (54.5 − psi_f)/0.62); güç payı hesabında hover torku hava yoğunluğuyla artar
  (+15·(0.935 − σ)/0.935 psi). Neden: fl10 3 M'de 1000 → 1500 ft hover tırmanışında 17.9 s kesintisiz 56 psi üstü.
- Hava sıcaklığı: `physics_ext` `atmo` bölümü / `delta_T_C` seçeneği → JSBSim `atmosphere/delta-T`; `density_obs` (obs 43:
  (ρ/ρ0 − 0.9)/0.05). Seçenekler (eğitimde kapalı): `wind_shear` (MIL-F-8785C log profili), `turb_from_wind` (türbülans
  W20'si ortalama rüzgârdan).
- Gust erteleme: JSBSim 1−cos gust'ı sürerken gelen gust öncekinin bitişine ertelenir (JSBSim yönü güncellemiyor, genliği
  basamakla değiştiriyor, süreyi yeniden başlatmıyordu).
- Reset hover PID'inin 2. evresi (yalnızca 1. evre başarısızsa): güçlü yan rüzgârda (15–25 kt) hover artık kuruluyor.
- Canlı: yerde başlayıp görev yokken iniş bayrağı 1 (ajan kızaklar ~0.7 ft'te süzülüyordu).
- `max_airspeed_kt` = Vne 170 kt.
- `torque_feedback_climb`, `pen_turn_rate` yalnızca ödülü değiştirir. `torque_density_climb` görevlerin **süre
  hedeflerini de** değiştirir (sıcak günde / yüksekte tırmanış hedefi daha yavaş → son sınır daha geç: 9700 lbs, +30 °C'de
  kalkışın son sınırı 35 → 67 s). Bu yüzden modeller arası karşılaştırmada hepsi `torque_density_climb` **kapalı**
  ölçütlerle değerlendirildi (`--env '{"torque_density_climb": false}'`; aşağıdaki tablolar).

### 8.4. Eğitim koşuları (hepsi ince ayar modu: lr 1e-4, KL 0.02; 2 env; `--no-promote`; tek seed)

| koşu | başlangıç | seviye / env ayarları | adım (süre) | ne oldu |
|---|---|---|---|---|
| fl10 | flight_final; gözlem 42 → 43 (`widen_takeoff_obs.py --task flight`, yoğunluk sütunu sıfır) | F10 | 3.0 M (79 dk) | seçim 18/21 · 104/105; held-out sıcak gün grubu görev 19/31 → 27/31; yerinde dönüş kayması 21 → 16 ft (ADS-33 dışı) → **`flight_f10.zip`** |
| fl10b | fl10 3 M | F10, `pen_turn_rate` 1.0 | 1.0 M (28 dk) | dönüş kayması iyileşmedi (16 → 23 ft) |
| fl11 | fl10b 1 M | F11; `pen_turn_rate` 2.0, `torque_feedback_climb`, `torque_density_climb` (bundan sonra hep açık) | 3.5 M (92 dk) | yerinde dönüş 0/8 → 7/8 (kayma 6.4 ft); **ama seçim 8/21 · 89/105** — iniş 6/13, ileri uçuşta irtifa 1/8, hızlanmada 56 psi üstü 2 kat |
| fl12 | fl11 3.5 M | F11 + pirouette rampası | 1.5 M (39 dk) | pirouette eğitim başarısı %3–6 → %85; dönüş kayması 4.3 ft (1 M); **F6 inişi 0/8** |
| fl13 | fl12 1.5 M | F12 | 1.0 M (28 dk) | seçim 5/21 (0.5 M; iniş 0/13) → durduruldu |
| fl14 | fl13 1 M | F13 | 2.0 M (52 dk) | iniş geri geldi (F6a / F6 8/8, 0.5 M'den itibaren); seçim 11 / 15 / 13 / 14 (21'de; 0.5 / 1 / 1.5 / 2 M), 2 M'de bir senaryoda düşme → **`flight_v2.zip` = fl14 1.5 M** |
| fl15 | fl14 2 M | F13 | 1.5 M (43 dk) | 56 psi üstü 17–25 s'ye indi, ama ileri uçuş hız / irtifa değişimleri yavaşladı (seçim istenen 88–95/105) ve yerinde dönüş 3/8'e düştü (1.5 M) |

Soy: flight_final (11.8 M) + fl10 3.0 + fl10b 1.0 + fl11 3.5 + fl12 1.5 + fl13 1.0 + fl14 1.5 = **~23 M adım, ~10 saat
CPU**. Seçim yalnızca seçim takımı + hassasiyet ölçümü + iniş kapısı (F6a / F6 8 episode) + seed 70000'li F8 / F10 ile
yapıldı; held-out takım ve seed 500000'li seviyeler son modelde bir kez koşuldu (istisna: fl10 3 M'in held-out sonucu
tork çalışmasına yön verdi).

**İniş bozulmasının nedeni (ölçüldü):** F11 soyunda ajan yere oturduktan sonra collective action'ını −0.85'te tutuyordu
(collective 0.21, kızaklarda ağırlık %61–64 < %70 → iniş bandı hiç sağlanmıyor); fl10'da −1.0 (collective 0.02, %96).
Ödül bunu açıkça cezalandırıyor (yerde adım başına ~1.5 daha az ödül) — öğrenilmiş bir kayma, ödül çelişkisi değil.
Eğitim istatistiği bunu gizledi: stokastik policy'nin gürültüsü collective'i zaman zaman aşağı itiyordu (F11'de
"iniş %78–89"), deterministik F6'da 0/8. **Ders:** her anlık görüntüde bütün becerilerin deterministik kontrolü (iniş
dahil). Çare: F13 (F6a oturma okulu + F6 tekrarı) → 0.5 M adımda F6a / F6 8/8.

**Tork geri beslemesi yere yakın riskli (ölçüldü, fl14 2 M):** 9700 lbs, 25 kt rüzgâr + orta türbülans, alçak hover'dan
hızlanma: ilk 2 s'de tork 59–60 psi → geri besleme istenen tırmanışı 1.5 ft/s'ye kısıyor → ajan torku düşürürken
−8…−9 ft/s ile 39 ft'ten 14 ft'e indi → `low_altitude`. Aynı senaryo flight_final / flight_f10 / flight_v2'de güvenli.
Öneri (uygulanmadı — yeniden eğitim gerektirir): geri beslemeyi yere yakın (< 50 ft) ve hızlanmanın ETL öncesinde kapatmak.

### 8.5. Hassasiyet — `docs/flight/precision_probe.py` (`precision_2026-10-01.json`)

Sakin hava, kızaklar 10 ft; yerinde dönüş +180°, −180°, −270°, +360° ve pirouette (100 ft, 45 s + 4 s rampa) iki yöne;
8800 / 9700 lbs → 8 dönüş + 4 pirouette; F10 toleransları (hassasiyet ×0.5: dönüşte konum ≤ 10 ft, pirouette'te
hareketli hedefe ≤ 10 ft). Kayma = görev sırasında hedef noktadan en büyük yatay uzaklık.

| model | yerinde dönüş başarı | kayma ort. (en büyük) | en büyük yaw hızı ort. | pirouette başarı | hareketli hedefe uzaklık ort. (en büyük) |
|---|---|---|---|---|---|
| flight_final | 0/8 | 21.3 ft (30.2) | 36 °/s | 0/4 | 15.1 ft (19.6) |
| flight_f10 (fl10 3 M) | 0/8 | 15.9 ft (25.7) | 34 °/s | 1/4 | 12.9 ft (15.8) |
| fl11 3.5 M | 7/8 | 6.4 ft (11.7) | 32 °/s | 2/4 | 10.2 ft (11.9) |
| fl12 1 M | 8/8 | 4.3 ft (6.6) | 29 °/s | 2/4 | 10.0 ft (11.9) |
| **flight_v2 (fl14 1.5 M)** | **8/8** | **5.8 ft (7.6)** | 27 °/s | **4/4** | **8.5 ft (9.1)** |
| fl15 1.5 M | 3/8 | 10.2 ft (11.9) | 24 °/s | 3/4 | 9.4 ft (10.3) |

(pirouette'i rampasız görmüş modeller için de ölçüm rampalı — eğitimsiz.)

### 8.6. Sonuçlar — flight_final / flight_f10 / flight_v2

Karşılaştırma ölçütü hepsinde aynı (`torque_density_climb` kapalı); deterministik. Held-out takım (`--suite test`, 35
senaryo, 4 grup: zincir / öğe / doğal zarf / sıcak gün — F1–F10'un örneklemediği kombinasyonlar, ayrı seed'ler) ve
held-out seviye istatistikleri (seed 500000+, seviye başına 10 episode) son model seçildikten sonra koşuldu (seçimde
kullanılmadı; aynı model iki ölçütle koşuldu, tablolar `torque_density_climb` kapalı olanı).

**Held-out takım + seviyeler** (`eval_test_final.json`, `eval_test_f10.json`, `eval_test_v2.json`):

| ölçüt | flight_final | flight_f10 | **flight_v2** |
|---|---|---|---|
| güvenli (düşme / sınır aşımı yok) | 35/35 | 35/35 | **35/35** |
| test/zincir (tüm görevler · görev) | 7/8 · 51/52 | 7/8 · 51/52 | 5/8 · 43/52 |
| test/öğe | 8/9 · 17/19 | 8/9 · 22/22 | 7/9 · 21/22 |
| test/doğal (doğal zarfın uçları) | 9/11 · 21/23 | 10/11 · 22/23 | 10/11 · 22/23 |
| test/sıcak (−10…+30 °C) | 2/7 · 19/31 | 3/7 · 27/31 | **6/7 · 30/31** |
| toplam: tüm görevler · görev (istenen) · görev (yeterli+) | 26/35 · 108/125 · 120/125 | 28/35 · 122/128 · 128/128 | 28/35 · 116/128 · 126/128 |
| 56 psi üstü toplam (en uzun kesintisiz; tepe) | 50 s (9.6 s; 60.9) | 76 s (17.9 s; 60.6) | **30 s (6.7 s; 61.5)** |
| en büyük kayma: yerinde dönüş / pirouette | 32.3 / 19.3 ft | 19.6 / 15.4 ft | **5.8 / 9.9 ft** |
| seviyeler F3 / F5 / F6 / F7 / F8 / F10 (episode, %) | 100 / 100 / 100 / 100 / 90 / 50 | 100 / 100 / 100 / 100 / 90 / 60 (1 `low_altitude`) | 80 / 90 / 100 / 80 / 80 / 50 |

(flight_final / flight_f10'un iki pirouette senaryosu rampalı görevle yeniden koşuldu; `n` farkı: flight_final'de
kesilen pencereler sayılmıyor.)

**Seçim takımı** (21 senaryo; `eval_final.json`, `eval_secim_f10.json`, `eval_secim_fl11.json`, `eval_secim_v2.json`):

| ölçüt | flight_final | flight_f10 | fl11 3.5 M | **flight_v2** |
|---|---|---|---|---|
| güvenli | 21/21 | 21/21 | 21/21 | 21/21 |
| tüm görevler · görev (istenen) · görev (yeterli+) | 17/21 · 102/105 · 105/105 | 18/21 · 104/105 · 105/105 | 8/21 · 89/105 · 97/105 | 13/21 · 98/105 · 105/105 |
| kısa zincir / uzun zincir (tüm görevler) | 6/6 · 4/6 | 6/6 · 5/6 | 2/6 · 0/6 | 6/6 · 0/6 |
| iniş | 13/13 | 13/13 | 6/13 | 13/13 |
| ileri uçuşta hız / irtifa değişimi (istenen · yeterli+) | 6/8 · 8/8 / 7/8 · 8/8 | 8/8 · 8/8 / 7/8 · 8/8 | 6/8 · 8/8 / 1/8 · 7/8 | 4/8 · 8/8 / 6/8 · 8/8 |
| 56 psi üstü toplam (en uzun) | 43 s (8.5) | 40 s (5.7) | 76 s (7.8) | 46 s (6.7) |

**Hassasiyet** (8.5): yerinde dönüş 0/8 → **8/8** (kayma 21.3 → **5.8 ft**), pirouette 0/4 → **4/4** (15.1 → **8.5 ft**).

**ADS-33 karnesi** (`evaluate_ads33.py --heavy`, sakin hava, 8800 / 9700 lbs; pirouette env görevi + rampa, 38 + 4 s;
`docs/ads33/karne_flight_final_gorev.json`, `karne_flight_f10.json`, `karne_flight_v2.json`):

| MTE (×4 / ×2) | flight_final | flight_f10 | **flight_v2** |
|---|---|---|---|
| hover | 4 yeterli | 1 istenen, 3 yeterli | **2 istenen, 2 yeterli** |
| hovering turn (180°) | 4 yetersiz (konum 9–22 ft) | 4 yetersiz (6–16 ft) | **3 yeterli** (konum 4.1–4.9 ft), 1 yetersiz (irtifa 5.1 ft) |
| vertical (±25 ft) | 1 istenen, 1 yeterli | 2 yeterli | 2 yeterli |
| pirouette | 4 yeterli | 4 yeterli | 4 yeterli |
| landing | 1 istenen, 1 yeterli | 2 yeterli | 2 yetersiz (temas yanal 3.0–3.6 ft) |
| toplam istenen / yeterli / yetersiz | 2 / 10 / 4 | 1 / 11 / 4 | 2 / 11 / 3 |

**Doğal zarf + yönlendirici** (`check_natural_limits.py`, flight_v2): 40 kt'ta −20 kt → 20 kt; 100 kt'ta +45 kt → 130 kt
(kırpma); 900 ft'te +800 ft → 1500 ft (kırpma); 250 ft'te −300 ft → 50 ft (kırpma); hover 30 ft'te bob −25 ft → 12 ft;
yönlendirici: 80 kt'ta "in" → duruş + iniş (kızak 4), heading 270 → 271°, hız 0 + 100 ft → hover 101 ft, hover'da
Δψ +90 / Δh +200 → 89° / 302 ft, hover'da 60 kt + heading 90 → 59 kt / 91°, pirouette → tur tamam; hiçbiri episode'u
bitirmedi.

### 8.7. Tork sınırı öğrenmeyi bozuyor mu? (kullanıcı sorusu)

**Hayır — sınır kalmalı.** Ölçümler:
- Güç tavanı (56 psi, `fcs/throttle-max-norm`) ve tork cezası açıkken bütün görevler öğrenildi; flight_v2 seçim
  takımında bütün görevleri en az "yeterli" yapıyor (105/105), held-out takımda güvenli 35/35.
- Aşım küçük ve kısa: toplam sürenin ~%0.5–1.4'ü; türbülanssız tepe 57–60 psi (%102–107); 60 psi üstü yalnızca orta
  türbülansta anlık. 56 psi üstü geçiş, devirin düşmesiyle (rotor kinetik enerjisi; en düşük 306–311 rpm) — güç tavanı
  zaten motor gücünü kesiyor.
- Aşımın yeri (`evaluate_flight.py` pencere başına yeni alanlar `torque_max`, `t_over56_s`): 9500–9700 lbs'de hover →
  ileri uçuş hızlanmasının ilk 5–8 s'si (0–25 kt, ETL öncesi, en çok güç isteyen bölge) ve 1000 → 1500 ft hover
  tırmanışı. Hızlanmayı daha çabuk yapan model ETL'ye daha erken geçip daha az aşıyor (fl10: 4 s'de 15 kt, aşım 4.5 s;
  fl14 0.5 M: 4 s'de 11 kt, 9.6 s).
- Ödül yönlendirmesi işe yarıyor: 1500 ft hover tırmanışında kesintisiz aşım 17.9 s (flight_f10) → 1.1 s (flight_v2);
  held-out takımda 56 psi üstü toplam 50 s (flight_final) → 30 s (flight_v2; en uzun kesintisiz 9.6 → 6.7 s).
- Neden değiştirmeyelim: 56 psi gerçek AH-1S sınırı (%100; %88'e kadar sürekli) ve mentor şartı; gevşetmek modeli
  gerçek dışı yapar, ajan zaten öğreniyor. Bir sonraki adım gerekirse ödülde: hızlanmanın ETL öncesinde "güç farkında
  ivme" (tırmanma değil önce hızlan — ağır helikopterin gerçek kalkış tekniği) ve geri beslemeyi yere yakın kapatmak
  (8.4).

### 8.8. Bilinen sınırlar / açık konular

- **flight_v2'nin bedeli:** ileri uçuşta hız değişimleri biraz yavaş — seçim takımında hız değişimi 4/8 "istenen"
  (8/8 "yeterli"), uzun zincirlerin hiçbiri "tüm görevler istenen" değil; held-out seviyelerde F3 / F5 / F7 / F8
  episode başarısı 1–2 / 10 düşük (başarısızlıklar ileri uçuş görevlerinde, güvenlik değil). Türbülanslı kısa held-out
  zincirde ileri uçuşta dönüş 2 kez başarısız. ADS-33 inişinde temas noktası yanal 3.0–3.6 ft (yetersiz; flight_final
  istenen / yeterli). Genel görevlerde zamanlama öncelikliyse `flight_f10.zip` (hassasiyeti flight_final gibi).
- Hassasiyet ile genel beceriler arasında denge kırılgan: aynı soydaki anlık görüntüler arasında yerinde dönüş 3/8–8/8,
  seçim istenen 88–99/105 oynuyor (tek seed, deterministik ölçümler 8–21 senaryo).
- Pirouette ADS-33'te 4/4 "yeterli", "istenen" değil (radyal 11–11.5 ft > 10 ft ya da irtifa > 3 ft).
- `torque_density_climb` görev süre hedeflerini değiştiriyor (8.3) — karşılaştırmalarda kapalı ölçüt.
- Held-out takım bir kez (fl10 3 M'de) tork çalışmasına yön verdi; sonuç modeli seçimi ondan bağımsız.

### 8.9. Çalıştırma

```bash
python physics_ext.py && python helicopter_env_flight.py                  # birim / duman testleri
python docs/flight/check_natural_limits.py                                # doğal zarf + yönlendirici (flight_v2)
python docs/flight/precision_probe.py models_flight/flight_v2.zip models_flight/flight_final.zip
# değerlendirme — karşılaştırma ölçütü: torque_density_climb kapalı
python evaluate_flight.py --model models_flight/flight_v2.zip --suite secim --env '{"torque_density_climb": false}'
python evaluate_flight.py --model models_flight/flight_v2.zip --suite test --env '{"torque_density_climb": false}' \
    --json docs/flight/eval_test_v2.json
python evaluate_flight.py --model models_flight/flight_v2.zip --levels F3,F5,F6,F7,F8,F10 --episodes 10 --no-scenarios \
    --seed0 500000 --env '{"torque_density_climb": false}'
python evaluate_ads33.py --model models_flight/flight_v2.zip --heavy --json docs/ads33/karne_flight_v2.json
python docs/flight/compare_eval.py flight_final=docs/flight/eval_test_final.json flight_v2=docs/flight/eval_test_v2.json
# eğitim soyu (8.4; her koşu bir öncekinin modelinden, ince ayar)
python widen_takeoff_obs.py --task flight --model models_flight/flight_final.zip --out runs/fl10/init.zip \
    --env '{"density_obs": true}'
python train_command_curriculum.py --task flight --out runs/fl10 --init-model runs/fl10/init.zip \
    --level F10 --no-promote --total-steps 3000000 --n-envs 2 \
    --n-steps 4096 --batch-size 512 --net 256,256 --eval-freq 1000000 --eval-episodes 10 --eval-levels F10,F8,F6 \
    --snapshot-freq 1000000 --fine-from F6a
# fl10b: --level F10 --env-overrides '{"pen_turn_rate": 1.0}' (1 M) · fl11: --level F11 --env-overrides
# '{"pen_turn_rate": 2.0, "torque_feedback_climb": true, "torque_density_climb": true}' (3.5 M) · fl12: F11 (1.5 M) ·
# fl13: F12 (1 M) · fl14: F13 (1.5 M → flight_v2) — aynı bayraklarla, --init-model bir öncekinin son modeli
python command_viz.py --start ground --wind-kt 15 --turb light            # canlı 3D (varsayılan model flight_v2)
```

## 9. Ödül dengesi, rotor dt, 120 kt zarfı, iniş — `flight_v3` (2026-10-02, branch `reward-v3`; README 35)

Ölçüt (bütün tablolar): deterministik policy, `torque_density_climb` ve `next_at_deadline` kapalı (eski modellerle aynı
zamanlama), rotor dt modeli modelin kendi ayarı (flight_v2 "legacy", flight_v3 "sim"). `evaluate_flight.py`'nin düzeltilmiş
`episode_ok`'u (kesilen oto-tutma pencereleri sayılmaz).

### 9.1. Yeni ölçümler (bağımsız değerlendirme, flight_v2)

- **Osilasyon** (hover 100 ft, sakin, 30 s): konum RMS 0.94 ft (maks 1.1), irtifa maks 0.7 ft, heading 1.0°, roll / pitch
  std 0.5°; kumanda baskın frekansı 0.33 Hz, 1 Hz üstü güç %1–4 (15 kt yan rüzgâr + hafif türbülansta pedal %11). Yüksek
  frekanslı titreme yok.
- **Ödül yapısı:** getiri 412–1266, görev bonusu %1.5–1.9; track + guide pozitif getirinin ~%99'u.
- **Zamanlama:** ileri uçuş görevlerinde banda giriş / son sınır medyan 0.75–0.86 (flight_final 0.52–0.74).
- **Sağlamlık probe'u** (kısa zincir sakin / rüzgârlı ağır, 300 ft'ten iniş, 80 kt dönüşler): gözlem gürültüsü σ 0.02–0.05
  (≈0.5 ft, 1.4°, 1 ft/s) sorunsuz, σ 0.10'da bozulma; aksiyon gecikmesi 75 ms sorunsuz, 150 ms'de kalkış / duruş / hover
  tutma bozuluyor, 300 ms'de kalkışta düşme; trim bias'ları (collective +0.03, pedal +0.08, lateral −0.08) sorunsuz.
- **Hızlanma probe'u** (`scratchpad/probe_accel.py` mantığı; sakin, 9300 lbs): 20 → 120 kt'ta ilk 10 s'de 8.2 ft kayıp,
  36 kt hız artışı, %90 hız 62 s; iniş 300 ft'ten: 10–20 ft'te 6.1–6.6 ft/s, son 5 ft 4.5–5.1 ft/s, temas −3.3 ft/s.
- **İniş stres taraması** (24 iniş: 150 / 300 / 50 ft × sakin / 15 kt yan + hafif / 25 kt + orta + gust / 20 kt arkadan +
  hafif + gust × 8800–9700 lbs, 2 seed): flight_v2 19/24 başarılı, **4 güvensiz** (25 kt + orta türbülans + gust'ta
  150–300 ft'ten: 3 kuyruk çarpması, 1 devrilme — burun 3° → 13° kaldırılıp yer hızı frenleniyor). Bu zayıflık eskiden
  beri vardı (held-out T2_t_turb inişi flight_v2'de −4.54 ft/s).
- **Yeni seed'lerle seviyeler** (seed 900000+, 20 episode): flight_v2 F3 %75, F5 %90, F6 %100, F7 %85, F8 %75, F10 %65.

### 9.2. İnce ayar koşuları

`fl_v3`: flight_v2'den, F13, 3 M adım (70 dk, 3 env), lr 1e-4 + KL 0.02, `rotor_dt_mode: sim`, yeni ödül varsayılanları
(next_at_deadline, w_task_early 50, accel_h_allow_ft 30, pen_land_final 3). `fl_v3b`: fl_v3'ün 1.5 M modelinden,
`pen_land_att` 2 ile 1.5 M adım.

| model | seçim: tüm · görev | 56 psi üstü (en uzun) | banda giriş / son sınır (cruise, medyan) | temas medyan / en sert | son 5 ft alçalma (300 ft / 50 ft rüzgâr) | 60 → 100 kt banda giriş |
|---|---|---|---|---|---|---|
| flight_v2 | 15/21 · 98/105 | 46 s (6.7) | 0.75–0.86 | −2.6 / −3.6 | 4.5 / 5.1 ft/s | 35.9 s / 38.7 |
| fl_v3 0.5 M | 15/21 · 93/105 | 50 s (6.9) | 0.73 | −2.4 / −3.3 | 4.9 / 4.4 | 37.4 |
| fl_v3 1.0 M | 13/21 · 94/105 | 61 s (7.5) | 0.75 | −2.2 / −3.2 | 4.6 / 4.2 | 29.4 |
| fl_v3 1.5 M | 17/21 · 100/105 | 61 s (7.0) | 0.76 | −2.4 / −3.8 | 3.7 / 3.9 | 29.5 |
| fl_v3 2.0 M | 18/21 · 102/105 | 51 s (7.6) | 0.76 | −2.2 / −2.8 | 3.6 / 3.8 | 29.5 |
| fl_v3 2.5 M | 17/21 · 101/105 | 36 s (5.0) | 0.73 | −2.0 / −3.3 | 3.0 / 3.5 | 29.5 |
| fl_v3 3.0 M | 14/21 · 97/105 | 51 s (6.5) | 0.74 | −1.7 / −2.7 | 2.9 / 3.7 | 29.5 |

Held-out (35) ve iniş stresi: fl_v3 1.5 M güvenli 35/35, tüm 29/35, görev 117/128, sıcak gün 7/7 (flight_v2 29/35 ·
116/128 · 6/7); 2.0 M 34/35 (T2_t_turb inişinde kuyruk çarpması), 31/35 · 120/128; 2.5 M 34/35 (aynı senaryo), yerinde
dönüş hassasiyeti bozuldu (7/8, 7.1 ft). İniş stresinde adaylar 2–4 güvensiz (flight_v2 4) → gust'lı iniş ortak
zayıflık; `pen_land_att` bunun için eklendi (fl_v3b). Yeni seed'lerle seviyeler: 1.5 M F3 %80, F5 %95, F6 %100, F7 %85,
F8 %80, F10 %80; 2.0 M %90 / 95 / 100 / 90 / 85 / 80.

Hızlanma probe'u (20 → 120 kt): ilk 10 s irtifa kaybı 8.2 (v2) → 8.6 / 9.1 / 7.4 / 6.5 ft (1.5 / 2.0 / 2.5 / 3.0 M), hız
kazancı 35–36 kt aynı; hızlanma penceresi süre hedefi (2.5 ft/s² rampa) değişmediği için %90 hız süresi 62 s sabit —
"bir an önce hızlanma" için rampanın (`cruise_accel_fps2`) ve hızlanma süre hedefinin de büyütülmesi gerekir (yapılmadı;
F10 müfredatı 2.5 ft/s² ile).

### 9.3. Sonuç modeli

**Sonuç modeli `models_flight/flight_v3.zip` = fl_v3 1.5 M** (flight_v2 + 1.5 M adım ince ayar; sha256
`models_sha256.txt`'de; `rotor_dt_mode: sim` zip'inde). Seçim gerekçesi: held-out takımda düşme / sınır aşımı olmayan
tek aday (2.0 M ve 2.5 M'de T2_t_turb inişinde kuyruk çarpması; flight_v2 orada −4.54 ft/s ile zor iniyordu), seçim
takımı ve yeni seed'li seviyelerde flight_v2'den iyi, iniş daha yumuşak, hassasiyet korunmuş. 2.0 M daha yüksek görev
başarısına rağmen (held-out 31/35 · 120/128, seviyeler %90 / 95 / 100 / 90 / 85 / 80) o kuyruk çarpması yüzünden
seçilmedi; `runs/fl_v3/models/snap_02000k.zip` yeniden üretilebilir. İkinci tur (`fl_v3b`, `pen_land_att` ile 1.5 M)
gust'lı iniş kazalarını azaltmadı ve genel görevleri geriletti (0.5 M 14/21 · 95/105, 1.0 M 12/21 · 93/105, 1.5 M 15/21 · 93/105 — yeterli+ 97/105; iniş
stresi 2 / 4 / 3 güvensiz, 1.5 M'de 14/24 başarılı) → olumsuz sonuç, model alınmadı; ödül terimi varsayılan olarak açık kalıyor (ileride sıfırdan eğitimde
denenmek üzere), eski modellerin değerlendirmesine etkisi yok.

| ölçüt | flight_v2 | **flight_v3** |
|---|---|---|
| held-out (35): güvenli · tüm · görev (istenen / yeterli+) | 35/35 · 29/35 · 116 / 126 (128) | **35/35 · 29/35 · 117 / 127 (128)** |
| held-out: sıcak gün (tüm · görev) | 6/7 · 30/31 | **7/7 · 31/31** |
| held-out 56 psi üstü toplam (en uzun) | 31.6 s (6.7) | 45.0 s (7.2) |
| seçim (21): tüm · görev | 15/21 · 98/105 | **17/21 · 100/105** |
| seviyeler F3 / F5 / F6 / F7 / F8 / F10 (seed 900000+, 20 ep.) | 75 / 90 / 100 / 85 / 75 / 65 | **80 / 95 / 100 / 85 / 80 / 80** |
| temas hızı medyan / en sert (seçim + held-out) | −2.6 / −3.9 ft/s | **−2.5 / −3.8** (son 5 ft 3.7–3.9 ft/s ↔ 4.5–5.1) |
| ADS-33 (16 MTE): istenen / yeterli / yetersiz | 2 / 11 / 3 | 1 / 13 / 2 (iniş yanal 1.8 / 2.8 ft ↔ 3.0 / 3.6) |
| hassasiyet: yerinde dönüş 8 / pirouette 4 | 8/8, 5.8 ft / 4/4, 8.5 ft | 8/8, 5.3 ft / 4/4, 8.3 ft |
| iniş stres taraması (24) | 19/24, 4 güvensiz | 18/24, 3 güvensiz |
| 60 → 100 kt banda giriş / son sınır | 35.9 / 38.7 s | **29.5 / 38.6 s** |

Açık kalanlar: (1) 25 kt + orta türbülans + gust'ta 150–300 ft'ten iniş iki modelde de 2–4/24 kaza (kuyruk çarpması /
devrilme; burun yukarı frenleme) — `pen_land_att` ince ayarda çözmedi, sıfırdan eğitimde ya da gust'lı iniş okulu
seviyesiyle denenmeli; (2) hızlanma hızı rampayla (2.5 ft/s²) sınırlı — daha çevik hızlanma için `cruise_accel_fps2`
ve süre hedefi büyütülüp yeniden eğitilmeli; (3) tek seed, tek soy — farklar 10–35 senaryoluk takımlarda ±1–2 senaryo
gürültüsünün sınırında (F10 %65 → %80 ve iniş yumuşaması bunun üstünde, seçim takımındaki +2/21 değil).

Dosyalar: `docs/flight/eval_test_v3.json`, `eval_secim_v3.json`, `eval_levels_v3.json`, `precision_v3.json`,
`docs/ads33/karne_flight_v3.json`, `eval_test_v2_rev.json` (flight_v2, düzeltilmiş ölçüt), `accel_v2/v3.json`,
`landing_stress_v2_v3.json`; probe'lar `probe_accel.py`, `probe_landing.py`; koşular `runs/fl_v3`, `runs/fl_v3b`.

```bash
python -m pytest -q tests
python evaluate_flight.py --model models_flight/flight_v3.zip --suite test --env '{"torque_density_climb": false, "next_at_deadline": false}'
python docs/flight/probe_accel.py models_flight/flight_v3.zip /tmp/accel_v3.json
python docs/flight/probe_landing.py /tmp/landing.json models_flight/flight_v2.zip models_flight/flight_v3.zip
# ince ayar (fl_v3): flight_v2'den, yeni ödül varsayılanları + rotor dt "sim"
python train_command_curriculum.py --task flight --out runs/fl_v3 --init-model models_flight/flight_v2.zip --level F13 \
    --no-promote --total-steps 3000000 --n-envs 3 --n-steps 4096 --batch-size 512 --net 256,256 --eval-freq 500000 \
    --eval-episodes 10 --eval-levels F13,F10,F6 --snapshot-freq 500000 --fine-from F6a --env-overrides '{"rotor_dt_mode": "sim"}'
```

### 9.4. İleri uçuş komutları: sakin ↔ rüzgâr ↔ türbülans (`probe_cruise_wind.py`, `cruise_wind_v3_v2.json`)

Aynı 8 komut (±30 kt, ±90°, ±200 ft, birleşik +20 kt / +60° / +100 ft, 60 s tut), 300 ft / 80 kt / 9300 lbs.

| çevre | flight_v3 başarı | flight_v2 başarı | tutma hatası medyan (irtifa / heading / hız) v3 | roll std v3 | başarısız komutlar v3 |
|---|---|---|---|---|---|
| sakin | 7/8 | 7/8 | 4.0 ft / 1.0° / 1.4 ft/s | 1.9° | birleşik |
| 15 kt rüzgâr (sağ ön) | 5/8 | 4/8 | 3.0 ft / 2.0° / 1.5 ft/s | 2.1° | +30 kt (heading 4.3° > 3°), −200 ft, birleşik |
| 25 kt + orta türbülans + gust | 6/8 | 7/8 | 6.3 ft / 1.4° / 4.5 ft/s (bantlar 2×) | 4.7° | +90°, birleşik |

Okuma: (1) Birleşik komut (üç eksen birden) iki modelde ve üç çevrede de başarısız — en zayıf komut türü; (2) 15 kt
rüzgârda hız değişimi sırasında heading 3° bandının dışına kayıyor (yan rüzgârda burun tutma), tek başına rüzgâr
türbülanstan daha çok başarısızlık veriyor çünkü bantlar genişlemiyor; (3) türbülansta 60 s tutmada irtifa sapması
23–31 ft (2× band 24 ft'in sınırında); (4) flight_v3 ile flight_v2 arasında anlamlı fark yok.

### 9.5. Rüzgâr / gust altında iniş okulu, üç seed, regresyon kapısı — `flight_v4` (2026-10-02)

Kullanıcı endişesi: bir episode türü düzeltilirken diğeri bozulmamalı. Ajan tek bir sinir ağıdır; iniş eğitimi aynı
ağırlıkları değiştirir, ileri uçuş da etkilenir ("unutma"). Tekrar (rehearsal) bunu azaltır, sıfırlamaz. Çare: her yeni
modeli sabit bir **regresyon kapısından** geçirmek (`docs/flight/gate.py`): seçim takımı (21) + iniş stres taraması (24) +
held-out (35); her görev kategorisinde başarı referansın en çok 1 altına inebilir, güvensiz (düşme / sınır aşımı) sayısı
artamaz. Geçmeyen model alınmaz.

Deney: F14 (rüzgâr 0–25 kt, hafif / orta türbülans, gust altında iniş; %50 F13 / F10 / F8 / F9 tekrarı), flight_v3'ten
1.5 M adım, **üç seed** (1, 2, 3; `--seed` artık yüklenen modele de uygulanıyor), referans flight_v3.

| aday | iniş stresi (24) | güvensiz | seçim: tüm · görev | held-out: tüm · görev | geriledi | kapı |
|---|---|---|---|---|---|---|
| flight_v3 (referans) | 18/24 | 3 | 17/21 · 100/105 | 29/35 · 117/128 | — | — |
| seed 1, 1.0 M | 8/24 | 2 | iniş 2/13 | iniş 5/15 | iniş çöktü (collective yere oturunca tam inmiyor, fl11'deki hata) | KALDI |
| seed 2, 1.0 M | 18/24 | 0 | dönüş 9/14, irtifa 1/8 | dönüş 6/15 | ileri uçuş | KALDI |
| seed 3, 1.0 M | 18/24 | 5 | — | — | güvenlik | KALDI |
| seed 1, 1.5 M | 16/24 | 1 | hızlanma 14/14 ↑ | birleşik 13/13 ↑ | iniş stresi −2 | KALDI |
| seed 2, 1.5 M | 17/24 | 1 | dönüş 10/14, hız 3/8 | dönüş 5/15 | ileri uçuş | KALDI |
| **seed 3, 1.5 M** | 18/24 | 2 | 17/21 · 101/105 | **30/35 · 119/128** (yeterli+ 128/128) | yok | **GEÇTİ** |

Aynı eğitim, üç seed, üç farklı bozulma: ince ayar kararsız; kapı olmadan seçim şans işi. **Sonuç modeli
`models_flight/flight_v4.zip` = seed 3, 1.5 M.** Ek kontroller: yeni seed'li seviyeler F3 %80 / F5 %90 / F6 %100 / F7 %90 /
F8 %90 / F10 %85 (flight_v3 80 / 95 / 100 / 85 / 80 / 80); yerinde dönüş 8/8 (kayma 6.8 ft, flight_v3 5.3), pirouette 4/4
(8.5 ft); hızlanmada irtifa kaybı 0.1 ft (flight_v3 8.2; bunun yerine 13–18 ft tırmanıyor); iniş son 5 ft'te 2.4–2.6 ft/s
(3.7–3.9), temas medyanı −1.3 ft/s. Bedeli: 56 psi üstü süre held-out'ta 45 → 60 s, seçimde 61 → 78 s.

Gust'lı iniş kazaları (25 kt + orta türbülans + gust, 50–150 ft'ten): 2/24 (devrilme) — flight_v2 4, flight_v3 3; F14
okulu bunu azaltmadı, yalnızca kötüleştirmedi. Açık konu.

Dosyalar: `eval_secim_v4.json`, `eval_test_v4.json`, `landing_stress_v4.json`, `precision_v4.json`, `accel_v4.json`,
`eval_levels_v4.json`, kapı çıktıları `docs/flight/gate/`, koşular `runs/fl_v4_s1..s3`.

```bash
python docs/flight/gate.py --ref models_flight/flight_v3.zip --cand <aday.zip> --out /tmp/gate_aday     # yeni model kapısı
python train_command_curriculum.py --task flight --out runs/fl_v4_s3 --init-model models_flight/flight_v3.zip --level F14 \
    --no-promote --total-steps 1500000 --n-envs 1 --vec dummy --n-steps 4096 --batch-size 512 --net 256,256 --eval-freq 0 \
    --snapshot-freq 500000 --fine-from F6a --seed 3
```

### 9.6. Unutma problemi: araştırma, BC buffer, rejim uzmanları, büyük batch (2026-10-03)

**Sorun.** Bir beceri (iniş ya da ileri uçuş) iyileşince diğeri geriliyor; aynı eğitim farklı seed'lerde farklı şekilde
bozuluyor (9.5'te üç seed, üç farklı bozulma).

**Kök nedenler (ölçüldü):**
1. Tek aksiyon ağı bütün görevleri taşıyor; iniş, ağın başka hiçbir görevde istemediği bir şeyi istiyor (collective
   tam aşağı, aksiyon −1; yer etkisi, kızak teması).
2. **Güncelleme başına çok az episode:** episode ~2000 adım, PPO güncellemesi 4096 adım (1 env) → her güncelleme ~2
   episode görüyor; ağ o episode'ların görevine doğru kayıyor, sonraki güncellemede başka göreve. Seed değişkenliğinin
   ana kaynağı. (Oyunlarda / Isaac Gym'de her güncelleme binlerce episode'dan gelir.)
3. Soy boyunca 15'ten fazla sıralı ince ayar ve her birinde ödül değişikliği.

**Literatür** (Wołczyk vd. 2024 ICML; Rolnick vd. 2019 CLEAR; Kirkpatrick vd. 2017 EWC; Schwarz vd. 2018 Progress &
Compress; Hessel vd. 2019 PopArt; Henderson vd. 2018, Agarwal vd. 2021 seed sayısı): sıralı ince ayarda en etkili
yöntem davranış klonlama buffer'ı (öğretmen = eski model); seçenekler: EWC, parametre izolasyonu, görev başına
normalizasyon, birlikte (joint) eğitim. Makalelerde 5–10 seed.

**Mentor kuralı:** öğretmen / öğrenci (damıtma) yasak. BC buffer (`ppo_bc.py`, dondurulmuş eski model = öğretmen)
teknik olarak damıtmadır → sonuç modeli için kullanılmaz; kod deney kaydı olarak duruyor.

**Öğretmensiz yöntem: rejim uzmanları** (`regime_policy.py`, `make_regime_model.py`). Aksiyon ağı üç uzmana bölünür:
iniş (gözlem[24] = 1), ileri uçuş (gözlem[34] = 1), hover (diğer). Seçici öğrenilmez, bayraklar seçer. Değer ağı ortak.
Başlangıçta üç uzman flight_v4'ün aksiyon ağının kopyası (fark 0). Eğitimde bir rejim dondurulabilir
(`--freeze-regimes`): ağırlıkları değişmez (0.5 M adımda doğrulandı: iniş uzmanında değişim 0.000).

**Deney 1 — F15 mükemmellik (ileri uçuş irtifa + düz tırmanış), 1 env, 1 M adım:**

| aday | iniş stresi (24) | seçim: tüm · görev | held-out: tüm · görev | ne geriledi | kapı |
|---|---|---|---|---|---|
| flight_v4 (referans) | 18 | 17/21 · 101/105 | 30/35 · 119/128 | — | — |
| düz PPO s1 | 16 | 12/21 · 93/105 | | iniş 13→9 (seçim), 15→12 (held-out) | KALDI |
| düz PPO s2 | 18 | 17/21 · 99/105 | | güvensiz 2→8, ileri uçuş dönüşü | KALDI |
| rejim (iniş kilitli) s1 | **20** | 17/21 · 101/105 | 30/35 · **122**/128 | hover: 360° yerinde dönüşte yatış sınırı (1 güvensiz) | KALDI |
| rejim s2 | **19** | 6/21 · 71/105 | 10/35 · 83/125 | ileri uçuş heading tutma (bantta 3° aşımı) | KALDI |
| rejim s3 | **20** | 14/21 · 90/98 | 31/35 · 121/125 | ileri uçuş irtifa komutu, hover dönüş kayması 30 ft | KALDI |

Sonuç: kilitli rejim (iniş) üç seed'de de korundu ve iyileşti; bozulma yalnızca EĞİTİLEN uzmanların içinde (hover:
tırmanış ↔ yerinde dönüş; ileri uçuş: irtifa ↔ heading). Mükemmellik ölçümü (`probe_perfection.py`): düz uçuşta irtifa
sapması flight_v4 medyan 4.9 ft → düz s2 2.8 ft; tırmanışta yatay kayma değişmedi (5–6 ft).

**Deney 2 — tek rejim + büyük batch:** yalnızca ileri uçuş uzmanı eğitilir (hover ve iniş kilitli), 16 paralel ortam ×
512 adım (her güncellemede 16 episode; hız ~1940 adım/s, eski koşunun ~3 katı), 3 seed. [9.6 DENEY 2 DOLDURULACAK]
