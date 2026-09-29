# Tek ajanlı sürekli uçuş (Aşama 2) — `helicopter_env_flight.py`, `flight_curriculum.py`

Tek bir PPO ajanı (Stable-Baselines3, sıfırdan, öğretmen–öğrenci yok), tek ve sürekli bir episode'da: rotor warm-up
(scriptli) → kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δhız / Δheading / Δirtifa → duruş (hover) → isteğe bağlı
iniş. Dört kumanda doğrudan (collective, boylamsal / yanal cyclic, pedal); AFCS yalnızca SAS. Fizik baştan açık: repo
uçağı (kalibre yer etkisi), güç tavanı %100 tork (56 psi), tork gözlemi ve cezası, yakıt tüketimi (el kitabı grafiği;
bitince motor ayrılır). Rüzgâr / gust / türbülans seviyenin çevre aşamasından.

> Durum (2026-09-29): **eğitim bitti** — sonuç modeli `models_flight/flight_final.zip` (fl_v9 4.5 M). Sonuç tablosu ve
> şekiller bölüm 6'da.

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
| **fl_v9** (cila) | fl_v8'in F8 modeli, F9'da (`--no-promote`), 5 M adım (120 dk) | F9: F8 karışımı + havada başlangıç %40, 1–3 hover manevrası, sakin hover manevrası tekrarı (F2); 56 psi üstüne doğrusal ek ceza (`pen_torque_over_lin`) | eğitim başarısı F9 %45 → %70–85; yerinde dönüş %25 → %94, kayma %20 → %94, iniş %55 → %80. Takımda en iyi ara model **4.5 M** (tüm görevler 17/21, görev 102/105, iniş 13/13); 5 M'de iniş 8/13'e düştü (dalgalanma) → **sonuç modeli = fl_v9 4.5 M** |

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
