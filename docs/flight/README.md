# Tek ajanlı sürekli uçuş (Aşama 2) — `helicopter_env_flight.py`, `flight_curriculum.py`

Tek bir PPO ajanı (Stable-Baselines3, sıfırdan, öğretmen–öğrenci yok), tek ve sürekli bir episode'da: rotor warm-up
(scriptli) → kalkış → hover → ileri uçuşa geçiş → ileri uçuşta Δhız / Δheading / Δirtifa → duruş (hover) → isteğe bağlı
iniş. Dört kumanda doğrudan (collective, boylamsal / yanal cyclic, pedal); AFCS yalnızca SAS. Fizik baştan açık: repo
uçağı (kalibre yer etkisi), güç tavanı %100 tork (56 psi), tork gözlemi ve cezası, yakıt tüketimi (el kitabı grafiği;
bitince motor ayrılır). Rüzgâr / gust / türbülans seviyenin çevre aşamasından.

> Durum: **eğitim sürüyor** (bu dosya koşular ilerledikçe güncellenir). Sonuç tablosu ve şekiller bölüm 6'da.

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

| seviye | görev | çevre | eşik |
|---|---|---|---|
| F1 | hover tut (havada başla, 15–300 ft) — kalkış curriculum'unun K1'i | E0 sakin | %80 |
| F2 | kalkış 10–300 ft + 0–2 hover manevrası; %40 havada başlayıp manevra | E0 | %75 |
| F3 | ileri uçuş (40–100 kt, 150–800 ft): tek eksen Δ (hız ±10–25 kt / heading ±20–90° / irtifa ±50–200 ft) | E0 %70 · E1 %30 | %75 |
| F4 | ileri uçuş: birleşik Δ'lar + %30 kesen komut | E0 %60 · E1 %40 | %70 |
| F5 | geçişler: hover → 40–80 kt hızlanma (+0–200 ft) + 0–1 Δ → duruş; ileri uçuştan duruş | E0 / E1 | %70 |
| F6 | zincir: yerden kalkış → hızlanma → 1–3 Δ → duruş | E0 / E1 | %65 |
| F7 | iniş (alçak hover / kısa kalkış; %30 yerde hafif yüklü, %30 çok alçak hover) | E0 / E1 | %70 |
| F8 | karma: hover / kalkış / iniş / ileri uçuş / geçişler / zincir | E1 / E2 | %65 |
| F9 | son: karma + zincir, rüzgâr 0–25 kt, türbülans yok / hafif / orta, gust'lar | E1 %30 · E2 %40 · E3 %30 | %60 |

Çevre aşamaları: E0 sakin · E1 rüzgâr 0–10 kt · E2 0–15 kt + %60 hafif türbülans + gust (dakikada 0.5, 3–8 kt) ·
E3 5–25 kt + hafif / orta türbülans (%50 / %50) + gust (dakikada 1, 5–12 kt). Rüzgâr yönü rastgele (başlangıç heading'ine
göre). Tork cezası ve yakıt her seviyede açık; ağırlık 8800–9700 lbs (tank başına 150–600 lbs; OGE hover ≤ ~53 psi).
F2'den itibaren episode'ların %30–40'ı eski seviyelerden (rehearsal); seviye atlama: son 100 episode + görev türü
başına kapı (%80, en az 30 komut) ya da deterministik değerlendirme (`--promote-on-eval`).

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

Ortam: Claude'un cloud konteyneri, 2 çekirdek CPU, SB3 PPO, 2 paralel env (~500 adım/s). Ortak ayarlar: n_steps 4096 ×
2 env, batch 512, 10 epoch, lr 3e-4, γ 0.995, λ 0.95, σ0 = e^−1.2, ayrı pi / vf ağları (256×256, tanh), `--promote-on-eval`,
500 bin adımda bir deterministik değerlendirme (F2 / F4 / F6 + mevcut seviye, 12'şer episode; en iyisi `best.zip`).

| koşu | başlangıç | değişiklik | sonuç |
|---|---|---|---|
| fl_v0 | sıfırdan, 256×256 | trim ölçülen hava hızıyla (τ 2 s); F1 = hover tut + 0–2 manevra | F1'de 300 bin adımda başarı %6, **episode'ların %50'si 150 ft sapmayla** bitiyordu (kalkış env'inin K1'inde %2). Rastgele policy'yle: ölçülen hıza göre trim 5/16, referans hızına göre 1/16 kaçış. Durduruldu. |
| fl_v1 | sıfırdan, 256×256 | trim görevin referans hızıyla; F1 = yalnızca hover tut | **F1 160 bin, F2 337 bin adımda geçildi** (toplam ~0.5 M, 16 dk). F3'te (ileri uçuşta başlangıç) episode'ların %80'i 5 s içinde 40° pitch ile bitti: hover policy'si 90 kt'lık yer hızını (gözlemde ±5'te kırpılmış) "fren yap" diye okuyup cyclic'i tam geri çekiyordu. Durduruldu. |
| fl_v2 | fl_v1'in F2 modeli, F3'ten | kalkış env'inin hız girdileri hız hatası (hover'da aynı; ileri uçuşta referans hava hızına göre) | sürüyor |

## 6. Sonuçlar

(eğitim bitince)

## 7. Çalıştırma

```bash
python helicopter_env_flight.py                                    # duman testi (F1/F3/F5/F6/F9, sıfır action)
python docs/flight/check_flight_scripted.py --scenarios all --levels F3,F5,F6 --seeds 3 --zero
python train_command_curriculum.py --task flight --out runs/fl --total-steps 30000000 --n-envs 2 --n-steps 4096 \
    --batch-size 512 --net 256,256 --eval-freq 500000 --eval-episodes 12 --promote-on-eval --snapshot-freq 1000000
python evaluate_flight.py --model runs/fl/models/best.zip --json docs/flight/eval.json [--levels F3,F6,F9]
python command_viz.py --model models_flight/flight_final.zip --start ground --wind-kt 15 --turb light
```
