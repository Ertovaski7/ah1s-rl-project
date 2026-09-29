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

Ortam: Claude'un cloud konteyneri, 2 çekirdek CPU, SB3 PPO, 2 paralel env (~500 adım/s). Ortak ayarlar: n_steps 4096 ×
2 env, batch 512, 10 epoch, lr 3e-4, γ 0.995, λ 0.95, σ0 = e^−1.2, ayrı pi / vf ağları (256×256, tanh), `--promote-on-eval`,
500 bin adımda bir deterministik değerlendirme (F2 / F3 / F5 + mevcut seviye, 12'şer episode; en iyisi `best.zip`:
F2 / F3 / F5 ortalaması).

| koşu | başlangıç | değişiklik | sonuç |
|---|---|---|---|
| fl_v0 | sıfırdan, 256×256 | trim ölçülen hava hızıyla (τ 2 s); F1 = hover tut + 0–2 manevra | F1'de 300 bin adımda başarı %6, **episode'ların %50'si 150 ft sapmayla** bitiyordu (kalkış env'inin K1'inde %2). Rastgele policy'yle: ölçülen hıza göre trim 5/16, referans hızına göre 1/16 kaçış. Durduruldu. |
| fl_v1 | sıfırdan, 256×256 | trim görevin referans hızıyla; F1 = yalnızca hover tut | **F1 160 bin, F2 337 bin adımda geçildi** (toplam ~0.5 M, 16 dk). F3'te (ileri uçuşta başlangıç) episode'ların %80'i 5 s içinde 40° pitch ile bitti: hover policy'si 90 kt'lık yer hızını (gözlemde ±5'te kırpılmış) "fren yap" diye okuyup cyclic'i tam geri çekiyordu. Durduruldu. |
| fl_v2 | fl_v1'in F2 modeli, F3'ten | kalkış env'inin hız girdileri hız hatası (hover'da aynı; ileri uçuşta referans hava hızına göre) | pitch kaçışı bitti, ama ileri uçuş 0.76 M adımda **%0**; deterministik F2 (hover) %100 → %25 (unutma). Durduruldu. |
| diag_f3 | sıfırdan, yalnızca ileri uçuş (tanı) | — | "tut" hemen %97; heading komutları 0.5 M adımda %0 (dönüş yatışla yapılır; ajan bulamadı) → koordineli dönüş yönlendirmesi (istenen yatış = atan(r·V/g), 15→30 kt'ta devreye girer) ve ilk Δ seviyesinde yumuşak süre hedefleri |
| **fl_v3** | sıfırdan, 256×256, `--fine-from F7`, seed 4 | hover + ileri uçuş F1'den birlikte; dönüş yönlendirmesi; F2'de yumuşak Δ'lar | F1 0.25 M adımda (7.8 dk), F2 0.75 M adımda (20 dk) geçildi; 1 M adımda deterministik F2 %92, F3 %58, F5 %58. Sürüyor. |

## 6. Sonuçlar

(eğitim bitince)

## 7. Çalıştırma

```bash
python helicopter_env_flight.py                                    # duman testi (F1/F2/F4/F5/F8, sıfır action)
python docs/flight/check_flight_scripted.py --scenarios all --levels F3,F5,F6 --seeds 3 --zero
python train_command_curriculum.py --task flight --out runs/fl --total-steps 40000000 --n-envs 2 --n-steps 4096 \
    --batch-size 512 --net 256,256 --eval-freq 500000 --eval-episodes 12 --promote-on-eval --snapshot-freq 1000000 \
    --fine-from F7 --seed 4
python evaluate_flight.py --model runs/fl/models/best.zip --json docs/flight/eval.json [--levels F2,F5,F8]
python evaluate_ads33.py --model runs/fl/models/best.zip           # ADS-33 MTE karnesi (sakin hava)
python docs/flight/fig_flight.py --model runs/fl/models/best.zip --runs runs/fl
python command_viz.py --model models_flight/flight_final.zip --start ground --wind-kt 15 --turb light
```
