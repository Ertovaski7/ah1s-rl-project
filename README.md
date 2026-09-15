# AH-1S JSBSim Reinforcement Learning Uçuş Kontrol Projesi

Bu proje, **JSBSim içindeki `ah1s` helikopter modelini** kullanarak AH-1S için çok aşamalı bir uçuş görevinin **PPO tabanlı reinforcement learning (RL)**, **teacher–student / policy distillation**, **residual düzeltmeler** ve **AFCS destekli geçiş kontrolü** ile gerçekleştirilmesini amaçlar.

Temel görev zinciri şu şekildedir:

> **Stage 1: Kalkış ve 300 ft hover → Stage 2: İleri uçuş → Stage 3: Relative turn → Transition: Dönüş sonrası stabilizasyon → Post-turn Stage 2: Yeni heading üzerinde ileri uçuş**

En önemli tasarım kararı, fazlar arasında simülasyonun yeniden başlatılmamasıdır. **Aynı JSBSim FDM (Flight Dynamics Model) ve aynı helikopter state’i bir sonraki faza aktarılır.** Böylece her policy yalnızca ideal bir başlangıç durumunda değil, önceki fazın gerçek dinamik çıktısı üzerinden çalışır.

---

## 1. Projenin amacı

Bu projede helikopterin tek bir policy ile bütün görevi öğrenmesi yerine görev daha yönetilebilir alt problemlere ayrılmıştır:

1. Helikopteri yerden kaldırmak ve yaklaşık **300 ft** irtifaya çıkarmak.
2. 300 ft civarında irtifayı koruyarak ileri uçmak.
3. Mevcut heading’i referans alıp istenen açı kadar **relative turn** yapmak.
4. Dönüş sonrasında kalan lateral hız, roll, yaw-rate ve vertical-speed gibi dinamikleri sönümlemek.
5. Yeni heading üzerinde yeniden ileri uçuşa devam etmek.

Bu yaklaşım sayesinde her faz için farklı bir kontrol problemi tanımlanmış, daha sonra bu fazlar tek bir kesintisiz görev zincirinde birleştirilmiştir.

---

# 2. Genel algoritma akışı

```mermaid
flowchart TD
    A[JSBSim AH-1S modeli yüklenir] --> B[reset00.xml başlangıç koşulları]
    B --> C[Rotor warm-up ve governor aktivasyonu]
    C --> D[Low-level AFCS roll/pitch/yaw stabilizasyonu]

    D --> E[Stage 1 PPO / Distilled Policy]
    E --> F{300 ft civarında stabil mi?}
    F -- Hayır --> E
    F -- Evet --> G[Stage 2 Forward PPO]

    G --> H{İleri uçuş handoff mesafesine ulaşıldı mı?}
    H -- Hayır --> G
    H -- Evet --> I[Dönüş başlangıç heading'i kaydedilir]

    I --> J[Stage 3 Goal-Conditioned Relative Turn]
    J --> K[Base turn PPO + residual/specialist düzeltmeler]
    K --> L{Turn success koşulları 3 s korunuyor mu?}
    L -- Hayır --> J

    L -- Evet --> M[Yeni heading referansı alınır]
    M --> N[AFCS Transition / Stabilizasyon]
    N --> O[Roll, yaw-rate, lateral speed, vertical speed ve altitude toparlanır]

    O --> P[Post-turn Stage 2 PPO]
    P --> Q[Yeni heading üzerinde ileri uçuş]
    Q --> R[Full mission doğrulaması]

    E -. aynı FDM / reset yok .-> G
    G -. aynı FDM / reset yok .-> J
    J -. aynı FDM / reset yok .-> N
    N -. aynı FDM / reset yok .-> P
```

Özet olarak sistemin runtime veri akışı:

```text
JSBSim state
    ↓
Observation vector
    ↓
Aktif fazın PPO / control policy'si
    ↓
Normalized action [-1, 1]
    ↓
Physical control mapping
    ↓
collective / longitudinal cyclic / lateral cyclic / pedal
    ↓
JSBSim physics step
    ↓
Yeni state + reward + safety kontrolleri
```

---

# 3. AH-1S ve simülasyon tarafı

Projede gerçek uçuş donanımı yerine **JSBSim flight dynamics engine** ve JSBSim’in `ah1s` modeli kullanılmaktadır.

Temel simülasyon ayarlarından bazıları:

- JSBSim model adı: `ah1s`
- Başlangıç koşulu: `reset00.xml`
- Physics timestep: **0.0075 s**
- Her RL action için: **10 physics step**
- Etkin RL kontrol periyodu: yaklaşık **0.075 s**
- Kontrol frekansı: yaklaşık **13.33 Hz**
- Ana hedef irtifa: **300 ft AGL**

Rotor başlangıçta warm-up sürecinden geçirilir ve governor aktif hale getirilir. Roll, pitch ve yaw için AH-1S modelinin AFCS kanalları düşük seviye stabilizasyon amacıyla kullanılabilir. Stage 1’in temel yaklaşımında altitude kontrolü AFCS’ye bırakılmaz; irtifa davranışı PPO tarafından öğrenilir.

> Bu repository bir **simülasyon / araştırma projesidir**. Buradaki action mapping, trim değerleri ve kontrol mantıkları JSBSim modeline yöneliktir; gerçek AH-1S uçuş kontrol sistemi için operasyonel talimat olarak değerlendirilmemelidir.

---

# 4. Helikopter kontrol eksenleri ve action space

Policy’nin temel action vektörü dört boyutludur:

```python
Action = [a0, a1, a2, a3]
```

Tüm PPO action’ları önce normalize edilmiş olarak üretilir:

```text
-1.0 <= action[i] <= +1.0
```

Simülasyondaki karşılıkları:

| Action | Simülasyondaki kontrol | Temel etkisi |
|---|---|---|
| `a0` | Collective | Ana rotor toplam pitch/thrust; climb, descent ve altitude üzerinde güçlü etki |
| `a1` | Elevator / longitudinal cyclic komutu | Pitch ve ileri–geri hareket üzerinde etki |
| `a2` | Aileron / lateral cyclic komutu | Roll ve lateral hareket üzerinde etki |
| `a3` | Rudder / pedal komutu | Yaw / heading üzerinde etki |

Buradaki `elevator`, `aileron` ve `rudder` isimleri **JSBSim modelindeki FCS property adlarıdır**. Helikopter kontrolü açısından bunlar sırasıyla longitudinal cyclic, lateral cyclic ve pedal/yaw davranışına karşılık gelecek şekilde kullanılmıştır.

## Stage 1 action mapping

İlk Stage 1 geliştirmesinde PPO yalnızca collective üzerinde aktifti:

```text
a0 = -1  → collective ≈ 0.590
a0 =  0  → collective ≈ 0.620
a0 = +1  → collective ≈ 0.650
```

Temel formül:

```python
collective = 0.620 + 0.030 * a0
```

Daha sonraki distilled Stage 1 sürümünde student dört action üretir; cyclic ve pedal komutları hover trim değerleri etrafında sınırlı authority ile uygulanır ve action’lar smoothing işleminden geçirilir.

## Stage 2 action mapping

Stage 2’de collective ve longitudinal cyclic ileri uçuş için aktif olarak kullanılır. Lateral/yaw mapping daha sonra fiziksel residual olarak onarılmıştır:

```text
a2 → mevcut aileron trim + 0.026 * a2
a3 → mevcut rudder  trim + 0.040 * a3
```

Bu sayede dört PPO çıktısının tamamı fiziksel olarak JSBSim’e ulaşır.

## Stage 3 relative-turn action mapping

Turn environment içinde dört action da aktiftir:

```python
collective = clip(0.540 + 0.080 * a0, 0.460, 0.620)
elevator   = clip(-0.145 + 0.035 * a1, -0.180, -0.110)
aileron    = clip(0.19095 + 0.300 * a2, -1.0, 1.0)
rudder     = clip(0.39000 + 0.500 * a3, -1.0, 1.0)
```

Bu fazda lateral/yaw authority Stage 2’ye göre daha geniştir; çünkü policy’nin gerçek bir yön değiştirme manevrası üretmesi gerekir.

---

# 5. Observation space

Policy yalnızca altitude değerini görmez. Helikopterin hem translational hem rotational durumunu içeren normalize edilmiş bir observation kullanılır.

Temel 12-boyutlu observation:

| İndeks | Gözlem |
|---|---|
| 0 | Altitude error |
| 1 | Altitude |
| 2 | Vertical speed |
| 3 | Forward velocity |
| 4 | Lateral velocity |
| 5 | Pitch |
| 6 | Roll |
| 7 | Roll rate `p` |
| 8 | Pitch rate `q` |
| 9 | Yaw rate `r` |
| 10 | Heading error |
| 11 | Rotor RPM error |

Stage 1 distillation environment’ında bu gözleme ayrıca pozisyon ve yatay hareket bilgileri eklenerek **18-boyutlu** observation kullanılır:

```text
base 12
+ north
+ east
+ north velocity
+ east velocity
+ heading error
+ yaw rate
= 18 observation
```

Relative-turn environment’ında ise Stage 2 observation’ına hedef bilgileri eklenir ve toplam **16-boyutlu goal-conditioned observation** elde edilir:

```text
12 Stage-2 feature
+ target_turn / 360
+ remaining_turn / 360
+ cumulative_turn / 360
+ direction
= 16 observation
```

Bu sayede aynı turn policy yalnızca mevcut helikopter state’ini değil, **hangi açı kadar dönmesi gerektiğini de** gözlemleyebilir.

---

# 6. Stage 1 — Takeoff ve 300 ft hover

Stage 1’in görevi:

```text
motor / rotor hazır
    ↓
takeoff
    ↓
300 ft'e climb
    ↓
vertical speed'i azalt
    ↓
300 ft civarında stabil hover
```

İlk deneylerde collective-only PPO kullanıldı. Daha sonra teacher–student distillation ile yatay drift, heading ve küçük cyclic/pedal düzeltmeleri de student policy tarafından öğrenilebilir hale getirildi.

Stage 1 tamamlandığında helikopter resetlenmez. O anki:

- altitude,
- vertical speed,
- forward/lateral velocity,
- attitude,
- angular rate,
- heading,
- rotor state

bilgileri aynı FDM üzerinde Stage 2’ye taşınır.

---

# 7. Stage 2 — Forward flight

Stage 2’nin amacı yaklaşık 300 ft irtifayı korurken kontrollü ileri uçuş üretmektir.

Reward mantığında başlıca terimler:

- ileri yönde gerçek progress için pozitif reward,
- altitude error cezası,
- vertical speed cezası,
- hedef forward speed’den sapma cezası,
- lateral drift cezası,
- pitch / roll cezası,
- action değişimi için smoothness cezası.

Standalone Stage 2 eğitim environment’ında hedef forward distance **300 ft** olarak tanımlanmıştır. Full mission içindeki turn handoff senaryosunda ise turn environment ortak giriş koşulunu üretmek için yaklaşık **160 ft forward flight** sonrası dönüş fazını başlatır.

---

# 8. Stage 3 — Relative turn

## Relative turn nedir?

Relative turn, helikopterin dönüşe başladığı andaki heading’i referans alıp verilen açı kadar dönmesidir.

Örnek:

```text
Başlangıç heading = 180°
Relative command  = +50°
Yeni hedef yön    ≈ 230°
```

Projedeki işaret konvansiyonunda:

```text
+ açı → sağa / saat yönünde
- açı → sola / saat yönünün tersine
```

Burada önemli nokta sadece son heading’i karşılaştırmak değildir. `+360°` gibi tam tur manevrasında başlangıç ve bitiş heading’i aynı görünebilir. Bu nedenle sistem her step’te heading farkını unwrap ederek **cumulative turn** tutar:

```python
remaining_turn = requested_turn - cumulative_turn
```

Böylece `-50°`, `+50°`, `+200°` ve `+360°` gibi manevralar aynı goal-conditioned mantıkla takip edilebilir.

## Turn success kriterleri

Turn yalnızca açı yakalandığında başarılı sayılmaz. Stabilite de kontrol edilir.

Güncel turn environment kriterleri:

```text
|remaining turn| <= 1.5°
|roll|          <= 4°
|yaw rate|      <= 6°/s
285 ft <= altitude <= 315 ft
|vertical speed| <= 1.5 ft/s
forward speed >= 8 ft/s
```

Bu koşulların yaklaşık **3 saniye boyunca** korunması gerekir.

Safety sınırlarından bazıları:

```text
altitude < 275 ft veya > 325 ft
|roll| > 18°
|pitch| > 15°
|yaw rate| > 30°/s
forward speed < 1 ft/s
JSBSim failure
```

---

# 9. Turn policy neden tek modelden ibaret değil?

Helikopter dönüş problemi farklı giriş state’lerinde ve farklı hedef açılarda aynı zorlukta değildir. Bu nedenle final yaklaşım yalnızca tek bir PPO çıktısına dayanmak yerine **base policy + residual/specialist correction** yapısını kullanır.

Genel fikir:

```text
Base turn PPO action
        +
Live-entry residual adapter
        +
Hedefe özel gated correction gerekiyorsa
        +
Terminal correction gerekiyorsa
        ↓
Final action
        ↓
clip [-1, +1]
        ↓
JSBSim
```

Residual modeller ana policy’nin öğrendiği uçuş davranışını tamamen değiştirmek yerine küçük düzeltmeler üretir. Böylece mevcut çalışan policy korunurken belirli problemli giriş bölgeleri veya hedef açıları düzeltilebilir.

---

# 10. Teacher–Student / Policy Distillation

Teacher bu projede final runtime controller değildir.

Teacher’ın görevi eğitim sırasında student policy’ye daha iyi bir başlangıç davranışı sağlamaktır.

Stage 1 tarafında teacher bileşenleri örneğin:

- vertical control için daha önce öğrenilmiş PPO davranışı,
- yatay drift için tanımlanmış / kalibre edilmiş controller,
- teacher action ile student action arasında training-time blending

şeklinde kullanılabilir.

Training sırasında genel mantık:

```text
State
 ├──→ Teacher → target / reference action
 └──→ Student PPO → predicted action

Training scaffold / distillation
        ↓
Student giderek daha fazla kontrol authority alır
        ↓
Teacher blend azaltılır
        ↓
Teacher OFF
```

Final görevde amaç:

```text
Teacher = OFF
Student / learned policy = ON
```

Yani teacher **ağırlıklarını runtime’da student’a vermiyor** ve her step’te final uçuşu yönetmiyor. Teacher eğitim sırasında hedef davranış/action üretmek için kullanılan bir scaffold’dur. Environment reward’ları ise ayrıca görev hedefleri ve fiziksel hata metriklerinden hesaplanır.

---

# 11. Transition — dönüşten sonra neden ayrı bir faz var?

Bir dönüşün geometrik olarak tamamlanması ile helikopterin stabilize olması aynı şey değildir.

Örneğin `+50°` turn tamamlandığında heading hedefe ulaşmış olabilir; ancak helikopterde hâlâ:

- lateral velocity,
- roll,
- yaw rate,
- vertical speed,
- altitude deviation

kalabilir.

Bu durumda Stage 2 PPO’ya aniden geçmek kötü bir handoff oluşturabilir.

Bu nedenle turn capture’dan sonra **AFCS destekli transition** kullanılır.

Transition mantığı:

```text
Turn target yakalandı
        ↓
Yeni heading referansı sabitlenir
        ↓
Roll azaltılır
Yaw-rate azaltılır
Lateral speed azaltılır
Vertical speed azaltılır
Altitude ~300 ft civarına toparlanır
        ↓
Forward-flight için uygun state
        ↓
Stage 2 PPO yeniden devreye girer
```

Kısacası:

> **Turn capture açıyı tamamlar; transition helikopteri o yeni açı üzerinde tekrar düzgün uçabilir hale getirir.**

---

# 12. Post-turn Stage 2

Transition sonrasında aynı Stage 2 forward-flight policy yeniden kullanılır.

Fakat artık referans yön değişmiştir.

Örnek:

```text
Eski heading      = 180°
Relative turn     = +50°
Yeni heading      ≈ 230°
Post-turn Stage 2 = 230° doğrultusunda ileri uçuş
```

Bu faz, turn tamamlandıktan sonra helikopterin yalnızca hedef açıya ulaşmadığını, **o yeni yönde kararlı şekilde uçuşa devam edebildiğini** doğrular.

---

# 13. Neden fazlar arasında reset yok?

Bu projenin önemli noktalarından biri budur.

Kolay bir test yaklaşımında her stage ideal state’ten başlatılabilir. Ancak bu gerçek handoff problemlerini gizler.

Bu projede hedef:

```text
Stage 1 final state
    ↓
Stage 2 initial state
    ↓
Turn initial state
    ↓
Transition initial state
    ↓
Post-turn initial state
```

olarak gerçek state zincirini korumaktır.

Bu nedenle aynı JSBSim FDM üzerinde:

```text
NO RESET BETWEEN MISSION PHASES
```

prensibi uygulanır.

---

# 14. PPO bu projede nasıl çalışıyor?

PPO (Proximal Policy Optimization), continuous action space için kullanılan policy-gradient tabanlı bir RL algoritmasıdır.

Training döngüsü basitleştirilmiş haliyle:

```text
1. Observation al
2. PPO policy action üretir
3. Action physical control değerine map edilir
4. JSBSim physics ilerletilir
5. Yeni state okunur
6. Reward hesaplanır
7. Transition buffer'a kaydedilir
8. PPO policy/value network güncellenir
9. Yeni rollout başlar
```

Runtime sırasında ise training update yapılmaz:

```text
obs → model.predict(..., deterministic=True) → action → JSBSim
```

---

# 15. Reward tasarımının genel mantığı

Her stage’in reward’ı farklıdır; çünkü her fazın amacı farklıdır.

## Stage 1

Öncelikler:

- 300 ft’e yaklaşmak,
- vertical speed’i kontrol etmek,
- hover’da kalmak,
- drift’i azaltmak,
- attitude ve rotor durumunu güvenli tutmak.

## Stage 2

Öncelikler:

- forward progress,
- altitude hold,
- uygun forward speed,
- düşük lateral drift,
- düşük vertical speed,
- uygun pitch/roll,
- smooth action.

## Relative turn

Turn reward’ın ana bileşeni gerçek **remaining-turn error reduction** değeridir:

```python
progress = |previous_remaining| - |remaining|
reward += 2.0 * progress
```

Buna ek olarak altitude, vertical speed, forward speed, roll, yaw-rate ve action smoothness cezaları uygulanır.

Bu sayede policy sadece hızlı dönmeye değil, **dönüş sırasında helikopteri uçabilir ve güvenli bölgede tutmaya** zorlanır.

---

# 16. Doğrulama yaklaşımı

Turn sistemi yalnızca tek bir ideal başlangıç state’inde denenmemiştir. Full-entry robustness testlerinde Stage 2 uçuşu farklı sürelerde devam ettirilerek fiziksel olarak farklı turn-entry state’leri oluşturulmuştur.

Güncel V22 turn robustness testi:

```text
Targets: -50°, +50°, +200°, +360°
Seeds per target: 5
Toplam test: 20
PASS: 20 / 20
Safety failure: 0 / 20
```

Bu testte state teleport edilmemiş; Stage 2 fiziksel olarak birkaç ek step devam ettirilerek turn giriş koşulu değiştirilmiştir.

Full-mission tarafında `-50°`, `+50°`, `+200°` ve `+360°` için güncel rekonstüre edilmiş sistemde ayrı ayrı PASS kanıtı elde edilmiştir. Bunlar farklı targeted validator/tuning sürümlerinde elde edildiği için tek bir unified latest-config `4/4` koşusu ile aynı şey olarak değerlendirilmemelidir.

---

# 17. Önemli model / kod bileşenleri

Repository’deki başlıca dosyalar:

```text
helicopter_env_v2.py
    Temel AH-1S / JSBSim environment uyumluluk katmanı

helicopter_env_stage1_distill.py
    Stage 1 teacher-student / distilled environment

helicopter_env_stage2_refine_mapped.py
    Stage 2 dört-action physical mapping düzeltmesi

helicopter_env_turn_goal.py
    Goal-conditioned relative-turn environment

helicopter_env_turn_goal_full_entry.py
    Turn için full-mission benzeri gerçek giriş state'i

AH1S_STAGE1_FINAL_DISTILLED.zip
    Stage 1 final distilled model

AH1S_STAGE2_HYBRID_FINAL.zip
    Stage 2 forward-flight model

train_turn_full_entry_*.py
    Turn residual / specialist / terminal training scriptleri

test_turn_full_entry_*.py
    Turn robustness ve runtime validator scriptleri

validate_full_mission_*.py
    Stage 1 → Stage 2 → Turn → Transition → Post-turn görev doğrulaması

build_final_interactive_replay.py
build_interactive_replay_3d_dashboard.py
beautify_current_full_mission_replay.py
    Telemetry / interaktif uçuş replay araçları
```

Repository’de geliştirme tarihi boyunca çok sayıda `vN` dosyası bulunur. Bu numaralar çoğunlukla **deneme, repair veya validator versiyonlarını** ifade eder; görev stage numarası değildir.

---

# 18. Kurulum

Python bağımlılıkları:

```bash
pip install -r requirements.txt
```

`requirements.txt` içinde temel olarak:

```text
stable-baselines3
gymnasium
numpy
matplotlib
jsbsim
```

bulunur.

Google Colab kullanımında repository klonlandıktan / güncellendikten sonra modeller ve scriptler doğrudan çalıştırılabilir.

---

# 19. İnteraktif replay

Projede flight telemetry’nin sunum amaçlı incelenebilmesi için interaktif HTML replay araçları da bulunmaktadır.

Replay içinde amaçlanan görünüm:

```text
Stage 1 — Takeoff / Hover
Stage 2 — Forward Flight
Stage 3 — Relative Turn
Transition — AFCS Stabilization
Post-turn — Stage 2 PPO
```

3D trajectory, top view, side view, Play/Pause, scrub slider ve canlı telemetry değerleri ile aynı görevin zaman içinde nasıl geliştiği incelenebilir.

Önemli nokta: replay bir kontrol policy’si değildir; **kaydedilmiş gerçek simulator telemetry’sinin görselleştirilmesidir.**

---

# 20. Kısa teknik özet

```text
Simulator        : JSBSim
Aircraft model   : ah1s
RL algorithm     : PPO
Framework        : Stable-Baselines3 + Gymnasium
Action space     : Box(-1, 1, shape=(4,))
Core controls    : collective, longitudinal cyclic, lateral cyclic, pedal
Base observation : 12 features
Stage1 distilled : 18 features
Turn observation : 16 features, goal-conditioned
Target altitude  : ~300 ft AGL
Turn examples    : -50°, +50°, +200°, +360°
Runtime teacher  : OFF
Phase reset      : YOK — aynı FDM korunur
Post-turn logic  : AFCS transition → Stage 2 PPO
```

---

# 21. Tek cümlede proje

> **AH-1S helikopteri JSBSim üzerinde PPO tabanlı policy’lerle 300 ft’e kaldıran, ileri uçuran, mevcut heading’e göre parametrik relative turn yaptıran, dönüş sonrası dinamikleri AFCS transition ile sönümleyip yeni heading üzerinde tekrar ileri uçuşa devam ettiren kesintisiz çok-aşamalı bir RL uçuş kontrol sistemidir.**
