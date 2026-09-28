# Manevra curriculum'u — koşu kanıtları (2026-09-23)

Claude'un cloud ortamında (JSBSim 1.3.1, SB3 2.9, torch 2.14 CPU, 2 çekirdek) koşturuldu. Ayrıntılar: ana README bölüm 28.

| Dosya | İçerik |
|---|---|
| `progress_man_v1.csv`, `state_man_v1.json` | `man_v1`: PPO sıfırdan, M1 → M5, 1.68 M adım, 33.2 dk (γ 0.995, M3'ten itibaren lr 1e-4 + KL 0.02) |
| `eval_M5.txt`, `eval_M5.json` | `evaluate_maneuver_policy.py --level M5 --compare-old`: 16 tek komut (5–90 kt) 14/16 zamanında + eski komut modeliyle karşılaştırma |
| `fig_old_vs_maneuver.png` | aynı komut (Δψ +90°, Δh +100 ft; 900 ft, 15 ft/s): eski komut modeli ve manevra modeli — heading / irtifa, yatış / yunuslama, yaw hızı / dikey hız |

Model: `../../models_maneuver/maneuver_M5_final.zip` (= `man_v1/models/level_04_M5.zip`, M5'i %80 ile geçen ağ).

Not: M5'i geçen modelden M5'te ~380 bin adım daha ince ayar denendi (lr 1e-4, KL 0.02); eğitim başarısı %64–74'te kaldı,
iyileşme görülmediği için durduruldu ve kullanılmadı.
