# Dayanıklılık (robustness) — koşu kanıtları (2026-09-23)

Claude'un cloud ortamında (JSBSim 1.3.1, SB3 2.9, torch 2.14 CPU, 2 çekirdek) koşturuldu. Ayrıntılar: ana README bölüm 29.

Üç model karşılaştırıldı (dosya adlarındaki etiketler):

| Etiket | Model | Collective yetkisi |
|---|---|---|
| `M5` | `models_maneuver/maneuver_M5_final.zip` (bölüm 28) | trim ± 0.25 |
| `genisletilmis` | M5 modeli, `widen_collective.py` ile yeniden ölçeklenmiş, eğitimsiz (log'larda `m5_coll045.zip`; repoda yok, komutla birebir yeniden üretilir) | trim ± 0.45 |
| `dayanikli` | `models_maneuver/maneuver_robust_final.zip` (log'larda `best_1000k.zip`, aynı dosya): genişletilmiş model + S4'te 1 M adım ince ayar | trim ± 0.45 |

| Dosya | İçerik |
|---|---|
| `robustness_<etiket>.txt`, `.json` | `evaluate_robustness.py`: 49 koşu (23 sabit senaryo + aynaları + S1 / S2'den 10 rastgele episode); özet ve koşu başına pencereler |
| `tek_komut_<etiket>.txt`, `.json` | `evaluate_maneuver_policy.py --level M5`: 16 tek komut (5–90 kt, 900 ft) |
| `seviyeler_<etiket>.txt` | `level_stats.py`: M5, S1, S2, S3'ten 100'er deterministik episode (seed 50000–50099), komut türüne göre başarı |
| `probe_collective.py`, `.txt` | Kök neden probu: 55° yatışlı 180° dönüşte collective ±0.25 / ±0.45 ile irtifa kaybı, doygunluk, rotor devri |
| `progress_rob_v5.csv`, `state_rob_v5.json`, `eval_rob_v5.csv` | `rob_v5`: genişletilmiş modelden S4 ince ayarı; eğitim ilerlemesi ve 500 bin adımda bir deterministik değerlendirme (M5 / S2 / S3, 50'şer episode, seed 90000+) |
| `eval_rob_v3.csv`, `eval_rob_v4.csv` | M5 modelinden doğrudan ince ayar denemeleri (iyileşme yok; README 29.4) |
| `fig_robustness.png` | Üstte: ±180° testere (60 kt, ayna) iki modelle — heading, yatış, collective; altta test senaryosu özeti |

Özet (bağımsız testler, M5 → dayanıklı): test senaryolarında tüm komutlar 40/49 → 44/49 (güvenli 49/49 ikisinde de),
tek komutlar 14/16 → 16/16, seviyeler M5 %74 → %87, S1 %81 → %85, S2 %77 → %82, S3 %62 → %75.
