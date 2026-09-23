# Komut curriculum'u — ilk koşuların kanıtları (2026-09-22)

Claude'un cloud ortamında (JSBSim 1.3.1, SB3 2.9, torch 2.14 CPU, 2 çekirdek) koşturuldu. Ayrıntılar: ana README bölüm 26.

| Dosya | İçerik |
|---|---|
| `progress_cmd_v1.csv`, `state_cmd_v1.json` | v1 ayarı (`--env-config v1`): H1 → R1, 2.24 M adım, 37.7 dk |
| `progress_cmd_v2_run1.csv`, `state_cmd_v2_run1.json` | v2, sıfırdan: H1 → H6 (sonrasında V/A seviyelerinde unutma görüldü, koşu A4'te durduruldu) |
| `progress_cmd_v2_run2.csv`, `state_cmd_v2_run2.json` | v2, H6 modelinden devam: V1 → R1 (ince ayar lr 1e-4 + KL 0.02, eksen başına başarı kapısı) |
| `final_eval_v2.json` | 110 testlik doğrulama (v2 son, v2 H1, v1 son, a = 0) |
| `fig_mission_v1.png`, `fig_mission_v2.png` | tek uçuşta 6 ardışık komut; v1'de elevator titreşimi ve dönüşte hız / irtifa kaçırma, v2'de yok |

Modeller `../../models_command_curriculum/`:

- `v2_level_00_H1.zip`: PPO sıfırdan, yalnızca ±5° seviyesi (2.3 dk).
- `v2_level_05_H6.zip`: H1 → H6 sonu (run 2'nin başlangıcı).
- `v2_R1_final.zip`: son model, önerilen.
- `v1_R1_final.zip`: v1 son modeli, karşılaştırma için (kumanda titreşimi örneği).
