# Kalkış / hover / iniş — kanıtlar (README bölüm 30)

| Dosya | İçerik |
|---|---|
| `probe_takeoff.py`, `probe_takeoff.txt` | Ölçümler (policy yok): açık döngü kalkış, PID ile hover trimleri (8–1000 ft), 300 ft'te adım cevapları, ağırlık / CG trimleri |
| `fig_takeoff.py`, `fig_takeoff.png` | Şekil: açık döngü ↔ ajan kalkışı, dört kumanda (300 ft kalkış), iniş (kızak yüksekliği, dikey hız, ağırlık, collective) |
| `senaryolar_final.txt/.json` | 28 sabit senaryo, sonuç modeli (`models_takeoff/takeoff_final.zip`) |
| `senaryolar_K6.txt/.json` | Aynı senaryolar, iniş eğitiminden önceki model (`to_v1`, K6'yı geçen ağ) |
| `seviyeler_final.txt/.json` | K1…K9 (+K7a), seviye başına 40 deterministik episode, seed 70000+ (sonuç modeli) |
| `seviyeler_K6.txt/.json` | K5, K7, K8, K9 aynı şekilde (K6 modeli) |
| `progress_to_*.csv`, `eval_to_*.csv`, `curriculum_state_to_*.json` | Son modelin soyundaki ana koşular: `to_v1` (K1–K6), `to_v9` (K7a, K7), `to_v15` (K8), `to_v18` (K9 ince ayar; 2 M adımdaki ara model = sonuç modeli) |

Değerlendirmeler Claude'un cloud ortamında (JSBSim 1.3.1, SB3 2.9, torch CPU, 2 çekirdek), tek seed.
