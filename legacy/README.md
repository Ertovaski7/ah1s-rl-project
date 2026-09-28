# Eski sistem (legacy): heading ve durma görevleri

Projenin ilk sistemi: **Stage 1** (kalkış + 300 ft hover) → **Stage 2** (ileri uçuş) → **Turn** (relative turn) ya da
**Stage 3** (300 ft ilerideki hedefte durma + hover), hepsi aynı JSBSim FDM'de. Teacher–student distillation, residual
adapter ve gated patch'lerle kuruldu. Mentor kararıyla 2026-09-22'de bu yol bırakıldı; yerini repo kökündeki tek PPO
ajanlı sistem aldı (komut → manevra → dayanıklılık → kalkış).

Kod ve modeller olduğu gibi duruyor ve çalışıyor. Model yolları bu klasöre göre olduğu için komutlar buradan çalıştırılır:

```bash
cd legacy
python validate_stage3_stop_hover.py                        # Stage 1 → 2 → 3: hedefte durma + 5 s hover
python validate_final_continuous_mission_v1.py 20 -30 75    # heading görevi (headless)
python test_turn_full_entry_v22_v21_runtime.py              # turn stack: 20 randomized entry
python visualize_final_multiturn.py 20 -30 75               # heading görevi -> GIF
```

Colab'da canlı dashboard:

```python
%cd /content/ah1s-rl-project/legacy
%run run_colab_live_heading_dashboard.py
```

`training/` altındaki scriptler kendilerini bu klasöre göre ayarlar (`python legacy/training/<script>.py`). Modellerin
SHA-256 listesi repo kökündeki `models_sha256.txt`'de (kökten: `sha256sum -c models_sha256.txt`).

Ayrıntılı doküman: ana README bölüm 1.1–1.3 ve 2–25.
