# training/ klasöründen çalıştırılabilmesi için: repo kökünü import yoluna ekle
# ve çalışma dizinini köke al (model/sonuç yolları köke göredir).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _REPO_ROOT)
_os.chdir(_REPO_ROOT)

from pathlib import Path

src = Path("training/train_turn_full_entry_v7_gated_200_patch.py")
dst = Path("training/train_turn_full_entry_v13_gated_50_patch.py")

if not src.exists():
    raise FileNotFoundError(src)

text = src.read_text(encoding="utf-8")
repls = [
    ("V7_GATED_200", "V13_GATED_50"),
    ("V7_GATED", "V13_GATED"),
    ("V7 ", "V13 "),
    ("V7_", "V13_"),
    ("TARGET_200_NORM", "TARGET_50_NORM"),
    ("200.0 / 360.0", "50.0 / 360.0"),
    ("target_turn_deg=200.0", "target_turn_deg=50.0"),
    ("200.0 - env.cumulative_turn_deg", "50.0 - env.cumulative_turn_deg"),
    ("200.0 / 1.10", "50.0 / 1.10"),
    ("collect_shadow_200", "collect_shadow_50"),
    ("GATED_200_PATCH", "GATED_50_PATCH"),
    ("+200", "+50"),
]

for a, b in repls:
    text = text.replace(a, b)

# Make output filenames unique for the +50 patch.
text = text.replace(
    'AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt',
    'AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH.pt'
)
text = text.replace(
    'AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH_BEST.pt',
    'AH1S_TURN_FULL_ENTRY_V13_GATED_50_PATCH_BEST.pt'
)

dst.write_text(text, encoding="utf-8")
print("CREATED:", dst)
print("NEXT=RUN_V13")
