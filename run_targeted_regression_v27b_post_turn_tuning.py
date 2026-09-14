from pathlib import Path
import subprocess
import sys

SRC = Path("run_targeted_regression_v26c_post_turn_repairs.py")
TMP = Path("_run_targeted_regression_v27b_impl.py")

text = SRC.read_text(encoding="utf-8")

replacements = [
    ("0.545\\\\n", "0.562\\\\n", "feedback base"),
    ("+ 0.0008 * (300.0 - pre_ms[\"alt\"])\\\\n", "+ 0.0010 * (300.0 - pre_ms[\"alt\"])\\\\n", "altitude gain"),
    ("- 0.0060 * pre_ms[\"vs\"]\\\\n", "- 0.0045 * pre_ms[\"vs\"]\\\\n", "vertical-speed gain"),
    ("float(np.clip(feedback_target, 0.520, 0.565))", "float(np.clip(feedback_target, 0.548, 0.588))", "collective clip"),
    ("max_step = 0.0020 if (195.0 <= requested_turn <= 205.0) else 0.0005", "max_step = 0.0010 if (195.0 <= requested_turn <= 205.0) else 0.0005", "collective slew"),
    ("abs(ms[\"vs\"]) <= (0.6 if requested_turn < 0.0 else 1.0)", "abs(ms[\"vs\"]) <= (0.45 if requested_turn < 0.0 else 1.0)", "negative-turn VS criterion"),
    ("((293.0 <= ms[\"alt\"] <= 307.0) if requested_turn < 0.0 else (285.0 <= ms[\"alt\"] <= 315.0))", "((294.0 <= ms[\"alt\"] <= 306.0) if requested_turn < 0.0 else (285.0 <= ms[\"alt\"] <= 315.0))", "negative-turn altitude criterion"),
]

for old, new, label in replacements:
    if old not in text:
        raise RuntimeError(f"V27b patch failed: {label}")
    text = text.replace(old, new, 1)

# Keep V26c mechanics, but write/run through a separate temporary implementation.
text = text.replace(
    'PATCHED = Path("_run_targeted_regression_v26c_impl.py")',
    'PATCHED = Path("_run_targeted_regression_v27b_generated.py")',
    1,
)

TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
