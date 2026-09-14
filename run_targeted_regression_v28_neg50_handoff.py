from pathlib import Path
import subprocess
import sys

SRC = Path("run_targeted_regression_v26c_post_turn_repairs.py")
TMP = Path("_run_targeted_regression_v28_impl.py")

text = SRC.read_text(encoding="utf-8")

# Run only -50 in this pass.
text = text.replace(
    'text = replace_once(text, "ANGLES = [-50, 50, 200, 360]", "ANGLES = [-50, 200]", "targeted angle list")',
    'text = replace_once(text, "ANGLES = [-50, 50, 200, 360]", "ANGLES = [-50]", "targeted angle list")',
    1,
)

# Patch the generated final validator at a very stable control-flow point.
# For -50, do not leave the AFCS transition after a 1 s settle while still
# down at ~287 ft. Keep the transition running until altitude is at least 294 ft.
anchor = "    TMP_FINAL.write_text(text, encoding='utf-8')\n"
extra = '''    old_break = '        if settle_safety or settle_hold >= 1.0:\\n            break\\n'\n    new_break = (\n        '        if settle_safety or (settle_hold >= 1.0 and '\n        '(requested_turn >= 0.0 or ms[\"alt\"] >= 294.0)):\\n'\n        '            break\\n'\n    )\n    if old_break not in text:\n        raise RuntimeError('V28 final transform failed: transition break condition')\n    text = text.replace(old_break, new_break, 1)\n\n'''
if anchor not in text:
    raise RuntimeError("V28 patch failed: build_final write anchor not found")
text = text.replace(anchor, extra + anchor, 1)

text = text.replace(
    'PATCHED = Path("_run_targeted_regression_v26c_impl.py")',
    'PATCHED = Path("_run_targeted_regression_v28_generated.py")',
    1,
)

TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
