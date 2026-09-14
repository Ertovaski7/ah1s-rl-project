from pathlib import Path
import subprocess
import sys

SRC = Path("run_final_regression_v24_v22_stack.py")
TMP = Path("_run_final_regression_v24b_impl.py")

text = SRC.read_text(encoding="utf-8")
old = "    old_v10_path = 'V10_NEG_PATCH_PATH = Path(\"models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt\")\\n'\n"
new = "    old_v10_path = \"V10_NEG_PATCH_PATH = Path('models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt')\\n\"\n"
if old not in text:
    raise RuntimeError("V24b patch failed: old_v10_path definition not found")
text = text.replace(old, new, 1)
TMP.write_text(text, encoding="utf-8")

proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
