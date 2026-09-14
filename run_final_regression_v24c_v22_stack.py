from pathlib import Path
import subprocess
import sys

SRC = Path("run_final_regression_v24_v22_stack.py")
TMP = Path("_run_final_regression_v24c_impl.py")

text = SRC.read_text(encoding="utf-8")

# Fix 1: generated V11 uses single quotes for the V10 path.
old = "    old_v10_path = 'V10_NEG_PATCH_PATH = Path(\"models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt\")\\n'\n"
new = "    old_v10_path = \"V10_NEG_PATCH_PATH = Path('models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt')\\n\"\n"
if old not in text:
    raise RuntimeError("V24c patch failed: old_v10_path definition not found")
text = text.replace(old, new, 1)

# Fix 2: V19's historical source transform replaces the first exact
# `env2.previous_action = np.zeros(4, dtype=np.float32)` occurrence with
# `handoff_action.copy()`. In the materialized V11 validator the first
# occurrence is the Stage2 INITIAL attach, before handoff_action exists.
# Make only that first occurrence textually distinct but semantically identical,
# so V19's transform reaches the intended post-turn handoff occurrence.
needle = "    text = materialize_v11()\n"
inject = """    text = materialize_v11()\n\n    stage2_init = 'env2.previous_action = np.zeros(4, dtype=np.float32)\\n'\n    stage2_init_safe = 'env2.previous_action = np.zeros((4,), dtype=np.float32)\\n'\n    if stage2_init not in text:\n        raise RuntimeError('V24c transform failed: Stage2 previous_action init not found')\n    text = text.replace(stage2_init, stage2_init_safe, 1)\n"""
if needle not in text:
    raise RuntimeError("V24c patch failed: materialize_v11 hook not found")
text = text.replace(needle, inject, 1)

TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
