from pathlib import Path
import subprocess
import sys

V25 = Path("run_final_regression_v25_adaptive_transition.py")
GEN = Path("_run_final_regression_v25_impl.py")
PATCHED = Path("_run_targeted_regression_v29_impl.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"V29 patch failed: {label}")
    return text.replace(old, new, 1)


# Generate the proven V25 runner without executing it yet.
v25_src = V25.read_text(encoding="utf-8")
old_tail = (
    'TMP.write_text(text, encoding="utf-8")\n'
    'proc = subprocess.run([sys.executable, str(TMP)])\n'
    'raise SystemExit(proc.returncode)\n'
)
new_tail = 'TMP.write_text(text, encoding="utf-8")\n'
v25_src = replace_once(v25_src, old_tail, new_tail, "disable V25 immediate execution")
exec(compile(v25_src, str(V25), "exec"), {"__name__": "v29_prepare", "__file__": str(V25)})

if not GEN.exists():
    raise RuntimeError("V29 patch failed: V25 implementation was not generated")

text = GEN.read_text(encoding="utf-8")
text = replace_once(
    text,
    "ANGLES = [-50, 50, 200, 360]",
    "ANGLES = [-50]",
    "targeted angle list",
)

# Patch the RAW V19 wrapper before it is redirected/materialized. This avoids
# fragile matching against the already-generated final validator.
old_build = (
    "def build_transition() -> None:\n"
    "    text = V19.read_text(encoding='utf-8')\n"
    "    text = must_replace(\n"
)
insert = (
    "def build_transition() -> None:\n"
    "    text = V19.read_text(encoding='utf-8')\n"
    "    old_break = '        if settle_safety or settle_hold >= 1.0:\\n            break\\n'\n"
    "    new_break = ('        if settle_safety or (settle_hold >= 1.0 and '"
    "                 '(requested_turn >= 0.0 or ms[\\\"alt\\\"] >= 294.0)):\\n'"
    "                 '            break\\n')\n"
    "    if old_break not in text:\n"
    "        raise RuntimeError('V29 transform failed: raw V19 transition break condition')\n"
    "    text = text.replace(old_break, new_break, 1)\n"
    "    text = must_replace(\n"
)
text = replace_once(text, old_build, insert, "build_transition hook")

PATCHED.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(PATCHED)])
raise SystemExit(proc.returncode)
