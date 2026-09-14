from pathlib import Path
import subprocess
import sys

V26C = Path("run_targeted_regression_v26c_post_turn_repairs.py")
V26C_GEN = Path("_run_targeted_regression_v26c_impl.py")
PATCHED = Path("_run_targeted_regression_v28b_impl.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"V28b patch failed: {label}")
    return text.replace(old, new, 1)


# 1) Let V26c generate its runner, but do not execute it yet.
src = V26C.read_text(encoding="utf-8")
old_tail = '''PATCHED.write_text(text, encoding="utf-8")

# 3) Run only -50 and +200.
proc = subprocess.run([sys.executable, str(PATCHED)])
raise SystemExit(proc.returncode)
'''
new_tail = '''PATCHED.write_text(text, encoding="utf-8")
'''
src = replace_once(src, old_tail, new_tail, "disable V26c immediate execution")
exec(compile(src, str(V26C), "exec"), {"__name__": "v28b_prepare", "__file__": str(V26C)})

if not V26C_GEN.exists():
    raise RuntimeError("V28b patch failed: V26c generated runner not found")

# 2) Patch the generated runner itself. This is the level that contains
# build_final() and the TMP_FINAL.write_text anchor.
text = V26C_GEN.read_text(encoding="utf-8")
text = replace_once(text, "ANGLES = [-50, 200]", "ANGLES = [-50]", "targeted angle list")

anchor = "    TMP_FINAL.write_text(text, encoding='utf-8')\n"
extra = '''    # V28b: -50 must not leave transition merely because the 1 s settle hold
    # is satisfied while still down near 287 ft. Require >=294 ft before handoff.
    old_break = '        if settle_safety or settle_hold >= 1.0:\\n            break\\n'
    new_break = (
        '        if settle_safety or (settle_hold >= 1.0 and '
        '(requested_turn >= 0.0 or ms["alt"] >= 294.0)):\\n'
        '            break\\n'
    )
    if old_break not in text:
        raise RuntimeError('V28b final transform failed: transition break condition')
    text = text.replace(old_break, new_break, 1)

'''
text = replace_once(text, anchor, extra + anchor, "build_final write anchor")

PATCHED.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(PATCHED)])
raise SystemExit(proc.returncode)
