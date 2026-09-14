from pathlib import Path
import subprocess
import sys

SRC = Path("build_interactive_replay_current_final.py")
TMP = Path("_build_interactive_replay_current_final_v2_impl.py")

text = SRC.read_text(encoding="utf-8")

old_open = "text += r'''\n\nfrom pathlib import Path as _ReplayPath\n"
new_open = 'text += r"""\n\nfrom pathlib import Path as _ReplayPath\n'
if old_open not in text:
    raise RuntimeError("Replay v2 patch failed: outer replay block opening quote not found")
text = text.replace(old_open, new_open, 1)

old_close = "else:\n    print(\"WARNING: no replay telemetry recorded\")\n'''\n\nprint(\"=\" * 120)\n"
new_close = 'else:\n    print("WARNING: no replay telemetry recorded")\n"""\n\nprint("=" * 120)\n'
if old_close not in text:
    raise RuntimeError("Replay v2 patch failed: outer replay block closing quote not found")
text = text.replace(old_close, new_close, 1)

TMP.write_text(text, encoding="utf-8")

# Fail fast on syntax before starting JSBSim.
compile(text, str(TMP), "exec")

proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
