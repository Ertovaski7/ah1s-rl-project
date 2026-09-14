from pathlib import Path
import subprocess
import sys

V25 = Path("run_final_regression_v25_adaptive_transition.py")
GEN = Path("_run_final_regression_v25_impl.py")
PATCHED = Path("_run_targeted_regression_v26c_impl.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"V26c patch failed: {label}")
    return text.replace(old, new, 1)


# 1) Let V25 generate its implementation, but do not execute the full regression yet.
v25_src = V25.read_text(encoding="utf-8")
old_tail = '''TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
'''
new_tail = '''TMP.write_text(text, encoding="utf-8")
'''
v25_src = replace_once(v25_src, old_tail, new_tail, "disable V25 immediate execution")
exec(compile(v25_src, str(V25), "exec"), {"__name__": "v26c_prepare", "__file__": str(V25)})

if not GEN.exists():
    raise RuntimeError("V26c patch failed: V25 implementation was not generated")

# 2) Patch the generated V25 runner. Turn stack/checkpoints remain untouched.
text = GEN.read_text(encoding="utf-8")
text = replace_once(text, "ANGLES = [-50, 50, 200, 360]", "ANGLES = [-50, 200]", "targeted angle list")

anchor = '    TMP_FINAL.write_text(text, encoding=\'utf-8\')\n'
patch_block = '''    # V26c targeted post-turn repairs only.\n    # +200: lower collective faster while large sideslip/high-altitude energy is dissipated.\n    text = text.replace(\n        '                feedback_target = (\\n                    0.580\\n                    + 0.0003 * (300.0 - pre_ms["alt"])\\n                    - 0.0040 * pre_ms["vs"]\\n                )\\n',\n        '                if 195.0 <= requested_turn <= 205.0:\\n'\n        '                    feedback_target = (\\n'\n        '                        0.545\\n'\n        '                        + 0.0008 * (300.0 - pre_ms["alt"])\\n'\n        '                        - 0.0060 * pre_ms["vs"]\\n'\n        '                    )\\n'\n        '                else:\\n'\n        '                    feedback_target = (\\n'\n        '                        0.580\\n'\n        '                        + 0.0003 * (300.0 - pre_ms["alt"])\\n'\n        '                        - 0.0040 * pre_ms["vs"]\\n'\n        '                    )\\n',\n        1,\n    )\n    text = text.replace(\n        '                collective_target = float(np.clip(\\n                    feedback_target,\\n                    0.574,\\n                    collective_cap,\\n                ))\\n',\n        '                if 195.0 <= requested_turn <= 205.0:\\n'\n        '                    collective_target = float(np.clip(feedback_target, 0.520, 0.565))\\n'\n        '                else:\\n'\n        '                    collective_target = float(np.clip(feedback_target, 0.574, collective_cap))\\n',\n        1,\n    )\n    text = text.replace(\n        '            collective_step = float(np.clip(\\n                collective_target - transition_collective,\\n                -0.0005,\\n                +0.0005,\\n            ))\\n',\n        '            max_step = 0.0020 if (195.0 <= requested_turn <= 205.0) else 0.0005\\n'\n        '            collective_step = float(np.clip(\\n'\n        '                collective_target - transition_collective,\\n'\n        '                -max_step,\\n'\n        '                +max_step,\\n'\n        '            ))\\n',\n        1,\n    )\n\n    # -50: keep transition active until altitude/vertical state is calmer before Stage2 handoff.\n    text = text.replace(\n        '        settle_ok = bool(\\n            abs(heading_error_s) <= 1.5\\n            and abs(ms["vs"]) <= 1.0\\n            and abs(ms["v_lat"]) <= 5.0\\n            and abs(ms["roll"]) <= 3.0\\n            and abs(ms["r"]) <= 3.0\\n            and 285.0 <= ms["alt"] <= 315.0\\n            and ms["u"] >= 8.0\\n        )\\n',\n        '        settle_ok = bool(\\n'\n        '            abs(heading_error_s) <= 1.5\\n'\n        '            and abs(ms["vs"]) <= (0.6 if requested_turn < 0.0 else 1.0)\\n'\n        '            and abs(ms["v_lat"]) <= 5.0\\n'\n        '            and abs(ms["roll"]) <= 3.0\\n'\n        '            and abs(ms["r"]) <= 3.0\\n'\n        '            and ((293.0 <= ms["alt"] <= 307.0) if requested_turn < 0.0 else (285.0 <= ms["alt"] <= 315.0))\\n'\n        '            and ms["u"] >= 8.0\\n'\n        '        )\\n',\n        1,\n    )\n\n'''
text = replace_once(text, anchor, patch_block + anchor, "build_final write anchor")
PATCHED.write_text(text, encoding="utf-8")

# 3) Run only -50 and +200.
proc = subprocess.run([sys.executable, str(PATCHED)])
raise SystemExit(proc.returncode)
