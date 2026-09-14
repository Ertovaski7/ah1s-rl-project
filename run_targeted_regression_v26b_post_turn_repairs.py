from pathlib import Path
import subprocess
import sys

SRC = Path("run_final_regression_v25_adaptive_transition.py")
TMP = Path("_run_targeted_regression_v26b_impl.py")

text = SRC.read_text(encoding="utf-8")

# Inject V26 transforms into V25 immediately before its cosmetic-label section.
marker = "# Cosmetic label only.\n"
if marker not in text:
    raise RuntimeError("V26b patch failed: cosmetic marker not found")

extra = r'''
# V26b: targeted post-turn repairs, added to the generated V24 runner's build_final().
v26_anchor = """    text = text.replace(\n        '    transition_collective = 0.574 if plus50_feedback else 0.590\\n',\n        '    transition_collective = (0.590 if requested_turn < 0.0 else (0.574 if plus50_feedback else 0.590))\\n',\n        1,\n    )\n"""
if v26_anchor not in text:
    raise RuntimeError("V26b patch failed: V25 transition anchor not found")

v26_inject = v26_anchor + r'''
    # +200: much lower collective while large sideslip is being dissipated.
    text = text.replace(
        '                feedback_target = (\n                    0.580\n                    + 0.0003 * (300.0 - pre_ms["alt"])\n                    - 0.0040 * pre_ms["vs"]\n                )\n',
        '                if 195.0 <= requested_turn <= 205.0:\n'
        '                    feedback_target = (\n'
        '                        0.545\n'
        '                        + 0.0008 * (300.0 - pre_ms["alt"])\n'
        '                        - 0.0060 * pre_ms["vs"]\n'
        '                    )\n'
        '                else:\n'
        '                    feedback_target = (\n'
        '                        0.580\n'
        '                        + 0.0003 * (300.0 - pre_ms["alt"])\n'
        '                        - 0.0040 * pre_ms["vs"]\n'
        '                    )\n',
        1,
    )

    text = text.replace(
        '                collective_target = float(np.clip(\n                    feedback_target,\n                    0.574,\n                    collective_cap,\n                ))\n',
        '                if 195.0 <= requested_turn <= 205.0:\n'
        '                    collective_target = float(np.clip(feedback_target, 0.520, 0.565))\n'
        '                else:\n'
        '                    collective_target = float(np.clip(feedback_target, 0.574, collective_cap))\n',
        1,
    )

    text = text.replace(
        '            collective_step = float(np.clip(\n                collective_target - transition_collective,\n                -0.0005,\n                +0.0005,\n            ))\n',
        '            max_step = 0.0020 if (195.0 <= requested_turn <= 205.0) else 0.0005\n'
        '            collective_step = float(np.clip(\n'
        '                collective_target - transition_collective,\n'
        '                -max_step,\n'
        '                +max_step,\n'
        '            ))\n',
        1,
    )

    # -50: do not hand back to Stage2 while still low; require a quieter vertical state.
    text = text.replace(
        '        settle_ok = bool(\n            abs(heading_error_s) <= 1.5\n            and abs(ms["vs"]) <= 1.0\n            and abs(ms["v_lat"]) <= 5.0\n            and abs(ms["roll"]) <= 3.0\n            and abs(ms["r"]) <= 3.0\n            and 285.0 <= ms["alt"] <= 315.0\n            and ms["u"] >= 8.0\n        )\n',
        '        settle_ok = bool(\n'
        '            abs(heading_error_s) <= 1.5\n'
        '            and abs(ms["vs"]) <= (0.6 if requested_turn < 0.0 else 1.0)\n'
        '            and abs(ms["v_lat"]) <= 5.0\n'
        '            and abs(ms["roll"]) <= 3.0\n'
        '            and abs(ms["r"]) <= 3.0\n'
        '            and ((293.0 <= ms["alt"] <= 307.0) if requested_turn < 0.0 else (285.0 <= ms["alt"] <= 315.0))\n'
        '            and ms["u"] >= 8.0\n'
        '        )\n',
        1,
    )
'''
text = text.replace(v26_anchor, v26_inject, 1)

# Only run the two remaining failing cases in this pass.
text = text.replace("ANGLES = [-50, 50, 200, 360]", "ANGLES = [-50, 200]", 1)
'''

text = text.replace(marker, extra + marker, 1)
TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
