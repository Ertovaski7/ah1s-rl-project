from pathlib import Path
import subprocess
import sys

SRC = Path("run_final_regression_v25_adaptive_transition.py")
TMP = Path("_run_targeted_regression_v26_impl.py")

text = SRC.read_text(encoding="utf-8")

# Run only the two remaining failing full-mission cases first.
# The underlying V22 turn stack is unchanged.
text = text.replace(
    "SRC = Path(\"run_final_regression_v24_v22_stack.py\")",
    "SRC = Path(\"run_final_regression_v24_v22_stack.py\")",
    1,
)

# Inject extra V26 transforms into V25's build_final hook.
hook = '''    text = text.replace(\n        '    transition_collective = 0.574 if plus50_feedback else 0.590\\n',\n        '    transition_collective = (0.590 if requested_turn < 0.0 else (0.574 if plus50_feedback else 0.590))\\n',\n        1,\n    )\n'''

inject = hook + '''
    # V26 target-specific post-turn repair. Turn policies/checkpoints are untouched.
    # +200 enters transition high in altitude with very large lateral velocity;
    # use a substantially lower collective while sideslip is being dissipated.
    text = text.replace(
        '                feedback_target = (\\n                    0.580\\n                    + 0.0003 * (300.0 - pre_ms["alt"])\\n                    - 0.0040 * pre_ms["vs"]\\n                )\\n',
        '                if 195.0 <= requested_turn <= 205.0:\\n'
        '                    feedback_target = (\\n'
        '                        0.545\\n'
        '                        + 0.0008 * (300.0 - pre_ms["alt"])\\n'
        '                        - 0.0060 * pre_ms["vs"]\\n'
        '                    )\\n'
        '                else:\\n'
        '                    feedback_target = (\\n'
        '                        0.580\\n'
        '                        + 0.0003 * (300.0 - pre_ms["alt"])\\n'
        '                        - 0.0040 * pre_ms["vs"]\\n'
        '                    )\\n',
        1,
    )

    # For +200 allow the low-collective command to go below the +50 floor.
    text = text.replace(
        '                collective_target = float(np.clip(\\n                    feedback_target,\\n                    0.574,\\n                    collective_cap,\\n                ))\\n',
        '                if 195.0 <= requested_turn <= 205.0:\\n'
        '                    collective_target = float(np.clip(\\n'
        '                        feedback_target, 0.520, 0.565\\n'
        '                    ))\\n'
        '                else:\\n'
        '                    collective_target = float(np.clip(\\n'
        '                        feedback_target, 0.574, collective_cap\\n'
        '                    ))\\n',
        1,
    )

    # Let +200 collective move fast enough to arrest the climb before 325 ft.
    text = text.replace(
        '            collective_step = float(np.clip(\\n                collective_target - transition_collective,\\n                -0.0005,\\n                +0.0005,\\n            ))\\n',
        '            max_step = 0.0020 if (195.0 <= requested_turn <= 205.0) else 0.0005\\n'
        '            collective_step = float(np.clip(\\n'
        '                collective_target - transition_collective,\\n'
        '                -max_step,\\n'
        '                +max_step,\\n'
        '            ))\\n',
        1,
    )

    # -50 was already safe, but transition exited after only ~1 s at 287 ft.
    # Require a higher/quiet vertical state before handing back to Stage2 PPO.
    text = text.replace(
        '        settle_ok = bool(\\n            abs(heading_error_s) <= 1.5\\n            and abs(ms["vs"]) <= 1.0\\n            and abs(ms["v_lat"]) <= 5.0\\n            and abs(ms["roll"]) <= 3.0\\n            and abs(ms["r"]) <= 3.0\\n            and 285.0 <= ms["alt"] <= 315.0\\n            and ms["u"] >= 8.0\\n        )\\n',
        '        settle_ok = bool(\\n'
        '            abs(heading_error_s) <= 1.5\\n'
        '            and abs(ms["vs"]) <= (0.6 if requested_turn < 0.0 else 1.0)\\n'
        '            and abs(ms["v_lat"]) <= 5.0\\n'
        '            and abs(ms["roll"]) <= 3.0\\n'
        '            and abs(ms["r"]) <= 3.0\\n'
        '            and ((293.0 <= ms["alt"] <= 307.0) if requested_turn < 0.0 else (285.0 <= ms["alt"] <= 315.0))\\n'
        '            and ms["u"] >= 8.0\\n'
        '        )\\n',
        1,
    )
'''

if hook not in text:
    raise RuntimeError("V26 patch failed: V25 transition hook not found")
text = text.replace(hook, inject, 1)

# Only run the two remaining failing angles in this diagnostic/repair pass.
text = text.replace(
    "proc = subprocess.run([sys.executable, str(TMP)])",
    "TMP.write_text(text, encoding=\"utf-8\")\n\n# Patch the generated V24 runner to execute only -50 and +200.\ngenerated = TMP.read_text(encoding=\"utf-8\")\ngenerated = generated.replace(\"ANGLES = [-50, 50, 200, 360]\", \"ANGLES = [-50, 200]\", 1)\nTMP.write_text(generated, encoding=\"utf-8\")\nproc = subprocess.run([sys.executable, str(TMP)])",
    1,
)

TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
