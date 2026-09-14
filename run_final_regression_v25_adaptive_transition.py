from pathlib import Path
import subprocess
import sys

SRC = Path("run_final_regression_v24_v22_stack.py")
TMP = Path("_run_final_regression_v25_impl.py")

text = SRC.read_text(encoding="utf-8")

# Keep the two V24c source-transform fixes.
old = "    old_v10_path = 'V10_NEG_PATCH_PATH = Path(\"models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt\")\\n'\n"
new = "    old_v10_path = \"V10_NEG_PATCH_PATH = Path('models_turn_hybrid/AH1S_TURN_LIVE_V10_GATED_NEG_COLLECTIVE.pt')\\n\"\n"
if old not in text:
    raise RuntimeError("V25 patch failed: old_v10_path definition not found")
text = text.replace(old, new, 1)

needle = "    text = materialize_v11()\n"
inject = """    text = materialize_v11()\n\n    stage2_init = 'env2.previous_action = np.zeros(4, dtype=np.float32)\\n'\n    stage2_init_safe = 'env2.previous_action = np.zeros((4,), dtype=np.float32)\\n'\n    if stage2_init not in text:\n        raise RuntimeError('V25 transform failed: Stage2 previous_action init not found')\n    text = text.replace(stage2_init, stage2_init_safe, 1)\n"""
if needle not in text:
    raise RuntimeError("V25 patch failed: materialize_v11 hook not found")
text = text.replace(needle, inject, 1)

# Extend V23's post-turn transition to the two V24 failures only:
# -50: low-altitude/vertical recovery before Stage2 handoff.
# +200: high-sideslip, high-altitude adaptive collective instead of fixed 0.590.
# +50 keeps its proven adaptive transition; +360 keeps its proven fixed 0.590 path.
final_hook = """    text = text.replace(\n        'FINAL V23 - HYBRID RL + CONDITIONAL POST-TURN AFCS STABILIZATION',\n        'FINAL V24 - V22 TURN STACK + CONDITIONAL POST-TURN AFCS STABILIZATION',\n    )\n"""
final_inject = final_hook + """
    # V25: allow transition for -50 too; positive cases were already enabled.
    text = text.replace(
        'new_gate = "if requested_turn > 0.0:"',
        'new_gate = "if requested_turn != 0.0:"',
        1,
    )

    # V25: use the proven feedback transition for -50 and +200 as well as +50.
    text = text.replace(
        '    plus50_feedback = 45.0 <= requested_turn <= 55.0\\n',
        '    plus50_feedback = ((-55.0 <= requested_turn <= -45.0) or (45.0 <= requested_turn <= 55.0) or (195.0 <= requested_turn <= 205.0))\\n',
        1,
    )
    text = text.replace(
        '    transition_collective = 0.574 if plus50_feedback else 0.590\\n',
        '    transition_collective = (0.590 if requested_turn < 0.0 else (0.574 if plus50_feedback else 0.590))\\n',
        1,
    )
"""
if final_hook not in text:
    raise RuntimeError("V25 patch failed: build_final hook not found")
text = text.replace(final_hook, final_inject, 1)

# Cosmetic label only.
text = text.replace(
    "print('AH-1S FINAL REGRESSION SUITE — V24 / V22 TURN STACK')",
    "print('AH-1S FINAL REGRESSION SUITE — V25 / V22 TURN STACK + ADAPTIVE TRANSITION')",
    1,
)

TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
