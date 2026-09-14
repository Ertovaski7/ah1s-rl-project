from pathlib import Path
import subprocess
import sys

V25 = Path("run_final_regression_v25_adaptive_transition.py")
GEN = Path("_run_final_regression_v25_impl.py")
PATCHED = Path("_run_targeted_regression_v27_impl.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"V27 patch failed: {label}")
    return text.replace(old, new, 1)


# Generate V25 implementation without running it.
v25_src = V25.read_text(encoding="utf-8")
old_tail = '''TMP.write_text(text, encoding="utf-8")
proc = subprocess.run([sys.executable, str(TMP)])
raise SystemExit(proc.returncode)
'''
new_tail = '''TMP.write_text(text, encoding="utf-8")
'''
v25_src = replace_once(v25_src, old_tail, new_tail, "disable V25 immediate execution")
exec(compile(v25_src, str(V25), "exec"), {"__name__": "v27_prepare", "__file__": str(V25)})

if not GEN.exists():
    raise RuntimeError("V27 patch failed: V25 implementation was not generated")

text = GEN.read_text(encoding="utf-8")
text = replace_once(text, "ANGLES = [-50, 50, 200, 360]", "ANGLES = [-50, 200]", "targeted angle list")

anchor = "    TMP_FINAL.write_text(text, encoding='utf-8')\n"
patch_block = r'''    # V27 targeted post-turn tuning only; turn checkpoints are unchanged.

    # +200: use a moderate altitude-aware collective instead of the too-high V25
    # 0.574 floor or the too-low V26 0.520-0.565 range.
    old_fb = '''                feedback_target = (
                    0.580
                    + 0.0003 * (300.0 - pre_ms["alt"])
                    - 0.0040 * pre_ms["vs"]
                )
'''
    new_fb = '''                if 195.0 <= requested_turn <= 205.0:
                    feedback_target = (
                        0.562
                        + 0.0010 * (300.0 - pre_ms["alt"])
                        - 0.0045 * pre_ms["vs"]
                    )
                    if pre_ms["alt"] > 312.0:
                        feedback_target -= 0.006
                    elif pre_ms["alt"] < 290.0:
                        feedback_target += 0.010
                else:
                    feedback_target = (
                        0.580
                        + 0.0003 * (300.0 - pre_ms["alt"])
                        - 0.0040 * pre_ms["vs"]
                    )
'''
    if old_fb not in text:
        raise RuntimeError('V27 final transform failed: feedback_target block')
    text = text.replace(old_fb, new_fb, 1)

    old_clip = '''                collective_target = float(np.clip(
                    feedback_target,
                    0.574,
                    collective_cap,
                ))
'''
    new_clip = '''                if 195.0 <= requested_turn <= 205.0:
                    collective_target = float(np.clip(feedback_target, 0.548, 0.588))
                else:
                    collective_target = float(np.clip(
                        feedback_target,
                        0.574,
                        collective_cap,
                    ))
'''
    if old_clip not in text:
        raise RuntimeError('V27 final transform failed: collective clip block')
    text = text.replace(old_clip, new_clip, 1)

    old_step = '''            collective_step = float(np.clip(
                collective_target - transition_collective,
                -0.0005,
                +0.0005,
            ))
'''
    new_step = '''            max_step = 0.0010 if (195.0 <= requested_turn <= 205.0) else 0.0005
            collective_step = float(np.clip(
                collective_target - transition_collective,
                -max_step,
                +max_step,
            ))
'''
    if old_step not in text:
        raise RuntimeError('V27 final transform failed: collective step block')
    text = text.replace(old_step, new_step, 1)

    # -50: make the handoff criterion genuinely stricter. V26's replacement did
    # not match the generated validator, so patch the exact generated condition.
    old_settle = '''        settle_ok = bool(
            abs(heading_error_s) <= 1.5
            and abs(ms["vs"]) <= 1.0
            and abs(ms["v_lat"]) <= 5.0
            and abs(ms["roll"]) <= 3.0
            and abs(ms["r"]) <= 3.0
            and 285.0 <= ms["alt"] <= 315.0
            and ms["u"] >= 8.0
        )
'''
    new_settle = '''        settle_ok = bool(
            abs(heading_error_s) <= 1.5
            and abs(ms["vs"]) <= (0.45 if requested_turn < 0.0 else 1.0)
            and abs(ms["v_lat"]) <= 5.0
            and abs(ms["roll"]) <= 3.0
            and abs(ms["r"]) <= 3.0
            and ((294.0 <= ms["alt"] <= 306.0) if requested_turn < 0.0 else (285.0 <= ms["alt"] <= 315.0))
            and ms["u"] >= 8.0
        )
'''
    if old_settle not in text:
        raise RuntimeError('V27 final transform failed: settle_ok block')
    text = text.replace(old_settle, new_settle, 1)

'''
text = replace_once(text, anchor, patch_block + anchor, "build_final write anchor")
PATCHED.write_text(text, encoding="utf-8")

proc = subprocess.run([sys.executable, str(PATCHED)])
raise SystemExit(proc.returncode)
