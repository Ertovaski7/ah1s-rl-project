from pathlib import Path

SOURCE = Path("validate_plus50_v22_adaptive_transition.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old = '''    while settle_elapsed < transition_max_s:
        if plus50_adaptive:
            pre_ms = physical_metrics(fdm)
'''
new = '''    next_diag_t = 0.0
    while settle_elapsed < transition_max_s:
        if plus50_adaptive:
            pre_ms = physical_metrics(fdm)
'''
if old not in text:
    raise RuntimeError("timeline diagnostic: loop start not found")
text = text.replace(old, new, 1)

old2 = '''            transition_collective += collective_step

        fdm[\\"fcs/collective-cmd-norm\\"] = transition_collective
'''
new2 = '''            transition_collective += collective_step

            if settle_elapsed + 1e-9 >= next_diag_t:
                print(
                    "V22 TRANSITION TRACE | "
                    f"t={settle_elapsed:5.2f}s | alt={pre_ms['alt']:7.2f}ft | "
                    f"vs={pre_ms['vs']:+6.2f}fps | v_lat={pre_ms['v_lat']:+6.2f}fps | "
                    f"target={collective_target:.4f} | collective={transition_collective:.4f}"
                )
                next_diag_t += 1.0

        fdm[\\"fcs/collective-cmd-norm\\"] = transition_collective
'''
if old2 not in text:
    raise RuntimeError("timeline diagnostic: collective application block not found")
text = text.replace(old2, new2, 1)

text = text.replace(
    "V22 +50 ADAPTIVE TRANSITION -> SMOOTH STAGE2 HANDOFF",
    "V22 +50 ADAPTIVE TRANSITION — TIMELINE DIAGNOSTIC",
)

exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
