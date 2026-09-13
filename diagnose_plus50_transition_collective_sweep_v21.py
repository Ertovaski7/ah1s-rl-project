from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path("validate_full_mission_v19_360_transition_trim_0590.py")
CANDIDATES = [0.540, 0.550, 0.560, 0.570, 0.580, 0.590]
ANGLE = 50
LOG_DIR = Path("diagnostic_plus50_sweep_v21")


def first_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.strip()
    return ""


def make_variant(base_text: str, collective: float) -> str:
    text = base_text

    # V19 contains the generated validator body inside a triple-quoted
    # replacement string. Match only stable substrings so newline escaping
    # cannot break the sweep script.
    old_gate = "if abs(requested_turn) >= 300.0:"
    new_gate = "if requested_turn > 0.0:"
    if old_gate not in text:
        raise RuntimeError("V21 sweep: transition gate not found in V19 source")
    text = text.replace(old_gate, new_gate, 1)

    old_collective = "transition_collective = 0.590"
    new_collective = f"transition_collective = {collective:.3f}"
    if old_collective not in text:
        raise RuntimeError("V21 sweep: transition collective scalar not found in V19 source")
    text = text.replace(old_collective, new_collective, 1)

    text = text.replace(
        "V19 DIRECT - 360 TRANSITION TRIM 0.590 -> SMOOTH STAGE2 HANDOFF",
        f"V21 +50 DIAGNOSTIC - TRANSITION COLLECTIVE {collective:.3f}",
    )
    text = text.replace(
        "360 transition collective=0.590; max settle=20.0s.",
        f"+50 diagnostic transition collective={collective:.3f}; max settle=20.0s.",
    )
    return text


def main() -> int:
    if not BASE.exists():
        print(f"ERROR: base validator not found: {BASE}")
        return 2

    base_text = BASE.read_text(encoding="utf-8")
    LOG_DIR.mkdir(exist_ok=True)

    print("=" * 120)
    print("V21 +50 TRANSITION COLLECTIVE SWEEP")
    print("Fresh full mission per candidate; validation thresholds unchanged.")
    print("Only the +50 post-turn transition collective is varied.")
    print("Candidates:", ", ".join(f"{x:.3f}" for x in CANDIDATES))
    print("=" * 120)

    results = []

    for collective in CANDIDATES:
        print("\n" + "=" * 120)
        print(f"+50 SWEEP | transition_collective={collective:.3f}")
        print("=" * 120)

        variant = make_variant(base_text, collective)
        tmp_path = Path(tempfile.gettempdir()) / f"validate_plus50_c{collective:.3f}.py"
        tmp_path.write_text(variant, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            input=f"{ANGLE}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
        )
        output = proc.stdout

        log_path = LOG_DIR / f"plus50_collective_{collective:.3f}.log"
        log_path.write_text(output, encoding="utf-8")

        turn = first_line(output, "TURN CAPTURE:")
        transition = first_line(output, "POST-TURN TRANSITION TRIM |")
        post = first_line(output, "POST-TURN FORWARD:")
        full = first_line(output, "FULL MISSION PASS:")
        safety = first_line(output, "SAFETY FAILURE:")

        print(turn or "TURN CAPTURE: not found")
        print(transition or "POST-TURN TRANSITION TRIM: not found")
        print(post or "POST-TURN FORWARD: not found")
        print(safety or "SAFETY FAILURE: not found")
        print(full or "FULL MISSION PASS: not found")

        passed = proc.returncode == 0 and full == "FULL MISSION PASS: True"
        results.append((collective, passed, transition, post, safety, str(log_path)))

        if passed:
            print(f"\nPASS FOUND at collective={collective:.3f}; stopping sweep early.")
            break

    print("\n" + "=" * 120)
    print("V21 +50 SWEEP SUMMARY")
    print("=" * 120)
    print(f"{'COLLECTIVE':>12} | {'RESULT':^8} | TRANSITION")
    print("-" * 120)
    for collective, passed, transition, post, safety, log in results:
        print(f"{collective:12.3f} | {'PASS' if passed else 'FAIL':^8} | {transition or 'not found'}")
        print(f"{'':12} | {'':8} | {post or 'post not found'}")
        print(f"{'':12} | {'':8} | {safety or 'safety not found'} | {log}")

    winners = [x for x in results if x[1]]
    if winners:
        print(f"\nBEST VERIFIED +50 COLLECTIVE: {winners[0][0]:.3f}")
        return 0

    print("\nNO +50 FULL PASS FOUND IN THIS SWEEP.")
    print("Use the transition telemetry above to decide the next targeted change; do not loosen thresholds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
