from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VALIDATOR = Path("validate_full_mission_v20_positive_transition_trim.py")
ANGLES = [-50, 50, 200, 360]
LOG_DIR = Path("final_regression_logs_v20")


def extract_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.strip()
    return ""


def run_case(angle: int) -> dict:
    print("\n" + "=" * 120)
    print(f"FINAL REGRESSION V20 | requested_turn={angle:+d} deg")
    print("=" * 120)

    proc = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        input=f"{angle}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = proc.stdout
    print(output, end="" if output.endswith("\n") else "\n")

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"turn_{angle:+d}.log"
    log_path.write_text(output, encoding="utf-8")

    full_pass_line = extract_line(output, "FULL MISSION PASS:")
    criterion_line = extract_line(output, "FINAL CRITERION BREAKDOWN")
    turn_line = extract_line(output, "TURN CAPTURE:")
    transition_line = extract_line(output, "POST-TURN TRANSITION TRIM")
    post_line = extract_line(output, "POST-TURN FORWARD:")
    same_fdm_line = extract_line(output, "SAME FDM / NO RESET:")
    safety_line = extract_line(output, "SAFETY FAILURE:")

    passed = proc.returncode == 0 and full_pass_line == "FULL MISSION PASS: True"

    return {
        "angle": angle,
        "passed": passed,
        "returncode": proc.returncode,
        "turn": turn_line or "TURN CAPTURE: not found",
        "transition": transition_line or "POST-TURN TRANSITION TRIM: not used/not found",
        "post": post_line or "POST-TURN FORWARD: not found",
        "same_fdm": same_fdm_line or "SAME FDM / NO RESET: not found",
        "safety": safety_line or "SAFETY FAILURE: not found",
        "criterion": criterion_line or "FINAL CRITERION BREAKDOWN: not found",
        "log": str(log_path),
    }


def main() -> int:
    if not VALIDATOR.exists():
        print(f"ERROR: validator not found: {VALIDATOR}")
        return 2

    print("=" * 120)
    print("AH-1S FINAL REGRESSION SUITE — V20")
    print("Angles: -50, +50, +200, +360 deg")
    print("Positive turns use the proven 0.590 post-turn transition trim; -50 remains unchanged.")
    print("Each case launches the exact same final validator in a fresh process.")
    print("Per-case full logs are saved under final_regression_logs_v20/.")
    print("=" * 120)

    results = [run_case(angle) for angle in ANGLES]

    print("\n" + "=" * 120)
    print("FINAL REGRESSION MATRIX — V20")
    print("=" * 120)
    print(f"{'ANGLE':>8} | {'RESULT':^10} | {'RETURN':^6} | LOG")
    print("-" * 120)
    for r in results:
        result = "PASS" if r["passed"] else "FAIL"
        print(f"{r['angle']:+8d} | {result:^10} | {r['returncode']:^6d} | {r['log']}")

    all_pass = all(r["passed"] for r in results)
    print("-" * 120)
    print(f"FINAL 4/4 REGRESSION PASS: {all_pass}")

    if not all_pass:
        print("\nFAILED CASE DETAILS")
        for r in results:
            if r["passed"]:
                continue
            print("-" * 120)
            print(f"ANGLE {r['angle']:+d}")
            print(r["turn"])
            print(r["transition"])
            print(r["post"])
            print(r["same_fdm"])
            print(r["safety"])
            print(r["criterion"])
            print(f"Full log: {r['log']}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
