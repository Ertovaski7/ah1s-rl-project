from __future__ import annotations

import re
import subprocess
import sys


def main():
    target = input("Target angle for V11 diagnostic [-50]: ").strip() or "-50"
    proc = subprocess.run(
        [sys.executable, "validate_full_mission_v11_final_gated.py"],
        input=target + "\n",
        text=True,
        capture_output=True,
    )

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    lines = out.splitlines()

    print("=" * 120)
    print(f"V11 DIAGNOSTIC SUMMARY | target={target} | returncode={proc.returncode}")
    print("=" * 120)

    keys = [
        "STAGE1:",
        "STAGE2 ENTRY:",
        "TURN PROGRESS COORDINATE CALIBRATION",
        "TURN ENTRY OBS CHECK",
        "TURN CAPTURE:",
        "POST-TURN FORWARD:",
        "POST-TURN WINDOW",
        "SAME FDM / NO RESET:",
        "STAGE1 TAKEOFF + 300 FT HOVER:",
        "STAGE2 FORWARD ENTRY:",
        "TURN CAPTURE (",
        "POST-TURN FORWARD 10s:",
        "SAFETY FAILURE:",
        "TEACHER ACTIVE AT RUNTIME:",
        "CONTROLLER ACTIVE AT RUNTIME:",
        "V7 +200 GATE AT TURN ENTRY:",
        "V10 -50 GATE AT TURN ENTRY:",
        "VALIDATED COMMAND SET MEMBER:",
        "FULL MISSION PASS:",
    ]

    found = []
    for line in lines:
        if any(k in line for k in keys):
            found.append(line)

    if found:
        for line in found:
            print(line)
    else:
        print("No summary markers found; printing the final 120 lines.")

    print("\n" + "-" * 120)
    print("FINAL 120 RAW LINES")
    print("-" * 120)
    for line in lines[-120:]:
        print(line)

    print("\n" + "-" * 120)
    print("AUTOMATIC FAILURE CLASSIFICATION")
    print("-" * 120)

    text = "\n".join(lines)
    checks = {
        "stage1": "STAGE1 TAKEOFF + 300 FT HOVER: PASS" in text,
        "stage2": "STAGE2 FORWARD ENTRY: PASS" in text,
        "turn": bool(re.search(r"TURN CAPTURE \([^\n]+\): PASS", text)),
        "post": "POST-TURN FORWARD 10s: PASS" in text,
        "same_fdm": "SAME FDM / NO RESET: PASS" in text,
        "full": "FULL MISSION PASS: True" in text,
    }

    safety_match = re.findall(r"SAFETY FAILURE:\s*(True|False)", text)
    safety = safety_match[-1] if safety_match else "UNKNOWN"

    for name, ok in checks.items():
        print(f"{name:10s}: {'PASS' if ok else 'FAIL/UNKNOWN'}")
    print(f"{'safety':10s}: {safety}")

    if checks["turn"] and not checks["post"]:
        print("\nDiagnosis: turn capture succeeded; failure is isolated to post-turn continuation criteria.")
    elif not checks["turn"]:
        print("\nDiagnosis: turn capture itself failed; inspect TURN CAPTURE and preceding turn trajectory lines above.")
    elif checks["turn"] and checks["post"] and not checks["full"]:
        print("\nDiagnosis: flight phases passed but a continuity/summary criterion failed; inspect SAME FDM and gate lines.")
    elif checks["full"]:
        print("\nDiagnosis: full mission passed.")


if __name__ == "__main__":
    main()
