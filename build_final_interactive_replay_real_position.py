from __future__ import annotations

import builtins
from pathlib import Path

SOURCE = Path("build_final_interactive_replay.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old_init = '''    t0 = num(rows[0], "sim_t", 0.0)
    h0 = math.radians(num(rows[0], "heading_deg", 0.0))
    north = 0.0
    east = 0.0
    prev_t = t0
    out = []
'''
new_init = '''    t0 = num(rows[0], "sim_t", 0.0)
    required_position_columns = {
        "north_ft", "east_ft", "forward_position_ft", "cross_track_position_ft"
    }
    missing = [k for k in required_position_columns if k not in rows[0]]
    if missing:
        raise RuntimeError(
            "Real-position columns are missing from telemetry CSV: " + ", ".join(missing) +
            ". Run visualize_full_mission_final_real_position.py first."
        )
    prev_t = t0
    out = []
'''
if old_init not in text:
    raise RuntimeError("real-position replay: init anchor not found")
text = text.replace(old_init, new_init, 1)

old_geometry = '''        u = num(row, "forward_speed_fps", 0.0)
        v = num(row, "lateral_speed_fps", 0.0)
        psi = math.radians(num(row, "heading_deg", 0.0))

        # Replay-only horizontal geometry reconstructed from measured body-axis
        # speeds and heading. The validated V23 control results are unchanged.
        vn = u * math.cos(psi) - v * math.sin(psi)
        ve = u * math.sin(psi) + v * math.cos(psi)
        north += vn * dt
        east += ve * dt
        forward = north * math.cos(h0) + east * math.sin(h0)
        cross = -north * math.sin(h0) + east * math.cos(h0)
'''
new_geometry = '''        u = num(row, "forward_speed_fps", 0.0)
        v = num(row, "lateral_speed_fps", 0.0)

        # Exact replay geometry from JSBSim position telemetry.
        north = num(row, "north_ft", 0.0)
        east = num(row, "east_ft", 0.0)
        forward = num(row, "forward_position_ft", 0.0)
        cross = num(row, "cross_track_position_ft", 0.0)
'''
if old_geometry not in text:
    raise RuntimeError("real-position replay: geometry anchor not found")
text = text.replace(old_geometry, new_geometry, 1)

old_sub = '''<div class="title"><div><h1>AH-1S Final Mission Interactive Replay <span class="badge">V23 telemetry</span></h1><div class="sub">Validated flight data — replay only</div></div><div id="phaseTop" class="sub"></div></div>'''
new_sub = '''<div class="title"><div><h1>AH-1S Final Mission Interactive Replay <span class="badge">V23 telemetry</span></h1><div class="sub">Validated flight data — real JSBSim position replay</div></div><div id="phaseTop" class="sub"></div></div>'''
text = text.replace(old_sub, new_sub, 1)

print("=" * 120)
print("FINAL INTERACTIVE REPLAY — REAL JSBSIM POSITION")
print("3D/top/side geometry uses recorded JSBSim latitude/longitude-derived coordinates.")
print("=" * 120)

builtins.exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
