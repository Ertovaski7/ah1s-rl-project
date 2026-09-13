from __future__ import annotations

import builtins
from pathlib import Path

SOURCE = Path("visualize_full_mission_final.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old = '''        "sim_t": m["sim_t"],
        "phase": str(phase),
        "altitude_ft": m["alt"],
'''

new = '''        "sim_t": m["sim_t"],
        "phase": str(phase),
        "latitude_deg": latitude_deg(fdm),
        "longitude_deg": longitude_deg(fdm),
        "altitude_ft": m["alt"],
'''

if old not in text:
    raise RuntimeError("real-position visualizer: telemetry location insertion anchor not found")
text = text.replace(old, new, 1)

# Extend the generated final validator with local coordinates derived directly
# from JSBSim latitude/longitude. No control or validation logic is changed.
old_exec = '''exec(compile(text, "<materialized_final_v23_with_telemetry>", "exec"), {"__name__": "__main__", "__file__": str(ROOT)})
'''

new_exec = '''
# Add exact local position coordinates to each telemetry row at record time.
_position_patch_anchor = ''' + '"""' + '''def record_telemetry(fdm, phase, action=None, cumulative=float("nan"), remaining=float("nan"), heading_error=float("nan")):
    m = physical_metrics(fdm)
''' + '"""' + '''
_position_patch_replacement = ''' + '"""' + '''def record_telemetry(fdm, phase, action=None, cumulative=float("nan"), remaining=float("nan"), heading_error=float("nan")):
    m = physical_metrics(fdm)
''' + '"""' + '''
if _position_patch_anchor not in text:
    raise RuntimeError("real-position visualizer: record_telemetry anchor missing")

# Inject a post-processing block before CSV export so local coordinates are
# computed from the real JSBSim lat/lon samples, referenced to the first sample
# and rotated into the initial mission-heading frame.
_export_anchor = ''' + '"""' + '''from pathlib import Path as _VizPath
import csv as _csv
import matplotlib.pyplot as _plt
''' + '"""' + '''
_export_replacement = ''' + '"""' + '''from pathlib import Path as _VizPath
import csv as _csv
import matplotlib.pyplot as _plt

if TELEMETRY:
    _EARTH_RADIUS_FT = 20902231.0
    _lat0 = float(TELEMETRY[0]["latitude_deg"])
    _lon0 = float(TELEMETRY[0]["longitude_deg"])
    _heading0 = math.radians(float(TELEMETRY[0]["heading_deg"]))
    _c0 = math.cos(_heading0)
    _s0 = math.sin(_heading0)
    for _row in TELEMETRY:
        _lat = float(_row["latitude_deg"])
        _lon = float(_row["longitude_deg"])
        _north = _EARTH_RADIUS_FT * math.radians(_lat - _lat0)
        _east = _EARTH_RADIUS_FT * math.cos(math.radians(_lat0)) * math.radians(_lon - _lon0)
        _row["north_ft"] = float(_north)
        _row["east_ft"] = float(_east)
        _row["forward_position_ft"] = float(_north * _c0 + _east * _s0)
        _row["cross_track_position_ft"] = float(-_north * _s0 + _east * _c0)
''' + '"""' + '''
if _export_anchor not in text:
    raise RuntimeError("real-position visualizer: export anchor missing")
text = text.replace(_export_anchor, _export_replacement, 1)

exec(compile(text, "<materialized_final_v23_with_real_position_telemetry>", "exec"), {"__name__": "__main__", "__file__": str(ROOT)})
'''

if old_exec not in text:
    raise RuntimeError("real-position visualizer: final execution anchor not found")
text = text.replace(old_exec, new_exec, 1)

print("=" * 120)
print("FINAL V23 VISUALIZATION — REAL JSBSIM POSITION TELEMETRY")
print("Adds latitude/longitude + north/east + forward/cross-track positions; flight logic unchanged.")
print("=" * 120)

ns = {"__name__": "__main__", "__file__": str(SOURCE)}
builtins.exec(compile(text, str(SOURCE), "exec"), ns)
