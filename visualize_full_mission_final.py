from __future__ import annotations

import builtins
from pathlib import Path

ROOT = Path("validate_full_mission_final_v23.py")
if not ROOT.exists():
    raise FileNotFoundError(ROOT)


def capture_generated_source(source_text: str, source_name: str) -> str:
    """Execute one wrapper without running its generated validator.

    The wrapper still performs its normal text replacements. Its final
    compile(...)/exec(...) pair is intercepted so we can capture the generated
    source string instead of executing the flight simulation.
    """
    captured = {}

    def fake_compile(source, filename, mode, *args, **kwargs):
        captured["compiled_source"] = source
        return source

    def fake_exec(code, globals_arg=None, locals_arg=None):
        if isinstance(code, str):
            captured["generated"] = code
        elif "compiled_source" in captured:
            captured["generated"] = captured["compiled_source"]
        else:
            raise RuntimeError(f"visualizer: could not capture generated source from {source_name}")

    ns = {
        "__name__": "__main__",
        "__file__": source_name,
        "compile": fake_compile,
        "exec": fake_exec,
    }
    builtins.exec(source_text, ns)

    generated = captured.get("generated")
    if not isinstance(generated, str) or not generated.strip():
        raise RuntimeError(f"visualizer: wrapper {source_name} produced no source")
    return generated


# V23 -> V19 wrapper -> V11 wrapper -> actual validator source.
level_v23 = ROOT.read_text(encoding="utf-8")
level_v19 = capture_generated_source(level_v23, str(ROOT))
level_v11 = capture_generated_source(level_v19, "<materialized_v19>")
text = capture_generated_source(level_v11, "<materialized_v11>")

if "AH-1S FINAL FULL-MISSION VALIDATOR" not in text or "def hard_safe(m):" not in text:
    raise RuntimeError("visualizer: nested wrapper materialization did not reach the final validator")

# -----------------------------------------------------------------------------
# Telemetry helper: read-only instrumentation. No control/reward/model changes.
# -----------------------------------------------------------------------------
helper_anchor = "def hard_safe(m):\n"
helper = '''TELEMETRY = []


def record_telemetry(fdm, phase, action=None, cumulative=float("nan"), remaining=float("nan"), heading_error=float("nan")):
    m = physical_metrics(fdm)
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else np.full(4, np.nan, dtype=np.float32)
    except Exception:
        arr = np.full(4, np.nan, dtype=np.float32)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size), constant_values=np.nan)
    TELEMETRY.append({
        "sim_t": m["sim_t"],
        "phase": str(phase),
        "altitude_ft": m["alt"],
        "vertical_speed_fps": m["vs"],
        "forward_speed_fps": m["u"],
        "lateral_speed_fps": m["v_lat"],
        "roll_deg": m["roll"],
        "pitch_deg": m["pitch"],
        "yaw_rate_deg_s": m["r"],
        "heading_deg": m["hdg"],
        "rotor_rpm": m["rpm"],
        "collective_cmd": fdm_get(fdm, "fcs/collective-cmd-norm"),
        "elevator_cmd": fdm_get(fdm, "fcs/elevator-cmd-norm"),
        "aileron_cmd": fdm_get(fdm, "fcs/aileron-cmd-norm"),
        "rudder_cmd": fdm_get(fdm, "fcs/rudder-cmd-norm"),
        "policy_a0": float(arr[0]),
        "policy_a1": float(arr[1]),
        "policy_a2": float(arr[2]),
        "policy_a3": float(arr[3]),
        "cumulative_turn_deg": float(cumulative),
        "remaining_turn_deg": float(remaining),
        "heading_error_deg": float(heading_error),
    })


'''
text = text.replace(helper_anchor, helper + helper_anchor, 1)


def replace_once(old: str, new: str, label: str):
    global text
    if old not in text:
        raise RuntimeError(f"visualizer: {label} anchor not found in materialized validator")
    text = text.replace(old, new, 1)


replace_once(
    '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n''',
    '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n    record_telemetry(fdm, "Stage1 Takeoff/Hover", action1)\n''',
    "Stage1",
)

replace_once(
    '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n''',
    '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n    record_telemetry(fdm, "Stage2 Forward", action2)\n''',
    "Stage2",
)

replace_once(
    '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n''',
    '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n    record_telemetry(\n        fdm, "Turn", action_turn,\n        cumulative=float(turn_info.get("cumulative_turn_deg", float("nan"))),\n        remaining=float(turn_info.get("remaining_turn_deg", float("nan"))),\n    )\n''',
    "Turn",
)

replace_once(
    '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n''',
    '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n        record_telemetry(\n            fdm, "Post-turn AFCS Transition", None,\n            cumulative=float(requested_turn - heading_error_s),\n            remaining=float(heading_error_s),\n            heading_error=float(heading_error_s),\n        )\n''',
    "AFCS transition",
)

replace_once(
    '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n''',
    '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n    current_hdg_for_log = heading_deg(fdm)\n    heading_error_for_log = wrap_deg(post_target_heading - current_hdg_for_log)\n    record_telemetry(\n        fdm, "Post-turn Stage2", action_post,\n        cumulative=float(requested_turn - heading_error_for_log),\n        remaining=float(heading_error_for_log),\n        heading_error=float(heading_error_for_log),\n    )\n''',
    "post-turn Stage2",
)

# -----------------------------------------------------------------------------
# Export CSV + presentation figures after the final validator summary.
# -----------------------------------------------------------------------------
text += r'''

from pathlib import Path as _VizPath
import csv as _csv
import matplotlib.pyplot as _plt

_angle_tag = f"{requested_turn:+.0f}".replace("+", "plus").replace("-", "minus")
_out = _VizPath("visualization_final") / f"turn_{_angle_tag}"
_out.mkdir(parents=True, exist_ok=True)
_csv_path = _out / "full_mission_telemetry.csv"

if TELEMETRY:
    _fields = list(TELEMETRY[0].keys())
    with _csv_path.open("w", newline="", encoding="utf-8") as _f:
        _w = _csv.DictWriter(_f, fieldnames=_fields)
        _w.writeheader()
        _w.writerows(TELEMETRY)

    _t0 = float(TELEMETRY[0]["sim_t"])
    _t = np.asarray([float(r["sim_t"]) - _t0 for r in TELEMETRY], dtype=float)
    _phase = [r["phase"] for r in TELEMETRY]

    def _series(name):
        return np.asarray([float(r[name]) for r in TELEMETRY], dtype=float)

    def _phase_spans(ax):
        _start = 0
        for _i in range(1, len(_phase) + 1):
            if _i == len(_phase) or _phase[_i] != _phase[_start]:
                ax.axvspan(_t[_start], _t[_i - 1], alpha=0.06)
                _mid = (_t[_start] + _t[_i - 1]) / 2.0
                ax.text(_mid, 0.98, _phase[_start], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
                _start = _i

    def _save(filename, ylabel, columns, labels=None, hlines=None):
        fig, ax = _plt.subplots(figsize=(13, 5.5))
        labels = labels or columns
        for _c, _label in zip(columns, labels):
            ax.plot(_t, _series(_c), label=_label, linewidth=1.5)
        if hlines:
            for _value, _label in hlines:
                ax.axhline(_value, linestyle="--", linewidth=1.0, label=_label)
        _phase_spans(ax)
        ax.set_xlabel("Mission time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if len(columns) > 1 or hlines:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(_out / filename, dpi=180, bbox_inches="tight")
        _plt.close(fig)

    _save("01_altitude.png", "Altitude (ft)", ["altitude_ft"], hlines=[(300.0, "300 ft target"), (285.0, "qualification min"), (315.0, "qualification max")])
    _save("02_vertical_speed.png", "Vertical speed (ft/s)", ["vertical_speed_fps"], hlines=[(0.0, "zero vertical speed"), (1.5, "+1.5 limit"), (-1.5, "-1.5 limit")])
    _save("03_forward_lateral_speed.png", "Speed (ft/s)", ["forward_speed_fps", "lateral_speed_fps"], ["Forward speed", "Lateral speed"])
    _save("04_attitude_rates.png", "Degrees / degrees per second", ["roll_deg", "pitch_deg", "yaw_rate_deg_s"], ["Roll", "Pitch", "Yaw rate"])
    _save("05_heading_turn_progress.png", "Degrees", ["heading_deg", "cumulative_turn_deg", "remaining_turn_deg"], ["Heading", "Cumulative relative turn", "Remaining turn"])
    _save("06_rotor_rpm.png", "Rotor RPM", ["rotor_rpm"])
    _save("07_flight_controls.png", "Normalized physical command", ["collective_cmd", "elevator_cmd", "aileron_cmd", "rudder_cmd"], ["Collective", "Elevator", "Aileron", "Rudder"])

    fig, ax = _plt.subplots(figsize=(13, 2.6))
    _start = 0
    for _i in range(1, len(_phase) + 1):
        if _i == len(_phase) or _phase[_i] != _phase[_start]:
            _s = _t[_start]
            _e = _t[_i - 1]
            _width = max(_e - _s, 0.15)
            ax.barh([0.5], [_width], left=[_s], height=0.42, alpha=0.35, edgecolor="black")
            ax.text(_s + _width / 2.0, 0.5, _phase[_start], ha="center", va="center", fontsize=9)
            _start = _i
    ax.set_xlim(_t[0], _t[-1])
    ax.set_ylim(0.15, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("Mission time (s)")
    ax.set_title("AH-1S Final Mission Phase Timeline")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(_out / "00_mission_phase_timeline.png", dpi=180, bbox_inches="tight")
    _plt.close(fig)

    print("=" * 120)
    print("FINAL VISUALIZATION EXPORT")
    print("Telemetry rows:", len(TELEMETRY))
    print("CSV:", _csv_path)
    print("Figures:", _out)
    print("=" * 120)
else:
    print("WARNING: telemetry list is empty; no visualization files written.")
'''

print("=" * 120)
print("FINAL V23 FULL-MISSION VISUALIZATION")
print("Nested wrappers materialized first; final control/reward/model logic is unchanged.")
print("Outputs: visualization_final/turn_<angle>/full_mission_telemetry.csv + PNG figures")
print("=" * 120)

exec(compile(text, "<materialized_final_v23_with_telemetry>", "exec"), {"__name__": "__main__", "__file__": str(ROOT)})
