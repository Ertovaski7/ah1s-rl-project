from __future__ import annotations

from pathlib import Path

SOURCE = Path("validate_full_mission_final_v23.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

wrapper = SOURCE.read_text(encoding="utf-8")
exec_anchor = 'exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})\n'
if exec_anchor not in wrapper:
    raise RuntimeError("visualizer: final V23 exec anchor not found")

instrumentation = r"""
# ============================================================================
# VISUALIZATION-ONLY INSTRUMENTATION
# ============================================================================
telemetry_helper_anchor = "def hard_safe(m):\n"
telemetry_helper = '''TELEMETRY = []


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
if telemetry_helper_anchor not in text:
    raise RuntimeError("visualizer: helper insertion anchor not found")
text = text.replace(telemetry_helper_anchor, telemetry_helper + telemetry_helper_anchor, 1)

old = '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n'''
new = '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n    record_telemetry(fdm, "Stage1 Takeoff/Hover", action1)\n'''
if old not in text:
    raise RuntimeError("visualizer: Stage1 hook anchor not found")
text = text.replace(old, new, 1)

old = '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n'''
new = '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n    record_telemetry(fdm, "Stage2 Forward", action2)\n'''
if old not in text:
    raise RuntimeError("visualizer: Stage2 hook anchor not found")
text = text.replace(old, new, 1)

old = '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n'''
new = '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n    record_telemetry(\n        fdm, "Turn", action_turn,\n        cumulative=float(turn_info.get("cumulative_turn_deg", float("nan"))),\n        remaining=float(turn_info.get("remaining_turn_deg", float("nan"))),\n    )\n'''
if old not in text:
    raise RuntimeError("visualizer: Turn hook anchor not found")
text = text.replace(old, new, 1)

old = '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n'''
new = '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n        record_telemetry(\n            fdm, "Post-turn AFCS Transition", None,\n            cumulative=float(requested_turn - heading_error_s),\n            remaining=float(heading_error_s),\n            heading_error=float(heading_error_s),\n        )\n'''
if old not in text:
    raise RuntimeError("visualizer: transition hook anchor not found")
text = text.replace(old, new, 1)

old = '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n'''
new = '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n    current_hdg_for_log = heading_deg(fdm)\n    heading_error_for_log = wrap_deg(post_target_heading - current_hdg_for_log)\n    record_telemetry(\n        fdm, "Post-turn Stage2", action_post,\n        cumulative=float(requested_turn - heading_error_for_log),\n        remaining=float(heading_error_for_log),\n        heading_error=float(heading_error_for_log),\n    )\n'''
if old not in text:
    raise RuntimeError("visualizer: post-turn Stage2 hook anchor not found")
text = text.replace(old, new, 1)

text += r'''\n\n# ============================================================================\n# TELEMETRY EXPORT + PRESENTATION FIGURES\n# ============================================================================\nfrom pathlib import Path as _VizPath\nimport csv as _csv\nimport matplotlib.pyplot as _plt\n\n_angle_tag = f"{requested_turn:+.0f}".replace("+", "plus").replace("-", "minus")\n_out = _VizPath("visualization_final") / f"turn_{_angle_tag}"\n_out.mkdir(parents=True, exist_ok=True)\n_csv_path = _out / "full_mission_telemetry.csv"\n\nif TELEMETRY:\n    _fields = list(TELEMETRY[0].keys())\n    with _csv_path.open("w", newline="", encoding="utf-8") as _f:\n        _w = _csv.DictWriter(_f, fieldnames=_fields)\n        _w.writeheader()\n        _w.writerows(TELEMETRY)\n\n    _t0 = float(TELEMETRY[0]["sim_t"])\n    _t = np.asarray([float(r["sim_t"]) - _t0 for r in TELEMETRY], dtype=float)\n    _phase = [r["phase"] for r in TELEMETRY]\n\n    def _series(name):\n        return np.asarray([float(r[name]) for r in TELEMETRY], dtype=float)\n\n    def _phase_spans(ax):\n        _start = 0\n        for _i in range(1, len(_phase) + 1):\n            if _i == len(_phase) or _phase[_i] != _phase[_start]:\n                ax.axvspan(_t[_start], _t[_i - 1], alpha=0.06)\n                _mid = (_t[_start] + _t[_i - 1]) / 2.0\n                ax.text(_mid, 0.98, _phase[_start], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)\n                _start = _i\n\n    def _save(filename, ylabel, columns, labels=None, hlines=None):\n        fig, ax = _plt.subplots(figsize=(13, 5.5))\n        labels = labels or columns\n        for _c, _label in zip(columns, labels):\n            ax.plot(_t, _series(_c), label=_label, linewidth=1.5)\n        if hlines:\n            for _value, _label in hlines:\n                ax.axhline(_value, linestyle="--", linewidth=1.0, label=_label)\n        _phase_spans(ax)\n        ax.set_xlabel("Mission time (s)")\n        ax.set_ylabel(ylabel)\n        ax.grid(True, alpha=0.25)\n        if len(columns) > 1 or hlines:\n            ax.legend(loc="best")\n        fig.tight_layout()\n        fig.savefig(_out / filename, dpi=180, bbox_inches="tight")\n        _plt.close(fig)\n\n    _save("01_altitude.png", "Altitude (ft)", ["altitude_ft"], hlines=[(300.0, "300 ft target"), (285.0, "qualification min"), (315.0, "qualification max")])\n    _save("02_vertical_speed.png", "Vertical speed (ft/s)", ["vertical_speed_fps"], hlines=[(0.0, "zero vertical speed"), (1.5, "+1.5 limit"), (-1.5, "-1.5 limit")])\n    _save("03_forward_lateral_speed.png", "Speed (ft/s)", ["forward_speed_fps", "lateral_speed_fps"], ["Forward speed", "Lateral speed"])\n    _save("04_attitude_rates.png", "Degrees / degrees per second", ["roll_deg", "pitch_deg", "yaw_rate_deg_s"], ["Roll", "Pitch", "Yaw rate"])\n    _save("05_heading_turn_progress.png", "Degrees", ["heading_deg", "cumulative_turn_deg", "remaining_turn_deg"], ["Heading", "Cumulative relative turn", "Remaining turn"])\n    _save("06_rotor_rpm.png", "Rotor RPM", ["rotor_rpm"])\n    _save("07_flight_controls.png", "Normalized physical command", ["collective_cmd", "elevator_cmd", "aileron_cmd", "rudder_cmd"], ["Collective", "Elevator", "Aileron", "Rudder"])\n\n    fig, ax = _plt.subplots(figsize=(13, 2.6))\n    _start = 0\n    for _i in range(1, len(_phase) + 1):\n        if _i == len(_phase) or _phase[_i] != _phase[_start]:\n            _s = _t[_start]\n            _e = _t[_i - 1]\n            _width = max(_e - _s, 0.15)\n            ax.barh([0.5], [_width], left=[_s], height=0.42, alpha=0.35, edgecolor="black")\n            ax.text(_s + _width / 2.0, 0.5, _phase[_start], ha="center", va="center", fontsize=9)\n            _start = _i\n    ax.set_xlim(_t[0], _t[-1])\n    ax.set_ylim(0.15, 0.85)\n    ax.set_yticks([])\n    ax.set_xlabel("Mission time (s)")\n    ax.set_title("AH-1S Final Mission Phase Timeline")\n    ax.grid(True, axis="x", alpha=0.25)\n    fig.tight_layout()\n    fig.savefig(_out / "00_mission_phase_timeline.png", dpi=180, bbox_inches="tight")\n    _plt.close(fig)\n\n    print("=" * 120)\n    print("FINAL VISUALIZATION EXPORT")\n    print("Telemetry rows:", len(TELEMETRY))\n    print("CSV:", _csv_path)\n    print("Figures:", _out)\n    print("=" * 120)\nelse:\n    print("WARNING: telemetry list is empty; no visualization files written.")\n'''
"""

wrapper = wrapper.replace(exec_anchor, instrumentation + "\n" + exec_anchor, 1)

print("=" * 120)
print("FINAL V23 FULL-MISSION VISUALIZATION")
print("Control/reward/model logic is unchanged; telemetry hooks only.")
print("Outputs: visualization_final/turn_<angle>/full_mission_telemetry.csv + PNG figures")
print("=" * 120)

exec(compile(wrapper, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
