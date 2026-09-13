from __future__ import annotations

from pathlib import Path

SOURCE = Path("validate_full_mission_final_v23.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

wrapper = SOURCE.read_text(encoding="utf-8")
exec_anchor = 'exec(compile(text, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})\n'
if exec_anchor not in wrapper:
    raise RuntimeError("visualizer: final V23 exec anchor not found")

instrumentation = r'''
# ============================================================================
# VISUALIZATION-ONLY INSTRUMENTATION
# This block does not alter any control action, reward, state, threshold,
# model, AFCS setting, phase transition, or stopping criterion. It only records
# the already-computed live-FDM telemetry and writes CSV/PNG artifacts.
# ============================================================================
telemetry_helper_anchor = "def hard_safe(m):\n"
telemetry_helper = '''TELEMETRY = []\n\n\ndef record_telemetry(fdm, phase, action=None, cumulative=float(\"nan\"), remaining=float(\"nan\"), heading_error=float(\"nan\")):\n    m = physical_metrics(fdm)\n    try:\n        arr = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else np.full(4, np.nan, dtype=np.float32)\n    except Exception:\n        arr = np.full(4, np.nan, dtype=np.float32)\n    if arr.size < 4:\n        arr = np.pad(arr, (0, 4 - arr.size), constant_values=np.nan)\n    TELEMETRY.append({\n        \"sim_t\": m[\"sim_t\"],\n        \"phase\": str(phase),\n        \"altitude_ft\": m[\"alt\"],\n        \"vertical_speed_fps\": m[\"vs\"],\n        \"forward_speed_fps\": m[\"u\"],\n        \"lateral_speed_fps\": m[\"v_lat\"],\n        \"roll_deg\": m[\"roll\"],\n        \"pitch_deg\": m[\"pitch\"],\n        \"yaw_rate_deg_s\": m[\"r\"],\n        \"heading_deg\": m[\"hdg\"],\n        \"rotor_rpm\": m[\"rpm\"],\n        \"collective_cmd\": fdm_get(fdm, \"fcs/collective-cmd-norm\"),\n        \"elevator_cmd\": fdm_get(fdm, \"fcs/elevator-cmd-norm\"),\n        \"aileron_cmd\": fdm_get(fdm, \"fcs/aileron-cmd-norm\"),\n        \"rudder_cmd\": fdm_get(fdm, \"fcs/rudder-cmd-norm\"),\n        \"policy_a0\": float(arr[0]),\n        \"policy_a1\": float(arr[1]),\n        \"policy_a2\": float(arr[2]),\n        \"policy_a3\": float(arr[3]),\n        \"cumulative_turn_deg\": float(cumulative),\n        \"remaining_turn_deg\": float(remaining),\n        \"heading_error_deg\": float(heading_error),\n    })\n\n\n'''
if telemetry_helper_anchor not in text:
    raise RuntimeError("visualizer: helper insertion anchor not found")
text = text.replace(telemetry_helper_anchor, telemetry_helper + telemetry_helper_anchor, 1)

# Stage 1 telemetry.
old = '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n'''
new = '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n    record_telemetry(fdm, \"Stage1 Takeoff/Hover\", action1)\n'''
if old not in text:
    raise RuntimeError("visualizer: Stage1 hook anchor not found")
text = text.replace(old, new, 1)

# Stage 2 telemetry.
old = '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n'''
new = '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n    record_telemetry(fdm, \"Stage2 Forward\", action2)\n'''
if old not in text:
    raise RuntimeError("visualizer: Stage2 hook anchor not found")
text = text.replace(old, new, 1)

# Turn telemetry.
old = '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n'''
new = '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n    record_telemetry(\n        fdm, \"Turn\", action_turn,\n        cumulative=float(turn_info.get(\"cumulative_turn_deg\", float(\"nan\"))),\n        remaining=float(turn_info.get(\"remaining_turn_deg\", float(\"nan\"))),\n    )\n'''
if old not in text:
    raise RuntimeError("visualizer: Turn hook anchor not found")
text = text.replace(old, new, 1)

# Positive-turn AFCS transition telemetry. This anchor exists only in final V23
# generated text after the V19/V23 transformations have been applied.
old = '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n'''
new = '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n        record_telemetry(\n            fdm, \"Post-turn AFCS Transition\", None,\n            cumulative=float(requested_turn - heading_error_s),\n            remaining=float(heading_error_s),\n            heading_error=float(heading_error_s),\n        )\n'''
if old not in text:
    raise RuntimeError("visualizer: transition hook anchor not found")
text = text.replace(old, new, 1)

# Final Stage2 post-turn telemetry.
old = '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n'''
new = '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n    current_hdg_for_log = heading_deg(fdm)\n    heading_error_for_log = wrap_deg(post_target_heading - current_hdg_for_log)\n    record_telemetry(\n        fdm, \"Post-turn Stage2\", action_post,\n        cumulative=float(requested_turn - heading_error_for_log),\n        remaining=float(heading_error_for_log),\n        heading_error=float(heading_error_for_log),\n    )\n'''
if old not in text:
    raise RuntimeError("visualizer: post-turn Stage2 hook anchor not found")
text = text.replace(old, new, 1)

# Export after the validator's final summary. No control code is changed.
text += r'''\n\n# ============================================================================\n# TELEMETRY EXPORT + PRESENTATION FIGURES\n# ============================================================================\nfrom pathlib import Path as _VizPath\nimport csv as _csv\nimport matplotlib.pyplot as _plt\n\n_angle_tag = f\"{requested_turn:+.0f}\".replace(\"+\", \"plus\").replace(\"-\", \"minus\")\n_out = _VizPath(\"visualization_final\") / f\"turn_{_angle_tag}\"\n_out.mkdir(parents=True, exist_ok=True)\n\n_csv_path = _out / \"full_mission_telemetry.csv\"\nif TELEMETRY:\n    _fields = list(TELEMETRY[0].keys())\n    with _csv_path.open(\"w\", newline=\"\", encoding=\"utf-8\") as _f:\n        _w = _csv.DictWriter(_f, fieldnames=_fields)\n        _w.writeheader()\n        _w.writerows(TELEMETRY)\n\n    _t0 = float(TELEMETRY[0][\"sim_t\"])\n    _t = np.asarray([float(r[\"sim_t\"]) - _t0 for r in TELEMETRY], dtype=float)\n    _phase = [r[\"phase\"] for r in TELEMETRY]\n\n    def _series(name):\n        return np.asarray([float(r[name]) for r in TELEMETRY], dtype=float)\n\n    def _phase_spans(ax):\n        if not TELEMETRY:\n            return\n        _start = 0\n        for _i in range(1, len(_phase) + 1):\n            if _i == len(_phase) or _phase[_i] != _phase[_start]:\n                ax.axvspan(_t[_start], _t[_i - 1], alpha=0.06)\n                _mid = (_t[_start] + _t[_i - 1]) / 2.0\n                ax.text(_mid, 0.98, _phase[_start], transform=ax.get_xaxis_transform(),\n                        ha=\"center\", va=\"top\", fontsize=8, rotation=0)\n                _start = _i\n\n    def _save_single(filename, ylabel, columns, labels=None, hlines=None):\n        fig, ax = _plt.subplots(figsize=(13, 5.5))\n        labels = labels or columns\n        for c, label in zip(columns, labels):\n            ax.plot(_t, _series(c), label=label, linewidth=1.5)\n        if hlines:\n            for value, label in hlines:\n                ax.axhline(value, linestyle=\"--\", linewidth=1.0, label=label)\n        _phase_spans(ax)\n        ax.set_xlabel(\"Mission time (s)\")\n        ax.set_ylabel(ylabel)\n        ax.grid(True, alpha=0.25)\n        if len(columns) > 1 or hlines:\n            ax.legend(loc=\"best\")\n        fig.tight_layout()\n        fig.savefig(_out / filename, dpi=180, bbox_inches=\"tight\")\n        _plt.close(fig)\n\n    _save_single(\n        \"01_altitude.png\", \"Altitude (ft)\", [\"altitude_ft\"],\n        hlines=[(300.0, \"300 ft target\"), (285.0, \"qualification min\"), (315.0, \"qualification max\")],\n    )\n    _save_single(\n        \"02_vertical_speed.png\", \"Vertical speed (ft/s)\", [\"vertical_speed_fps\"],\n        hlines=[(0.0, \"zero vertical speed\"), (1.5, \"+1.5 limit\"), (-1.5, \"-1.5 limit\")],\n    )\n    _save_single(\n        \"03_forward_lateral_speed.png\", \"Speed (ft/s)\",\n        [\"forward_speed_fps\", \"lateral_speed_fps\"], [\"Forward speed\", \"Lateral speed\"],\n    )\n    _save_single(\n        \"04_attitude_rates.png\", \"Degrees / degrees per second\",\n        [\"roll_deg\", \"pitch_deg\", \"yaw_rate_deg_s\"], [\"Roll\", \"Pitch\", \"Yaw rate\"],\n    )\n    _save_single(\n        \"05_heading_turn_progress.png\", \"Degrees\",\n        [\"heading_deg\", \"cumulative_turn_deg\", \"remaining_turn_deg\"],\n        [\"Heading\", \"Cumulative relative turn\", \"Remaining turn\"],\n    )\n    _save_single(\n        \"06_rotor_rpm.png\", \"Rotor RPM\", [\"rotor_rpm\"],\n    )\n    _save_single(\n        \"07_flight_controls.png\", \"Normalized physical command\",\n        [\"collective_cmd\", \"elevator_cmd\", \"aileron_cmd\", \"rudder_cmd\"],\n        [\"Collective\", \"Elevator\", \"Aileron\", \"Rudder\"],\n    )\n\n    # Compact architecture/timeline figure for direct use in the presentation.\n    fig, ax = _plt.subplots(figsize=(13, 2.6))\n    _phase_order = []\n    for _p in _phase:\n        if not _phase_order or _phase_order[-1] != _p:\n            _phase_order.append(_p)\n    _y = 0.5\n    _starts = []\n    _ends = []\n    _labels = []\n    _start = 0\n    for _i in range(1, len(_phase) + 1):\n        if _i == len(_phase) or _phase[_i] != _phase[_start]:\n            _starts.append(_t[_start])\n            _ends.append(_t[_i - 1])\n            _labels.append(_phase[_start])\n            _start = _i\n    for _s, _e, _label in zip(_starts, _ends, _labels):\n        _width = max(_e - _s, 0.15)\n        ax.barh([_y], [_width], left=[_s], height=0.42, alpha=0.35, edgecolor=\"black\")\n        ax.text(_s + _width / 2.0, _y, _label, ha=\"center\", va=\"center\", fontsize=9)\n    ax.set_xlim(_t[0], _t[-1])\n    ax.set_ylim(0.15, 0.85)\n    ax.set_yticks([])\n    ax.set_xlabel(\"Mission time (s)\")\n    ax.set_title(\"AH-1S Final Mission Phase Timeline\")\n    ax.grid(True, axis=\"x\", alpha=0.25)\n    fig.tight_layout()\n    fig.savefig(_out / \"00_mission_phase_timeline.png\", dpi=180, bbox_inches=\"tight\")\n    _plt.close(fig)\n\n    print(\"=" * 120)\n    print(\"FINAL VISUALIZATION EXPORT\")\n    print(\"Telemetry rows:\", len(TELEMETRY))\n    print(\"CSV:\", _csv_path)\n    print(\"Figures:\", _out)\n    print(\"=" * 120)\nelse:\n    print(\"WARNING: telemetry list is empty; no visualization files written.\")\n'''
'''

wrapper = wrapper.replace(exec_anchor, instrumentation + "\n" + exec_anchor, 1)

print("=" * 120)
print("FINAL V23 FULL-MISSION VISUALIZATION")
print("Control/reward/model logic is unchanged; telemetry hooks only.")
print("Outputs: visualization_final/turn_<angle>/full_mission_telemetry.csv + PNG figures")
print("=" * 120)

exec(compile(wrapper, str(SOURCE), "exec"), {"__name__": "__main__", "__file__": str(SOURCE)})
