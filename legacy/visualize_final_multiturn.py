from __future__ import annotations

"""
FINAL MULTI-TURN VISUALIZER
===========================

Matplotlib/PillowWriter visualizer in the same style as the older visualize_*.py
files.  It runs the CURRENT validated controller stack and produces a GIF.

Mission:
    Stage1 takeoff -> ~300 ft hover
    -> Stage2 forward flight
    -> arbitrary relative turn command(s)
    -> V11 bumpless recovery
    -> Stage2 forward continuation
    -> next command

Important:
- ONE continuous JSBSim FDM.
- No reset/teleport between phases.
- Runtime teacher OFF.
- Flight/control logic is reused from the already validated final stack.
- This file is visualization only; it does not train or alter model weights.

Examples:
    python visualize_final_multiturn.py
    python visualize_final_multiturn.py 20 -30 75
    python visualize_final_multiturn.py 75 -90 120 150 --forward-seconds 3

Output:
    ah1s_final_multiturn.gif
"""

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import validate_live_multiturn_same_fdm_v1 as live
import validate_final_continuous_mission_v1 as finalv
import test_turn_arbitrary_angles_v1 as arb


DEFAULT_TARGETS = [20.0, -30.0, 75.0, -90.0, 120.0, 150.0]
EARTH_RADIUS_FT = 20_902_231.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="*", type=float)
    p.add_argument("--forward-seconds", type=float, default=3.0)
    p.add_argument("--output", default="ah1s_final_multiturn.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=900)
    return p.parse_args()


def fdm_float(fdm, key, default=float("nan")):
    try:
        return float(fdm[key])
    except Exception:
        return float(default)


@dataclass
class Recorder:
    rows: list[dict] = field(default_factory=list)
    phase: str = "INITIALIZING"
    requested: float | None = None

    def set_phase(self, phase: str, requested: float | None = None):
        self.phase = str(phase)
        self.requested = None if requested is None else float(requested)

    def capture(self, fdm):
        self.rows.append(
            {
                "sim_t": fdm_float(fdm, "simulation/sim-time-sec", 0.0),
                "phase": self.phase,
                "requested": self.requested,
                "latitude_deg": fdm_float(
                    fdm,
                    "position/lat-geod-deg",
                    fdm_float(fdm, "position/lat-gc-deg", 0.0),
                ),
                "longitude_deg": fdm_float(
                    fdm,
                    "position/long-gc-deg",
                    fdm_float(fdm, "position/long-geod-deg", 0.0),
                ),
                "altitude": fdm_float(fdm, "position/h-agl-ft", 0.0),
                "vertical_speed": fdm_float(fdm, "velocities/h-dot-fps", 0.0),
                "forward_velocity": fdm_float(fdm, "velocities/u-aero-fps", 0.0),
                "lateral_velocity": fdm_float(fdm, "velocities/v-aero-fps", 0.0),
                "roll": fdm_float(fdm, "attitude/roll-rad", 0.0),
                "pitch": fdm_float(fdm, "attitude/pitch-rad", 0.0),
                "heading": fdm_float(
                    fdm,
                    "attitude/heading-true-rad",
                    fdm_float(fdm, "attitude/psi-rad", 0.0),
                ),
                "yaw_rate": fdm_float(fdm, "velocities/r-rad_sec", 0.0),
            }
        )


class RecordingFDM:
    """Transparent JSBSim proxy that records every N physics ticks."""

    def __init__(self, raw_fdm, recorder: Recorder, every_runs: int = 10):
        object.__setattr__(self, "_raw_fdm", raw_fdm)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_every_runs", max(1, int(every_runs)))
        object.__setattr__(self, "_run_counter", 0)

    def __getattr__(self, name):
        return getattr(self._raw_fdm, name)

    def __getitem__(self, key):
        return self._raw_fdm[key]

    def __setitem__(self, key, value):
        self._raw_fdm[key] = value

    def run(self, *args, **kwargs):
        ok = self._raw_fdm.run(*args, **kwargs)
        count = self._run_counter + 1
        object.__setattr__(self, "_run_counter", count)
        if count % self._every_runs == 0:
            self._recorder.capture(self)
        return ok


# -----------------------------------------------------------------------------
# Initial mission with telemetry recording.  This mirrors the already validated
# build_initial_live_mission() logic; the only addition is recorder.capture().
# -----------------------------------------------------------------------------
def build_initial_recorded_mission(recorder: Recorder):
    print("\n[1/2] Stage1: takeoff -> stable 300 ft hover")
    recorder.set_phase("Stage 1 - Takeoff / Hover")

    env1 = live.Env1(teacher_model_path=None, training_mode=False)
    obs1, _ = env1.reset(seed=live.SEED)
    raw_fdm = live.get_fdm(env1)
    original_fdm_id = id(raw_fdm)
    recorder.capture(raw_fdm)

    dt1 = live.env_control_dt(env1)
    stable_time = 0.0

    for _ in range(int(live.STAGE1_MAX_TIME / dt1)):
        action, _ = live.stage1_model.predict(obs1, deterministic=True)
        obs1, _, terminated, truncated, info = env1.step(action)
        recorder.capture(raw_fdm)

        alt = live.info_float(info, "altitude")
        vs = live.info_float(info, "vertical_speed")
        vn = live.info_float(info, "vn", 0.0)
        ve = live.info_float(info, "ve", 0.0)
        drift = live.info_float(info, "drift", 999.0)

        stable = (
            295.0 <= alt <= 305.0
            and abs(vs) <= 0.5
            and float(np.hypot(vn, ve)) <= 1.0
            and drift <= 3.0
        )
        stable_time = stable_time + dt1 if stable else 0.0

        if stable_time >= live.HANDOFF_STABLE_TIME:
            break
        if truncated or (terminated and not bool(info.get("success", False))):
            raise RuntimeError("Stage1 failed before stable handoff")

    if stable_time < live.HANDOFF_STABLE_TIME:
        raise RuntimeError("Stage1 stable hover handoff was not reached")

    live.print_tel("STAGE1 HANDOFF", raw_fdm)

    print("\n[2/2] Stage2: same FDM -> forward-flight turn entry")
    recorder.set_phase("Stage 2 - Forward Flight")

    env2 = live.Env2(
        aileron_scale=live.AILERON_SCALE,
        rudder_scale=live.RUDDER_SCALE,
    )
    env2.reset(seed=live.SEED)
    env2.fdm = raw_fdm
    env2.phase = 1
    env2.forward_distance = 0.0
    env2.previous_action = np.zeros((4,), dtype=np.float32)
    env2.steps = 0

    raw_fdm["ap/afcs/pitch-channel-active-norm"] = 0.5
    raw_fdm["ap/afcs/roll-channel-active-norm"] = 1.0
    raw_fdm["ap/afcs/yaw-channel-active-norm"] = 1.0

    obs2 = np.asarray(env2._get_obs(), dtype=np.float32)
    dt2 = live.env_control_dt(env2)

    for _ in range(int(70.0 / dt2)):
        action, _ = live.stage2_model.predict(obs2, deterministic=True)
        obs2, _, terminated, truncated, _ = env2.step(action)
        obs2 = np.asarray(obs2, dtype=np.float32)
        recorder.capture(raw_fdm)

        if env2.forward_distance >= live.ENTRY_FORWARD_FT:
            break
        if terminated or truncated:
            raise RuntimeError("Stage2 failed before initial turn entry")

    if env2.forward_distance < live.ENTRY_FORWARD_FT:
        raise RuntimeError("Stage2 did not reach initial turn entry")
    if id(raw_fdm) != original_fdm_id or env2.fdm is not raw_fdm:
        raise RuntimeError("FDM identity changed during Stage1 -> Stage2 handoff")

    live.print_tel("STAGE2 TURN ENTRY", raw_fdm)
    print(f"Shared raw FDM id: {original_fdm_id}")
    return env1, env2, raw_fdm


def run_recorded_mission(targets, forward_seconds, recorder: Recorder):
    stack = arb.load_stack()
    env1 = env2 = None
    raw_fdm = None
    proxy = None
    results = []

    try:
        env1, env2, raw_fdm = build_initial_recorded_mission(recorder)

        # From this point onward, wrap the SAME raw JSBSim object.  All control
        # functions see the proxy as their shared FDM and the proxy delegates to
        # the already-running raw FDM while sampling physics telemetry.
        proxy = RecordingFDM(raw_fdm, recorder, every_runs=10)
        env2.fdm = proxy
        proxy_id = id(proxy)

        for i, target in enumerate(targets):
            recorder.set_phase(f"Turn / Recovery {target:+.0f} deg", target)
            recorder.capture(proxy)

            print("\n" + "-" * 150)
            print(f"VISUALIZED USER COMMAND {i+1}/{len(targets)}: {target:+.1f} deg")
            print("-" * 150)

            result = finalv.execute_command_same_fdm(
                stack,
                proxy,
                proxy_id,
                float(target),
            )
            recorder.capture(proxy)
            results.append(result)

            if not result["success"] or result["safety"]:
                raise RuntimeError(
                    f"Command {target:+.1f} failed: success={result['success']} "
                    f"safety={result['safety']} rem={result['remaining']:+.2f}"
                )

            if i < len(targets) - 1 and forward_seconds > 0.0:
                recorder.set_phase("Stage 2 - Forward Flight")
                live.fly_stage2_between_turns(
                    env2,
                    proxy,
                    float(forward_seconds),
                    proxy_id,
                )
                recorder.capture(proxy)

        return results

    finally:
        if env2 is not None:
            try:
                env2.fdm = None
                env2.close()
            except Exception:
                pass
        if env1 is not None:
            try:
                env1.close()
            except Exception:
                pass


def rotation_matrix(roll, pitch, yaw):
    rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=float,
    )
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]],
        dtype=float,
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=float,
    )
    return rz @ ry @ rx


def create_gif(rows, output, fps=20, max_frames=900):
    if not rows:
        raise RuntimeError("No telemetry recorded")

    lat0 = float(rows[0]["latitude_deg"])
    lon0 = float(rows[0]["longitude_deg"])

    xs = []
    ys = []
    zs = []
    rolls = []
    pitches = []
    headings = []
    vs_values = []
    fwd_values = []
    lat_values = []
    yaw_values = []
    times = []
    phases = []
    requested = []

    for row in rows:
        lat = float(row["latitude_deg"])
        lon = float(row["longitude_deg"])
        north = EARTH_RADIUS_FT * math.radians(lat - lat0)
        east = EARTH_RADIUS_FT * math.cos(math.radians(lat0)) * math.radians(lon - lon0)

        xs.append(east)
        ys.append(north)
        zs.append(float(row["altitude"]))
        rolls.append(float(row["roll"]))
        pitches.append(float(row["pitch"]))
        headings.append(float(row["heading"]))
        vs_values.append(float(row["vertical_speed"]))
        fwd_values.append(float(row["forward_velocity"]))
        lat_values.append(float(row["lateral_velocity"]))
        yaw_values.append(math.degrees(float(row["yaw_rate"])))
        times.append(float(row["sim_t"]))
        phases.append(str(row["phase"]))
        requested.append(row["requested"])

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    rolls = np.asarray(rolls, dtype=float)
    pitches = np.asarray(pitches, dtype=float)
    headings = np.asarray(headings, dtype=float)
    vs_values = np.asarray(vs_values, dtype=float)
    fwd_values = np.asarray(fwd_values, dtype=float)
    lat_values = np.asarray(lat_values, dtype=float)
    yaw_values = np.asarray(yaw_values, dtype=float)
    times = np.asarray(times, dtype=float)

    # Remove exact duplicate consecutive samples to keep GIF generation light.
    keep = np.ones(len(xs), dtype=bool)
    if len(xs) > 1:
        same_t = np.isclose(np.diff(times), 0.0)
        keep[1:][same_t] = False

    xs = xs[keep]
    ys = ys[keep]
    zs = zs[keep]
    rolls = rolls[keep]
    pitches = pitches[keep]
    headings = headings[keep]
    vs_values = vs_values[keep]
    fwd_values = fwd_values[keep]
    lat_values = lat_values[keep]
    yaw_values = yaw_values[keep]
    times = times[keep]
    phases = [p for p, k in zip(phases, keep) if k]
    requested = [r for r, k in zip(requested, keep) if k]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    margin_xy = 40.0
    x_min = float(np.min(xs) - margin_xy)
    x_max = float(np.max(xs) + margin_xy)
    y_min = float(np.min(ys) - margin_xy)
    y_max = float(np.max(ys) + margin_xy)

    if abs(x_max - x_min) < 80.0:
        mid = 0.5 * (x_min + x_max)
        x_min, x_max = mid - 40.0, mid + 40.0
    if abs(y_max - y_min) < 80.0:
        mid = 0.5 * (y_min + y_max)
        y_min, y_max = mid - 40.0, mid + 40.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0.0, max(350.0, float(np.max(zs)) + 20.0))
    ax.set_xlabel("East displacement from takeoff (ft)")
    ax.set_ylabel("North displacement from takeoff (ft)")
    ax.set_zlabel("Altitude AGL (ft)")
    ax.set_title("AH-1S FINAL MULTI-TURN — SAME FDM")

    plane_x = np.array([[x_min, x_max], [x_min, x_max]])
    plane_y = np.array([[y_min, y_min], [y_max, y_max]])
    plane_z = np.full((2, 2), 300.0)
    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.08)

    trail, = ax.plot([], [], [], linewidth=2)
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    body_collection = None
    nose_line = None
    rotor_line = None

    def update(frame):
        nonlocal body_collection, nose_line, rotor_line

        if body_collection is not None:
            body_collection.remove()
        if nose_line is not None:
            nose_line.remove()
        if rotor_line is not None:
            rotor_line.remove()

        x = xs[frame]
        y = ys[frame]
        z = zs[frame]
        roll = rolls[frame]
        pitch = pitches[frame]
        yaw = headings[frame]
        r = rotation_matrix(roll, pitch, yaw)

        trail.set_data(xs[: frame + 1], ys[: frame + 1])
        trail.set_3d_properties(zs[: frame + 1])

        length, width, height = 18.0, 5.0, 4.0
        vertices = np.array(
            [
                [-length / 2, -width / 2, -height / 2],
                [ length / 2, -width / 2, -height / 2],
                [ length / 2,  width / 2, -height / 2],
                [-length / 2,  width / 2, -height / 2],
                [-length / 2, -width / 2,  height / 2],
                [ length / 2, -width / 2,  height / 2],
                [ length / 2,  width / 2,  height / 2],
                [-length / 2,  width / 2,  height / 2],
            ],
            dtype=float,
        )
        vertices = vertices @ r.T
        vertices[:, 0] += x
        vertices[:, 1] += y
        vertices[:, 2] += z

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]
        body_collection = Poly3DCollection(faces, alpha=0.8)
        ax.add_collection3d(body_collection)

        nose = r @ np.array([15.0, 0.0, 0.0])
        nose_line, = ax.plot(
            [x, x + nose[0]],
            [y, y + nose[1]],
            [z, z + nose[2]],
            linewidth=3,
        )

        rotor_a = r @ np.array([0.0, -14.0, 2.5])
        rotor_b = r @ np.array([0.0, 14.0, 2.5])
        rotor_line, = ax.plot(
            [x + rotor_a[0], x + rotor_b[0]],
            [y + rotor_a[1], y + rotor_b[1]],
            [z + rotor_a[2], z + rotor_b[2]],
            linewidth=2,
        )

        cmd = requested[frame]
        cmd_text = "-" if cmd is None else f"{float(cmd):+.1f} deg"
        info_text.set_text(
            f"Phase: {phases[frame]}\n"
            f"Relative command: {cmd_text}\n"
            f"Sim time: {times[frame]:.1f} s\n"
            f"Altitude: {z:.1f} ft\n"
            f"Heading: {math.degrees(yaw) % 360.0:.1f} deg\n"
            f"Roll: {math.degrees(roll):+.1f} deg\n"
            f"Pitch: {math.degrees(pitch):+.1f} deg\n"
            f"VS: {vs_values[frame]:+.2f} ft/s\n"
            f"Forward: {fwd_values[frame]:.2f} ft/s\n"
            f"Lateral: {lat_values[frame]:+.2f} ft/s\n"
            f"Yaw rate: {yaw_values[frame]:+.2f} deg/s"
        )

        return trail, body_collection, nose_line, rotor_line, info_text

    max_frames = max(50, int(max_frames))
    stride = max(1, int(math.ceil(len(xs) / max_frames)))
    frames = list(range(0, len(xs), stride))
    if frames[-1] != len(xs) - 1:
        frames.append(len(xs) - 1)

    print(f"Telemetry samples: {len(xs)} | GIF frames: {len(frames)} | stride={stride}")
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000.0 / max(1, fps),
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print("GIF created:", output)


def main():
    args = parse_args()
    targets = args.targets or DEFAULT_TARGETS

    if any(abs(float(t)) < 1.0 or abs(float(t)) > 360.0 for t in targets):
        raise ValueError("Each target must satisfy 1 <= |target| <= 360 deg")
    if args.forward_seconds < 0.0:
        raise ValueError("--forward-seconds must be >= 0")

    print("=" * 130)
    print("AH-1S FINAL MULTI-TURN VISUALIZER")
    print("Matplotlib 3D + PillowWriter GIF | SAME FDM | runtime teacher OFF")
    print("Commands:", [float(x) for x in targets])
    print("=" * 130)

    recorder = Recorder()
    results = run_recorded_mission(targets, args.forward_seconds, recorder)

    print("\nMISSION RESULTS")
    for r in results:
        print(
            f"  target={r['target']:+7.1f} | PASS={r['success']} | "
            f"safety={r['safety']} | rem={r['remaining']:+.2f} | "
            f"heading={r['start_heading']:.2f}->{r['end_heading']:.2f}"
        )

    create_gif(
        recorder.rows,
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
