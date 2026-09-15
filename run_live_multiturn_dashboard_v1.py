from __future__ import annotations

"""
LIVE MULTI-TURN DASHBOARD V1
============================

Final interactive runtime for Colab/demo use:
- Stage1 takeoff -> Stage2 forward flight happens once.
- One JSBSim FDM remains alive for the whole session.
- User submits RELATIVE turn commands while the simulation is already running.
- Arbitrary command execution reuses the validated final continuous-mission
  controller (V11 supervisory primitives + bumpless recovery).
- Between commands, Stage2 keeps flying forward on the latest heading.
- Runtime teacher remains OFF.

Concurrency rule
----------------
The simulator thread is the ONLY thread that touches JSBSim/FDM.  Gradio
callbacks only enqueue commands and read a copied telemetry snapshot.
"""

import argparse
import contextlib
import io
import math
import queue
import threading
import time
import traceback
from collections import deque

import validate_live_multiturn_same_fdm_v1 as live
import validate_final_continuous_mission_v1 as finalv
import test_turn_arbitrary_angles_v1 as arb


FORWARD_CHUNK_SIM_S = 0.30
FORWARD_CHUNK_WALL_S = 0.30
MIN_FORWARD_BETWEEN_COMMANDS_S = 3.0
MAX_PENDING_COMMANDS = 3


class LiveEngine:
    def __init__(self):
        self.command_queue: queue.Queue[float] = queue.Queue(maxsize=MAX_PENDING_COMMANDS)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

        self.stack = None
        self.env1 = None
        self.env2 = None
        self.fdm = None
        self.fdm_id = None
        self.heading_ref = None
        self.forward_remaining = 0.0

        self.events = deque(maxlen=20)
        self.snapshot_data = {
            "phase": "STARTING",
            "status": "Runtime thread has not started yet.",
            "sim_t": 0.0,
            "alt": float("nan"),
            "heading": float("nan"),
            "roll": float("nan"),
            "pitch": float("nan"),
            "vs": float("nan"),
            "v": float("nan"),
            "lat_v": float("nan"),
            "yaw_rate": float("nan"),
            "requested": None,
            "remaining": None,
            "safety": "UNKNOWN",
            "last_result": "-",
            "queue": 0,
            "fdm_id": "-",
        }

    def _log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.events.appendleft(f"[{stamp}] {text}")

    def _telemetry_snapshot(self):
        if self.fdm is None:
            return None
        s = live.telemetry(self.fdm)
        return {
            "sim_t": float(s["sim_t"]),
            "alt": float(s["alt"]),
            "heading": float(s["heading"]),
            "roll": float(s["roll"]),
            "pitch": float(s["pitch"]),
            "vs": float(s["vs"]),
            "v": float(s["v"]),
            "lat_v": float(s["lat_v"]),
            "yaw_rate": float(s["yaw_rate"]),
        }

    def _publish(
        self,
        *,
        phase: str | None = None,
        status: str | None = None,
        requested=None,
        remaining=None,
        safety: str | None = None,
        last_result: str | None = None,
    ):
        tel = self._telemetry_snapshot()
        with self.lock:
            if tel is not None:
                self.snapshot_data.update(tel)
            if phase is not None:
                self.snapshot_data["phase"] = phase
            if status is not None:
                self.snapshot_data["status"] = status
            self.snapshot_data["requested"] = requested
            self.snapshot_data["remaining"] = remaining
            if safety is not None:
                self.snapshot_data["safety"] = safety
            if last_result is not None:
                self.snapshot_data["last_result"] = last_result
            self.snapshot_data["queue"] = int(self.command_queue.qsize())
            self.snapshot_data["fdm_id"] = str(self.fdm_id) if self.fdm_id is not None else "-"

    def snapshot(self):
        with self.lock:
            d = dict(self.snapshot_data)
            d["events"] = "\n".join(self.events) if self.events else "-"
        return d

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="ah1s-simulator", daemon=True)
        self.thread.start()

    def enqueue(self, value):
        try:
            target = float(value)
        except (TypeError, ValueError):
            return False, "Geçerli bir sayı gir. Örn: 20, -30, 75."

        if not math.isfinite(target) or abs(target) < 1.0 or abs(target) > 360.0:
            return False, "Komut 1° ile 360° arasında olmalı. Sol dönüş için negatif değer kullan."

        snap = self.snapshot()
        if snap["phase"] in {"ERROR", "STOPPED"}:
            return False, f"Runtime komut kabul etmiyor: {snap['phase']}"

        try:
            self.command_queue.put_nowait(target)
        except queue.Full:
            return False, "Komut kuyruğu dolu. Önce mevcut dönüşlerin bitmesini bekle."

        self._log(f"Queued relative command {target:+.1f} deg")
        self._publish()
        return True, f"{target:+.1f}° relatif dönüş kuyruğa eklendi."

    def request_stop(self):
        self.stop_event.set()
        self._log("Stop requested from UI")
        return "Stop istendi. Simülasyon güvenli noktada kapatılacak."

    def _run(self):
        try:
            self._publish(
                phase="INITIALIZING",
                status="Turn modelleri yükleniyor, ardından Stage1 ve Stage2 başlatılacak...",
                safety="CHECKING",
            )
            self._log("Loading verified turn stack")
            self.stack = arb.load_stack()

            self._log("Stage1 takeoff + Stage2 forward entry starting")
            self.env1, self.env2, self.fdm, self.fdm_id = live.build_initial_live_mission()
            self.heading_ref = live.heading_deg(self.fdm)
            self.forward_remaining = 0.0

            self._log(f"Live mission ready on shared FDM id={self.fdm_id}")
            self._publish(
                phase="FORWARD / READY",
                status="Stage1 ve Stage2 tamamlandı. Relatif dönüş komutu girebilirsin.",
                safety="OK",
            )

            while not self.stop_event.is_set():
                # Execute a queued command only after the mandatory post-command
                # forward interval has elapsed.
                if self.forward_remaining <= 1e-9:
                    try:
                        target = self.command_queue.get_nowait()
                    except queue.Empty:
                        target = None

                    if target is not None:
                        start_hdg = live.heading_deg(self.fdm)
                        self._log(
                            f"Executing {target:+.1f} deg from live heading {start_hdg:.2f} deg"
                        )
                        self._publish(
                            phase="TURN / RECOVERY",
                            status=(
                                f"{target:+.1f}° relatif komut yürütülüyor. "
                                "Aynı FDM üzerinde primitive + bumpless recovery çalışıyor."
                            ),
                            requested=target,
                            remaining=target,
                            safety="CHECKING",
                        )

                        result = finalv.execute_command_same_fdm(
                            self.stack,
                            self.fdm,
                            self.fdm_id,
                            float(target),
                        )

                        self.heading_ref = live.heading_deg(self.fdm)
                        result_text = (
                            f"target={result['target']:+.1f}°, PASS={result['success']}, "
                            f"safety={result['safety']}, remaining={result['remaining']:+.2f}°, "
                            f"heading={result['start_heading']:.2f}°→{result['end_heading']:.2f}°"
                        )

                        if not result["success"] or result["safety"]:
                            self._log("COMMAND FAILED: " + result_text)
                            self._publish(
                                phase="ERROR",
                                status="Turn command failed or violated the safety envelope.",
                                requested=target,
                                remaining=result["remaining"],
                                safety="FAIL",
                                last_result=result_text,
                            )
                            break

                        self._log("COMMAND PASS: " + result_text)
                        self.forward_remaining = MIN_FORWARD_BETWEEN_COMMANDS_S
                        self._publish(
                            phase="FORWARD STABILIZATION",
                            status=(
                                f"Komut tamamlandı. En az {MIN_FORWARD_BETWEEN_COMMANDS_S:.1f}s "
                                "ileri uçuş yapılacak; sonraki komut varsa sonra başlayacak."
                            ),
                            requested=None,
                            remaining=None,
                            safety="OK",
                            last_result=result_text,
                        )
                        continue

                # Continuous forward flight while waiting.  Reuse the same
                # validated short Stage2 continuation in small chunks.  Its
                # stdout is suppressed so the notebook is not flooded.
                self.fdm["ap/afcs/psi-trim-rad"] = math.radians(float(self.heading_ref))
                with contextlib.redirect_stdout(io.StringIO()):
                    live.fly_stage2_between_turns(
                        self.env2,
                        self.fdm,
                        FORWARD_CHUNK_SIM_S,
                        self.fdm_id,
                    )

                self.forward_remaining = max(
                    0.0, self.forward_remaining - FORWARD_CHUNK_SIM_S
                )

                if self.forward_remaining > 1e-9:
                    phase = "FORWARD STABILIZATION"
                    status = (
                        f"Yeni heading üzerinde ileri uçuş: "
                        f"{self.forward_remaining:.1f}s zorunlu geçiş kaldı."
                    )
                else:
                    phase = "FORWARD / READY"
                    status = "İleri uçuş devam ediyor. Yeni relatif dönüş komutu bekleniyor."

                self._publish(
                    phase=phase,
                    status=status,
                    requested=None,
                    remaining=None,
                    safety="OK",
                )
                time.sleep(FORWARD_CHUNK_WALL_S)

        except Exception as exc:
            self._log(f"Runtime exception: {type(exc).__name__}: {exc}")
            self._log(traceback.format_exc().splitlines()[-1])
            self._publish(
                phase="ERROR",
                status=f"{type(exc).__name__}: {exc}",
                safety="FAIL",
            )
        finally:
            # env1 owns the shared live FDM.  Detach env2 first, exactly like
            # the validators, so JSBSim is closed only once.
            if self.env2 is not None:
                try:
                    self.env2.fdm = None
                    self.env2.close()
                except Exception:
                    pass
            if self.env1 is not None:
                try:
                    self.env1.close()
                except Exception:
                    pass

            if self.snapshot().get("phase") != "ERROR":
                self._publish(
                    phase="STOPPED",
                    status="Simülasyon durduruldu.",
                    safety="STOPPED",
                )
            self._log("Simulator thread exited")


def fmt(x, digits=2):
    try:
        v = float(x)
        if not math.isfinite(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def build_ui(engine: LiveEngine):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            'Gradio bulunamadı. Colab\'da önce: !pip -q install "gradio>=5,<6"'
        ) from exc

    def submit(value):
        ok, message = engine.enqueue(value)
        return message

    def stop_runtime():
        return engine.request_stop()

    def refresh():
        s = engine.snapshot()
        return (
            s["phase"],
            s["status"],
            fmt(s["sim_t"]),
            fmt(s["alt"]),
            fmt(s["heading"]),
            fmt(s["roll"]),
            fmt(s["pitch"]),
            fmt(s["vs"]),
            fmt(s["v"]),
            fmt(s["lat_v"]),
            fmt(s["yaw_rate"]),
            "-" if s["requested"] is None else f"{float(s['requested']):+.1f}",
            "-" if s["remaining"] is None else f"{float(s['remaining']):+.2f}",
            s["safety"],
            s["last_result"],
            str(s["queue"]),
            s["fdm_id"],
            s["events"],
        )

    with gr.Blocks(title="AH-1S Live Same-FDM Controller") as demo:
        gr.Markdown(
            "# AH-1S Live Same-FDM Controller\n"
            "**ONE live JSBSim FDM · runtime teacher OFF · relative turn commands**"
        )
        gr.Markdown(
            "Pozitif açı sağa, negatif açı sola relatif dönüş komutudur. "
            "Örn. `20`, `-30`, `75`, `120`."
        )

        with gr.Row():
            command = gr.Number(label="Relative Turn Command (deg)", value=20.0)
            submit_btn = gr.Button("Queue Turn Command", variant="primary")
            stop_btn = gr.Button("Stop Runtime", variant="stop")

        command_feedback = gr.Textbox(label="Command feedback", interactive=False)

        with gr.Row():
            phase = gr.Textbox(label="Phase", interactive=False)
            status = gr.Textbox(label="Status", interactive=False, lines=2)
            safety = gr.Textbox(label="Safety", interactive=False)

        with gr.Row():
            sim_t = gr.Textbox(label="Sim Time (s)", interactive=False)
            altitude = gr.Textbox(label="Altitude (ft)", interactive=False)
            heading = gr.Textbox(label="Heading (deg)", interactive=False)
            vs = gr.Textbox(label="Vertical Speed (ft/s)", interactive=False)

        with gr.Row():
            forward_v = gr.Textbox(label="Forward Speed (ft/s)", interactive=False)
            lateral_v = gr.Textbox(label="Lateral Speed (ft/s)", interactive=False)
            roll = gr.Textbox(label="Roll (deg)", interactive=False)
            pitch = gr.Textbox(label="Pitch (deg)", interactive=False)
            yaw_rate = gr.Textbox(label="Yaw Rate (deg/s)", interactive=False)

        with gr.Row():
            requested = gr.Textbox(label="Requested Turn (deg)", interactive=False)
            remaining = gr.Textbox(label="Remaining (deg)", interactive=False)
            queued = gr.Textbox(label="Queued Commands", interactive=False)
            fdm_id = gr.Textbox(label="Shared FDM id", interactive=False)

        last_result = gr.Textbox(label="Last command result", interactive=False, lines=2)
        event_log = gr.Textbox(label="Event log", interactive=False, lines=12)

        submit_btn.click(submit, inputs=command, outputs=command_feedback)
        command.submit(submit, inputs=command, outputs=command_feedback)
        stop_btn.click(stop_runtime, outputs=command_feedback)

        outputs = [
            phase,
            status,
            sim_t,
            altitude,
            heading,
            roll,
            pitch,
            vs,
            forward_v,
            lateral_v,
            yaw_rate,
            requested,
            remaining,
            safety,
            last_result,
            queued,
            fdm_id,
            event_log,
        ]
        timer = gr.Timer(value=0.5, active=True)
        timer.tick(refresh, outputs=outputs)
        demo.load(refresh, outputs=outputs)

    return demo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-share",
        action="store_true",
        help="Disable Gradio public share link (normally needed in Colab).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    engine = LiveEngine()
    demo = build_ui(engine)
    engine.start()

    print("=" * 110)
    print("AH-1S LIVE DASHBOARD V1")
    print("The simulator runs in one background thread; UI callbacks never touch JSBSim directly.")
    print("Wait until Phase becomes FORWARD / READY, then submit relative angles from the browser UI.")
    print("=" * 110)

    demo.launch(
        share=not args.no_share,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
