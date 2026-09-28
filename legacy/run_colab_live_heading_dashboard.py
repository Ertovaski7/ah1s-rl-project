from __future__ import annotations

"""
COLAB-NATIVE LIVE TARGET-HEADING DASHBOARD
==========================================

Run inside the Colab kernel with:

    %run run_colab_live_heading_dashboard.py

No Gradio, Hugging Face, localhost server or public share link is used.  The
HTML dashboard talks only to callbacks registered in the current Colab kernel.

Flight architecture:
    Stage1 takeoff/hover -> Stage2 forward -> user-entered absolute heading
    -> validated relative-turn controller -> bumpless recovery -> forward

The same live JSBSim FDM is preserved for the whole session.  Runtime teacher
is OFF.  The dashboard shows genuine simulator telemetry, not a GIF or a
prerecorded replay.
"""

import builtins
import contextlib
import io
import math
import queue
import threading
import time
import traceback
from collections import deque

import test_turn_arbitrary_angles_v1 as arb
import validate_final_continuous_mission_v1 as finalv
import validate_live_multiturn_same_fdm_v1 as live


EARTH_RADIUS_FT = 20_902_231.0
FORWARD_CHUNK_SIM_S = 0.30
FORWARD_CHUNK_WALL_S = 0.20
MIN_FORWARD_BETWEEN_COMMANDS_S = 3.0
UI_PROGRESS_WALL_S = 0.035
MAX_PENDING_COMMANDS = 2


def wrap_signed_deg(angle: float) -> float:
    """Wrap to (-180, 180], choosing +180 for the ambiguous opposite heading."""
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped <= -180.0 + 1e-9 else wrapped


def fdm_float(fdm, keys, default=float("nan")) -> float:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        try:
            value = float(fdm[key])
            if math.isfinite(value):
                return value
        except Exception:
            pass
    return float(default)


class ColabLiveEngine:
    """Single-owner JSBSim runtime with thread-safe snapshot callbacks."""

    def __init__(self):
        self.command_queue: queue.Queue[float] = queue.Queue(
            maxsize=MAX_PENDING_COMMANDS
        )
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
        self.has_turned = False

        self.origin_lat = None
        self.origin_lon = None
        self.origin_heading_rad = None
        self.origin_sim_t = None

        self.events = deque(maxlen=16)
        self.snapshot_data = {
            "seq": 0,
            "phase": "STARTING",
            "status": "Python runtime hazırlanıyor...",
            "sim_t": 0.0,
            "mission_time": 0.0,
            "alt": None,
            "heading": None,
            "roll": None,
            "pitch": None,
            "vs": None,
            "v": None,
            "lat_v": None,
            "yaw_rate": None,
            "forward": 0.0,
            "cross": 0.0,
            "target_heading": None,
            "relative_command": None,
            "heading_error": None,
            "safety": "CHECKING",
            "last_result": "-",
            "queue": 0,
        }

    def _log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.events.appendleft(f"[{stamp}] {text}")

    def _telemetry(self, fdm):
        raw = live.telemetry(fdm)
        lat = fdm_float(
            fdm, ("position/lat-geod-deg", "position/lat-gc-deg"), 0.0
        )
        lon = fdm_float(
            fdm, ("position/long-gc-deg", "position/long-geod-deg"), 0.0
        )
        heading = float(raw["heading"])
        sim_t = float(raw["sim_t"])

        if self.origin_lat is None:
            self.origin_lat = lat
            self.origin_lon = lon
            self.origin_heading_rad = math.radians(heading)
            self.origin_sim_t = sim_t

        north = EARTH_RADIUS_FT * math.radians(lat - self.origin_lat)
        east = (
            EARTH_RADIUS_FT
            * math.cos(math.radians(self.origin_lat))
            * math.radians(lon - self.origin_lon)
        )
        c0 = math.cos(self.origin_heading_rad)
        s0 = math.sin(self.origin_heading_rad)

        return {
            "sim_t": sim_t,
            "mission_time": sim_t - self.origin_sim_t,
            "alt": float(raw["alt"]),
            "heading": heading,
            "roll": float(raw["roll"]),
            "pitch": float(raw["pitch"]),
            "vs": float(raw["vs"]),
            "v": float(raw["v"]),
            "lat_v": float(raw["lat_v"]),
            "yaw_rate": float(raw["yaw_rate"]),
            "forward": float(north * c0 + east * s0),
            "cross": float(-north * s0 + east * c0),
        }

    def _publish(
        self,
        *,
        fdm=None,
        phase=None,
        status=None,
        target_heading="KEEP",
        relative_command="KEEP",
        safety=None,
        last_result=None,
    ):
        # Called only from the simulator thread whenever FDM is provided.
        tel = self._telemetry(fdm) if fdm is not None else None
        with self.lock:
            if tel is not None:
                self.snapshot_data.update(tel)
            if phase is not None:
                self.snapshot_data["phase"] = str(phase)
            if status is not None:
                self.snapshot_data["status"] = str(status)
            if target_heading != "KEEP":
                self.snapshot_data["target_heading"] = target_heading
            if relative_command != "KEEP":
                self.snapshot_data["relative_command"] = relative_command
            if safety is not None:
                self.snapshot_data["safety"] = str(safety)
            if last_result is not None:
                self.snapshot_data["last_result"] = str(last_result)

            target = self.snapshot_data["target_heading"]
            heading = self.snapshot_data["heading"]
            self.snapshot_data["heading_error"] = (
                wrap_signed_deg(float(target) - float(heading))
                if target is not None and heading is not None
                else None
            )
            self.snapshot_data["queue"] = int(self.command_queue.qsize())
            self.snapshot_data["seq"] += 1

    def snapshot(self):
        with self.lock:
            data = dict(self.snapshot_data)
            data["events"] = "\n".join(self.events) if self.events else "-"
        # Colab callback JSON cannot represent NaN/Infinity reliably.
        for key, value in list(data.items()):
            if isinstance(value, float) and not math.isfinite(value):
                data[key] = None
        return data

    def enqueue_heading(self, value):
        try:
            target = float(value)
        except (TypeError, ValueError):
            return {"ok": False, "message": "Geçerli bir heading gir: 0–359°."}
        if not math.isfinite(target) or not 0.0 <= target <= 360.0:
            return {"ok": False, "message": "Heading 0° ile 360° arasında olmalı."}
        target %= 360.0

        snap = self.snapshot()
        if snap["phase"] in {"ERROR", "STOPPED"}:
            return {
                "ok": False,
                "message": f"Runtime komut kabul etmiyor: {snap['phase']}",
            }
        try:
            self.command_queue.put_nowait(target)
        except queue.Full:
            return {
                "ok": False,
                "message": "Komut kuyruğu dolu; mevcut dönüşün bitmesini bekle.",
            }
        self._log(f"Target heading queued: {target:.1f} deg")
        with self.lock:
            self.snapshot_data["queue"] = int(self.command_queue.qsize())
        return {"ok": True, "message": f"Hedef {target:.1f}° kuyruğa eklendi."}

    def request_stop(self):
        self.stop_event.set()
        self._log("Stop requested")
        return {"ok": True, "message": "Simülasyonun durması istendi."}

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._run, name="ah1s-colab-simulator", daemon=True
        )
        self.thread.start()

    def _initial_progress(self, phase, fdm):
        self.fdm = fdm
        self.fdm_id = id(fdm)
        self._publish(
            fdm=fdm,
            phase=phase,
            status="Başlangıç uçuşu aynı canlı FDM üzerinde ilerliyor...",
            safety="CHECKING",
        )
        time.sleep(UI_PROGRESS_WALL_S)

    def _turn_progress(self, phase, target, relative):
        self._publish(
            fdm=self.fdm,
            phase=phase,
            status=f"Hedef {target:.1f}°; gerçek heading canlı güncelleniyor.",
            target_heading=target,
            relative_command=relative,
            safety="CHECKING",
        )
        time.sleep(UI_PROGRESS_WALL_S)

    def _run(self):
        try:
            self._publish(
                phase="INITIALIZING",
                status="Policy modelleri yükleniyor...",
                safety="CHECKING",
            )
            self._log("Loading validated controller stack")
            self.stack = arb.load_stack()

            self._log("Starting Stage1 and Stage2")
            self.env1, self.env2, self.fdm, self.fdm_id = (
                live.build_initial_live_mission(
                    progress_callback=self._initial_progress
                )
            )
            self.heading_ref = live.heading_deg(self.fdm)
            self._publish(
                fdm=self.fdm,
                phase="FORWARD / READY",
                status="Hazır. Target Heading alanına 0–359° arasında değer gir.",
                safety="OK",
            )
            self._log(f"Ready on shared FDM id={self.fdm_id}")

            while not self.stop_event.is_set():
                if self.forward_remaining <= 1e-9:
                    try:
                        target = self.command_queue.get_nowait()
                    except queue.Empty:
                        target = None

                    if target is not None:
                        start_heading = live.heading_deg(self.fdm)
                        relative = wrap_signed_deg(target - start_heading)
                        self._log(
                            f"Executing {start_heading:.2f} -> {target:.2f}; "
                            f"relative={relative:+.2f}"
                        )
                        self._publish(
                            fdm=self.fdm,
                            phase="TURN / PLANNING",
                            status=(
                                f"Current {start_heading:.1f}° → target {target:.1f}°; "
                                f"hesaplanan dönüş {relative:+.1f}°."
                            ),
                            target_heading=target,
                            relative_command=relative,
                            safety="CHECKING",
                        )

                        if abs(relative) < 1.0:
                            result_text = (
                                f"Zaten hedefte: heading={start_heading:.2f}°, "
                                f"error={relative:+.2f}°"
                            )
                            self.heading_ref = target
                            self.forward_remaining = MIN_FORWARD_BETWEEN_COMMANDS_S
                            self._publish(
                                fdm=self.fdm,
                                phase="POST-TURN — STAGE 2",
                                status="Helikopter hedef heading toleransı içinde.",
                                safety="OK",
                                last_result=result_text,
                            )
                            continue

                        result = finalv.execute_command_same_fdm(
                            self.stack,
                            self.fdm,
                            self.fdm_id,
                            relative,
                            progress_callback=lambda phase: self._turn_progress(
                                phase, target, relative
                            ),
                        )
                        self.heading_ref = live.heading_deg(self.fdm)
                        self.has_turned = True
                        result_text = (
                            f"target={target:.1f}°, relative={relative:+.1f}°, "
                            f"PASS={result['success']}, safety={result['safety']}, "
                            f"heading={result['start_heading']:.2f}°→"
                            f"{result['end_heading']:.2f}°, rem="
                            f"{result['remaining']:+.2f}°"
                        )
                        if not result["success"] or result["safety"]:
                            self._log("COMMAND FAILED: " + result_text)
                            self._publish(
                                fdm=self.fdm,
                                phase="ERROR",
                                status="Dönüş başarısız oldu veya safety sınırı aşıldı.",
                                safety="FAIL",
                                last_result=result_text,
                            )
                            break

                        self._log("COMMAND PASS: " + result_text)
                        self.forward_remaining = MIN_FORWARD_BETWEEN_COMMANDS_S
                        self._publish(
                            fdm=self.fdm,
                            phase="POST-TURN — STAGE 2",
                            status="Dönüş tamamlandı; yeni heading üzerinde ileri uçuş.",
                            safety="OK",
                            last_result=result_text,
                        )
                        continue

                self.fdm["ap/afcs/psi-trim-rad"] = math.radians(
                    float(self.heading_ref)
                )
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
                phase = (
                    "POST-TURN — STAGE 2"
                    if self.has_turned
                    else "STAGE 2 — FORWARD FLIGHT"
                )
                status = (
                    f"Yeni heading üzerinde stabilizasyon: "
                    f"{self.forward_remaining:.1f}s kaldı."
                    if self.forward_remaining > 0.0
                    else "İleri uçuş sürüyor; yeni target heading girebilirsin."
                )
                self._publish(
                    fdm=self.fdm,
                    phase=phase,
                    status=status,
                    safety="OK",
                )
                time.sleep(FORWARD_CHUNK_WALL_S)

        except Exception as exc:
            self._log(f"{type(exc).__name__}: {exc}")
            self._log(traceback.format_exc().splitlines()[-1])
            self._publish(
                fdm=self.fdm,
                phase="ERROR",
                status=f"{type(exc).__name__}: {exc}",
                safety="FAIL",
            )
        finally:
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
            if self.snapshot()["phase"] != "ERROR":
                self._publish(
                    phase="STOPPED",
                    status="Simülasyon durduruldu.",
                    safety="STOPPED",
                )
            self._log("Simulator thread exited")


COLAB_HTML = r"""
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
#ah1s-live{--bg:#0b1016;--panel:#111820;--border:#283441;--text:#edf4fa;--muted:#96a8b9;--blue:#2f9bff;--orange:#ff8a00;--green:#19ec91;--purple:#b974ff;--yellow:#ffd447;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif;border-radius:12px;padding:16px;min-height:980px}
#ah1s-live *{box-sizing:border-box}#ah1s-live .head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px;gap:18px;flex-wrap:wrap}#ah1s-live h1{font-size:22px;margin:0}#ah1s-live .sub{color:var(--muted);font-size:13px;margin-top:4px}#ah1s-live .ok{color:var(--green);font-weight:700}#ah1s-live .command{display:flex;gap:8px;align-items:center;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-bottom:12px;flex-wrap:wrap}#ah1s-live .command label{font-weight:700}#ah1s-live input[type=number]{width:160px;background:#0e151d;color:white;border:1px solid #405064;border-radius:7px;padding:9px 10px;font-size:16px}#ah1s-live button{background:#192431;color:#fff;border:1px solid #405064;border-radius:7px;padding:9px 14px;font-weight:700;cursor:pointer}#ah1s-live button.primary{background:#146b5b;border-color:#20b996}#ah1s-live button.stop{background:#612b34;border-color:#9a4655}#ah1s-feedback{color:#cbd7e1;font-size:13px;flex:1}#ah1s-live .grid{display:grid;grid-template-columns:1.55fr 1fr;gap:12px}#ah1s-live .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px}#ah1s-live .right{display:grid;grid-template-rows:1fr 1fr;gap:12px}#ah1s-3d{height:590px}#ah1s-top,#ah1s-side{height:285px}#ah1s-live .legend{display:flex;flex-wrap:wrap;gap:12px;margin:2px 0 8px}#ah1s-live .leg{font-size:12px;color:#d5dee7}#ah1s-live .dot{display:inline-block;width:18px;height:4px;border-radius:4px;margin-right:6px;vertical-align:middle}#ah1s-live .telemetry{margin-top:12px}#ah1s-live .telemetry h3{font-size:13px;margin:0 0 8px;color:#bcc9d4;letter-spacing:.07em}#ah1s-live .cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}#ah1s-live .card{background:#0e151d;border:1px solid var(--border);border-radius:9px;padding:9px}#ah1s-live .label{font-size:10px;color:#8498aa;text-transform:uppercase}#ah1s-live .value{font-size:17px;font-weight:700;margin-top:4px}#ah1s-live .phasecard{grid-column:span 2}#ah1s-live .statuscard{grid-column:span 3}#ah1s-live .log{white-space:pre-wrap;color:#9fb0be;font-family:ui-monospace,monospace;font-size:11px;max-height:125px;overflow:auto;margin-top:8px}@media(max-width:1000px){#ah1s-live .grid{grid-template-columns:1fr}#ah1s-live .cards{grid-template-columns:repeat(2,1fr)}#ah1s-live .phasecard,#ah1s-live .statuscard{grid-column:span 2}}
</style>
<div id="ah1s-live">
  <div class="head"><div><h1>AH-1S — LIVE TARGET-HEADING MISSION</h1><div class="sub">One continuous JSBSim FDM · runtime teacher OFF · Colab-native · no Hugging Face / no Gradio</div></div><div class="ok" id="ah1s-indicator">● CONNECTING</div></div>
  <div class="command"><label for="ah1s-target">Target Heading</label><input id="ah1s-target" type="number" min="0" max="360" step="1" value="90"><span>°</span><button class="primary" id="ah1s-send">Fly to Heading</button><button class="stop" id="ah1s-stop">Stop</button><span id="ah1s-feedback">Stage 1 ve Stage 2 hazırlanıyor...</span></div>
  <div class="legend"><span class="leg"><i class="dot" style="background:var(--blue)"></i>Stage 1 — Takeoff/Hover</span><span class="leg"><i class="dot" style="background:var(--orange)"></i>Stage 2 — Forward</span><span class="leg"><i class="dot" style="background:var(--green)"></i>Stage 3 — Turn</span><span class="leg"><i class="dot" style="background:var(--purple)"></i>AFCS Recovery</span><span class="leg"><i class="dot" style="background:var(--yellow)"></i>Post-turn Stage 2</span></div>
  <div class="grid"><div class="panel"><div id="ah1s-3d"></div></div><div class="right"><div class="panel"><div id="ah1s-top"></div></div><div class="panel"><div id="ah1s-side"></div></div></div></div>
  <div class="panel telemetry"><h3>LIVE TELEMETRY</h3><div class="cards">
    <div class="card phasecard"><div class="label">Mission phase</div><div id="ah1s-phase" class="value">STARTING</div></div><div class="card statuscard"><div class="label">Status</div><div id="ah1s-status" class="value">Python runtime hazırlanıyor...</div></div>
    <div class="card"><div class="label">Mission time</div><div id="ah1s-time" class="value">—</div></div><div class="card"><div class="label">Altitude</div><div id="ah1s-alt" class="value">—</div></div><div class="card"><div class="label">Vertical speed</div><div id="ah1s-vs" class="value">—</div></div><div class="card"><div class="label">Forward distance</div><div id="ah1s-forward" class="value">—</div></div><div class="card"><div class="label">Cross-track</div><div id="ah1s-cross" class="value">—</div></div>
    <div class="card"><div class="label">Forward speed</div><div id="ah1s-v" class="value">—</div></div><div class="card"><div class="label">Lateral speed</div><div id="ah1s-lat" class="value">—</div></div><div class="card"><div class="label">Roll / Pitch</div><div id="ah1s-att" class="value">—</div></div><div class="card"><div class="label">Current heading</div><div id="ah1s-heading" class="value">—</div></div><div class="card"><div class="label">Target heading</div><div id="ah1s-target-card" class="value">—</div></div>
    <div class="card"><div class="label">Heading error</div><div id="ah1s-error" class="value">—</div></div><div class="card"><div class="label">Computed turn</div><div id="ah1s-relative" class="value">—</div></div><div class="card"><div class="label">Yaw rate</div><div id="ah1s-yaw" class="value">—</div></div><div class="card"><div class="label">Safety</div><div id="ah1s-safety" class="value">CHECKING</div></div><div class="card"><div class="label">Queued commands</div><div id="ah1s-queue" class="value">0</div></div>
  </div><div id="ah1s-events" class="log">—</div></div>
</div>
<script>
(function bootAH1S(){
  if(!window.Plotly || !window.google || !google.colab || !google.colab.kernel){setTimeout(bootAH1S,120);return;}
  const H=[], seen={seq:-1};
  const colors={stage1:'#2f9bff',stage2:'#ff8a00',turn:'#19ec91',recovery:'#b974ff',post:'#ffd447'};
  const keys=['stage1','stage2','turn','recovery','post'];
  const phaseKey=p=>{p=String(p||'').toLowerCase();if(p.includes('stage 1'))return'stage1';if(p.includes('post-turn'))return'post';if(p.includes('recovery')||p.includes('afcs'))return'recovery';if(p.includes('turn'))return'turn';return'stage2'};
  const finite=x=>Number.isFinite(Number(x)), fmt=(x,n=1)=>finite(x)?Number(x).toFixed(n):'—', set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v};
  const base={paper_bgcolor:'#111820',plot_bgcolor:'#111820',font:{color:'#e8f0f7'},margin:{l:55,r:18,t:42,b:45},showlegend:false,uirevision:'ah1s-live'};
  function grouped(axisY){return keys.map(k=>({x:H.map(r=>phaseKey(r.phase)===k?r.forward:null),y:H.map(r=>phaseKey(r.phase)===k?(axisY==='cross'?r.cross:r.alt):null),mode:'lines',type:'scatter',line:{color:colors[k],width:4},connectgaps:false,name:k}))}
  function grouped3d(){return keys.map(k=>({x:H.map(r=>phaseKey(r.phase)===k?r.forward:null),y:H.map(r=>phaseKey(r.phase)===k?r.cross:null),z:H.map(r=>phaseKey(r.phase)===k?r.alt:null),mode:'lines',type:'scatter3d',line:{color:colors[k],width:6},connectgaps:false,name:k}))}
  function draw(s){
    const marker3={x:[s.forward||0],y:[s.cross||0],z:[s.alt||0],mode:'markers',type:'scatter3d',marker:{size:7,color:'#ffffff',symbol:'diamond'},name:'Aircraft'};
    const markerTop={x:[s.forward||0],y:[s.cross||0],mode:'markers',type:'scatter',marker:{size:10,color:'#ffffff',symbol:'diamond'},name:'Aircraft'};
    const markerSide={x:[s.forward||0],y:[s.alt||0],mode:'markers',type:'scatter',marker:{size:10,color:'#ffffff',symbol:'diamond'},name:'Aircraft'};
    Plotly.react('ah1s-3d',[...grouped3d(),marker3],{...base,title:'3D LIVE MISSION',scene:{bgcolor:'#111820',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Cross-track (ft)',gridcolor:'#293746'},zaxis:{title:'Altitude AGL (ft)',gridcolor:'#293746'},aspectmode:'data'}});
    Plotly.react('ah1s-top',[...grouped('cross'),markerTop],{...base,title:'TOP VIEW — LIVE MISSION',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Cross-track (ft)',gridcolor:'#293746',scaleanchor:'x',scaleratio:1}});
    Plotly.react('ah1s-side',[...grouped('alt'),markerSide],{...base,title:'SIDE VIEW — ALTITUDE PROFILE',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Altitude AGL (ft)',gridcolor:'#293746'}});
  }
  function updateCards(s){set('ah1s-phase',s.phase);set('ah1s-status',s.status);set('ah1s-time',fmt(s.mission_time,1)+' s');set('ah1s-alt',fmt(s.alt,1)+' ft');set('ah1s-vs',fmt(s.vs,2)+' ft/s');set('ah1s-forward',fmt(s.forward,1)+' ft');set('ah1s-cross',fmt(s.cross,1)+' ft');set('ah1s-v',fmt(s.v,2)+' ft/s');set('ah1s-lat',fmt(s.lat_v,2)+' ft/s');set('ah1s-att',fmt(s.roll,1)+'° / '+fmt(s.pitch,1)+'°');set('ah1s-heading',fmt(s.heading,1)+'°');set('ah1s-target-card',finite(s.target_heading)?fmt(s.target_heading,1)+'°':'—');set('ah1s-error',finite(s.heading_error)?((Number(s.heading_error)>=0?'+':'')+fmt(s.heading_error,1)+'°'):'—');set('ah1s-relative',finite(s.relative_command)?((Number(s.relative_command)>=0?'+':'')+fmt(s.relative_command,1)+'°'):'—');set('ah1s-yaw',fmt(s.yaw_rate,2)+'°/s');set('ah1s-safety',s.safety);set('ah1s-queue',String(s.queue));set('ah1s-events',s.events||'—');set('ah1s-indicator',s.phase==='ERROR'?'● ERROR':s.phase==='STOPPED'?'● STOPPED':'● CURRENT TELEMETRY');document.getElementById('ah1s-indicator').style.color=s.phase==='ERROR'?'#ff6b6b':s.phase==='STOPPED'?'#ffd447':'#19ec91'}
  async function call(name,args=[]){const r=await google.colab.kernel.invokeFunction(name,args,{});return r.data['application/json'];}
  async function poll(){try{const s=await call('ah1s_live.snapshot');updateCards(s);if(s.seq!==seen.seq&&finite(s.alt)){seen.seq=s.seq;H.push(s);if(H.length>5000)H.shift();draw(s)}}catch(e){set('ah1s-feedback','Bağlantı hatası: '+e)}}
  document.getElementById('ah1s-send').onclick=async()=>{const v=Number(document.getElementById('ah1s-target').value);if(!finite(v)||v<0||v>360){set('ah1s-feedback','0–360 arasında heading gir.');return}set('ah1s-feedback','Komut gönderiliyor...');try{const r=await call('ah1s_live.enqueue',[v]);set('ah1s-feedback',r.message)}catch(e){set('ah1s-feedback','Komut hatası: '+e)}};
  document.getElementById('ah1s-target').onkeydown=e=>{if(e.key==='Enter')document.getElementById('ah1s-send').click()};
  document.getElementById('ah1s-stop').onclick=async()=>{const r=await call('ah1s_live.stop');set('ah1s-feedback',r.message)};
  draw({forward:0,cross:0,alt:0});poll();setInterval(poll,250);
})();
</script>
"""


def main():
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise RuntimeError(
            "Bu dosya Colab kernel içinde `%run run_colab_live_heading_dashboard.py` "
            "ile çalıştırılmalıdır."
        ) from exc

    previous = getattr(builtins, "_ah1s_colab_live_engine", None)
    if previous is not None:
        try:
            previous.request_stop()
        except Exception:
            pass

    engine = ColabLiveEngine()
    builtins._ah1s_colab_live_engine = engine

    output.register_callback(
        "ah1s_live.snapshot", lambda: JSON(engine.snapshot())
    )
    output.register_callback(
        "ah1s_live.enqueue", lambda value: JSON(engine.enqueue_heading(value))
    )
    output.register_callback(
        "ah1s_live.stop", lambda: JSON(engine.request_stop())
    )

    display(HTML(COLAB_HTML))
    engine.start()
    print(
        "AH-1S Colab-native dashboard started. "
        "Wait for FORWARD / READY, then enter a target heading."
    )


if __name__ == "__main__":
    main()
