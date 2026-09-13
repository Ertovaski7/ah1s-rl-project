from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def angle_tag(angle: float) -> str:
    return f"{angle:+.0f}".replace("+", "plus").replace("-", "minus")


def f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_replay(rows):
    if not rows:
        raise RuntimeError("Telemetry CSV is empty")

    t0 = f(rows[0], "sim_t", 0.0)
    h0 = math.radians(f(rows[0], "heading_deg", 0.0))

    north = 0.0
    east = 0.0
    prev_t = t0
    out = []

    for i, row in enumerate(rows):
        sim_t = f(row, "sim_t", t0)
        dt = max(0.0, sim_t - prev_t) if i else 0.0
        prev_t = sim_t

        u = f(row, "forward_speed_fps", 0.0)
        v = f(row, "lateral_speed_fps", 0.0)
        psi = math.radians(f(row, "heading_deg", 0.0))

        # Approximate local horizontal trajectory reconstructed from body-axis
        # aerodynamic velocities and measured heading. This is replay geometry
        # only; original V23 control/validation results are unchanged.
        vn = u * math.cos(psi) - v * math.sin(psi)
        ve = u * math.sin(psi) + v * math.cos(psi)
        north += vn * dt
        east += ve * dt

        forward = north * math.cos(h0) + east * math.sin(h0)
        cross = -north * math.sin(h0) + east * math.cos(h0)

        phase = row.get("phase", "Unknown")
        phase_display = {
            "Stage1 Takeoff/Hover": "STAGE 1 — TAKEOFF / HOVER",
            "Stage2 Forward": "STAGE 2 — FORWARD FLIGHT",
            "Turn": "STAGE 3 — RELATIVE TURN",
            "Post-turn AFCS Transition": "TRANSITION — AFCS STABILIZATION",
            "Post-turn Stage2": "POST-TURN — STAGE 2 PPO",
        }.get(phase, phase)

        out.append({
            "phase": phase,
            "phase_display": phase_display,
            "time_s": sim_t - t0,
            "north_ft": north,
            "east_ft": east,
            "forward_ft": forward,
            "cross_track_ft": cross,
            "altitude_ft": f(row, "altitude_ft"),
            "vertical_speed_fps": f(row, "vertical_speed_fps"),
            "forward_speed_fps": u,
            "lateral_speed_fps": v,
            "roll_deg": f(row, "roll_deg"),
            "pitch_deg": f(row, "pitch_deg"),
            "yaw_rate_deg_s": f(row, "yaw_rate_deg_s"),
            "heading_deg": f(row, "heading_deg"),
            "heading_error_deg": f(row, "heading_error_deg"),
            "rotor_rpm": f(row, "rotor_rpm"),
            "collective_cmd": f(row, "collective_cmd"),
            "elevator_cmd": f(row, "elevator_cmd"),
            "aileron_cmd": f(row, "aileron_cmd"),
            "rudder_cmd": f(row, "rudder_cmd"),
            "cumulative_turn_deg": f(row, "cumulative_turn_deg"),
            "remaining_turn_deg": f(row, "remaining_turn_deg"),
        })
    return out


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AH-1S Final Mission Interactive Replay</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0d1117;--panel:#111820;--border:#2a3139;--text:#e6edf3;--muted:#9ba7b4;--accent:#ff7a00;--green:#00c853}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:18px}.title{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px}
.title h1{font-size:22px;margin:0}.title .sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:1.6fr 1fr;gap:12px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px}
#scene3d{height:560px}.right{display:grid;grid-template-rows:1fr 1fr;gap:12px}.smallplot{height:270px}
.controls{display:flex;gap:8px;align-items:center;margin-top:10px}.controls button{background:#1e2936;color:#fff;border:1px solid #394653;border-radius:6px;padding:8px 14px;cursor:pointer}.controls button:hover{background:#263545}
#slider{flex:1}.time{min-width:84px;text-align:right;font-weight:700}.caption{margin-top:6px;color:var(--muted);font-size:12px}
.telemetry{margin-top:12px}.telemetry h3{margin:0 0 8px 0;font-size:13px;color:#b9c3cd;letter-spacing:.08em}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.card{background:#0f151c;border:1px solid var(--border);border-radius:9px;padding:9px}.label{font-size:11px;color:#8fa0b0}.value{font-size:18px;font-weight:700;margin-top:3px}.phase .value{font-size:16px;color:#fff}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#17202a;border:1px solid #33414f;color:#dce4ec;font-size:11px;margin-left:8px}
@media(max-width:950px){.grid{grid-template-columns:1fr}.right{grid-template-columns:1fr;grid-template-rows:auto}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<div class="wrap">
  <div class="title"><div><h1>AH-1S Final Mission Interactive Replay <span class="badge">V23 telemetry</span></h1><div class="sub">Same validated flight data — replay only</div></div><div id="phaseTop" class="sub"></div></div>
  <div class="grid">
    <div class="panel"><div id="scene3d"></div><div class="controls"><button id="playBtn">Play</button><button id="pauseBtn">Pause</button><input id="slider" type="range" min="0" value="0" step="1"/><div class="time" id="timeText">0.0 s</div></div><div class="caption">Stage 1 → Stage 2 → Turn → AFCS Transition → Post-turn Stage 2</div></div>
    <div class="right"><div class="panel"><div id="topView" class="smallplot"></div></div><div class="panel"><div id="sideView" class="smallplot"></div></div></div>
  </div>
  <div class="panel telemetry"><h3>LIVE TELEMETRY</h3><div class="cards">
    <div class="card phase"><div class="label">Phase</div><div class="value" id="phase"></div></div>
    <div class="card"><div class="label">Time</div><div class="value" id="time"></div></div>
    <div class="card"><div class="label">Forward</div><div class="value" id="forward"></div></div>
    <div class="card"><div class="label">Cross-track</div><div class="value" id="cross"></div></div>
    <div class="card"><div class="label">Altitude</div><div class="value" id="alt"></div></div>
    <div class="card"><div class="label">Vertical speed</div><div class="value" id="vs"></div></div>
    <div class="card"><div class="label">Forward speed</div><div class="value" id="u"></div></div>
    <div class="card"><div class="label">Lateral speed</div><div class="value" id="v"></div></div>
    <div class="card"><div class="label">Pitch</div><div class="value" id="pitch"></div></div>
    <div class="card"><div class="label">Roll</div><div class="value" id="roll"></div></div>
    <div class="card"><div class="label">Heading / error</div><div class="value" id="heading"></div></div>
    <div class="card"><div class="label">Rotor RPM</div><div class="value" id="rpm"></div></div>
    <div class="card"><div class="label">Collective</div><div class="value" id="collective"></div></div>
    <div class="card"><div class="label">Elevator</div><div class="value" id="elevator"></div></div>
    <div class="card"><div class="label">Aileron / Rudder</div><div class="value" id="latctrl"></div></div>
    <div class="card"><div class="label">Turn progress</div><div class="value" id="turn"></div></div>
  </div></div>
</div>
<script>
const data = __DATA__;
const slider=document.getElementById('slider'); slider.max=Math.max(0,data.length-1);
let idx=0,timer=null;
const bg='#111820',paper='#111820',grid='#2a3139',text='#dce4ec',accent='#ff7a00',green='#00c853';
const xs=data.map(d=>d.forward_ft), ys=data.map(d=>d.cross_track_ft), zs=data.map(d=>d.altitude_ft);
const bounds=(a)=>{let lo=Math.min(...a),hi=Math.max(...a);if(!isFinite(lo)||!isFinite(hi)){lo=0;hi=1} if(Math.abs(hi-lo)<1){lo-=1;hi+=1} const p=(hi-lo)*.08;return [lo-p,hi+p]};
const xr=bounds(xs),yr=bounds(ys),zr=[Math.min(0,Math.min(...zs)-10),Math.max(...zs)+20];
const common={paper_bgcolor:paper,plot_bgcolor:bg,font:{color:text},margin:{l:55,r:15,t:35,b:45},showlegend:false};
Plotly.newPlot('scene3d',[{type:'scatter3d',mode:'lines',x:xs,y:ys,z:zs,line:{color:accent,width:4}},{type:'scatter3d',mode:'markers',x:[xs[0]],y:[ys[0]],z:[zs[0]],marker:{size:6,color:green}},{type:'scatter3d',mode:'markers',x:[xs[0]],y:[ys[0]],z:[zs[0]],marker:{size:7,color:'#00b0ff'}}],{...common,title:'3D FLIGHT PATH',scene:{bgcolor:bg,xaxis:{title:'Forward (ft)',gridcolor:grid,range:xr},yaxis:{title:'Cross-track (ft)',gridcolor:grid,range:yr},zaxis:{title:'Altitude AGL (ft)',gridcolor:grid,range:zr},aspectmode:'manual',aspectratio:{x:1.5,y:1,z:1.1}}});
Plotly.newPlot('topView',[{type:'scatter',mode:'lines',x:xs,y:ys,line:{color:accent,width:3}},{type:'scatter',mode:'markers',x:[xs[0]],y:[ys[0]],marker:{size:9,color:'#00b0ff'}}],{...common,title:'TOP VIEW',xaxis:{title:'Forward (ft)',gridcolor:grid,range:xr},yaxis:{title:'Cross-track (ft)',gridcolor:grid,range:yr}});
Plotly.newPlot('sideView',[{type:'scatter',mode:'lines',x:xs,y:zs,line:{color:accent,width:3}},{type:'scatter',mode:'markers',x:[xs[0]],y:[zs[0]],marker:{size:9,color:'#00b0ff'}}],{...common,title:'SIDE VIEW',xaxis:{title:'Forward (ft)',gridcolor:grid,range:xr},yaxis:{title:'Altitude AGL (ft)',gridcolor:grid,range:zr}});
function fmt(v,n=2){return Number.isFinite(v)?v.toFixed(n):'—'}
function setText(id,s){document.getElementById(id).textContent=s}
function render(i){idx=Math.max(0,Math.min(data.length-1,i));slider.value=idx;const d=data[idx];
  Plotly.restyle('scene3d',{x:[[d.forward_ft]],y:[[d.cross_track_ft]],z:[[d.altitude_ft]]},[2]);
  Plotly.restyle('topView',{x:[[d.forward_ft]],y:[[d.cross_track_ft]]},[1]);
  Plotly.restyle('sideView',{x:[[d.forward_ft]],y:[[d.altitude_ft]]},[1]);
  setText('phase',d.phase_display);setText('phaseTop',d.phase_display);setText('timeText',fmt(d.time_s,1)+' s');setText('time',fmt(d.time_s,2)+' s');
  setText('forward',fmt(d.forward_ft)+' ft');setText('cross',fmt(d.cross_track_ft)+' ft');setText('alt',fmt(d.altitude_ft)+' ft');setText('vs',fmt(d.vertical_speed_fps)+' ft/s');
  setText('u',fmt(d.forward_speed_fps)+' ft/s');setText('v',fmt(d.lateral_speed_fps)+' ft/s');setText('pitch',fmt(d.pitch_deg)+'°');setText('roll',fmt(d.roll_deg)+'°');
  setText('heading',fmt(d.heading_deg,1)+'° / '+fmt(d.heading_error_deg,1)+'°');setText('rpm',fmt(d.rotor_rpm,1));setText('collective',fmt(d.collective_cmd,3));
  setText('elevator',fmt(d.elevator_cmd,3));setText('latctrl',fmt(d.aileron_cmd,3)+' / '+fmt(d.rudder_cmd,3));setText('turn',fmt(d.cumulative_turn_deg,1)+'° / rem '+fmt(d.remaining_turn_deg,1)+'°');
}
function play(){if(timer)return;timer=setInterval(()=>{if(idx>=data.length-1){clearInterval(timer);timer=null;return}render(idx+1)},35)}
function pause(){if(timer){clearInterval(timer);timer=null}}
document.getElementById('playBtn').onclick=play;document.getElementById('pauseBtn').onclick=pause;slider.oninput=e=>{pause();render(Number(e.target.value))};render(0);
</script></body></html>'''


def main():
    raw = input("Relative turn angle for replay (-50, 50, 200, 360): ").strip()
    angle = float(raw)
    tag = angle_tag(angle)
    base = Path("visualization_final") / f"turn_{tag}"
    csv_path = base / "full_mission_telemetry.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}\nRun visualize_full_mission_final.py for this angle first.")

    rows = load_rows(csv_path)
    replay = build_replay(rows)
    html = HTML.replace("__DATA__", json.dumps(replay, ensure_ascii=False, allow_nan=False))
    out = base / "final_interactive_replay.html"
    out.write_text(html, encoding="utf-8")

    print("=" * 120)
    print("FINAL INTERACTIVE REPLAY CREATED")
    print("Telemetry rows:", len(replay))
    print("HTML:", out)
    print("=" * 120)


if __name__ == "__main__":
    main()
