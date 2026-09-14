from __future__ import annotations

import json
import re
from pathlib import Path


def tag(angle: float) -> str:
    return f"{angle:+.0f}".replace("+", "plus").replace("-", "minus")


def load_current_replay(angle: float):
    src = Path("interactive_replay_final") / f"turn_{tag(angle)}_interactive_replay.html"
    if not src.exists():
        raise FileNotFoundError(
            f"Current replay telemetry HTML not found: {src}\n"
            "Create the current replay for this angle first."
        )
    text = src.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"const\s+D\s*=\s*(\[.*?\])\s*;\s*const\s+requested\s*=\s*([-+0-9.eE]+)\s*;", text, re.S)
    if not m:
        raise RuntimeError("Could not extract current telemetry payload from replay HTML")
    data = json.loads(m.group(1))
    requested = float(m.group(2))
    if not data:
        raise RuntimeError("Telemetry payload is empty")
    return src, data, requested


def phase_key(name: str) -> str:
    s = (name or "").lower()
    if "stage1" in s or "takeoff" in s or "hover" in s:
        return "stage1"
    if "post-turn afcs" in s or "transition" in s:
        return "transition"
    if "post-turn" in s:
        return "post"
    if "turn" in s:
        return "turn"
    if "stage2" in s or "forward" in s:
        return "stage2"
    return "other"


def decorate(data):
    colors = {
        "stage1": "#2f9bff",
        "stage2": "#ff8a00",
        "turn": "#48e06f",
        "transition": "#b974ff",
        "post": "#ffd447",
        "other": "#8aa0b2",
    }
    labels = {
        "stage1": "STAGE 1 — TAKEOFF / HOVER",
        "stage2": "STAGE 2 — FORWARD FLIGHT",
        "turn": "STAGE 3 — RELATIVE TURN",
        "transition": "TRANSITION — AFCS STABILIZATION",
        "post": "POST-TURN — STAGE 2 PPO",
        "other": "OTHER",
    }
    for r in data:
        k = phase_key(str(r.get("phase", "")))
        r["phase_key"] = k
        r["phase_display"] = labels[k]
        r["phase_color"] = colors[k]
    return data


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AH-1S Full Mission Replay</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1016;--panel:#111820;--border:#283441;--text:#edf4fa;--muted:#96a8b9;--blue:#2f9bff;--orange:#ff8a00;--green:#48e06f;--purple:#b974ff;--yellow:#ffd447}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif}.wrap{max-width:1600px;margin:auto;padding:16px}.head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px}.head h1{font-size:22px;margin:0}.sub{color:var(--muted);font-size:13px;margin-top:4px}.ok{color:var(--green);font-weight:700}.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:12px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px}.right{display:grid;grid-template-rows:1fr 1fr;gap:12px}.plot3d{height:590px}.small{height:285px}.legend{display:flex;flex-wrap:wrap;gap:12px;margin:2px 0 8px}.leg{font-size:12px;color:#d5dee7}.dot{display:inline-block;width:18px;height:4px;border-radius:4px;margin-right:6px;vertical-align:middle}.controls{display:flex;gap:8px;align-items:center;margin-top:8px}.controls button,.controls select{background:#192431;color:#fff;border:1px solid #344657;border-radius:7px;padding:8px 12px}.controls input{flex:1}.time{font-weight:700;min-width:95px;text-align:right}.caption{font-size:12px;color:var(--muted);margin-top:6px}.telemetry{margin-top:12px}.telemetry h3{font-size:13px;margin:0 0 8px;color:#bcc9d4;letter-spacing:.07em}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.card{background:#0e151d;border:1px solid var(--border);border-radius:9px;padding:9px}.label{font-size:10px;color:#8498aa;text-transform:uppercase}.value{font-size:17px;font-weight:700;margin-top:4px}.phasecard{grid-column:span 2}.phasebar{height:8px;border-radius:999px;margin-top:7px}.timeline{display:flex;height:13px;border-radius:999px;overflow:hidden;margin-top:10px;border:1px solid #31404e}.seg{height:100%}
@media(max-width:1000px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<div class="head"><div><h1>AH-1S — FULL MISSION INTERACTIVE REPLAY</h1><div class="sub">One continuous JSBSim FDM · current validated model stack · requested turn: <b id="req"></b>°</div></div><div class="ok">● CURRENT TELEMETRY</div></div>
<div class="legend"><span class="leg"><i class="dot" style="background:var(--blue)"></i>Stage 1 — Takeoff/Hover</span><span class="leg"><i class="dot" style="background:var(--orange)"></i>Stage 2 — Forward</span><span class="leg"><i class="dot" style="background:var(--green)"></i>Stage 3 — Turn</span><span class="leg"><i class="dot" style="background:var(--purple)"></i>AFCS Transition</span><span class="leg"><i class="dot" style="background:var(--yellow)"></i>Post-turn Stage 2</span></div>
<div class="grid"><div class="panel"><div id="scene3d" class="plot3d"></div><div class="controls"><button id="play">▶ Play</button><button id="pause">⏸ Pause</button><button id="restart">↺ Restart</button><input id="slider" type="range" min="0" value="0" step="1"><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><div id="timeText" class="time">0.0 s</div></div><div id="timeline" class="timeline"></div><div class="caption">Stage 1 → Stage 2 → Relative Turn → AFCS Transition → Post-turn Stage 2</div></div>
<div class="right"><div class="panel"><div id="topView" class="small"></div></div><div class="panel"><div id="sideView" class="small"></div></div></div></div>
<div class="panel telemetry"><h3>LIVE TELEMETRY</h3><div class="cards">
<div class="card phasecard"><div class="label">Mission phase</div><div id="phase" class="value">—</div><div id="phasebar" class="phasebar"></div></div>
<div class="card"><div class="label">Mission time</div><div id="time" class="value">—</div></div><div class="card"><div class="label">Altitude</div><div id="alt" class="value">—</div></div><div class="card"><div class="label">Vertical speed</div><div id="vs" class="value">—</div></div>
<div class="card"><div class="label">Forward distance</div><div id="forward" class="value">—</div></div><div class="card"><div class="label">Cross-track</div><div id="cross" class="value">—</div></div><div class="card"><div class="label">Forward speed</div><div id="u" class="value">—</div></div><div class="card"><div class="label">Lateral speed</div><div id="v" class="value">—</div></div>
<div class="card"><div class="label">Roll / Pitch</div><div id="att" class="value">—</div></div><div class="card"><div class="label">Heading</div><div id="hdg" class="value">—</div></div><div class="card"><div class="label">Heading error</div><div id="herr" class="value">—</div></div><div class="card"><div class="label">Rotor RPM</div><div id="rpm" class="value">—</div></div><div class="card"><div class="label">Turn progress</div><div id="turn" class="value">—</div></div>
</div></div></div>
<script>
const D=__DATA__; const requested=__REQUESTED__; document.getElementById('req').textContent=requested.toFixed(0);
const f=(r,k,d=0)=>Number.isFinite(r[k])?r[k]:d, fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
const xs=D.map(r=>f(r,'forward_position_ft',f(r,'forward_ft',0))), ys=D.map(r=>f(r,'lateral_from_initial_axis_ft',f(r,'cross_track_ft',0))), zs=D.map(r=>f(r,'altitude_ft',0));
const C={stage1:'#2f9bff',stage2:'#ff8a00',turn:'#48e06f',transition:'#b974ff',post:'#ffd447',other:'#8aa0b2'};
const phaseOrder=['stage1','stage2','turn','transition','post'];
function traces3d(){let a=[];for(const k of phaseOrder){let x=[],y=[],z=[];D.forEach((r,i)=>{if(r.phase_key===k){x.push(xs[i]);y.push(ys[i]);z.push(zs[i])}});if(x.length)a.push({type:'scatter3d',mode:'lines',x,y,z,name:k,line:{color:C[k],width:6}})}a.push({type:'scatter3d',mode:'markers',x:[xs[0]],y:[ys[0]],z:[zs[0]],name:'Aircraft',marker:{size:7,color:'#ffffff',symbol:'diamond'}});return a}
function traces2d(yarr){let a=[];for(const k of phaseOrder){let x=[],y=[];D.forEach((r,i)=>{if(r.phase_key===k){x.push(xs[i]);y.push(yarr[i])}});if(x.length)a.push({type:'scatter',mode:'lines',x,y,name:k,line:{color:C[k],width:4}})}a.push({type:'scatter',mode:'markers',x:[xs[0]],y:[yarr[0]],name:'Aircraft',marker:{size:10,color:'#ffffff',symbol:'diamond'}});return a}
const base={paper_bgcolor:'#111820',plot_bgcolor:'#111820',font:{color:'#e8f0f7'},margin:{l:55,r:18,t:42,b:45},showlegend:false};
Plotly.newPlot('scene3d',traces3d(),{...base,title:'3D MISSION REPLAY',scene:{bgcolor:'#111820',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Cross-track (ft)',gridcolor:'#293746'},zaxis:{title:'Altitude AGL (ft)',gridcolor:'#293746'},aspectmode:'data'}});
Plotly.newPlot('topView',traces2d(ys),{...base,title:'TOP VIEW — FULL MISSION',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Cross-track (ft)',gridcolor:'#293746',scaleanchor:'x',scaleratio:1}});
Plotly.newPlot('sideView',traces2d(zs),{...base,title:'SIDE VIEW — ALTITUDE PROFILE',xaxis:{title:'Forward (ft)',gridcolor:'#293746'},yaxis:{title:'Altitude AGL (ft)',gridcolor:'#293746'}});
const slider=document.getElementById('slider');slider.max=D.length-1;let idx=0,timer=null;
const marker3d=phaseOrder.filter(k=>D.some(r=>r.phase_key===k)).length; const marker2d=marker3d;
function set(id,s){document.getElementById(id).textContent=s}
function render(i){idx=Math.max(0,Math.min(D.length-1,i));slider.value=idx;const r=D[idx],x=xs[idx],y=ys[idx],z=zs[idx];Plotly.restyle('scene3d',{x:[[x]],y:[[y]],z:[[z]],'marker.color':[[r.phase_color||'#fff']]},[marker3d]);Plotly.restyle('topView',{x:[[x]],y:[[y]],'marker.color':[[r.phase_color||'#fff']]},[marker2d]);Plotly.restyle('sideView',{x:[[x]],y:[[z]],'marker.color':[[r.phase_color||'#fff']]},[marker2d]);set('phase',r.phase_display||r.phase||'—');document.getElementById('phasebar').style.background=r.phase_color||'#8aa0b2';set('timeText',fmt(r.mission_time_s,1)+' s');set('time',fmt(r.mission_time_s,2)+' s');set('alt',fmt(r.altitude_ft,1)+' ft');set('vs',fmt(r.vertical_speed_fps,2)+' ft/s');set('forward',fmt(x,1)+' ft');set('cross',fmt(y,1)+' ft');set('u',fmt(r.forward_speed_fps,2)+' ft/s');set('v',fmt(r.lateral_speed_fps,2)+' ft/s');set('att',fmt(r.roll_deg,1)+'° / '+fmt(r.pitch_deg,1)+'°');set('hdg',fmt(r.heading_deg,1)+'°');set('herr',fmt(r.heading_error_deg,1)+'°');set('rpm',fmt(r.rotor_rpm,1));set('turn',fmt(r.cumulative_turn_deg,1)+'° / rem '+fmt(r.remaining_turn_deg,1)+'°')}
function play(){if(timer)return;timer=setInterval(()=>{if(idx>=D.length-1){clearInterval(timer);timer=null;return}let step=Math.max(1,Math.round(Number(document.getElementById('speed').value)));render(Math.min(D.length-1,idx+step))},35)}function pause(){if(timer){clearInterval(timer);timer=null}}
document.getElementById('play').onclick=play;document.getElementById('pause').onclick=pause;document.getElementById('restart').onclick=()=>{pause();render(0)};slider.oninput=e=>{pause();render(Number(e.target.value))};
let runs=[];let start=0;for(let i=1;i<=D.length;i++){if(i===D.length||D[i].phase_key!==D[start].phase_key){runs.push([start,i-1,D[start].phase_key]);start=i}}const tl=document.getElementById('timeline');runs.forEach(([a,b,k])=>{let e=document.createElement('div');e.className='seg';e.style.width=((b-a+1)/D.length*100)+'%';e.style.background=C[k]||C.other;e.title=D[a].phase_display;tl.appendChild(e)});render(0);
</script></body></html>'''


def main():
    raw = input("Relative turn angle (-50, 50, 200, 360): ").strip()
    angle = float(raw)
    src, data, requested = load_current_replay(angle)
    data = decorate(data)

    found = []
    for r in data:
        k = r["phase_key"]
        if k not in found:
            found.append(k)
    required = ["stage1", "stage2", "turn", "transition", "post"]
    missing = [k for k in required if k not in found]

    print("SOURCE CURRENT REPLAY:", src)
    print("PHASES FOUND:", found)
    if missing:
        raise RuntimeError(f"Current telemetry is not full mission; missing phases: {missing}")

    outdir = Path("interactive_replay_pretty")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"turn_{tag(angle)}_FULL_MISSION.html"
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    out.write_text(HTML.replace("__DATA__", payload).replace("__REQUESTED__", repr(requested)), encoding="utf-8")
    print("=" * 110)
    print("FULL STAGE-BY-STAGE REPLAY READY")
    print("HTML:", out)
    print("Stages: Stage1 -> Stage2 -> Turn -> AFCS Transition -> Post-turn Stage2")
    print("Uses telemetry already produced by the current working-model replay; no fake trajectory is generated.")
    print("=" * 110)


if __name__ == "__main__":
    main()
