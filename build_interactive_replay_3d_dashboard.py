from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

BASE_BUILDER = Path("build_interactive_replay_current_final_v2.py")
BASE_OUT = Path("interactive_replay_final")
OUT_DIR = Path("interactive_replay_3d")


def angle_tag(angle: float) -> str:
    return f"{angle:+.0f}".replace("+", "plus").replace("-", "minus")


def main() -> int:
    if not BASE_BUILDER.exists():
        raise FileNotFoundError(BASE_BUILDER)

    raw = input("Relative turn angle (-50, 50, 200, 360): ").strip()
    angle = float(raw)

    # First create the validated telemetry-backed self-contained replay.
    proc = subprocess.run(
        [sys.executable, str(BASE_BUILDER)],
        input=f"{angle:g}\n",
        text=True,
    )
    if proc.returncode != 0:
        return proc.returncode

    tag = angle_tag(angle)
    base_html = BASE_OUT / f"turn_{tag}_interactive_replay.html"
    if not base_html.exists():
        raise FileNotFoundError(
            f"Base replay was not created: {base_html}. Check the builder output above."
        )

    src = base_html.read_text(encoding="utf-8")
    m = re.search(
        r"const D=(\[.*?\]); const requested=([-+0-9.eE]+);",
        src,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not extract telemetry JSON from base replay HTML")

    data = json.loads(m.group(1))
    requested = float(m.group(2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"turn_{tag}_3d_replay.html"

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AH-1S 3D Mission Replay</title>
<style>
:root{
  --bg:#090e14;--panel:#111922;--panel2:#161f29;--line:#2a3947;
  --text:#f1f5f9;--muted:#9aa8b5;--accent:#ff4aa2;--cyan:#54d7ff;
  --green:#69df8d;--yellow:#ffd166;--orange:#ff9f43;
}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#080c12,#0a1118);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;overflow-x:hidden}
.wrap{max-width:1480px;margin:0 auto;padding:16px}
.top{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px;gap:16px}
.top h1{margin:0;font-size:23px;font-weight:800;letter-spacing:.01em}
.sub{color:var(--muted);font-size:13px;margin-top:4px}
.badge{background:#182431;border:1px solid #31485b;border-radius:999px;padding:7px 12px;color:#dce7ef;font-size:12px;font-weight:700}
.main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(330px,.62fr);gap:12px}
.panel{background:rgba(17,25,34,.96);border:1px solid #263645;border-radius:10px;box-shadow:0 14px 36px #0007}
.sceneWrap{padding:8px;position:relative}
#scene{width:100%;height:680px;display:block;border-radius:8px;background:radial-gradient(circle at 50% 35%,#1c2731 0,#111820 56%,#0b1117 100%);cursor:grab}
#scene.drag{cursor:grabbing}
.sceneLabel{position:absolute;left:18px;top:16px;background:#0c131aaa;border:1px solid #283845;border-radius:7px;padding:7px 9px;color:#dce4eb;font-size:12px;backdrop-filter:blur(3px)}
.sceneHelp{position:absolute;right:18px;bottom:16px;background:#0c131aaa;border:1px solid #283845;border-radius:7px;padding:7px 9px;color:#96a5b3;font-size:11px}
.side{padding:12px;display:flex;flex-direction:column;gap:10px}
.side h3{font-size:13px;color:#c9d4dd;margin:0 0 2px;font-weight:800;letter-spacing:.04em;text-transform:uppercase}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.card{background:var(--panel2);border:1px solid #253442;border-radius:8px;padding:9px 10px;min-height:72px}
.card.wide{grid-column:1/-1}.k{font-size:10px;color:#8fa0ae;text-transform:uppercase;letter-spacing:.07em}.v{font-size:18px;font-weight:800;margin-top:5px;white-space:nowrap}.small{font-size:13px;color:#b8c4ce;margin-top:3px}
.phase{border-left:4px solid var(--accent)}
.controls{margin-top:12px;padding:11px 12px;display:grid;grid-template-columns:auto auto 1fr auto auto;gap:10px;align-items:center}
button,select{background:#192532;color:#eef5fa;border:1px solid #33495b;border-radius:7px;padding:8px 12px;font-weight:800;cursor:pointer}
button:hover,select:hover{background:#203140}
input[type=range]{width:100%;accent-color:#ff4aa2}
.timecode{font-variant-numeric:tabular-nums;color:#c8d3dc;font-size:12px;min-width:100px;text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#9fb0bd;margin-top:2px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
@media(max-width:980px){.main{grid-template-columns:1fr}#scene{height:560px}.controls{grid-template-columns:auto auto 1fr auto}.timecode{grid-column:1/-1;text-align:left}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>AH-1S · 3D Mission Replay</h1>
      <div class="sub">Continuous JSBSim FDM · runtime teacher OFF · requested relative turn <b id="req"></b>°</div>
    </div>
    <div class="badge">Interactive · Play / Pause / Scrub / Rotate</div>
  </div>

  <div class="main">
    <div class="panel sceneWrap">
      <canvas id="scene"></canvas>
      <div class="sceneLabel">3D Flight Path · X: lateral · Y: forward · Z: altitude</div>
      <div class="sceneHelp">Drag = rotate · Mouse wheel = zoom</div>
    </div>

    <div class="panel side">
      <h3>Flight Telemetry</h3>
      <div class="grid">
        <div class="card wide phase"><div class="k">Current Phase</div><div class="v" id="phase">—</div></div>
        <div class="card"><div class="k">Altitude</div><div class="v" id="alt">—</div><div class="small">ft</div></div>
        <div class="card"><div class="k">Vertical Speed</div><div class="v" id="vs">—</div><div class="small">ft/s</div></div>
        <div class="card"><div class="k">Forward Speed</div><div class="v" id="u">—</div><div class="small">ft/s</div></div>
        <div class="card"><div class="k">Lateral Speed</div><div class="v" id="vl">—</div><div class="small">ft/s</div></div>
        <div class="card"><div class="k">Heading</div><div class="v" id="hdg">—</div><div class="small">deg</div></div>
        <div class="card"><div class="k">Turn Progress</div><div class="v" id="turn">—</div><div class="small">completed / target</div></div>
        <div class="card"><div class="k">Roll</div><div class="v" id="roll">—</div><div class="small">deg</div></div>
        <div class="card"><div class="k">Pitch</div><div class="v" id="pitch">—</div><div class="small">deg</div></div>
        <div class="card"><div class="k">Yaw Rate</div><div class="v" id="yaw">—</div><div class="small">deg/s</div></div>
        <div class="card"><div class="k">Rotor RPM</div><div class="v" id="rpm">—</div></div>
      </div>
      <div class="legend">
        <span><i class="dot" style="background:#ffd166"></i>Stage1</span>
        <span><i class="dot" style="background:#69df8d"></i>Stage2</span>
        <span><i class="dot" style="background:#ff4aa2"></i>Turn</span>
        <span><i class="dot" style="background:#ff9f43"></i>Transition</span>
        <span><i class="dot" style="background:#54d7ff"></i>Post-turn</span>
      </div>
    </div>
  </div>

  <div class="panel controls">
    <button id="play">▶ Play</button>
    <button id="restart">↺ Restart</button>
    <input id="slider" type="range" min="0" value="0" step="1">
    <select id="speed">
      <option value="0.5">0.5×</option><option value="1" selected>1×</option>
      <option value="2">2×</option><option value="4">4×</option><option value="8">8×</option>
    </select>
    <div class="timecode" id="timecode">0.0 / 0.0 s</div>
  </div>
</div>

<script>
const D=__DATA__;
const REQUESTED=__REQUESTED__;
document.getElementById('req').textContent=REQUESTED.toFixed(0);
const canvas=document.getElementById('scene'); const ctx=canvas.getContext('2d');
const slider=document.getElementById('slider'); slider.max=Math.max(0,D.length-1);
const play=document.getElementById('play'); const speed=document.getElementById('speed');
const DPR=Math.min(window.devicePixelRatio||1,2);
let W=0,H=0;
function resize(){const r=canvas.getBoundingClientRect();W=r.width;H=r.height;canvas.width=Math.round(W*DPR);canvas.height=Math.round(H*DPR);ctx.setTransform(DPR,0,0,DPR,0,0);draw(idx)}
window.addEventListener('resize',resize);
const num=(r,k,d=0)=>Number.isFinite(r[k])?r[k]:d;
const finite=(x)=>Number.isFinite(x);
const fmt=(x,n=1)=>finite(x)?x.toFixed(n):'—';
const phaseColor=p=>p.includes('Stage1')?'#ffd166':p.includes('Transition')?'#ff9f43':p.includes('Post-turn')?'#54d7ff':p.includes('Turn')?'#ff4aa2':'#69df8d';

// Mission coordinate bounds.
const xs=D.map(r=>num(r,'lateral_from_initial_axis_ft'));
const ys=D.map(r=>num(r,'forward_position_ft'));
const zs=D.map(r=>num(r,'altitude_ft'));
let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),zmin=Math.min(...zs),zmax=Math.max(...zs);
if(xmax-xmin<40){xmin-=20;xmax+=20} if(ymax-ymin<40){ymin-=20;ymax+=20} if(zmax-zmin<30){zmin-=15;zmax+=15}
const xmid=(xmin+xmax)/2, ymid=(ymin+ymax)/2;
const baseRange=Math.max(xmax-xmin,ymax-ymin,120);
let az=-0.72, elev=0.58, zoom=1.0;
function world(r){return {x:(num(r,'lateral_from_initial_axis_ft')-xmid)/baseRange*560,y:(num(r,'forward_position_ft')-ymid)/baseRange*560,z:(num(r,'altitude_ft')-zmin)/(zmax-zmin)*180};}
function project(p){
  const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(elev),se=Math.sin(elev);
  const x1=p.x*ca-p.y*sa, y1=p.x*sa+p.y*ca;
  const sy=p.z*ce-y1*se; const depth=p.z*se+y1*ce;
  const persp=1/(1+Math.max(-220,depth)*0.0007);
  return {x:W*0.5+x1*zoom*persp,y:H*0.65-sy*zoom*persp,depth};
}
function line3(a,b,color,w=1,alpha=1){const A=project(a),B=project(b);ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);ctx.stroke();ctx.globalAlpha=1;}
function grid(){
  const gx=baseRange?280:280, gy=280;
  for(let i=-5;i<=5;i++){
    let t=i/5;
    line3({x:t*gx,y:-gy,z:0},{x:t*gx,y:gy,z:0},'#26343f',1,.7);
    line3({x:-gx,y:t*gy,z:0},{x:gx,y:t*gy,z:0},'#26343f',1,.7);
  }
  line3({x:-gx,y:0,z:0},{x:gx,y:0,z:0},'#667784',1.3,.75);
  line3({x:0,y:-gy,z:0},{x:0,y:gy,z:0},'#667784',1.3,.75);
  line3({x:0,y:0,z:0},{x:0,y:0,z:220},'#667784',1.3,.75);
  const O=project({x:0,y:0,z:0}); ctx.fillStyle='#aab6bf';ctx.font='12px Segoe UI';ctx.fillText('Z',project({x:0,y:0,z:220}).x+6,project({x:0,y:0,z:220}).y);ctx.fillText('X',project({x:gx,y:0,z:0}).x+4,project({x:gx,y:0,z:0}).y);ctx.fillText('Y',project({x:0,y:gy,z:0}).x+4,project({x:0,y:gy,z:0}).y);ctx.fillStyle='#748491';ctx.fillText('0',O.x+5,O.y+13);
}
function drawPath(limit){
  if(!D.length)return;
  let prev=world(D[0]);
  for(let i=1;i<D.length;i++){
    const cur=world(D[i]); const active=i<=limit;
    const c=active?phaseColor(D[i].phase||''):'#3a4752';
    line3(prev,cur,c,active?3.1:1.4,active?1:.35); prev=cur;
  }
}
function helicopter(r){
  const P=project(world(r)); const heading=(num(r,'heading_deg')-num(D[0],'heading_deg'))*Math.PI/180;
  ctx.save();ctx.translate(P.x,P.y);ctx.rotate(heading-az);ctx.strokeStyle='#f5f7fa';ctx.fillStyle='#dce6ee';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(0,-12);ctx.lineTo(7,8);ctx.lineTo(0,5);ctx.lineTo(-7,8);ctx.closePath();ctx.fill();ctx.stroke();
  ctx.strokeStyle='#99a8b3';ctx.beginPath();ctx.moveTo(-14,0);ctx.lineTo(14,0);ctx.stroke();ctx.beginPath();ctx.arc(0,0,13,0,Math.PI*2);ctx.stroke();ctx.restore();
}
function draw(i){
  if(!ctx||!D.length)return;ctx.clearRect(0,0,W,H);grid();drawPath(i);helicopter(D[i]);
  ctx.fillStyle='#9aa8b5';ctx.font='11px Segoe UI';ctx.fillText(`View azimuth ${(az*180/Math.PI).toFixed(0)}° · elevation ${(elev*180/Math.PI).toFixed(0)}°`,14,H-14);
}
function update(i){
  idx=Math.max(0,Math.min(D.length-1,Number(i)||0)); slider.value=idx; const r=D[idx];
  document.getElementById('phase').textContent=r.phase||'—';
  document.getElementById('alt').textContent=fmt(r.altitude_ft,1);
  document.getElementById('vs').textContent=fmt(r.vertical_speed_fps,2);
  document.getElementById('u').textContent=fmt(r.forward_speed_fps,1);
  document.getElementById('vl').textContent=fmt(r.lateral_speed_fps,1);
  document.getElementById('hdg').textContent=fmt(r.heading_deg,1);
  const done=finite(r.cumulative_turn_deg)?r.cumulative_turn_deg:(REQUESTED-(finite(r.remaining_turn_deg)?r.remaining_turn_deg:REQUESTED));
  document.getElementById('turn').textContent=fmt(done,1)+'° / '+REQUESTED.toFixed(0)+'°';
  document.getElementById('roll').textContent=fmt(r.roll_deg,1);
  document.getElementById('pitch').textContent=fmt(r.pitch_deg,1);
  document.getElementById('yaw').textContent=fmt(r.yaw_rate_deg_s,2);
  document.getElementById('rpm').textContent=fmt(r.rotor_rpm,0);
  const now=num(r,'mission_time_s'), total=num(D[D.length-1],'mission_time_s');
  document.getElementById('timecode').textContent=fmt(now,1)+' / '+fmt(total,1)+' s';
  draw(idx);
}
let idx=0,playing=false,last=null,acc=0;
function tick(ts){
  if(!playing)return;if(last===null)last=ts;acc+=(ts-last)/1000*Number(speed.value);last=ts;
  while(idx<D.length-1){const dt=Math.max(.02,num(D[idx+1],'mission_time_s')-num(D[idx],'mission_time_s'));if(acc<dt)break;acc-=dt;idx++;update(idx)}
  if(idx>=D.length-1){playing=false;play.textContent='▶ Play';return} requestAnimationFrame(tick);
}
play.onclick=()=>{playing=!playing;play.textContent=playing?'⏸ Pause':'▶ Play';last=null;if(playing)requestAnimationFrame(tick)};
document.getElementById('restart').onclick=()=>{playing=false;play.textContent='▶ Play';idx=0;acc=0;update(0)};
slider.oninput=e=>{playing=false;play.textContent='▶ Play';idx=Number(e.target.value);acc=0;update(idx)};

// Drag to rotate, wheel to zoom.
let dragging=false,lx=0,ly=0;
canvas.addEventListener('pointerdown',e=>{dragging=true;lx=e.clientX;ly=e.clientY;canvas.classList.add('drag');canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;az+=(e.clientX-lx)*0.008;elev=Math.max(.15,Math.min(1.2,elev+(e.clientY-ly)*0.006));lx=e.clientX;ly=e.clientY;draw(idx)});
canvas.addEventListener('pointerup',e=>{dragging=false;canvas.classList.remove('drag')});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.55,Math.min(2.2,zoom*(e.deltaY>0?.92:1.08)));draw(idx)},{passive:false});

resize();update(0);
</script>
</body>
</html>'''

    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__REQUESTED__", repr(requested))
    out.write_text(html, encoding="utf-8")

    print("=" * 96)
    print("3D INTERACTIVE REPLAY READY")
    print("HTML:", out)
    print("Controls: Play/Pause + timeline scrub + speed + mouse rotate + wheel zoom")
    print("No localhost required; open the HTML directly in a browser.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
