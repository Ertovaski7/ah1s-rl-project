from __future__ import annotations

import builtins
import json
import math
from pathlib import Path

ROOT = Path("_v24_final_validator.py")
if not ROOT.exists():
    raise FileNotFoundError(
        "_v24_final_validator.py not found. Run the latest final/targeted regression first so the current validated final source exists."
    )


def capture_generated_source(source_text: str, source_name: str) -> str:
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
            raise RuntimeError(f"replay: could not capture generated source from {source_name}")

    ns = {
        "__name__": "__main__",
        "__file__": source_name,
        "compile": fake_compile,
        "exec": fake_exec,
    }
    builtins.exec(source_text, ns)
    generated = captured.get("generated")
    if not isinstance(generated, str) or not generated.strip():
        raise RuntimeError(f"replay: wrapper {source_name} produced no generated source")
    return generated


# Materialize wrappers until the real validator is reached.
text = ROOT.read_text(encoding="utf-8")
for depth in range(8):
    if "AH-1S FINAL FULL-MISSION VALIDATOR" in text and "def hard_safe(m):" in text:
        break
    text = capture_generated_source(text, f"<interactive_replay_level_{depth}>")
else:
    raise RuntimeError("replay: could not materialize the final full-mission validator")

# -----------------------------------------------------------------------------
# Read-only telemetry instrumentation. Flight/control logic is unchanged.
# -----------------------------------------------------------------------------
helper_anchor = "def hard_safe(m):\n"
helper = r'''TELEMETRY = []


def record_replay_telemetry(fdm, phase, action=None, cumulative=float("nan"), remaining=float("nan"), heading_error=float("nan")):
    m = physical_metrics(fdm)
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else np.full(4, np.nan, dtype=np.float32)
    except Exception:
        arr = np.full(4, np.nan, dtype=np.float32)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size), constant_values=np.nan)
    TELEMETRY.append({
        "sim_t": float(m["sim_t"]),
        "phase": str(phase),
        "latitude_deg": float(latitude_deg(fdm)),
        "longitude_deg": float(longitude_deg(fdm)),
        "altitude_ft": float(m["alt"]),
        "vertical_speed_fps": float(m["vs"]),
        "forward_speed_fps": float(m["u"]),
        "lateral_speed_fps": float(m["v_lat"]),
        "roll_deg": float(m["roll"]),
        "pitch_deg": float(m["pitch"]),
        "yaw_rate_deg_s": float(m["r"]),
        "heading_deg": float(m["hdg"]),
        "rotor_rpm": float(m["rpm"]),
        "collective_cmd": float(fdm_get(fdm, "fcs/collective-cmd-norm")),
        "elevator_cmd": float(fdm_get(fdm, "fcs/elevator-cmd-norm")),
        "aileron_cmd": float(fdm_get(fdm, "fcs/aileron-cmd-norm")),
        "rudder_cmd": float(fdm_get(fdm, "fcs/rudder-cmd-norm")),
        "policy_a0": float(arr[0]),
        "policy_a1": float(arr[1]),
        "policy_a2": float(arr[2]),
        "policy_a3": float(arr[3]),
        "cumulative_turn_deg": float(cumulative),
        "remaining_turn_deg": float(remaining),
        "heading_error_deg": float(heading_error),
    })


'''
if helper_anchor not in text:
    raise RuntimeError("replay: hard_safe anchor not found")
text = text.replace(helper_anchor, helper + helper_anchor, 1)


def replace_once(old: str, new: str, label: str):
    global text
    if old not in text:
        raise RuntimeError(f"replay: {label} anchor not found")
    text = text.replace(old, new, 1)


replace_once(
    '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n''',
    '''    obs1, _, terminated, truncated, info1 = env1.step(action1)\n    stage1_elapsed += dt1\n    record_replay_telemetry(fdm, "Stage1 Takeoff / Hover", action1)\n''',
    "Stage1",
)

replace_once(
    '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n''',
    '''    obs2, _, terminated, truncated, info2 = env2.step(action2)\n    obs2 = np.asarray(obs2, dtype=np.float32)\n    stage2_elapsed = (step + 1) * dt2\n    record_replay_telemetry(fdm, "Stage2 Forward", action2)\n''',
    "Stage2",
)

replace_once(
    '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n''',
    '''    obs_turn, _, terminated, truncated, turn_info = turn_env.step(action_turn)\n    obs_turn = np.asarray(obs_turn, dtype=np.float32)\n    turn_elapsed = (step + 1) * turn_env.CONTROL_DT\n    record_replay_telemetry(\n        fdm, "Goal-conditioned Turn", action_turn,\n        cumulative=float(turn_info.get("cumulative_turn_deg", float("nan"))),\n        remaining=float(turn_info.get("remaining_turn_deg", float("nan"))),\n    )\n''',
    "Turn",
)

# Transition exists in current final configuration.
replace_once(
    '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n''',
    '''        ms = physical_metrics(fdm)\n        current_hdg = heading_deg(fdm)\n        heading_error_s = wrap_deg(post_target_heading - current_hdg)\n        settle_elapsed += turn_env.CONTROL_DT\n        record_replay_telemetry(\n            fdm, "Post-turn AFCS Transition", None,\n            cumulative=float(requested_turn - heading_error_s),\n            remaining=float(heading_error_s),\n            heading_error=float(heading_error_s),\n        )\n''',
    "AFCS transition",
)

replace_once(
    '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n''',
    '''    obs_post, _, terminated_post, truncated_post, info_post = env2.step(action_post)\n    obs_post = np.asarray(obs_post, dtype=np.float32)\n\n    m = physical_metrics(fdm)\n    current_hdg_for_log = heading_deg(fdm)\n    heading_error_for_log = wrap_deg(post_target_heading - current_hdg_for_log)\n    record_replay_telemetry(\n        fdm, "Post-turn Stage2", action_post,\n        cumulative=float(requested_turn - heading_error_for_log),\n        remaining=float(heading_error_for_log),\n        heading_error=float(heading_error_for_log),\n    )\n''',
    "Post-turn Stage2",
)

# -----------------------------------------------------------------------------
# Append self-contained HTML export. No localhost / no external JS dependency.
# -----------------------------------------------------------------------------
text += r'''

from pathlib import Path as _ReplayPath
import json as _json
import math as _math


def _finite_or_none(v):
    try:
        x = float(v)
        return x if _math.isfinite(x) else None
    except Exception:
        return None


if TELEMETRY:
    _R = 20902231.0
    _lat0 = float(TELEMETRY[0]["latitude_deg"])
    _lon0 = float(TELEMETRY[0]["longitude_deg"])
    _hdg0 = _math.radians(float(TELEMETRY[0]["heading_deg"]))
    _c0, _s0 = _math.cos(_hdg0), _math.sin(_hdg0)
    _t0 = float(TELEMETRY[0]["sim_t"])

    for _row in TELEMETRY:
        _lat = float(_row["latitude_deg"])
        _lon = float(_row["longitude_deg"])
        _north = _R * _math.radians(_lat - _lat0)
        _east = _R * _math.cos(_math.radians(_lat0)) * _math.radians(_lon - _lon0)
        _row["mission_time_s"] = float(_row["sim_t"] - _t0)
        _row["north_ft"] = float(_north)
        _row["east_ft"] = float(_east)
        _row["forward_position_ft"] = float(_north * _c0 + _east * _s0)
        _row["lateral_from_initial_axis_ft"] = float(-_north * _s0 + _east * _c0)

    _clean = []
    for _row in TELEMETRY:
        _clean.append({k: (v if isinstance(v, str) else _finite_or_none(v)) for k, v in _row.items()})

    _angle_tag = f"{requested_turn:+.0f}".replace("+", "plus").replace("-", "minus")
    _out = _ReplayPath("interactive_replay_final")
    _out.mkdir(parents=True, exist_ok=True)
    _html_path = _out / f"turn_{_angle_tag}_interactive_replay.html"
    _data_json = _json.dumps(_clean, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    _requested = float(requested_turn)

    _html = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AH-1S Interactive Mission Replay</title>
<style>
:root{--bg:#071019;--panel:#0d1b27;--panel2:#122535;--text:#e8f1f7;--muted:#91a7b7;--accent:#60d9ff;--good:#82e6a4;--line:#294557}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:18px}.title{display:flex;justify-content:space-between;align-items:end;gap:16px;margin-bottom:14px}.title h1{margin:0;font-size:24px}.subtitle{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:1.45fr .75fr;gap:14px}.panel{background:var(--panel);border:1px solid #173247;border-radius:14px;padding:12px;box-shadow:0 12px 30px #0005}
#map{width:100%;height:620px;background:linear-gradient(180deg,#0a1721,#08131c);border-radius:10px;display:block}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:9px}.card{background:var(--panel2);border-radius:10px;padding:10px}.k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.v{font-size:21px;font-weight:700;margin-top:4px}.phase{grid-column:1/-1;border-left:4px solid var(--accent)}
.controls{display:grid;grid-template-columns:auto auto 1fr auto;gap:10px;align-items:center;margin-top:12px}.btn,select{background:#173247;color:var(--text);border:1px solid #31536a;border-radius:8px;padding:9px 12px;font-weight:700;cursor:pointer}input[type=range]{width:100%}
.chart{width:100%;height:170px;background:#08131c;border-radius:10px;margin-top:12px;display:block}.legend{color:var(--muted);font-size:12px;margin-top:6px}.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#123044;color:var(--accent);font-weight:700}
@media(max-width:900px){.grid{grid-template-columns:1fr}#map{height:480px}}
</style>
</head>
<body><div class="wrap">
<div class="title"><div><h1>AH-1S Interactive Mission Replay</h1><div class="subtitle">One continuous JSBSim FDM · runtime teacher OFF · requested turn: <span id="req"></span>°</div></div><div class="badge">Play / Pause / Scrub</div></div>
<div class="grid">
  <div class="panel"><svg id="map" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet"></svg><div class="legend">Top view: Forward axis ↑ · horizontal axis = lateral displacement from initial mission axis</div></div>
  <div class="panel"><div class="cards">
    <div class="card phase"><div class="k">Phase</div><div class="v" id="phase">—</div></div>
    <div class="card"><div class="k">Mission time</div><div class="v" id="time">0.0 s</div></div>
    <div class="card"><div class="k">Altitude</div><div class="v" id="alt">—</div></div>
    <div class="card"><div class="k">Forward speed</div><div class="v" id="spd">—</div></div>
    <div class="card"><div class="k">Vertical speed</div><div class="v" id="vs">—</div></div>
    <div class="card"><div class="k">Heading</div><div class="v" id="hdg">—</div></div>
    <div class="card"><div class="k">Roll / Pitch</div><div class="v" id="att">—</div></div>
    <div class="card"><div class="k">Turn remaining</div><div class="v" id="rem">—</div></div>
  </div>
  <canvas id="altChart" class="chart" width="700" height="170"></canvas>
  <canvas id="headingChart" class="chart" width="700" height="170"></canvas>
  </div>
</div>
<div class="panel" style="margin-top:14px"><div class="controls"><button class="btn" id="play">▶ Play</button><button class="btn" id="restart">↺ Restart</button><input id="slider" type="range" min="0" max="1" value="0" step="1"><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></div></div>
</div>
<script>
const D=__DATA__; const requested=__REQUESTED__; document.getElementById('req').textContent=requested.toFixed(0);
const svg=document.getElementById('map'), NS='http://www.w3.org/2000/svg'; const slider=document.getElementById('slider'); slider.max=Math.max(0,D.length-1);
const val=(o,k,d=0)=>Number.isFinite(o[k])?o[k]:d, fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
const xs=D.map(r=>val(r,'lateral_from_initial_axis_ft')), ys=D.map(r=>val(r,'forward_position_ft')); let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys); if(xmax-xmin<20){xmin-=10;xmax+=10} if(ymax-ymin<20){ymin-=10;ymax+=10} const pad=60;
const X=x=>pad+(x-xmin)/(xmax-xmin)*(1000-2*pad), Y=y=>620-pad-(y-ymin)/(ymax-ymin)*(620-2*pad);
function line(x1,y1,x2,y2,stroke,w=1,op=1){let e=document.createElementNS(NS,'line');Object.assign(e,{ });e.setAttribute('x1',x1);e.setAttribute('y1',y1);e.setAttribute('x2',x2);e.setAttribute('y2',y2);e.setAttribute('stroke',stroke);e.setAttribute('stroke-width',w);e.setAttribute('opacity',op);svg.appendChild(e);return e}
for(let i=0;i<=10;i++){let xx=pad+i*(1000-2*pad)/10, yy=pad+i*(620-2*pad)/10;line(xx,pad,xx,620-pad,'#173247',1,.55);line(pad,yy,1000-pad,yy,'#173247',1,.55)}
let path=document.createElementNS(NS,'path');let d=D.map((r,i)=>(i?'L':'M')+X(val(r,'lateral_from_initial_axis_ft'))+' '+Y(val(r,'forward_position_ft'))).join(' ');path.setAttribute('d',d);path.setAttribute('fill','none');path.setAttribute('stroke','#35566a');path.setAttribute('stroke-width','3');path.setAttribute('opacity','.65');svg.appendChild(path);
let done=document.createElementNS(NS,'path');done.setAttribute('fill','none');done.setAttribute('stroke','#60d9ff');done.setAttribute('stroke-width','5');done.setAttribute('stroke-linecap','round');svg.appendChild(done);
let heli=document.createElementNS(NS,'polygon');heli.setAttribute('points','0,-16 10,12 0,7 -10,12');heli.setAttribute('fill','#82e6a4');heli.setAttribute('stroke','#d8ffe4');heli.setAttribute('stroke-width','2');svg.appendChild(heli);
function updatePath(i){let dd=D.slice(0,i+1).map((r,j)=>(j?'L':'M')+X(val(r,'lateral_from_initial_axis_ft'))+' '+Y(val(r,'forward_position_ft'))).join(' ');done.setAttribute('d',dd)}
function chart(canvas,key,label,target){const c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;c.clearRect(0,0,W,H);c.fillStyle='#08131c';c.fillRect(0,0,W,H);let a=D.map(r=>r[key]).filter(Number.isFinite);if(!a.length)return;let mn=Math.min(...a),mx=Math.max(...a);if(target!==null){mn=Math.min(mn,target);mx=Math.max(mx,target)}if(mx-mn<1){mn-=.5;mx+=.5}const xp=i=>45+i/(D.length-1)*(W-60),yp=v=>H-25-(v-mn)/(mx-mn)*(H-45);c.strokeStyle='#294557';c.lineWidth=1;for(let g=0;g<5;g++){let yy=15+g*(H-40)/4;c.beginPath();c.moveTo(40,yy);c.lineTo(W-10,yy);c.stroke()} if(target!==null){c.strokeStyle='#82e6a4';c.setLineDash([6,5]);c.beginPath();c.moveTo(40,yp(target));c.lineTo(W-10,yp(target));c.stroke();c.setLineDash([])} c.strokeStyle='#60d9ff';c.lineWidth=2;c.beginPath();D.forEach((r,i)=>{let v=r[key];if(!Number.isFinite(v))return;let x=xp(i),y=yp(v);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.fillStyle='#91a7b7';c.font='12px Arial';c.fillText(label,10,14);canvas._map={xp,yp,key};}
chart(document.getElementById('altChart'),'altitude_ft','Altitude (ft) — target 300',300);chart(document.getElementById('headingChart'),'heading_deg','Heading (deg)',null);
function marker(canvas,i){chart(canvas,canvas._map.key,canvas.id==='altChart'?'Altitude (ft) — target 300':'Heading (deg)',canvas.id==='altChart'?300:null);let c=canvas.getContext('2d'),m=canvas._map,v=D[i][m.key];if(Number.isFinite(v)){c.fillStyle='#82e6a4';c.beginPath();c.arc(m.xp(i),m.yp(v),5,0,Math.PI*2);c.fill()}}
function render(i){i=Math.max(0,Math.min(D.length-1,Number(i)||0));slider.value=i;const r=D[i];updatePath(i);let x=X(val(r,'lateral_from_initial_axis_ft')),y=Y(val(r,'forward_position_ft'));heli.setAttribute('transform',`translate(${x} ${y}) rotate(${val(r,'heading_deg')})`);document.getElementById('phase').textContent=r.phase||'—';document.getElementById('time').textContent=fmt(r.mission_time_s,1)+' s';document.getElementById('alt').textContent=fmt(r.altitude_ft,1)+' ft';document.getElementById('spd').textContent=fmt(r.forward_speed_fps,1)+' ft/s';document.getElementById('vs').textContent=fmt(r.vertical_speed_fps,2)+' ft/s';document.getElementById('hdg').textContent=fmt(r.heading_deg,1)+'°';document.getElementById('att').textContent=fmt(r.roll_deg,1)+'° / '+fmt(r.pitch_deg,1)+'°';document.getElementById('rem').textContent=fmt(r.remaining_turn_deg,1)+'°';marker(document.getElementById('altChart'),i);marker(document.getElementById('headingChart'),i)}
let playing=false, last=null, acc=0, idx=0; const playBtn=document.getElementById('play');function tick(ts){if(!playing)return;if(last===null)last=ts;let dt=(ts-last)/1000*Number(document.getElementById('speed').value);last=ts;acc+=dt;while(idx<D.length-1){let step=Math.max(.02,val(D[idx+1],'mission_time_s')-val(D[idx],'mission_time_s'));if(acc<step)break;acc-=step;idx++;render(idx)}if(idx>=D.length-1){playing=false;playBtn.textContent='▶ Play';return}requestAnimationFrame(tick)}
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'⏸ Pause':'▶ Play';last=null;if(playing)requestAnimationFrame(tick)};document.getElementById('restart').onclick=()=>{playing=false;playBtn.textContent='▶ Play';idx=0;acc=0;render(0)};slider.oninput=e=>{idx=Number(e.target.value);acc=0;render(idx)};render(0);
</script></body></html>'''.replace('__DATA__', _data_json).replace('__REQUESTED__', str(_requested))

    _html_path.write_text(_html, encoding="utf-8")
    print("=" * 120)
    print("INTERACTIVE REPLAY READY")
    print("HTML:", _html_path)
    print("Telemetry rows:", len(TELEMETRY))
    print("No localhost required. Open the HTML file directly in a browser.")
    print("=" * 120)
else:
    print("WARNING: no replay telemetry recorded")
'''

print("=" * 120)
print("CURRENT FINAL MISSION -> INTERACTIVE HTML REPLAY")
print("Read-only telemetry instrumentation; control/model logic unchanged.")
print("=" * 120)

exec(compile(text, "<current_final_interactive_replay>", "exec"), {"__name__": "__main__", "__file__": str(ROOT)})
