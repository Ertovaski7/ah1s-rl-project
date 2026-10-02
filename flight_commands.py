from __future__ import annotations

"""
FLIGHT COMMANDS — arayüz için komut yönlendirici (command router), 2026-10-01
============================================================================

Arayüzü yazacak kişi env'in iç görev türlerini (takeoff / climb_to / turn / bob / move / cruise / stop / land /
pirouette) bilmek zorunda kalmasın diye TEK giriş noktası. Kullanıcı yalnızca uçuş komutu verir:

    hız      : mutlak (speed_kt) ya da Δ (dspeed_kt)       — 0 = hover; 10–130 kt ileri uçuş (hava hızı)
    heading  : mutlak (heading_deg, en kısa yön) ya da Δ (dheading_deg, tek komutta ±360°)
    irtifa   : mutlak (alt_ft) ya da Δ (dalt_ft), CG AGL  — hover 12–1500 ft, ileri uçuş 50–1500 ft
    eylemler : kalk (takeoff), in (land), dur / hover (stop), pirouette

Yönlendirici helikopterin o anki rejimine (yerde / hover / ileri uçuş) göre komutu env görev(ler)ine çevirir, doğal
sınırlara KIRPAR (asla ters çevirmez) ve gerekirse ara görev ekler:

    rejim        | hız komutu                       | heading            | irtifa          | iniş / hover eylemleri
    yerde        | —  (önce kalkış)                 | —                  | kalkış (h)      | —
    hover        | ≥ 10 kt → (dönüş) + hızlanma     | yerinde dönüş      | climb_to (h)    | land / hold / pirouette
    ileri uçuş   | < 10 kt → duruş (hover)          | cruise Δψ          | cruise h        | önce duruş, sonra görev
                 | 10–130 kt → cruise u             |                    |                 |

Neden (2026-09-30 ölçümleri): env'e doğrudan verilen komutlar (1) zarf dışına düşünce ters çevriliyordu (40 kt'ta
"−20 kt" → 60 kt), (2) rejime uymayınca episode bitiyordu (80 kt'ta hover dönüşü → 0.1 s'de speed_limit) ya da istenmeyen
hızlanma oluyordu (hover'da cruise Δψ → 30 kt).

Kullanım:
    from flight_commands import route_command, route_action, apply
    res = route_command(env, dspeed_kt=+20, dheading_deg=-90)      # ya da route_action(env, "land")
    if res.ok:
        apply(env, res)                                              # ilk görev hemen, diğerleri sırayla
    print(res.message)                                               # arayüzde gösterilecek açıklama (Türkçe)
"""

import math
from dataclasses import dataclass, field

from flight_curriculum import CMD_MAX_ALT_FT, CMD_MAX_KT, CMD_MIN_ALT_FT, CMD_MIN_KT, KT
from takeoff_curriculum import MAX_TARGET_H_FT, MIN_HOVER_H_FT

CRUISE_REGIME_KT = 25.0          # bu hava hızının üstü (ya da ileri uçuş penceresi) → ileri uçuş rejimi
MAX_DPSI_DEG = 360.0


@dataclass
class CommandResult:
    ok: bool
    tasks: list = field(default_factory=list)
    message: str = ""
    regime: str = ""
    notes: list = field(default_factory=list)       # kırpma / dönüştürme açıklamaları


def _wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def regime(env) -> str:
    """'ground' | 'hover' | 'cruise' — kızaklar yerde (≥ 3 nokta, alçak) → yerde; ileri uçuş penceresi açık ya da hava hızı
    ≥ 25 kt → ileri uçuş; yoksa hover."""
    s = env._state()
    if s["wow"] >= 3 and s["hs"] < 1.0:
        return "ground"
    if env._in_cruise() or abs(env.u_meas) >= CRUISE_REGIME_KT * KT:
        return "cruise"
    return "hover"


def _fmt_clip(name: str, want: float, got: float, unit: str) -> str:
    return f"{name} {want:.0f} {unit} doğal sınıra kırpıldı → {got:.0f} {unit}"


def route_command(env, speed_kt=None, dspeed_kt=None, heading_deg=None, dheading_deg=None, alt_ft=None,
                  dalt_ft=None) -> CommandResult:
    """Hız / heading / irtifa komutunu (mutlak ya da Δ; verilmeyen eksen değişmez) env görevlerine çevirir."""
    reg = regime(env)
    s = env._state()
    notes = []
    # --- heading
    dpsi = None
    if heading_deg is not None:
        dpsi = _wrap180(float(heading_deg) - s["psi_deg"])
    elif dheading_deg is not None:
        dpsi = float(dheading_deg)
        if abs(dpsi) > MAX_DPSI_DEG:
            notes.append(_fmt_clip("Δheading", dpsi, math.copysign(MAX_DPSI_DEG, dpsi), "°"))
            dpsi = math.copysign(MAX_DPSI_DEG, dpsi)
    if dpsi is not None and abs(dpsi) < 0.5:
        dpsi = None
    # --- irtifa (CG AGL)
    h_t = None
    if alt_ft is not None:
        h_t = float(alt_ft)
    elif dalt_ft is not None:
        h_t = s["h"] + float(dalt_ft)
    # --- hız (hava hızı, kt)
    v_now = 0.0 if reg != "cruise" else env.u_meas / KT
    v_t = None
    if speed_kt is not None:
        v_t = float(speed_kt)
    elif dspeed_kt is not None:
        v_t = v_now + float(dspeed_kt)
    if v_t is not None and v_t < 0.0:
        notes.append(_fmt_clip("hız", v_t, 0.0, "kt"))
        v_t = 0.0

    if reg == "ground":
        if h_t is None:
            return CommandResult(False, [], "Helikopter yerde: önce kalkış (irtifa komutu ya da 'kalk') ver.", reg, notes)
        h_c = min(max(h_t, MIN_HOVER_H_FT), MAX_TARGET_H_FT)
        if abs(h_c - h_t) > 0.5:
            notes.append(_fmt_clip("irtifa", h_t, h_c, "ft"))
        tasks = [dict(kind="takeoff", h=h_c)]
        msg = f"Kalkış → {h_c:.0f} ft hover"
        if dpsi is not None or (v_t is not None and v_t > 0.0):
            notes.append("yerde hız / heading komutu yok sayıldı (önce kalkış; sonra yeni komut ver)")
        return CommandResult(True, tasks, _msg(msg, notes), reg, notes)

    if reg == "hover":
        tasks, parts = [], []
        if v_t is not None and v_t >= CMD_MIN_KT:
            if dpsi is not None:                                  # önce burnu çevir, sonra hızlan (eğitimdeki hızlanma düz)
                tasks.append(dict(kind="turn", dpsi=dpsi))
                parts.append(f"yerinde dönüş {dpsi:+.0f}°")
            u_c = min(v_t, CMD_MAX_KT)
            if u_c < v_t:
                notes.append(_fmt_clip("hız", v_t, u_c, "kt"))
            dh = 0.0
            if h_t is not None:
                h_c = min(max(h_t, CMD_MIN_ALT_FT), CMD_MAX_ALT_FT)
                if abs(h_c - h_t) > 0.5:
                    notes.append(_fmt_clip("ileri uçuş irtifası", h_t, h_c, "ft"))
                dh = h_c - s["h"]
            tasks.append(dict(kind="cruise", u_kt=u_c, dh=dh, accel=True))
            parts.append(f"{u_c:.0f} kt'a hızlanma" + (f", Δh {dh:+.0f} ft" if abs(dh) > 0.5 else "")
                         + (f" (en az {CMD_MIN_ALT_FT:.0f} ft)" if s["h"] + dh < CMD_MIN_ALT_FT else ""))
            return CommandResult(True, tasks, _msg("Hover → ileri uçuş: " + ", sonra ".join(parts), notes), reg, notes)
        if v_t is not None and 0.0 < v_t < CMD_MIN_KT:
            notes.append(f"{CMD_MIN_KT:.0f} kt altı hız komutu → hover'da kalınıyor")
        if dpsi is not None:
            tasks.append(dict(kind="turn", dpsi=dpsi))
            parts.append(f"yerinde dönüş {dpsi:+.0f}°")
        if h_t is not None:
            h_c = min(max(h_t, MIN_HOVER_H_FT), MAX_TARGET_H_FT)
            if abs(h_c - h_t) > 0.5:
                notes.append(_fmt_clip("irtifa", h_t, h_c, "ft"))
            if abs(h_c - s["h"]) > 1.0:
                tasks.append(dict(kind="climb_to", h=h_c))
                parts.append(f"irtifa → {h_c:.0f} ft")
        if not tasks:
            tasks = [dict(kind="hold")]
            parts = ["hover tut"]
        return CommandResult(True, tasks, _msg("Hover: " + ", sonra ".join(parts), notes), reg, notes)

    # --- ileri uçuş
    if v_t is not None and v_t < CMD_MIN_KT:                    # 0 / çok yavaş → duruş, sonra hover komutları
        tasks = [dict(kind="stop", decel=2.5)]
        parts = ["duruş (hover)"]
        if dpsi is not None:
            tasks.append(dict(kind="turn", dpsi=dpsi))
            parts.append(f"yerinde dönüş {dpsi:+.0f}°")
        if h_t is not None:
            h_c = min(max(h_t, MIN_HOVER_H_FT), MAX_TARGET_H_FT)
            if abs(h_c - h_t) > 0.5:
                notes.append(_fmt_clip("irtifa", h_t, h_c, "ft"))
            tasks.append(dict(kind="climb_to", h=h_c))
            parts.append(f"irtifa → {h_c:.0f} ft")
        return CommandResult(True, tasks, _msg("İleri uçuş → " + ", sonra ".join(parts), notes), reg, notes)
    task, parts = dict(kind="cruise"), []
    if v_t is not None:
        u_c = min(max(v_t, CMD_MIN_KT), CMD_MAX_KT)
        if abs(u_c - v_t) > 0.05:
            notes.append(_fmt_clip("hız", v_t, u_c, "kt"))
        task["u_kt"] = u_c
        parts.append(f"hız → {u_c:.0f} kt")
    if dpsi is not None:
        task["dpsi"] = dpsi
        if heading_deg is not None:                       # mutlak yön (rota tutmada rota bu olur)
            task["psi_abs"] = float(heading_deg) % 360.0
        parts.append(f"heading {dpsi:+.0f}°")
    if h_t is not None:
        h_c = min(max(h_t, CMD_MIN_ALT_FT), CMD_MAX_ALT_FT)
        if abs(h_c - h_t) > 0.5:
            notes.append(_fmt_clip("ileri uçuş irtifası", h_t, h_c, "ft"))
        task["h"] = h_c
        parts.append(f"irtifa → {h_c:.0f} ft")
    if len(task) == 1:
        return CommandResult(False, [], "Komut boş: hız, heading ya da irtifa ver.", reg, notes)
    return CommandResult(True, [task], _msg("İleri uçuş: " + ", ".join(parts), notes), reg, notes)


def route_action(env, action: str, **kw) -> CommandResult:
    """Eylemler: 'takeoff' (h), 'land', 'stop' / 'hover', 'pirouette' (radius, circle_s, direction)."""
    reg = regime(env)
    a = action.lower()
    if a == "takeoff":
        h = float(kw.get("h", kw.get("alt_ft", 50.0)))
        if reg == "ground":
            return route_command(env, alt_ft=h)
        return route_command(env, alt_ft=h)                      # havada: o irtifaya çık / in
    if a == "land":
        if reg == "ground":
            return CommandResult(False, [], "Zaten yerde.", reg)
        tasks = ([dict(kind="stop", decel=2.5)] if reg == "cruise" else []) + [dict(kind="land")]
        return CommandResult(True, tasks, "İniş" + (" (önce duruş)" if reg == "cruise" else ""), reg)
    if a in ("stop", "hover", "hold"):
        if reg == "ground":
            return CommandResult(False, [], "Yerde: önce kalkış.", reg)
        tasks = [dict(kind="stop", decel=2.5)] if reg == "cruise" else [dict(kind="hold")]
        return CommandResult(True, tasks, "Duruş → hover" if reg == "cruise" else "Hover tut", reg)
    if a == "pirouette":
        if reg == "ground":
            return CommandResult(False, [], "Yerde: önce kalkış.", reg)
        p = dict(kind="pirouette", radius=float(kw.get("radius", 100.0)), circle_s=float(kw.get("circle_s", 45.0)),
                 direction=float(kw.get("direction", 1.0)))
        tasks = ([dict(kind="stop", decel=2.5)] if reg == "cruise" else []) + [p]
        return CommandResult(True, tasks, f"Pirouette: {p['radius']:.0f} ft yarıçap, {p['circle_s']:.0f} s"
                             + (" (önce duruş)" if reg == "cruise" else ""), reg)
    return CommandResult(False, [], f"Bilinmeyen eylem: {action}", reg)


def route_task(env, task: dict) -> CommandResult:
    """Eski arayüz görevlerini (env görev sözlüğü) rejime göre güvenli hale getirir: ileri uçuşta hover görevi → önce duruş
    ya da ileri uçuş karşılığı; hover'da cruise Δ → yönlendirici. Diğerleri olduğu gibi."""
    reg = regime(env)
    k = task.get("kind")
    if k == "cruise" and not task.get("accel"):
        if reg == "cruise":
            return route_command(env, dspeed_kt=task.get("du_kt"), dheading_deg=task.get("dpsi"),
                                 dalt_ft=task.get("dh"), speed_kt=task.get("u_kt"), alt_ft=task.get("h"))
        return route_command(env, dspeed_kt=task.get("du_kt"), dheading_deg=task.get("dpsi"), dalt_ft=task.get("dh"),
                             speed_kt=task.get("u_kt"), alt_ft=task.get("h"))
    if k == "cruise" and task.get("accel"):
        if reg == "cruise":
            return route_command(env, speed_kt=task.get("u_kt"), dalt_ft=task.get("dh"))
        return route_command(env, speed_kt=task.get("u_kt"), dalt_ft=task.get("dh"))
    if reg == "cruise":
        if k == "turn":
            return route_command(env, dheading_deg=task.get("dpsi"))
        if k == "bob":
            return route_command(env, dalt_ft=task.get("dh"))
        if k in ("takeoff", "climb_to"):
            return route_command(env, alt_ft=task.get("h"))
        if k in ("land", "hold", "pirouette"):
            return route_action(env, {"hold": "stop"}.get(k, k), **{x: y for x, y in task.items() if x != "kind"})
        if k == "move":
            return CommandResult(True, [dict(kind="stop", decel=2.5), dict(task)], "İleri uçuş: önce duruş, sonra kayma",
                                 reg)
        if k == "stop":
            return CommandResult(True, [dict(task)], "Duruş → hover", reg)
    if reg == "ground" and k not in ("takeoff", "climb_to", "recover"):
        return CommandResult(False, [], "Helikopter yerde: önce kalkış.", reg)
    if reg == "ground" and k == "climb_to":
        return route_command(env, alt_ft=task.get("h"))
    if reg == "hover" and k == "stop":
        return CommandResult(True, [dict(kind="hold")], "Zaten hover'da: hover tut", reg)
    return CommandResult(True, [dict(task)], "", reg)


def apply(env, res: CommandResult) -> float | None:
    """Görevleri sıraya koyar: yeni komut önceki planı iptal eder; ilk görev bir sonraki adımda, diğerleri bir öncekinin
    bitişinde (başarı ya da son sınır) verilir. Dönen değer: ilk görevin zamanı (s)."""
    if not res.ok or not res.tasks:
        return None
    env.pending = []
    env.next_issue_t = float("inf")
    t = env.queue_task(dict(res.tasks[0]))
    for d in res.tasks[1:]:
        env.pending.append(dict(d))
    return t


def _msg(main: str, notes: list) -> str:
    return main + ("" if not notes else " — " + "; ".join(notes))


__all__ = ["CommandResult", "route_command", "route_action", "route_task", "apply", "regime", "CRUISE_REGIME_KT"]
