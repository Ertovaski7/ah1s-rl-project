"""Kalkış probları (ölçüm; policy yok): yerde başlangıç, açık-döngü kalkış, PID ile hover trimleri, adım cevapları, ağırlık / CG.

python docs/takeoff/probe_takeoff.py openloop|trims|steps|weight   (repo kökünden)
"""
import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from helicopter_env_command import CONTROL_DT, PHYSICS_STEPS, wrap_deg  # noqa: E402
from helicopter_env_maneuver import HelicopterEnvManeuver  # noqa: E402

R_FT = 20_902_231.0


class Probe:
    def __init__(self, fuel=(0.0, 0.0)):
        self.env = HelicopterEnvManeuver(level="M1")
        e = self.env
        e._create_fdm()
        f = e.fdm
        f["propulsion/tank[0]/contents-lbs"] = fuel[0]
        f["propulsion/tank[1]/contents-lbs"] = fuel[1]
        e._warmup_rotor()
        self.f = f
        s = e._read_state()
        self.psi0 = s["psi_deg"]
        e._set_sas(self.psi0)
        self.lat0 = float(f["position/lat-geod-rad"])
        self.lon0 = float(f["position/long-gc-rad"])
        self.t = 0.0
        self.kth, self.kq = 3.0, 1.6
        self.i = np.zeros(4)
        self.c = np.array([0.0, float(f["fcs/elevator-cmd-norm"]), float(f["fcs/aileron-cmd-norm"]), float(f["fcs/rudder-cmd-norm"])])

    def st(self):
        f = self.f
        s = self.env._read_state()
        lat, lon = float(f["position/lat-geod-rad"]), float(f["position/long-gc-rad"])
        n = (lat - self.lat0) * R_FT
        e = (lon - self.lon0) * R_FT * math.cos(self.lat0)
        vn, ve = float(f["velocities/v-north-fps"]), float(f["velocities/v-east-fps"])
        ps = math.radians(s["psi_deg"])
        s.update(n=n, e=e, vn=vn, ve=ve, ug=vn * math.cos(ps) + ve * math.sin(ps), vg=-vn * math.sin(ps) + ve * math.cos(ps),
                 wow=sum(float(f[f"gear/unit[{i}]/WOW"]) for i in range(4)), fz=float(f["forces/fbz-gear-lbs"]),
                 w=float(f["inertia/weight-lbs"]))
        return s

    def step(self, c):
        self.c = np.array(c, dtype=float)
        self.env._write_controls(np.array([np.clip(c[0], 0, 1), *np.clip(c[1:], -1, 1)]))
        ok = self.env._run_physics_plain()
        self.t += CONTROL_DT
        return ok

    def pid(self, s, h_ref, psi_ref, n_ref=0.0, e_ref=0.0, vs_max=(-6.0, 8.0), free=()):
        """Probe PID (yalnızca ölçüm için). Konum → hız → attitude → cyclic; heading → pedal; irtifa → dikey hız → collective."""
        ff = self.env._sfd()
        ps = math.radians(s["psi_deg"])
        dn, de = n_ref - s["n"], e_ref - s["e"]
        xf, yr = dn * math.cos(ps) + de * math.sin(ps), -dn * math.sin(ps) + de * math.cos(ps)
        u_des, v_des = np.clip(0.12 * xf, -6, 6), np.clip(0.12 * yr, -6, 6)
        eu, ev = u_des - s["ug"], v_des - s["vg"]
        vs_des = float(np.clip(0.4 * (h_ref - s["h"]), *vs_max))
        evs = vs_des - s["vs"]
        eps = wrap_deg(psi_ref - s["psi_deg"])
        dt = CONTROL_DT
        air = s["wow"] < 0.5
        if air:
            self.i[0] = np.clip(self.i[0] + 0.01 * evs * dt, -0.3, 0.3)
            self.i[1] = np.clip(self.i[1] - 0.004 * eu * dt, -0.3, 0.3)
            self.i[2] = np.clip(self.i[2] + 0.004 * ev * dt, -0.3, 0.3)
            self.i[3] = np.clip(self.i[3] - 0.003 * eps * dt, -0.3, 0.3)
        th_ref = np.clip(-0.02 * eu, -0.25, 0.25) + self.i[1]
        ph_ref = np.clip(0.03 * ev, -0.25, 0.25) + self.i[2]
        c = np.array([
            ff[0] + 0.03 * evs + self.i[0],
            ff[1] + self.kth * (s["theta"] - th_ref) + self.kq * s["q"],
            ff[2] + 1.5 * (ph_ref - s["phi"]) - 0.5 * s["p"],
            ff[3] - 0.02 * eps + 0.8 * s["r"] + self.i[3]])
        for k in free:
            c[k] = self.c[k]
        return c


def openloop():
    """Kalkış: collective 0 → 0.62 (4 s rampa), cyclic / pedal SFD trimde sabit (geri besleme yok)."""
    p = Probe()
    ff = p.env._sfd()
    print(f"SFD trim (0 kt): coll {ff[0]:.3f} elev {ff[1]:.3f} ail {ff[2]:.3f} rud {ff[3]:.3f}")
    s = p.st()
    t_lift = None
    while p.t < 16.0:
        coll = min(0.62, 0.62 * p.t / 4.0)
        p.step([coll, ff[1], ff[2], ff[3]])
        s = p.st()
        if t_lift is None and s["wow"] < 0.5:
            t_lift = p.t
            print(f"liftoff t={p.t:.2f}s coll={coll:.3f}")
        if abs(p.t - round(p.t)) < CONTROL_DT / 2 and round(p.t) % 2 == 0:
            print(f"t={p.t:5.1f} coll={coll:.3f} h={s['h']:6.1f} vs={s['vs']:+5.1f} wow={s['wow']:.0f} fz={s['fz']:7.0f} "
                  f"psi={wrap_deg(s['psi_deg'] - p.psi0):+6.1f} r={math.degrees(s['r']):+6.1f} phi={math.degrees(s['phi']):+5.1f} "
                  f"th={math.degrees(s['theta']):+5.1f} N={s['n']:+6.1f} E={s['e']:+6.1f} rpm={s['rpm']:.0f}")


def trims():
    """PID ile yerden kalkış ve farklı yüksekliklerde hover: ortalama kumandalar."""
    p = Probe()
    psi_ref = p.psi0
    p.kth, p.kq = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0, float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
    plan = [8, 12, 20, 35, 60, 150, 400, 1000]
    s = p.st()
    t_lift = None
    for h_ref in plan:
        hist = []
        t_start = p.t
        while p.t - t_start < (30.0 if h_ref < 300 else 90.0):
            c = p.pid(s, h_ref, psi_ref)
            if s["wow"] > 0.5 and p.t < 8:                  # yerde: collective yumuşak rampa
                c[0] = min(c[0], 0.08 * p.t + 0.3)
            p.step(c)
            s = p.st()
            if t_lift is None and s["wow"] < 0.5:
                t_lift = p.t
                print(f"liftoff t={p.t:.2f}s coll={c[0]:.3f}")
            if p.t - t_start > (20.0 if h_ref < 300 else 80.0):
                hist.append(np.r_[c, s["rpm"], s["h"], s["phi"], s["theta"], s["n"], s["e"]])
        h = np.array(hist).mean(axis=0)
        print(f"hover h≈{h[5]:6.1f} (ref {h_ref:4d}): coll {h[0]:.4f} elev {h[1]:+.4f} ail {h[2]:+.4f} rud {h[3]:+.4f} rpm {h[4]:.1f} "
              f"phi {math.degrees(h[6]):+.2f} th {math.degrees(h[7]):+.2f} pos N {h[8]:+.1f} E {h[9]:+.1f}")


def steps():
    """300 ft hover'da adım cevapları: bir eksen açık döngü adım, diğerleri PID."""
    for axis, mag in ((0, 0.03), (0, 0.06), (1, 0.05), (1, 0.15), (2, 0.05), (2, 0.15), (3, 0.05), (3, 0.15)):
        p = Probe()
        s = p.st()
        while p.t < 55.0:
            c = p.pid(s, 300.0, p.psi0)
            if s["wow"] > 0.5 and p.t < 8:
                c[0] = min(c[0], 0.08 * p.t + 0.3)
            p.step(c)
            s = p.st()
        base = p.c.copy()
        s0 = dict(s)
        rec = []
        t0 = p.t
        while p.t - t0 < 3.0:
            c = p.pid(s, 300.0, p.psi0)
            c[axis] = base[axis] + mag
            p.step(c)
            s = p.st()
            rec.append((p.t - t0, math.degrees(s["p"]), math.degrees(s["q"]), math.degrees(s["r"]), s["vs"],
                        math.degrees(s["phi"] - s0["phi"]), math.degrees(s["theta"] - s0["theta"]), wrap_deg(s["psi_deg"] - s0["psi_deg"])))
        r = np.array(rec)
        def at(tq, col):
            return r[np.argmin(np.abs(r[:, 0] - tq)), col]
        name = ["collective", "long.cyclic", "lat.cyclic", "pedal"][axis]
        print(f"{name:11s} +{mag:.2f}: max|p| {np.abs(r[:,1]).max():5.1f} max|q| {np.abs(r[:,2]).max():5.1f} max|r| {np.abs(r[:,3]).max():5.1f} °/s "
              f"vs@1s {at(1,4):+5.1f} @3s {at(3,4):+5.1f} ft/s | Δφ@1s {at(1,5):+5.1f} Δθ@1s {at(1,6):+5.1f} Δψ@1s {at(1,7):+5.1f} "
              f"| Δφ@2s {at(2,5):+5.1f} Δθ@2s {at(2,6):+5.1f} Δψ@2s {at(2,7):+5.1f}")


def weight():
    for fuel in ((0, 0), (445, 445), (890, 890), (890, 0), (0, 890)):
        p = Probe(fuel=fuel)
        s = p.st()
        hist = []
        while p.t < 70.0:
            c = p.pid(s, 300.0, p.psi0)
            if s["wow"] > 0.5 and p.t < 8:
                c[0] = min(c[0], 0.08 * p.t + 0.3)
            p.step(c)
            s = p.st()
            if p.t > 55:
                hist.append(np.r_[c, s["rpm"], s["h"], s["theta"]])
        h = np.array(hist).mean(axis=0)
        print(f"fuel {fuel}: W {s['w']:.0f} lbs CG x {float(p.f['inertia/cg-x-in']):.1f} in → coll {h[0]:.4f} elev {h[1]:+.4f} "
              f"ail {h[2]:+.4f} rud {h[3]:+.4f} rpm {h[4]:.1f} h {h[5]:.1f} th {math.degrees(h[6]):+.2f}")


if __name__ == "__main__":
    {"openloop": openloop, "trims": trims, "steps": steps, "weight": weight}[sys.argv[1]]()
