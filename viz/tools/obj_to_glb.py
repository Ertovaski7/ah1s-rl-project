#!/usr/bin/env python3
"""
OBJ (+MTL) → GLB — komut uçuşu görselleştirmesinin helikopter modeli
=====================================================================

`viz/heli_bell.glb` bu betikle üretildi (kaynak: Heli_bell.obj / .mtl, low-poly
Bell). Yalnızca numpy kullanır.

Ne yapar
  * zemin düzlemini (`tanah_*`) atar,
  * gövde, ana rotor ve kuyruk rotoru ayrı node olur; rotor node'larının orijini
    göbekte (hub) durur → görüntüleyici onları kendi ekseninde döndürebilir,
  * modelin orijini yaklaşık ağırlık merkezine (ana rotor mili altı) taşınır,
  * birim metre, gövde boyu AH-1S'inki kadar (13.6 m) ölçeklenir,
  * glTF yönü: burun +Z, yukarı +Y (kaynak dosya da böyle).

Kullanım
  python viz/tools/obj_to_glb.py Heli_bell.obj --out viz/heli_bell.glb

Başka bir model (ör. low-poly AH-1 Cobra) için: DEFAULT_ROLES'taki nesne adı öneklerini o
OBJ'deki gövde / ana rotor / kuyruk rotoru nesnelerinin adlarına göre düzenle; görüntüleyici
node adlarını ("main_rotor", "tail_rotor") kullanır.
"""

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np

BODY, MAIN_ROTOR, TAIL_ROTOR = "body", "main_rotor", "tail_rotor"
DEFAULT_ROLES = {"body_helicopter": BODY, "baling_baling_1": MAIN_ROTOR, "baling-baling_2": TAIL_ROTOR}
SKIP_PREFIX = ("tanah",)                     # zemin düzlemi


def parse_mtl(path: Path) -> dict:
    mats, cur = OrderedDict(), None
    if not path.exists():
        return mats
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            cur = mats.setdefault(" ".join(t[1:]), {"Kd": [0.6, 0.6, 0.6], "d": 1.0})
        elif cur is not None and t[0] == "Kd":
            cur["Kd"] = [float(x) for x in t[1:4]]
        elif cur is not None and t[0] == "d":
            cur["d"] = float(t[1])
    return mats


def parse_obj(path: Path):
    """→ V (n,3), N (m,3), objects {ad: {malzeme: [(v, n) köşe listesi olan yüzler]}}"""
    V, N, objects = [], [], OrderedDict()
    cur_obj, cur_mtl = "default", None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0] == "#":
            continue
        t = line.split()
        k = t[0]
        if k == "v":
            V.append((float(t[1]), float(t[2]), float(t[3])))
        elif k == "vn":
            N.append((float(t[1]), float(t[2]), float(t[3])))
        elif k in ("o", "g"):
            cur_obj = " ".join(t[1:]) or "default"
        elif k == "usemtl":
            cur_mtl = " ".join(t[1:])
        elif k == "f":
            corners = []
            for tok in t[1:]:
                parts = tok.split("/")
                vi = int(parts[0])
                vi = vi - 1 if vi > 0 else len(V) + vi
                ni = -1
                if len(parts) >= 3 and parts[2]:
                    ni = int(parts[2])
                    ni = ni - 1 if ni > 0 else len(N) + ni
                corners.append((vi, ni))
            objects.setdefault(cur_obj, OrderedDict()).setdefault(cur_mtl, []).append(corners)
    return np.asarray(V, dtype=np.float64), np.asarray(N, dtype=np.float64), objects


def role_of(name: str):
    if name.lower().startswith(SKIP_PREFIX):
        return None
    for prefix, role in DEFAULT_ROLES.items():
        if name.startswith(prefix):
            return role
    return BODY


def mean_shift_center(P: np.ndarray, start: np.ndarray, axes: tuple, radius: float) -> np.ndarray:
    """Göbek merkezini bul: başlangıç noktası çevresindeki köşelerin ortalamasına yakınsa."""
    c = np.asarray(start, dtype=np.float64).copy()
    for _ in range(20):
        d = np.hypot(P[:, axes[0]] - c[axes[0]], P[:, axes[1]] - c[axes[1]])
        m = P[d < radius]
        if len(m) == 0:
            break
        new = c.copy()
        new[list(axes)] = m[:, list(axes)].mean(0)
        if np.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def face_normals_fallback(P, tris):
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


class GlbWriter:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.bin.extend(data)
        self.views.append(view)
        return len(self.views) - 1

    def vec3(self, arr: np.ndarray, with_bounds: bool) -> int:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        acc = {"bufferView": self._view(arr.tobytes(), 34962), "componentType": 5126,
               "count": int(len(arr)), "type": "VEC3"}
        if with_bounds:
            acc["min"] = [float(x) for x in arr.min(0)]
            acc["max"] = [float(x) for x in arr.max(0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def indices(self, idx: np.ndarray, n_vertices: int) -> int:
        dtype, comp = (np.uint16, 5123) if n_vertices <= 65535 else (np.uint32, 5125)
        arr = np.ascontiguousarray(idx.reshape(-1), dtype=dtype)
        self.accessors.append({"bufferView": self._view(arr.tobytes(), 34963), "componentType": comp,
                               "count": int(arr.size), "type": "SCALAR"})
        return len(self.accessors) - 1

    def write(self, path: Path, gltf: dict):
        gltf["bufferViews"] = self.views
        gltf["accessors"] = self.accessors
        while len(self.bin) % 4:
            self.bin.append(0)
        gltf["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<I4s", len(js), b"JSON"))
            f.write(js)
            f.write(struct.pack("<I4s", len(self.bin), b"BIN\x00"))
            f.write(bytes(self.bin))
        return total


def material_def(name: str, m: dict) -> dict:
    kd = [min(1.0, max(0.0, x)) for x in m["Kd"]]
    alpha = float(m.get("d", 1.0))
    mat = {"name": name, "doubleSided": True,
           "pbrMetallicRoughness": {"baseColorFactor": kd + [alpha], "metallicFactor": 0.1,
                                    "roughnessFactor": 0.6}}
    if alpha < 0.999:                        # kokpit camı
        mat["alphaMode"] = "BLEND"
        mat["pbrMetallicRoughness"].update(roughnessFactor=0.15, metallicFactor=0.0)
    return mat


def convert(obj_path: Path, out_path: Path, length_m: float, cg_height_frac: float) -> dict:
    V, N, objects = parse_obj(obj_path)
    mtls = parse_mtl(obj_path.with_suffix(".mtl"))

    # rol başına üçgenler: {rol: {malzeme: [(vi, ni) üçlüleri]}}
    parts = OrderedDict()
    for name, by_mtl in objects.items():
        role = role_of(name)
        if role is None:
            continue
        dst = parts.setdefault(role, OrderedDict())
        for mtl, faces in by_mtl.items():
            tri = dst.setdefault(mtl or "default", [])
            for corners in faces:
                for i in range(1, len(corners) - 1):   # fan üçgenleme
                    tri.append((corners[0], corners[i], corners[i + 1]))

    def verts_of(role):
        idx = {c[0] for tris in parts[role].values() for t in tris for c in t}
        return V[sorted(idx)]

    body_v = verts_of(BODY)
    rotor_v = verts_of(MAIN_ROTOR)
    tail_v = verts_of(TAIL_ROTOR)
    hub = mean_shift_center(rotor_v, rotor_v.mean(0), (0, 2), 0.9)
    hub[1] = 0.5 * (rotor_v[:, 1].min() + rotor_v[:, 1].max())
    tail_hub = mean_shift_center(tail_v, tail_v.mean(0), (1, 2), 0.3)
    tail_hub[0] = tail_v[:, 0].mean()
    skid = body_v[:, 1].min()
    top = body_v[:, 1].max()
    cg = np.array([0.0, skid + cg_height_frac * (top - skid), hub[2]])
    scale = length_m / float(body_v[:, 2].max() - body_v[:, 2].min())

    w = GlbWriter()
    materials, mat_index = [], {}
    used = [m for tris_by_mtl in parts.values() for m in tris_by_mtl]
    for name in dict.fromkeys(used):             # yalnızca kullanılan malzemeler, sırası korunur
        mat_index[name] = len(materials)
        materials.append(material_def(name, mtls.get(name, {"Kd": [0.6, 0.6, 0.6], "d": 1.0})))
    mat_index.setdefault("default", mat_index[used[0]])

    pivots = {BODY: cg, MAIN_ROTOR: hub, TAIL_ROTOR: tail_hub}
    meshes, nodes, stats = [], [{"name": "heli", "children": []}], {}
    nmax = len(N) + 1
    for role in (BODY, MAIN_ROTOR, TAIL_ROTOR):
        if role not in parts:
            continue
        tris_by_mtl = parts[role]
        all_c = np.array([c for tris in tris_by_mtl.values() for t in tris for c in t], dtype=np.int64)
        keys = all_c[:, 0] * nmax + (all_c[:, 1] + 1)
        uniq, inverse = np.unique(keys, return_inverse=True)
        vi = uniq // nmax
        ni = uniq % nmax - 1
        P = (V[vi] - pivots[role]) * scale
        if len(N) and np.all(ni >= 0):
            Nn = N[ni]
        else:                                   # normal yoksa yüz normali
            Nn = np.zeros_like(P)
            tri_all = inverse.reshape(-1, 3)
            fn = face_normals_fallback(P, tri_all)
            for k in range(3):
                np.add.at(Nn, tri_all[:, k], fn)
        Nn = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
        pos_acc = w.vec3(P, True)
        nor_acc = w.vec3(Nn, False)
        prims, cursor = [], 0
        for mtl, tris in tris_by_mtl.items():
            n = len(tris) * 3
            idx = inverse[cursor:cursor + n]
            cursor += n
            prims.append({"attributes": {"POSITION": pos_acc, "NORMAL": nor_acc},
                          "indices": w.indices(idx, len(P)),
                          "material": mat_index.get(mtl, mat_index["default"])})
        meshes.append({"name": role, "primitives": prims})
        node = {"name": role, "mesh": len(meshes) - 1}
        if role != BODY:
            node["translation"] = [float(x) for x in (pivots[role] - cg) * scale]
            node["extras"] = {"spin_axis": [0, 1, 0] if role == MAIN_ROTOR else [1, 0, 0]}
        nodes.append(node)
        nodes[0]["children"].append(len(nodes) - 1)
        stats[role] = dict(vertices=int(len(P)), triangles=int(len(inverse) // 3))

    gltf = {"asset": {"version": "2.0", "generator": "ah1s-rl-project viz/tools/obj_to_glb.py"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes,
            "materials": materials,
            "extras": {"units": "m", "forward": "+Z", "up": "+Y", "origin": "approx. CG under main rotor mast",
                       "source": obj_path.name, "scale_from_source": scale,
                       "main_rotor_radius_m": float(np.max(np.hypot(rotor_v[:, 0] - hub[0], rotor_v[:, 2] - hub[2])) * scale),
                       "skid_below_origin_m": float((cg[1] - skid) * scale)}}
    size = w.write(out_path, gltf)
    return dict(size_bytes=size, scale=scale, hub=hub.round(3).tolist(), tail_hub=tail_hub.round(3).tolist(),
                cg=cg.round(3).tolist(), parts=stats, extras=gltf["extras"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj", type=Path)
    ap.add_argument("--out", type=Path, default=Path("viz/heli_bell.glb"))
    ap.add_argument("--length-m", type=float, default=13.6, help="gövde boyu (AH-1S ≈ 13.6 m)")
    ap.add_argument("--cg-height-frac", type=float, default=0.43,
                    help="ağırlık merkezi yüksekliği: kızak altı → gövde tepesi oranı")
    args = ap.parse_args(argv)
    info = convert(args.obj, args.out, args.length_m, args.cg_height_frac)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
