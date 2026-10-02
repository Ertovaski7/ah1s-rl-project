// =====================================================================
// AH-1S Cobra: low-poly prosedürel 3D model (Three.js)
// =====================================================================
// Ölçüler JSBSim AH-1S modelinden (aircraft/ah1s/ah1s.xml, Engines/ah1s_rotor.xml, Engines/ah1s_tail_rotor.xml,
// Systems/rotor_control.xml):
//   ana rotor göbeği FS 176 / WL 153 in; Ø 44 ft, 2 kanat, veter 2.25 ft, burulma −0.175 rad, nominal 324 rpm
//   kuyruk rotoru FS 496.67 / BL +16 (sağda) / WL 119 in; Ø 8.5 ft, 2 kanat, veter 0.7 ft, nominal 1660 rpm
//   kızak temas noktaları FS 115.9 / 235, BL ±42, WL −4…−5 in; kuyruk tamponu FS 490 / WL 40 in; CG FS 172
//   stub kanat açıklığı 10.75 ft
// Gövde biçimi fotoğraflardan: dar tandem kokpit, düz panelli kanopi, burunda TSU nişangâhı, M197 taret,
// stub kanatlarda roket podu (iç) + 4'lü TOW (dış), motor girişleri, egzoz, süpürülmüş dikey stabilize, ventral fin.
//
// Çerçeve: metre, +Z burun, +Y yukarı, +X sol (iskele); glTF ile aynı.
// Orijin: CG istasyonu (FS 172); yerde CG yüksekliği 6.3 ft (takeoff_curriculum.GROUND_H_FT) → kızak altı y = −1.92 m.
//
// Gruplar (her biri ayrı THREE.Group, adıyla bulunur):
//   body        gövde, kanopi, kuyruk, kanatlar, dış yükler, kızaklar, mast, swashplate (dönmeyen parçalar)
//   main_rotor  pivot ana rotor göbeğinde: tpp (disk eğimi, cyclic) → spin (azimut, +Y ekseni) → kanat hatvesi (collective)
//   tail_rotor  pivot kuyruk rotoru göbeğinde: spin (azimut, gövdenin yanal ekseni X)
//
// Arayüz:
//   const heli = createAH1S({ livery: "tr" });   scene.add(heli.root);   her karede heli.update(dt)
//   heli.setState({ rotorRPM, collective, cyclic })   alanlar isteğe bağlı; sonlu sayı olmayan değer yok sayılır
//     rotorRPM    ana rotor devri, rpm (varsayılan nominal 324); kuyruk rotoru orantılı döner: × 1660 / 324
//     collective  0…1 (fcs/collective-cmd-norm) → kanat kök hatvesi θ0 = 0.14 + 0.22·c rad (8.0°…20.6°)
//     cyclic      { lon, lat } ya da [lon, lat], −1…+1 (fcs/elevator-cmd-norm, fcs/aileron-cmd-norm);
//                 + lon disk öne, + lat disk sağa; eğim = kazanç × komut (boylamsal 0.125 rad, yanal 0.05 rad)
//   heli.update(dt)    rotor azimutlarını Ω·dt ilerletir (dt: saniye, gerçek zaman)
//   heli.getState()    durum + türetilmiş değerler (kuyruk devri, 0.75R hatvesi, disk eğimi)
//   heli.setLivery("tr" | "od"), heli.setDiscColor(renk), heli.dispose()

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.169.0/+esm";

const IN = 0.0254, FT = 0.3048;
export const AH1S = Object.freeze({
  MR_RPM: 324, TR_RPM: 1660, TR_RATIO: 1660 / 324,
  MR_RADIUS: 22 * FT, MR_CHORD: 2.25 * FT, MR_TWIST: -0.175,
  TR_RADIUS: 4.25 * FT, TR_CHORD: 0.7 * FT, TR_PITCH: 0.12,
  COLL_GAIN: 0.22, COLL_BIAS: 0.14, COLL_MAX: 0.7, LON_GAIN: 0.125, LAT_GAIN: 0.05,
  GROUND_H: 6.3 * FT,
});
export const LIVERIES = Object.freeze({
  tr: { name: "Kara Kuvvetleri kamuflajı", colors: [0xb08c60, 0x66673f, 0x3d3b2d] },
  od: { name: "Olive drab", colors: [0x4e5434, 0x4a5032, 0x454a2f] },
});

// JSBSim yapısal çerçevesi (inç; x geriye, y sağa, z yukarı) → model çerçevesi
const jsb = (x, y, z) => new THREE.Vector3(-y * IN, (z + 4.5) * IN - AH1S.GROUND_H, (172 - x) * IN);
const MR_HUB = jsb(176, 0, 153), TR_HUB = jsb(496.67, 16, 119);
// tasarım tablolarının koordinatı: x (+ sol, m), h (yerden, m), zn (TSU ön yüzünden geriye, m)
// TSU ön yüzü mastın 4.90 m önünde → gövde boyu ≈ 13.55 m (AH-1S: 13.59 m)
const NOSE_Z = MR_HUB.z + 4.9;
const P = (x, h, zn) => new THREE.Vector3(x, h - AH1S.GROUND_H, NOSE_Z - zn);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------------
// geometri yardımcıları: her parça indekssiz, yalnızca konum taşıyan BufferGeometry
// (flatShading normali türevlerden hesaplar); aynı malzemedekiler tek geometride birleşir
// ---------------------------------------------------------------------
function posOnly(g) {
  const n = g.index ? g.toNonIndexed() : g;
  for (const k of Object.keys(n.attributes)) if (k !== "position") n.deleteAttribute(k);
  if (n !== g) g.dispose();
  return n;
}
function tris(pts) {
  const a = new Float32Array(pts.length * 3); pts.forEach((p, i) => p.toArray(a, i * 3));
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(a, 3)); return g;
}
function merge(list) {
  let n = 0; for (const g of list) n += g.attributes.position.array.length;
  const a = new Float32Array(n); let o = 0;
  for (const g of list) { a.set(g.attributes.position.array, o); o += g.attributes.position.array.length; g.dispose(); }
  const out = new THREE.BufferGeometry(); out.setAttribute("position", new THREE.BufferAttribute(a, 3));
  out.computeBoundingSphere(); return out;
}
const _m = new THREE.Matrix4(), _one = new THREE.Vector3(1, 1, 1), Y_UP = new THREE.Vector3(0, 1, 0);
const place = (g, pos, quat = new THREE.Quaternion()) => posOnly(g).applyMatrix4(_m.compose(pos, quat, _one));
const mirrorX = g => g.clone().applyMatrix4(new THREE.Matrix4().makeScale(-1, 1, 1));
const withMirror = g => [g, mirrorX(g)];
// halkalar arası yüzey (aynı nokta sayısı); closed=false → halka açık çizgi (ör. gövdeye oturan kanopi)
function loft(rings, { closed = true, capStart = true, capEnd = true } = {}) {
  const out = [], m = rings[0].length, segs = closed ? m : m - 1;
  for (let s = 0; s + 1 < rings.length; s++) {
    const A = rings[s], B = rings[s + 1];
    for (let k = 0; k < segs; k++) { const k1 = (k + 1) % m; out.push(A[k], B[k], B[k1], A[k], B[k1], A[k1]); }
  }
  const cap = R => { const c = R.reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(1 / R.length);
    for (let k = 0; k < R.length; k++) out.push(c, R[k], R[(k + 1) % R.length]); };
  if (capStart) cap(rings[0]);
  if (capEnd) cap(rings[rings.length - 1]);
  return tris(out);
}
function box(sx, sy, sz, pos, quat) { return place(new THREE.BoxGeometry(sx, sy, sz), pos, quat); }
// a → b ekseninde silindir (rA a ucunda, rB b ucunda)
function cyl(a, b, rA, rB = rA, seg = 8, open = false) {
  const d = new THREE.Vector3().subVectors(b, a);
  const q = new THREE.Quaternion().setFromUnitVectors(Y_UP, d.clone().normalize());
  return place(new THREE.CylinderGeometry(rB, rA, d.length(), seg, 1, open), a.clone().add(b).multiplyScalar(0.5), q);
}
function disc(center, normal, r, seg = 12, r0 = 0) {
  const g = r0 > 0 ? new THREE.RingGeometry(r0, r, seg) : new THREE.CircleGeometry(r, seg);
  return place(g, center, new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal.clone().normalize()));
}
function tube(points, r, tubular = 24, radial = 6) {
  return posOnly(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), tubular, r, radial, false));
}
// altıgen profil: hücum kenarından (u = 0) firar kenarına (u = c), kalınlık t
const foil = (c, t) => [[0, 0], [0.12 * c, 0.5 * t], [0.55 * c, 0.5 * t], [c, 0], [0.55 * c, -0.5 * t], [0.12 * c, -0.5 * t]];
// kesitleri { le, c, t } olan yüzey; u yönü (veter) ve v yönü (kalınlık) bütün kesitlerde ortak
function surface(sections, uDir, vDir) {
  return loft(sections.map(s => foil(s.c, s.t).map(([u, v]) => s.le.clone().addScaledVector(uDir, u).addScaledVector(vDir, v))));
}
const AFT = new THREE.Vector3(0, 0, -1), UP = new THREE.Vector3(0, 1, 0), LEFT = new THREE.Vector3(1, 0, 0);
// (zn, h) çokgeninden x = ±t/2 kalınlıkta levha
function plate(pts, t, x0 = 0) {
  return loft([pts.map(([zn, h]) => P(x0 + t / 2, h, zn)), pts.map(([zn, h]) => P(x0 - t / 2, h, zn))]);
}
// köşeleri pahlı dikdörtgen kesit (8 nokta), zn istasyonunda
function rectRing(zn, hw, h0, h1, ch) {
  return [P(hw - ch, h0, zn), P(hw, h0 + ch, zn), P(hw, h1 - ch, zn), P(hw - ch, h1, zn),
          P(-hw + ch, h1, zn), P(-hw, h1 - ch, zn), P(-hw, h0 + ch, zn), P(-hw + ch, h0, zn)];
}

// ---------------------------------------------------------------------
// gövde istasyonları: zn, alt yükseklik hb, alt yarı genişlik wb, alt omuz hc, en geniş yarı genişlik w,
// üst omuz hs, üst yarı genişlik wt, üst yükseklik ht (kokpitte ht / wt = kanopi pervazı)
// ---------------------------------------------------------------------
const STATIONS = [
  //  zn     hb    wb    hc    w     hs    wt    ht
  [0.30, 1.02, 0.15, 1.10, 0.25, 1.36, 0.21, 1.44],     // burun (TSU'nun arkası)
  [0.75, 0.83, 0.18, 0.95, 0.35, 1.38, 0.31, 1.47],
  [1.15, 0.69, 0.20, 0.85, 0.42, 1.40, 0.37, 1.49],     // ön cam tabanı
  [1.70, 0.61, 0.21, 0.80, 0.46, 1.38, 0.40, 1.48],
  [2.40, 0.58, 0.22, 0.78, 0.48, 1.40, 0.41, 1.50],     // nişancı
  [3.10, 0.57, 0.22, 0.78, 0.48, 1.52, 0.41, 1.63],     // pilot (koltuk yükseltilmiş)
  [3.80, 0.56, 0.22, 0.78, 0.48, 1.70, 0.41, 1.82],
  [4.15, 0.56, 0.22, 0.78, 0.48, 1.92, 0.39, 2.26],     // kanopi sonu → mast kaportası
  [4.50, 0.56, 0.22, 0.78, 0.48, 2.24, 0.33, 2.98],
  [5.40, 0.56, 0.22, 0.78, 0.48, 2.30, 0.33, 3.02],     // motor
  [6.30, 0.62, 0.21, 0.86, 0.46, 2.28, 0.31, 2.86],
  [7.00, 0.84, 0.18, 1.04, 0.40, 2.16, 0.25, 2.56],     // egzoz altı
  [7.60, 1.10, 0.15, 1.28, 0.32, 2.06, 0.18, 2.36],     // kuyruk konisi kökü
  [9.60, 1.50, 0.10, 1.62, 0.22, 2.08, 0.11, 2.25],
  [12.0, 1.86, 0.07, 1.92, 0.14, 2.14, 0.07, 2.24],
  [13.0, 2.03, 0.05, 2.07, 0.10, 2.19, 0.05, 2.26],
  [13.35, 2.12, 0.03, 2.14, 0.05, 2.20, 0.03, 2.24],    // kuyruk ucu
];
function bodyAt(zn) {
  const S = STATIONS; let i = 0; while (i < S.length - 2 && S[i + 1][0] < zn) i++;
  const a = S[i], b = S[i + 1], w = clamp((zn - a[0]) / (b[0] - a[0]), 0, 1);
  return a.map((v, k) => v + (b[k] - v) * w);
}
const fuseRing = ([zn, hb, wb, hc, w, hs, wt, ht]) =>
  [P(wb, hb, zn), P(w, hc, zn), P(w, hs, zn), P(wt, ht, zn), P(-wt, ht, zn), P(-w, hs, zn), P(-w, hc, zn), P(-wb, hb, zn)];
// kanopi istasyonları: zn, üst yarı genişlik, üst yükseklik (null: pervazda, ön camın tabanı)
const CANOPY = [
  [1.15, null, null],
  [1.92, 0.21, 2.17],        // nişancı ön camının üstü
  [2.45, 0.22, 2.22],
  [2.55, 0.22, 2.56],        // pilot ön camı (basamak)
  [3.70, 0.21, 2.62],
  [4.10, 0.19, 2.55],        // arka panel
];
function canopyRings() {
  return CANOPY.map(([zn, tw, th]) => {
    const [, , , , , , wt, ht] = bodyAt(zn);
    const top = th == null ? [wt * 0.55, ht] : [tw, th];
    return [P(wt, ht, zn), P(top[0], top[1], zn), P(-top[0], top[1], zn), P(-wt, ht, zn)];
  });
}

// ---------------------------------------------------------------------
// malzemeler
// ---------------------------------------------------------------------
const CAMO_GLSL = `
varying vec3 vCamoPos;
uniform vec3 uCamo0; uniform vec3 uCamo1; uniform vec3 uCamo2;
float camoHash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float camoNoise(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(camoHash(i), camoHash(i + vec3(1, 0, 0)), f.x), mix(camoHash(i + vec3(0, 1, 0)), camoHash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(camoHash(i + vec3(0, 0, 1)), camoHash(i + vec3(1, 0, 1)), f.x), mix(camoHash(i + vec3(0, 1, 1)), camoHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
vec3 camoColor(vec3 p) {
  p *= vec3(0.5, 0.62, 0.3);                          // iri lekeler, gövde boyunca uzun
  float a = 0.8 * camoNoise(p) + 0.2 * camoNoise(p * 2.6 + 5.3);
  float b = 0.82 * camoNoise(p * 0.85 + 41.7) + 0.18 * camoNoise(p * 2.2 + 17.9);
  return mix(mix(uCamo0, uCamo1, step(0.5, a)), uCamo2, step(0.6, b));
}`;
function camoMaterial(uniforms) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.82, metalness: 0.05, flatShading: true, side: THREE.DoubleSide });
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 vCamoPos;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvCamoPos = position;");
    sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\n" + CAMO_GLSL)
      .replace("vec4 diffuseColor = vec4( diffuse, opacity );", "vec4 diffuseColor = vec4( diffuse * camoColor( vCamoPos ), opacity );");
  };
  m.customProgramCacheKey = () => "ah1s-camo";
  return m;
}
function makeMaterials(livery) {
  const std = (color, roughness = 0.7, metalness = 0.1, extra = {}) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, flatShading: true, side: THREE.DoubleSide, ...extra });
  const uniforms = { uCamo0: { value: new THREE.Color() }, uCamo1: { value: new THREE.Color() }, uCamo2: { value: new THREE.Color() } };
  const M = {
    uniforms, camo: camoMaterial(uniforms),
    metal: std(0x2b2e2b, 0.55, 0.35),                     // rotor başı, mast, namlular
    stores: std(0x4b4e3d, 0.75, 0.1),                     // roket podu, TOW tüpleri
    skid: std(0x3c3f33, 0.6, 0.25),
    hole: std(0x0d0f10, 0.9, 0.0),                        // giriş / egzoz / tüp ağızları
    glass: std(0x1f3038, 0.14, 0.05),
    frame: new THREE.LineBasicMaterial({ color: 0x121411 }),
    mrBlade: std(0x2c2f2c, 0.62, 0.1, { transparent: true }),
    trBlade: std(0x2c2f2c, 0.62, 0.1, { transparent: true }),
    mrDisc: new THREE.MeshBasicMaterial({ color: 0x1b1d1a, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
    trDisc: new THREE.MeshBasicMaterial({ color: 0x1b1d1a, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
  };
  M.setLivery = name => {                               // kanopi çerçevesi gövde boyasının koyu tonu
    const L = LIVERIES[name] || LIVERIES.tr; L.colors.forEach((c, i) => uniforms[`uCamo${i}`].value.setHex(c));
    M.frame.color.setHex(L.colors[0]).multiplyScalar(0.62);
  };
  M.setLivery(livery);
  return M;
}

// ---------------------------------------------------------------------
// gövde (dönmeyen parçalar)
// ---------------------------------------------------------------------
function buildBody(M) {
  const camo = [], metal = [], stores = [], skid = [], hole = [], glass = [], frame = [];
  // ana gövde + kuyruk konisi
  camo.push(loft(STATIONS.map(fuseRing)));
  // TSU nişangâhı (burun): pahlı kutu, önde koyu pencere
  camo.push(loft([rectRing(-0.02, 0.18, 1.07, 1.39, 0.05), rectRing(0.06, 0.245, 1.0, 1.45, 0.07), rectRing(0.42, 0.245, 1.0, 1.45, 0.07)]));
  glass.push(loft([[P(0.13, 1.13, -0.03), P(0.13, 1.33, -0.03), P(-0.13, 1.33, -0.03), P(-0.13, 1.13, -0.03)]], { capEnd: false }));
  // kanopi (düz paneller) + çerçeve çizgileri; arka paneli kaporta kapatır
  const cr = canopyRings();
  glass.push(loft(cr, { closed: false, capStart: false, capEnd: false }));
  camo.push(tris([cr.at(-1)[0], cr.at(-1)[1], cr.at(-1)[2], cr.at(-1)[0], cr.at(-1)[2], cr.at(-1)[3]]));
  for (const r of cr) for (let k = 0; k < 3; k++) frame.push(r[k], r[k + 1]);
  for (let k = 0; k < 4; k++) for (let s = 0; s + 1 < cr.length; s++) frame.push(cr[s][k], cr[s + 1][k]);
  // tel kesici (pilot ön camının üstü)
  metal.push(plate([[2.62, 2.58], [2.3, 2.66], [2.34, 2.72], [2.7, 2.62]], 0.025));
  // M197 taret: döner tabla, gövde, üç namlu
  camo.push(cyl(P(0, 0.52, 1.1), P(0, 0.74, 1.1), 0.24, 0.24, 10));
  metal.push(box(0.17, 0.17, 0.6, P(0, 0.5, 0.92)));
  for (let k = 0; k < 3; k++) {
    const a = k * 2 * Math.PI / 3, dx = 0.045 * Math.cos(a), dh = 0.045 * Math.sin(a);
    metal.push(cyl(P(dx, 0.5 + dh, 0.65), P(dx, 0.5 + dh, -0.32), 0.022, 0.022, 5));
  }
  metal.push(cyl(P(0, 0.5, 0.08), P(0, 0.5, -0.02), 0.075, 0.075, 8));
  // motor hava girişleri (mastın iki yanında, öne bakar)
  for (const x of [0.37, -0.37]) {
    camo.push(cyl(P(x, 2.55, 5.05), P(x, 2.55, 4.42), 0.15, 0.19, 8));
    hole.push(disc(P(x, 2.55, 4.415), new THREE.Vector3(0, 0, 1), 0.155, 8));
  }
  // mast kaportası, mast, swashplate
  camo.push(cyl(P(0, 2.94, 4.9), P(0, 3.2, 4.9), 0.27, 0.19, 10));
  metal.push(cyl(P(0, 3.1, 4.9), MR_HUB.clone().add(new THREE.Vector3(0, 0.02, 0)), 0.085, 0.075, 8));
  metal.push(cyl(P(0, 3.4, 4.9), P(0, 3.47, 4.9), 0.26, 0.26, 12));
  // egzoz (motorun arkasında, ağzı geriye ve hafif yukarı) + IR karıştırıcı (ALQ-144)
  camo.push(cyl(P(0, 2.6, 6.25), P(0, 2.74, 7.45), 0.27, 0.29, 10));
  hole.push(disc(P(0, 2.742, 7.46), new THREE.Vector3(0, 0.12, -1.2), 0.24, 10));
  metal.push(cyl(P(0, 2.86, 5.95), P(0, 3.2, 5.95), 0.12, 0.12, 10));
  metal.push(place(new THREE.SphereGeometry(0.12, 10, 4, 0, Math.PI * 2, 0, Math.PI / 2), P(0, 3.2, 5.95)));
  // stub kanatlar + pilonlar + dış yükler (iç: 19'lu roket podu, dış: 4'lü TOW)
  const wing = surface([{ le: P(0.3, 1.55, 4.55), c: 0.75, t: 0.12 }, { le: P(1.64, 1.55, 4.64), c: 0.6, t: 0.08 }], AFT, UP);
  camo.push(...withMirror(wing));
  const sideParts = { camo: [], stores: [], hole: [] };
  for (const [x, len] of [[0.95, 0.5], [1.45, 0.45]]) sideParts.camo.push(box(0.07, 0.14, len, P(x, 1.45, 4.92)));
  sideParts.stores.push(cyl(P(0.95, 1.18, 5.85), P(0.95, 1.18, 4.25), 0.2, 0.2, 10));
  sideParts.stores.push(cyl(P(0.95, 1.18, 4.25), P(0.95, 1.18, 4.1), 0.2, 0.15, 10));
  sideParts.hole.push(disc(P(0.95, 1.18, 4.095), new THREE.Vector3(0, 0, 1), 0.13, 10));
  for (const dx of [-0.09, 0.09]) for (const dh of [-0.09, 0.09]) {
    sideParts.stores.push(cyl(P(1.45 + dx, 1.2 + dh, 5.5), P(1.45 + dx, 1.2 + dh, 4.3), 0.085, 0.085, 8));
    sideParts.hole.push(disc(P(1.45 + dx, 1.2 + dh, 4.295), new THREE.Vector3(0, 0, 1), 0.065, 8));
  }
  sideParts.stores.push(box(0.36, 0.06, 0.12, P(1.45, 1.31, 4.6)), box(0.36, 0.06, 0.12, P(1.45, 1.31, 5.2)));
  for (const g of sideParts.camo) camo.push(...withMirror(g));
  for (const g of sideParts.stores) stores.push(...withMirror(g));
  for (const g of sideParts.hole) hole.push(...withMirror(g));
  // yatay stabilize (kuyruk konisinin ortasında), dikey stabilize (süpürülmüş), ventral fin + kuyruk tamponu
  camo.push(...withMirror(surface([{ le: P(0, 1.86, 9.3), c: 0.62, t: 0.07 }, { le: P(1.05, 1.86, 9.38), c: 0.5, t: 0.05 }], AFT, UP)));
  camo.push(surface([{ le: P(0, 2.12, 11.4), c: 1.92, t: 0.16 }, { le: P(0, 3.6, 12.78), c: 0.78, t: 0.09 }], AFT, LEFT));
  camo.push(plate([[12.2, 2.0], [13.1, 2.08], [12.98, 1.13], [12.76, 1.13]], 0.05));
  skid.push(cyl(P(0, 1.13, 12.72), P(0, 1.13, 13.02), 0.025, 0.025, 6));
  // kuyruk rotoru dişli kutusu (dikey stabilizenin sağ yüzünde) + mili
  const trH = TR_HUB.y + AH1S.GROUND_H, trZn = NOSE_Z - TR_HUB.z;
  camo.push(box(0.15, 0.28, 0.36, P(-0.1, trH, trZn)));
  metal.push(cyl(P(-0.16, trH, trZn), P(TR_HUB.x + 0.03, trH, trZn), 0.035, 0.035, 6));
  // kızaklar: iki boru (ucu yukarı kıvrık) + iki köprü borusu (kemer)
  const SK = -jsb(0, 42, 0).x;                  // 42 in
  for (const x of [SK, -SK]) {
    skid.push(tube([P(x, 0.42, 2.42), P(x, 0.24, 2.58), P(x, 0.08, 2.82), P(x, 0.045, 3.2), P(x, 0.045, 5.0), P(x, 0.045, 6.82)], 0.045, 28, 6));
  }
  for (const zn of [3.6, 6.0]) {
    const half = [P(SK, 0.05, zn), P(SK - 0.03, 0.3, zn), P(SK - 0.2, 0.52, zn), P(0.55, 0.61, zn), P(0.0, 0.63, zn)];
    skid.push(tube([...half, ...half.slice(0, -1).reverse().map(p => p.clone().setX(-p.x))], 0.045, 24, 6));
  }

  const body = new THREE.Group(); body.name = "body";
  const add = (list, mat, name) => { if (!list.length) return; const mesh = new THREE.Mesh(merge(list), mat); mesh.name = name; body.add(mesh); };
  add(camo, M.camo, "body_paint"); add(metal, M.metal, "body_metal"); add(stores, M.stores, "body_stores");
  add(skid, M.skid, "body_skids"); add(hole, M.hole, "body_openings"); add(glass, M.glass, "body_glass");
  const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(frame), M.frame); lines.name = "canopy_frame"; body.add(lines);
  return body;
}

// ---------------------------------------------------------------------
// kanatlar: +X boyunca, veter Z'de (hücum kenarı −Z, hatve ekseni çeyrek veterde), burulma X ekseni etrafında
// ---------------------------------------------------------------------
function bladeGeometry(sections, twist, R) {
  return loft(sections.map(({ r, le, c, t }) => {
    const a = twist * r / R, ca = Math.cos(a), sa = Math.sin(a);
    return foil(c, t).map(([u, v]) => { const z = le + u; return new THREE.Vector3(r, v * ca - z * sa, v * sa + z * ca); });
  }));
}

function buildMainRotor(M) {
  const R = AH1S.MR_RADIUS, c = AH1S.MR_CHORD, le = -0.25 * c, te = 0.75 * c;
  const group = new THREE.Group(); group.name = "main_rotor"; group.position.copy(MR_HUB);
  const tpp = new THREE.Group(); tpp.name = "main_rotor_tpp"; tpp.rotation.order = "ZXY"; group.add(tpp);
  const spin = new THREE.Group(); spin.name = "main_rotor_spin"; tpp.add(spin);
  // göbek: tahterevalli boyunduruğu, kanat tutucuları, mast somunu, hatve kolları ve çubukları (dönen swashplate'e iner)
  const hub = [box(0.9, 0.1, 0.28, new THREE.Vector3()), cyl(new THREE.Vector3(0, -0.04, 0), new THREE.Vector3(0, 0.17, 0), 0.075, 0.06, 8)];
  for (const s of [1, -1]) {
    hub.push(box(0.34, 0.13, 0.24, new THREE.Vector3(0.5 * s, 0, 0)));
    hub.push(box(0.06, 0.05, 0.2, new THREE.Vector3(0.44 * s, -0.04, -0.2 * s)));
    hub.push(cyl(new THREE.Vector3(0.44 * s, -0.05, -0.29 * s), new THREE.Vector3(0.4 * s, -0.66, -0.27 * s), 0.016, 0.016, 5));
  }
  const hubMesh = new THREE.Mesh(merge(hub), M.metal); hubMesh.name = "main_rotor_hub"; spin.add(hubMesh);
  const blade = bladeGeometry([
    { r: 0.32, le: -0.09, c: 0.2, t: 0.1 }, { r: 0.78, le: -0.1, c: 0.24, t: 0.07 }, { r: 1.08, le, c, t: 0.06 },
    { r: R - 0.42, le, c, t: 0.06 }, { r: R, le: te - 0.36, c: 0.36, t: 0.035 },                  // süpürülmüş, daralan uç
  ], AH1S.MR_TWIST, R);
  const pitch = [];
  for (let i = 0; i < 2; i++) {
    const arm = new THREE.Group(); arm.rotation.y = i * Math.PI; spin.add(arm);
    const p = new THREE.Group(); p.name = `main_rotor_blade_${i}`; arm.add(p); pitch.push(p);
    p.add(new THREE.Mesh(i ? blade.clone() : blade, M.mrBlade));
  }
  const blur = new THREE.Mesh(disc(new THREE.Vector3(), UP, R, 64, 0.6), M.mrDisc); blur.name = "main_rotor_disc"; blur.renderOrder = 2; tpp.add(blur);
  return { group, tpp, spin, pitch, blur };
}

function buildTailRotor(M) {
  const R = AH1S.TR_RADIUS, c = AH1S.TR_CHORD;
  const group = new THREE.Group(); group.name = "tail_rotor"; group.position.copy(TR_HUB);
  const spin = new THREE.Group(); spin.name = "tail_rotor_spin"; group.add(spin);
  const hub = [cyl(new THREE.Vector3(0.05, 0, 0), new THREE.Vector3(-0.08, 0, 0), 0.06, 0.045, 8), box(0.07, 0.34, 0.1, new THREE.Vector3(-0.01, 0, 0))];
  const hubMesh = new THREE.Mesh(merge(hub), M.metal); hubMesh.name = "tail_rotor_hub"; spin.add(hubMesh);
  // kanat +X'te kurulur, Z ekseni etrafında 90° döndürülür: açıklık +Y, kaldırma −X (sağa itki), hücum kenarı −Z
  const blade = bladeGeometry([{ r: 0.1, le: -0.04, c: 0.12, t: 0.04 }, { r: 0.28, le: -0.25 * c, c, t: 0.03 }, { r: R, le: -0.25 * c, c, t: 0.022 }], 0, R)
    .applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI / 2));
  for (let i = 0; i < 2; i++) {
    const arm = new THREE.Group(); arm.rotation.x = i * Math.PI; spin.add(arm);
    const p = new THREE.Group(); p.name = `tail_rotor_blade_${i}`; p.rotation.y = AH1S.TR_PITCH; arm.add(p);
    p.add(new THREE.Mesh(i ? blade.clone() : blade, M.trBlade));
  }
  const blur = new THREE.Mesh(disc(new THREE.Vector3(-0.005, 0, 0), LEFT, R, 40, 0.12), M.trDisc); blur.name = "tail_rotor_disc"; blur.renderOrder = 2; group.add(blur);
  return { group, spin, blur };
}

// ---------------------------------------------------------------------
// model
// ---------------------------------------------------------------------
export class AH1SModel {
  constructor({ livery = "tr" } = {}) {
    this.mat = makeMaterials(livery); this.livery = LIVERIES[livery] ? livery : "tr";
    this.root = new THREE.Group(); this.root.name = "ah1s";
    this.body = buildBody(this.mat);
    this._mr = buildMainRotor(this.mat); this._tr = buildTailRotor(this.mat);
    this.mainRotor = this._mr.group; this.tailRotor = this._tr.group;
    this.root.add(this.body, this.mainRotor, this.tailRotor);
    this.state = { rotorRPM: AH1S.MR_RPM, collective: 0, cyclic: { lon: 0, lat: 0 } };
    this.azimuth = { main: 0.55, tail: 0.9 };            // rad (duran rotor çapraz görünsün)
    this._mr.spin.rotation.y = this.azimuth.main; this._tr.spin.rotation.x = -this.azimuth.tail;
    this._blur = { main: 0, tail: 0 };
    this._apply(); this._applyBlur();
  }
  setState({ rotorRPM, collective, cyclic } = {}) {
    const ok = Number.isFinite, S = this.state;
    if (ok(rotorRPM)) S.rotorRPM = Math.max(0, rotorRPM);
    if (ok(collective)) S.collective = clamp(collective, 0, 1);
    if (cyclic != null) {
      const lon = Array.isArray(cyclic) ? cyclic[0] : cyclic.lon, lat = Array.isArray(cyclic) ? cyclic[1] : cyclic.lat;
      if (ok(lon)) S.cyclic.lon = clamp(lon, -1, 1);
      if (ok(lat)) S.cyclic.lat = clamp(lat, -1, 1);
    }
    this._apply();
    return this;
  }
  getState() {
    const S = this.state, th = this._theta0();
    return { rotorRPM: S.rotorRPM, tailRPM: S.rotorRPM * AH1S.TR_RATIO, collective: S.collective, cyclic: { ...S.cyclic },
      rootPitchDeg: th * 180 / Math.PI, pitch75Deg: (th + 0.75 * AH1S.MR_TWIST) * 180 / Math.PI,
      tiltDeg: { lon: AH1S.LON_GAIN * S.cyclic.lon * 180 / Math.PI, lat: AH1S.LAT_GAIN * S.cyclic.lat * 180 / Math.PI } };
  }
  _theta0() { return clamp(AH1S.COLL_BIAS + AH1S.COLL_GAIN * this.state.collective, 0, AH1S.COLL_MAX); }
  _apply() {
    const th = this._theta0();
    for (const p of this._mr.pitch) p.rotation.x = th;                    // + hücum kenarı yukarı
    this._mr.tpp.rotation.x = AH1S.LON_GAIN * this.state.cyclic.lon;      // + ön kenar aşağı
    this._mr.tpp.rotation.z = AH1S.LAT_GAIN * this.state.cyclic.lat;      // + sağ (−X) kenar aşağı
  }
  // dt: gerçek zaman (s). Ana rotor üstten bakınca saat yönünün tersine döner; kuyruk rotorunda üst kanat geriye gider.
  update(dt) {
    if (!(dt > 0)) return this;
    dt = Math.min(dt, 0.25);
    const om = this.state.rotorRPM * Math.PI / 30, TAU = Math.PI * 2;
    this.azimuth.main = (this.azimuth.main + om * dt) % TAU;
    this.azimuth.tail = (this.azimuth.tail + om * AH1S.TR_RATIO * dt) % TAU;
    this._mr.spin.rotation.y = this.azimuth.main;
    this._tr.spin.rotation.x = -this.azimuth.tail;
    // iki kanatlı rotor bir karede ~90°'den fazla dönünce göz yönünü kaçırır (stroboskop): kanatlar soluklaşır, bulanık disk koyulaşır
    const k = Math.min(1, dt * 6), deg = x => x * dt * 180 / Math.PI;
    this._blur.main += (smooth(25, 75, deg(om)) - this._blur.main) * k;
    this._blur.tail += (smooth(25, 75, deg(om * AH1S.TR_RATIO)) - this._blur.tail) * k;
    this._applyBlur();
    return this;
  }
  _applyBlur() {
    const run = smooth(0, 0.6, this.state.rotorRPM / AH1S.MR_RPM), M = this.mat;
    M.mrBlade.opacity = 1 - 0.7 * this._blur.main * run;
    M.trBlade.opacity = 1 - 0.8 * this._blur.tail * run;
    M.mrDisc.opacity = run * (0.06 + 0.14 * this._blur.main);
    M.trDisc.opacity = run * (0.06 + 0.16 * this._blur.tail);
    this._mr.blur.visible = M.mrDisc.opacity > 0.004; this._tr.blur.visible = M.trDisc.opacity > 0.004;
  }
  setLivery(name) { if (LIVERIES[name]) { this.livery = name; this.mat.setLivery(name); } return this; }
  setDiscColor(color) { this.mat.mrDisc.color.set(color); this.mat.trDisc.color.set(color); return this; }
  dispose() {
    this.root.traverse(o => o.geometry?.dispose());
    for (const m of Object.values(this.mat)) if (m && m.isMaterial) m.dispose();
  }
}
export const createAH1S = opts => new AH1SModel(opts);
