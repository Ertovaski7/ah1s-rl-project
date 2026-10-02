// =====================================================================
// Arazi: hafif dağlık, kuru otlu yamaçlar, meşe benekleri, vadilerde köyler (Three.js, birim ft)
// =====================================================================
// Yalnızca görsel arka plan: JSBSim fiziği düz zeminde kalır. Sayfa helikopteri kayıttaki AGL irtifasıyla bu arazinin
// üstüne koyar; helikopter tepelerin içinden geçmez, kızaklar zemine oturur.
//
// Sahne çerçevesi: X doğu, Z güney (−kuzey), Y yukarı, ft. Pist orijinde, çevresi düz vadi tabanı (zemin 0).
// Her şey deterministik (sabit tohum): aynı noktada hep aynı yükseklik, köy, yol, ağaç.
// Çizilen pencere helikopteri izler (merkez 4000 ft katlarına oturur): sonsuz arazi, sabit maliyet.
// Uzak ağaçlar zemin shader'ında benek; yakındakiler (±4500 ft) aynı beneklerin üstünde low-poly 3D ağaç.
//
// Arayüz:
//   const terrain = createTerrain();   scene.add(terrain.group);
//   terrain.update(X, Z, cx, cz)  her karede: helikopterin ve kameranın konumu; gerekince pencereyi / ağaçları yeniden kurar
//                                 (yakın 3D ağaçlar kamerayı izler: serbest kamerayla uzağa bakınca da çıplak benek kalmaz)
//   terrain.groundAt(X, Z)        kafes yüzeyinin yüksekliği (ft; pencerenin çekirdeğinde çizilen üçgenlerle aynı)
//   terrain.surfaceAt(X, Z)       o an çizilen yüzey (çekirdek dışında seyrek ağ; uzak nesneler ve kamera tabanı için)
//   terrain.groundSmooth(X, Z)    ±500 ft ağırlıklı ortalama (irtifada yumuşak yerleştirme)
//   terrain.normalAt(X, Z, out)   yüzey normali
//   terrain.stats()               üçgen ve örnek sayıları

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.169.0/+esm";

const CELL = 200;                  // ft, kafes aralığı (çekirdek bölge)
const CORE = 12000;                // ft, kafesin kesin olduğu çekirdek yarı genişlik
const EXTENT = 60000;              // ft, arazinin yarı genişliği (sayfanın sisi 42 000 ft'te kapatır)
const GROWTH = 1.16;               // çekirdeğin dışında hücre büyüme oranı
const SNAP = 4000;                 // ft, pencere merkezi bu katlara oturur
const RELIEF = 1800;               // ft, sırt yüksekliği ölçeği
const PAD_FLAT = 1500, PAD_BLEND = 5500;
const MACRO = 9000;                // ft, köy hücresi
const VILLAGE_R = 30000;           // ft, köylerin kurulduğu yarıçap
const MAX_VIL = 24;                // shader'a giden köy sayısı
const TREE_CELL = 50, TREE_R = 4500, TREE_SNAP = 1000;
const ROAD_STEP = 2;               // yol araması: kafes hücresi katı (400 ft)
const OFF = 1048576;               // hash girdileri pozitif kalsın

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------------
// gürültü: tam sayı hash (GLSL'deki eşiyle bit bit aynı), gradyan ve değer gürültüsü
// ---------------------------------------------------------------------
function ihashU(i, j, s) {
  let h = (Math.imul(i + OFF, 0x27d4eb2d) ^ Math.imul(j + OFF, 0x165667b1) ^ Math.imul(s, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
const ihash = (i, j, s) => ihashU(i, j, s) / 4294967296;
const GX = new Float64Array(16), GY = new Float64Array(16);
for (let k = 0; k < 16; k++) { GX[k] = Math.cos(k * Math.PI / 8); GY[k] = Math.sin(k * Math.PI / 8); }
const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
function gnoise(x, y, s) {                       // ≈ −1…1
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const k00 = ihashU(xi, yi, s) & 15, k10 = ihashU(xi + 1, yi, s) & 15, k01 = ihashU(xi, yi + 1, s) & 15, k11 = ihashU(xi + 1, yi + 1, s) & 15;
  const n00 = GX[k00] * xf + GY[k00] * yf, n10 = GX[k10] * (xf - 1) + GY[k10] * yf;
  const n01 = GX[k01] * xf + GY[k01] * (yf - 1), n11 = GX[k11] * (xf - 1) + GY[k11] * (yf - 1);
  const u = fade(xf), v = fade(yf);
  return 1.4 * (n00 + (n10 - n00) * u + (n01 - n00) * v + (n00 - n10 - n01 + n11) * u * v);
}
function fbm(x, y, oct, s) {
  let a = 0, amp = 1, f = 1, n = 0;
  for (let k = 0; k < oct; k++) { a += amp * gnoise(x * f, y * f, s + k); n += amp; amp *= 0.5; f *= 2.07; }
  return a / n;
}
function vn(x, y, s) {                           // 0…1, shader'daki vn ile aynı
  const xi = Math.floor(x), yi = Math.floor(y), u = x - xi, v = y - yi, uu = u * u * (3 - 2 * u), vv = v * v * (3 - 2 * v);
  const a = ihash(xi, yi, s), b = ihash(xi + 1, yi, s), c = ihash(xi, yi + 1, s), d = ihash(xi + 1, yi + 1, s);
  return (a + (b - a) * uu) + ((c + (d - c) * uu) - (a + (b - a) * uu)) * vv;
}

// arazi yüksekliği (ft): bükülmüş alanda sırtlı (ridged) gürültü; düşük değerler sıkıştırılır → geniş vadi tabanları
export function terrainHeight(X, Z) {
  const x = X * 1e-3, z = Z * 1e-3;                                        // kft
  const qx = x + 2.4 * fbm(x * 0.08, z * 0.08, 2, 101), qz = z + 2.4 * fbm(x * 0.08 + 7.3, z * 0.08 - 3.1, 2, 103);
  let r = 0, amp = 1, f = 0.075, w = 1, n = 0;
  for (let k = 0; k < 4; k++) {
    let v = 1 - Math.abs(gnoise(qx * f, qz * f, 201 + k)); v *= v; v *= w; w = Math.min(1, v * 1.8);
    r += v * amp; n += amp; amp *= 0.45; f *= 2.15;
  }
  r /= n;
  const zone = 0.5 + 0.5 * fbm(x * 0.03, z * 0.03, 2, 301);                // geniş ölçek: dağlık / ova
  const h = RELIEF * Math.pow(r, 1.7) * (0.45 + 0.85 * zone) + 60 * r * fbm(x * 0.6, z * 0.6, 2, 401);
  return Math.max(0, h) * smooth(PAD_FLAT, PAD_BLEND, Math.hypot(X, Z));
}

// ---------------------------------------------------------------------
// zemin shader'ı (MeshLambertMaterial'e eklenir): renk, kaya, vadi yeşili, köy tarlaları, meşe benekleri
// ---------------------------------------------------------------------
const TERRAIN_GLSL = `
varying vec3 vTerrPos;
varying vec3 vTerrN;
uniform vec4 uVil[${MAX_VIL}];
uniform int uVilN;
uniform vec3 uGrassA, uGrassB, uSoil, uRock, uGreen, uTree, uFieldG, uFieldS, uFieldP, uVilGround;
float ihash(ivec2 c, int s) {
  uint h = (uint(c.x + ${OFF}) * 0x27d4eb2du) ^ (uint(c.y + ${OFF}) * 0x165667b1u) ^ (uint(s) * 0x9e3779b1u);
  h = (h ^ (h >> 15u)) * 0x85ebca6bu;
  h = (h ^ (h >> 13u)) * 0xc2b2ae35u;
  h ^= h >> 16u;
  return float(h) / 4294967296.0;
}
float vn(vec2 p, int s) {
  vec2 i = floor(p), f = p - i, u = f * f * (3.0 - 2.0 * f); ivec2 c = ivec2(i);
  float a = ihash(c, s), b = ihash(c + ivec2(1, 0), s), cc = ihash(c + ivec2(0, 1), s), d = ihash(c + ivec2(1, 1), s);
  return mix(mix(a, b, u.x), mix(cc, d, u.x), u.y);
}
float villageMask(vec2 p, out float core) {
  float fld = 0.0; core = 0.0;
  for (int k = 0; k < ${MAX_VIL}; k++) {
    if (k >= uVilN) break;
    vec4 v = uVil[k]; float d = length(p - v.xy);
    fld = max(fld, 1.0 - smoothstep(v.z * 1.0, v.z * 2.2, d));
    core = max(core, 1.0 - smoothstep(v.z * 0.6, v.z * 1.05, d));
  }
  return fld;
}
float treeDensity(vec3 P, vec3 N, float vil) {
  float f = 0.65 * vn(P.xz / 3000.0, 601) + 0.35 * vn(P.xz / 800.0, 602);
  float d = 0.08 + 0.62 * smoothstep(0.32, 0.72, f) + 0.22 * clamp(-N.z * 2.5, 0.0, 1.0);
  d -= 0.8 * max(0.0, (1.0 - N.y) - 0.2);
  d *= 0.35 + 0.65 * smoothstep(20.0, 160.0, P.y);
  d *= smoothstep(500.0, 900.0, length(P.xz));
  d *= 1.0 - 0.9 * vil;
  return clamp(d, 0.0, 0.8);
}
vec3 terrainColor(vec3 P) {
  vec3 N = normalize(vTerrN);                 // renk için yumuşak normal (ışık yine yüzey yüzey, low-poly)
  float slope = 1.0 - N.y;
  float g1 = vn(P.xz / 1400.0, 801), g2 = vn(P.xz / 330.0, 802), g3 = vn(P.xz / 90.0, 803);
  vec3 col = mix(uGrassA, uGrassB, smoothstep(0.3, 0.7, 0.6 * g1 + 0.4 * g2));
  col = mix(col, uSoil, 0.6 * smoothstep(0.6, 0.8, 0.5 * g2 + 0.5 * g3));
  col = mix(col, uRock, smoothstep(0.16, 0.3, slope + 0.12 * (g3 - 0.5)));
  col = mix(col, uGreen, 0.5 * (1.0 - smoothstep(25.0, 140.0, P.y)) * smoothstep(0.35, 0.65, g1));
  float core, vil = villageMask(P.xz, core);
  float plot = vn(P.xz / 140.0, 811), band = fract(P.y / 12.0);
  vec3 fcol = plot < 0.36 ? uFieldG : (plot < 0.68 ? uFieldS : uFieldP);
  fcol *= 0.86 + 0.14 * smoothstep(0.0, 0.1, band) * (1.0 - smoothstep(0.82, 0.94, band));   // teras basamakları
  col = mix(col, fcol, 0.85 * vil * (1.0 - smoothstep(0.12, 0.24, slope)));
  col = mix(col, uVilGround, 0.55 * core);
  // meşe benekleri: 50 ft hücrede en çok bir ağaç; uzakta ortalama örtüye karışır (titreşmesin)
  vec2 cp = P.xz / ${TREE_CELL}.0; ivec2 ci = ivec2(floor(cp)); vec2 cf = cp - floor(cp);
  float dens = treeDensity(P, N, max(vil, core));
  vec2 c = vec2(0.3) + 0.4 * vec2(ihash(ci, 702), ihash(ci, 703));
  float rr = 0.18 + 0.12 * ihash(ci, 704);
  float dotv = (1.0 - smoothstep(rr * 0.75, rr, length(cf - c))) * step(ihash(ci, 701), dens);
  float t = mix(dotv, dens * 0.18, smoothstep(0.08, 0.45, length(fwidth(cp))));
  return mix(col, uTree, 0.92 * t);
}`;

function terrainMaterial(uniforms) {
  const m = new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true });
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 vTerrPos;\nvarying vec3 vTerrN;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvTerrPos = position; vTerrN = normal;");
    sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\n" + TERRAIN_GLSL)
      .replace("vec4 diffuseColor = vec4( diffuse, opacity );", "vec4 diffuseColor = vec4( diffuse * terrainColor( vTerrPos ), opacity );");
  };
  m.customProgramCacheKey = () => "ah1s-terrain";
  return m;
}

// çekirdekte CELL aralıklı, dışarıda büyüyen eksen ofsetleri (pencere merkezine göre)
function axisOffsets() {
  const core = []; for (let v = -CORE; v <= CORE; v += CELL) core.push(v);
  const out = []; let step = CELL, v = CORE;
  while (v < EXTENT) { step *= GROWTH; v = Math.min(EXTENT, v + step); out.push(v); }
  return [...out.map(o => -o).reverse(), ...core, ...out];
}

// ikili yığın (A*)
class Heap {
  constructor() { this.k = []; this.v = []; }
  push(key, val) { const k = this.k, v = this.v; let i = k.length; k.push(key); v.push(val);
    while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= key) break; k[i] = k[p]; v[i] = v[p]; i = p; } k[i] = key; v[i] = val; }
  pop() { const k = this.k, v = this.v, top = v[0], lk = k.pop(), lv = v.pop(); const n = k.length;
    if (n) { let i = 0; for (;;) { let c = 2 * i + 1; if (c >= n) break; if (c + 1 < n && k[c + 1] < k[c]) c++; if (k[c] >= lk) break; k[i] = k[c]; v[i] = v[c]; i = c; }
      k[i] = lk; v[i] = lv; }
    return top; }
  get size() { return this.k.length; }
}

const HOUSE_COLORS = [0xe4ded1, 0xd8cfbd, 0xcfc4ad, 0xbdb5a5, 0xe9e5dc, 0xa99f8e, 0xc9bda4].map(c => new THREE.Color(c));
const ROOF_COLORS = [0x5b84ad, 0x8e4a3c, 0x7b8a8f].map(c => new THREE.Color(c));
const OAK_COLORS = [0x3d4a2a, 0x46532f, 0x364126, 0x4b5631].map(c => new THREE.Color(c));
const GARDEN_COLORS = [0x5f7a35, 0x6c873c, 0x55702f].map(c => new THREE.Color(c));

export class Terrain {
  constructor() {
    this.group = new THREE.Group(); this.group.name = "terrain";
    this.cache = new Map();                      // kafes yükseklikleri
    this.villages = new Map();                   // makro hücre → köy (ya da null)
    this.layouts = new Map();                    // köy → evler / bahçeler
    this.roads = new Map();                      // köy çifti → şerit üçgenleri
    this.axis = axisOffsets();
    this.uniforms = {
      uVil: { value: Array.from({ length: MAX_VIL }, () => new THREE.Vector4(1e9, 1e9, 1, 0)) }, uVilN: { value: 0 },
      uGrassA: { value: new THREE.Color(0xb39a6c) }, uGrassB: { value: new THREE.Color(0xc5ad7f) }, uSoil: { value: new THREE.Color(0xd2bf98) },
      uRock: { value: new THREE.Color(0x968b78) }, uGreen: { value: new THREE.Color(0x7c8048) }, uTree: { value: new THREE.Color(0x353f24) },
      uFieldG: { value: new THREE.Color(0x7f8c48) }, uFieldS: { value: new THREE.Color(0xcdbd8c) }, uFieldP: { value: new THREE.Color(0xa88b63) },
      uVilGround: { value: new THREE.Color(0xc9b791) },
    };
    this.mat = {
      ground: terrainMaterial(this.uniforms),
      road: new THREE.MeshLambertMaterial({ color: 0xd8c7a0, flatShading: true, side: THREE.DoubleSide }),
      inst: new THREE.MeshLambertMaterial({ color: 0xffffff, flatShading: true }),
    };
    this.ground = new THREE.Mesh(new THREE.BufferGeometry(), this.mat.ground); this.ground.name = "terrain_ground";
    this.ground.frustumCulled = false; this.group.add(this.ground);
    this.roadMesh = new THREE.Mesh(new THREE.BufferGeometry(), this.mat.road); this.roadMesh.name = "terrain_roads"; this.roadMesh.frustumCulled = false;
    this.group.add(this.roadMesh);
    const ico = new THREE.IcosahedronGeometry(1, 0), box = new THREE.BoxGeometry(1, 1, 1);
    const cone = new THREE.ConeGeometry(1, 1, 6); cone.translate(0, 0.5, 0);
    const shaft = new THREE.CylinderGeometry(1, 1, 1, 8); shaft.translate(0, 0.5, 0);
    const dome = new THREE.SphereGeometry(1, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2);
    this.inst = {};
    for (const [name, geo] of [["oaks", ico], ["houses", box], ["roofs", box], ["poplars", cone], ["gardens", ico], ["domes", dome], ["minarets", shaft], ["caps", cone]]) {
      const m = new THREE.InstancedMesh(geo, this.mat.inst, 1); m.count = 0; m.name = `terrain_${name}`; m.frustumCulled = false;
      this.inst[name] = m; this.group.add(m);
    }
    this.oakCache = new Map();                   // 1000 ft blok → ağaçlar
    this.center = null; this.treeCenter = null; this.activeVillages = []; this.jobs = [];
    this._n = new THREE.Vector3();
    this.update(0, 0);
    while (this.jobs.length) this.jobs.shift().call(this);   // ilk kurulum tek seferde
  }

  // ---------- yükseklik sorguları ----------
  _lat(i, j) {
    const k = i * 4194304 + j; let h = this.cache.get(k);
    if (h === undefined) { if (this.cache.size > 600000) this.cache.clear(); h = terrainHeight(i * CELL, j * CELL); this.cache.set(k, h); }
    return h;
  }
  groundAt(X, Z) {
    const gx = X / CELL, gz = Z / CELL, i = Math.floor(gx), j = Math.floor(gz), fx = gx - i, fz = gz - j;
    const h00 = this._lat(i, j), h11 = this._lat(i + 1, j + 1);
    if (fx >= fz) { const h10 = this._lat(i + 1, j); return h00 + (h10 - h00) * fx + (h11 - h10) * fz; }
    const h01 = this._lat(i, j + 1); return h00 + (h11 - h01) * fx + (h01 - h00) * fz;
  }
  normalAt(X, Z, out = new THREE.Vector3()) {
    const gx = X / CELL, gz = Z / CELL, i = Math.floor(gx), j = Math.floor(gz), fx = gx - i, fz = gz - j;
    const h00 = this._lat(i, j), h11 = this._lat(i + 1, j + 1);
    let dx, dz;
    if (fx >= fz) { const h10 = this._lat(i + 1, j); dx = h10 - h00; dz = h11 - h10; }
    else { const h01 = this._lat(i, j + 1); dx = h11 - h01; dz = h01 - h00; }
    return out.set(-dx / CELL, 1, -dz / CELL).normalize();
  }
  surfaceAt(X, Z) {
    const M = this.mesh;
    if (!M || (Math.abs(X - M.cx) <= CORE && Math.abs(Z - M.cz) <= CORE)) return this.groundAt(X, Z);
    const ax = this.axis, n = ax.length, find = v => { let lo = 0, hi = n - 2; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ax[m] <= v) lo = m; else hi = m - 1; } return lo; };
    const u = clamp(X - M.cx, ax[0], ax[n - 1]), w = clamp(Z - M.cz, ax[0], ax[n - 1]), i = find(u), j = find(w);
    const fx = (u - ax[i]) / (ax[i + 1] - ax[i]), fz = (w - ax[j]) / (ax[j + 1] - ax[j]), P = M.pos, y = (a, b) => P[(b * n + a) * 3 + 1];
    const h00 = y(i, j), h11 = y(i + 1, j + 1);
    if (fx >= fz) { const h10 = y(i + 1, j); return h00 + (h10 - h00) * fx + (h11 - h10) * fz; }
    const h01 = y(i, j + 1); return h00 + (h11 - h01) * fx + (h01 - h00) * fz;
  }
  _delta(X, Z) { const M = this.mesh; return !M || (Math.abs(X - M.cx) <= CORE && Math.abs(Z - M.cz) <= CORE) ? 0 : this.surfaceAt(X, Z) - this.groundAt(X, Z); }
  groundSmooth(X, Z) {
    const d = 500; let s = 0;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) s += (a ? 1 : 2) * (b ? 1 : 2) * this.groundAt(X + a * d, Z + b * d);
    return s / 16;
  }

  // ---------- pencere ----------
  // Yeniden kurulum işleri sıraya girer, her çağrıda en çok biri çalışır (takılma yerine birkaç karede biter).
  // Eski pencere bu sırada geçerli kalır: helikopter çekirdekten çıkmadan (±12 000 ft) yeni pencere hazır olur.
  update(X, Z, cx = X, cz = Z) {
    const queue = f => { if (!this.jobs.includes(f)) this.jobs.push(f); };
    if (!this.center || Math.abs(X - this.center.x) > SNAP || Math.abs(Z - this.center.z) > SNAP) {
      this.center = { x: Math.round(X / SNAP) * SNAP, z: Math.round(Z / SNAP) * SNAP };
      queue(this._groundRows); queue(this._buildGround); queue(this._buildVillages); queue(this._buildRoads);
    }
    if (!this.treeCenter || Math.abs(cx - this.treeCenter.x) > TREE_SNAP || Math.abs(cz - this.treeCenter.z) > TREE_SNAP) {
      this.treeCenter = { x: Math.round(cx / TREE_SNAP) * TREE_SNAP, z: Math.round(cz / TREE_SNAP) * TREE_SNAP };
      queue(this._buildOaks);
    }
    if (this.jobs.length) this.jobs.shift().call(this);
  }
  // zemin iki karede: önce satırların yarısı, sonra kalanı + değiştirme
  _groundRows(from = 0, to = this.axis.length >> 1) {
    const ax = this.axis, n = ax.length, { x: cx, z: cz } = this.center;
    if (from === 0 || !this._pending || this._pending.cx !== cx || this._pending.cz !== cz) this._pending = { cx, cz, pos: new Float32Array(n * n * 3) };
    const pos = this._pending.pos;
    for (let j = from; j < to; j++) for (let i = 0; i < n; i++) {
      const X = cx + ax[i], Z = cz + ax[j], k = (j * n + i) * 3;
      const core = Math.abs(ax[i]) <= CORE && Math.abs(ax[j]) <= CORE;
      pos[k] = X; pos[k + 1] = core ? this._lat(Math.round(X / CELL), Math.round(Z / CELL)) : terrainHeight(X, Z); pos[k + 2] = Z;
    }
  }
  _buildGround() {
    const n = this.axis.length;
    this._groundRows(this._pending && this._pending.cx === this.center.x && this._pending.cz === this.center.z ? n >> 1 : 0, n);
    const pos = this._pending.pos; this.mesh = { cx: this._pending.cx, cz: this._pending.cz, pos }; this._pending = null;
    const idx = new Uint32Array((n - 1) * (n - 1) * 6); let o = 0;
    for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {
      const v00 = j * n + i, v10 = v00 + 1, v01 = v00 + n, v11 = v01 + 1;
      idx[o++] = v00; idx[o++] = v11; idx[o++] = v10;                 // groundAt ile aynı köşegen
      idx[o++] = v00; idx[o++] = v01; idx[o++] = v11;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();                                       // renk mantığı için (yoğunluk, kaya, yön)
    this.ground.geometry.dispose(); this.ground.geometry = g;
    if (!this.jobs.includes(this._buildOaks)) this.jobs.push(this._buildOaks);   // uzak ağaçlar yeni yüzeye otursun
  }

  // ---------- köyler ----------
  smoothNormalAt(X, Z, out = new THREE.Vector3()) {           // köşe normallerine yakın (merkezi fark)
    const d = CELL;
    return out.set(-(this.groundAt(X + d, Z) - this.groundAt(X - d, Z)) / (2 * d), 1, -(this.groundAt(X, Z + d) - this.groundAt(X, Z - d)) / (2 * d)).normalize();
  }
  _slopeAt(X, Z) { const n = this.normalAt(X, Z, this._n); return Math.sqrt(1 - n.y * n.y) / n.y; }
  _villageAt(ci, cj) {
    const key = `${ci},${cj}`;
    if (this.villages.has(key)) return this.villages.get(key);
    let v = null;
    if (ihash(ci, cj, 501) < 0.55) {
      let best = null;
      for (let k = 0; k < 8; k++) {
        const X = (ci + 0.12 + 0.76 * ihash(ci, cj, 510 + k)) * MACRO, Z = (cj + 0.12 + 0.76 * ihash(ci, cj, 520 + k)) * MACRO;
        if (Math.hypot(X, Z) < 4500) continue;
        const h = this.groundAt(X, Z), sl = this._slopeAt(X, Z), score = h + 4000 * Math.max(0, sl - 0.06);
        if (!best || score < best.score) best = { X, Z, sl, score };
      }
      if (best && best.sl < 0.22) {
        const town = ihash(ci, cj, 530) > 0.86;
        const n = town ? 90 + Math.floor(70 * ihash(ci, cj, 531)) : 14 + Math.floor(36 * ihash(ci, cj, 532));
        v = { key, X: best.X, Z: best.Z, n, r: 42 * Math.sqrt(n), town, mosque: town || (n > 30 && ihash(ci, cj, 533) < 0.6), s: (ci * 7919 + cj * 104729) | 0 };
      }
    }
    this.villages.set(key, v);
    return v;
  }
  _villageMask(X, Z, list = this.activeVillages) {
    let m = 0;
    for (const v of list) {
      const d = Math.hypot(X - v.X, Z - v.Z);
      m = Math.max(m, 1 - smooth(v.r, v.r * 2.2, d), 1 - smooth(v.r * 0.6, v.r * 1.05, d));
    }
    return m;
  }
  _layout(v) {
    if (this.layouts.has(v.key)) return this.layouts.get(v.key);
    const R = (k, t) => ihash(v.s, k, t), houses = [], gardens = [], n = this._n;
    for (let t = 0; houses.length < v.n && t < v.n * 8; t++) {
      const a = 2 * Math.PI * R(t, 541), rr = v.r * Math.pow(R(t, 542), 0.75);
      const x = v.X + Math.cos(a) * rr, z = v.Z + Math.sin(a) * rr;
      this.normalAt(x, z, n); if (n.y < 0.93) continue;                       // ~21°'den dik yamaçta ev yok
      const two = R(t, 545) < 0.25, w = 26 + 18 * R(t, 543), d = 22 + 12 * R(t, 544), ht = two ? 20 : 11;
      if (houses.some(p => Math.hypot(p.x - x, p.z - z) < (Math.max(w, d) + Math.max(p.w, p.d)) * 0.6)) continue;
      const rot = Math.atan2(n.x, n.z) + (R(t, 546) < 0.5 ? 0 : Math.PI / 2) + 0.25 * (R(t, 547) - 0.5);
      const c = Math.cos(rot), s = Math.sin(rot); let lo = Infinity, hi = -Infinity;
      for (const [u, q] of [[-w / 2, -d / 2], [w / 2, -d / 2], [-w / 2, d / 2], [w / 2, d / 2]]) {
        const g = this.groundAt(x + c * u + s * q, z - s * u + c * q); lo = Math.min(lo, g); hi = Math.max(hi, g);
      }
      houses.push({ x, z, w, d, rot, yb: lo - 4, yt: hi + ht, color: HOUSE_COLORS[Math.floor(R(t, 548) * HOUSE_COLORS.length)],
        roof: R(t, 549) < 0.16 ? ROOF_COLORS[Math.floor(R(t, 550) * ROOF_COLORS.length)] : null });
    }
    let mosque = null;
    if (v.mosque && houses.length > 4) {                                     // merkeze en yakın ev → cami (kubbe + minare)
      let best = 0; houses.forEach((h, k) => { if (Math.hypot(h.x - v.X, h.z - v.Z) < Math.hypot(houses[best].x - v.X, houses[best].z - v.Z)) best = k; });
      const h = houses[best]; h.w = 52; h.d = 52; h.color = HOUSE_COLORS[4]; h.roof = null;
      const g = this.groundAt(h.x, h.z); h.yb = g - 6; h.yt = g + 24;
      const mx = h.x + Math.cos(h.rot) * 34 + Math.sin(h.rot) * 34, mz = h.z - Math.sin(h.rot) * 34 + Math.cos(h.rot) * 34;
      mosque = { x: h.x, z: h.z, top: h.yt, mx, mz, mg: this.groundAt(mx, mz) };
    }
    for (let t = 0; gardens.length < v.n * 1.3 && t < v.n * 6; t++) {      // kavaklar ve meyve bahçeleri
      const a = 2 * Math.PI * R(t, 561), rr = v.r * (0.5 + 1.1 * R(t, 562));
      const x = v.X + Math.cos(a) * rr, z = v.Z + Math.sin(a) * rr;
      if (this.normalAt(x, z, n).y < 0.9 || houses.some(p => Math.hypot(p.x - x, p.z - z) < 34)) continue;
      const poplar = R(t, 563) < 0.45, g = this.groundAt(x, z);
      gardens.push({ x, z, g, poplar, r: poplar ? 6 + 3 * R(t, 564) : 11 + 6 * R(t, 564), h: poplar ? 42 + 26 * R(t, 565) : 0,
        color: GARDEN_COLORS[Math.floor(R(t, 566) * GARDEN_COLORS.length)] });
    }
    const L = { houses, mosque, gardens };
    this.layouts.set(v.key, L);
    return L;
  }
  _buildVillages() {
    const { x: cx, z: cz } = this.center, list = [];
    const c0 = Math.floor((cx - VILLAGE_R) / MACRO), c1 = Math.floor((cx + VILLAGE_R) / MACRO);
    const r0 = Math.floor((cz - VILLAGE_R) / MACRO), r1 = Math.floor((cz + VILLAGE_R) / MACRO);
    for (let i = c0; i <= c1; i++) for (let j = r0; j <= r1; j++) {
      const v = this._villageAt(i, j); if (v && Math.hypot(v.X - cx, v.Z - cz) < VILLAGE_R) list.push(v);
    }
    list.sort((a, b) => Math.hypot(a.X - cx, a.Z - cz) - Math.hypot(b.X - cx, b.Z - cz));
    this.activeVillages = list.slice(0, MAX_VIL);
    const U = this.uniforms.uVil.value;
    U.forEach((u, k) => { const v = this.activeVillages[k]; if (v) u.set(v.X, v.Z, v.r, v.town ? 1 : 0); else u.set(1e9, 1e9, 1, 0); });
    this.uniforms.uVilN.value = this.activeVillages.length;
    // örnekler
    const H = [], RF = [], PO = [], GA = [], DO = [], MI = [], CA = [];
    for (const v of this.activeVillages) {
      const L = this._layout(v);
      for (const h of L.houses) {
        H.push([h.x, (h.yb + h.yt) / 2, h.z, h.w, h.yt - h.yb, h.d, h.rot, h.color]);
        if (h.roof) RF.push([h.x, h.yt + 0.6, h.z, h.w + 3, 1.2, h.d + 3, h.rot, h.roof]);
      }
      if (L.mosque) {
        const m = L.mosque;
        DO.push([m.x, m.top - 2, m.z, 22, 18, 22, 0, HOUSE_COLORS[4]]);
        MI.push([m.mx, m.mg - 4, m.mz, 4.5, 92, 4.5, 0, HOUSE_COLORS[4]]);
        CA.push([m.mx, m.mg + 88, m.mz, 5.2, 16, 5.2, 0, ROOF_COLORS[2]]);
      }
      for (const g of L.gardens) {
        if (g.poplar) PO.push([g.x, g.g - 2, g.z, g.r, g.h, g.r, 0, g.color]);
        else GA.push([g.x, g.g + g.r * 0.9, g.z, g.r, g.r * 0.85, g.r, 0, g.color]);
      }
    }
    this._fill("houses", H); this._fill("roofs", RF); this._fill("poplars", PO); this._fill("gardens", GA);
    this._fill("domes", DO); this._fill("minarets", MI); this._fill("caps", CA);
  }
  // örnek listesi: [x, y, z, sx, sy, sz, dönüş (Y), renk]; y kafes zeminine göre, çekirdek dışında çizilen yüzeye taşınır
  _fill(name, rows) {
    let m = this.inst[name];
    if (m.instanceMatrix.count < rows.length) {
      const cap = Math.max(16, Math.ceil(rows.length * 1.3)), nm = new THREE.InstancedMesh(m.geometry, m.material, cap);
      nm.name = m.name; nm.frustumCulled = false; this.group.remove(m); m.dispose(); this.group.add(nm); this.inst[name] = m = nm;
    }
    if (!m.instanceColor) m.setColorAt(0, HOUSE_COLORS[0]);
    const te = m.instanceMatrix.array, ce = m.instanceColor.array;
    for (let k = 0; k < rows.length; k++) {                        // T · R_y · S, sütun sıralı
      const [x, y0, z, sx, sy, sz, rot, color] = rows[k], y = y0 + this._delta(x, z), c = Math.cos(rot), sn = Math.sin(rot), o = k * 16;
      te[o] = c * sx; te[o + 1] = 0; te[o + 2] = -sn * sx; te[o + 3] = 0;
      te[o + 4] = 0; te[o + 5] = sy; te[o + 6] = 0; te[o + 7] = 0;
      te[o + 8] = sn * sz; te[o + 9] = 0; te[o + 10] = c * sz; te[o + 11] = 0;
      te[o + 12] = x; te[o + 13] = y; te[o + 14] = z; te[o + 15] = 1;
      ce[k * 3] = color.r; ce[k * 3 + 1] = color.g; ce[k * 3 + 2] = color.b;
    }
    m.count = rows.length; m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true;
  }

  // ---------- yollar: eğimden kaçınan A* (400 ft), yumuşatılmış, araziye oturtulmuş şerit ----------
  _buildRoads() {
    const vs = this.activeVillages, pairs = new Set();
    const ends = [{ key: "pad", X: 320, Z: 0 }, ...vs];
    for (const a of ends) {
      let best = null, bd = Infinity;
      for (const b of vs) { if (b === a) continue; const d = Math.hypot(a.X - b.X, a.Z - b.Z); if (d < bd) { bd = d; best = b; } }
      if (best && bd < 16000) pairs.add([a.key, best.key].sort().join("|"));
    }
    const byKey = new Map(ends.map(e => [e.key, e])), parts = [];
    for (const p of pairs) {
      if (!this.roads.has(p)) { const [a, b] = p.split("|").map(k => byKey.get(k)); this.roads.set(p, this._road(a, b)); }
      parts.push(this.roads.get(p));
    }
    let n = 0; for (const a of parts) n += a.length;
    const pos = new Float32Array(n); let o = 0; for (const a of parts) { pos.set(a, o); o += a.length; }
    for (let k = 0; k < n; k += 3) pos[k + 1] += this._delta(pos[k], pos[k + 2]);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.roadMesh.geometry.dispose(); this.roadMesh.geometry = g;
  }
  _road(a, b) {
    const S = CELL * ROAD_STEP, H = (i, j) => this._lat(i * ROAD_STEP, j * ROAD_STEP);
    const ai = Math.round(a.X / S), aj = Math.round(a.Z / S), bi = Math.round(b.X / S), bj = Math.round(b.Z / S);
    const pad = 12, i0 = Math.min(ai, bi) - pad, j0 = Math.min(aj, bj) - pad, W = Math.abs(ai - bi) + 2 * pad + 1, Hh = Math.abs(aj - bj) + 2 * pad + 1;
    const id = (i, j) => (j - j0) * W + (i - i0), cost = new Float64Array(W * Hh).fill(Infinity), from = new Int32Array(W * Hh).fill(-1);
    const heap = new Heap(), goal = id(bi, bj); cost[id(ai, aj)] = 0; heap.push(0, id(ai, aj));
    const D = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    while (heap.size) {
      const c = heap.pop(); if (c === goal) break;
      const ci = c % W + i0, cj = Math.floor(c / W) + j0, hc = H(ci, cj);
      for (const [di, dj] of D) {
        const ni = ci + di, nj = cj + dj; if (ni < i0 || nj < j0 || ni >= i0 + W || nj >= j0 + Hh) continue;
        const L = S * Math.hypot(di, dj), sl = Math.abs(H(ni, nj) - hc) / L;
        const nc = cost[c] + L * (1 + 30 * sl * sl) + (sl > 0.25 ? 25 * L : 0), k = id(ni, nj);
        if (nc < cost[k]) { cost[k] = nc; from[k] = c; heap.push(nc + Math.hypot(ni - bi, nj - bj) * S, k); }
      }
    }
    let pts = [];
    for (let c = goal; c >= 0; c = from[c]) pts.push([(c % W + i0) * S, (Math.floor(c / W) + j0) * S]);
    pts[0] = [b.X, b.Z]; pts[pts.length - 1] = [a.X, a.Z]; pts.reverse();
    for (let it = 0; it < 3; it++) {                                           // Chaikin yumuşatma
      const q = [pts[0]];
      for (let k = 0; k + 1 < pts.length; k++) { const [x0, z0] = pts[k], [x1, z1] = pts[k + 1];
        q.push([0.75 * x0 + 0.25 * x1, 0.75 * z0 + 0.25 * z1], [0.25 * x0 + 0.75 * x1, 0.25 * z0 + 0.75 * z1]); }
      q.push(pts.at(-1)); pts = q;
    }
    const res = [pts[0]];                                                      // 30 ft'te bir örnek
    for (let k = 1; k < pts.length; k++) {
      const [x0, z0] = res.at(-1), [x1, z1] = pts[k], d = Math.hypot(x1 - x0, z1 - z0);
      if (d < 30) continue; const m = Math.floor(d / 30);
      for (let s = 1; s <= m; s++) res.push([x0 + (x1 - x0) * s / m, z0 + (z1 - z0) * s / m]);
    }
    const half = 9, out = [];
    const edge = (k, side) => { const [x, z] = res[k], [xa, za] = res[Math.max(0, k - 1)], [xb, zb] = res[Math.min(res.length - 1, k + 1)];
      const tx = xb - xa, tz = zb - za, l = Math.hypot(tx, tz) || 1, ex = x - side * tz / l * half, ez = z + side * tx / l * half;
      return [ex, this.groundAt(ex, ez) + 2, ez]; };
    for (let k = 0; k + 1 < res.length; k++) {
      const l0 = edge(k, 1), r0 = edge(k, -1), l1 = edge(k + 1, 1), r1 = edge(k + 1, -1);
      out.push(...l0, ...r0, ...r1, ...l0, ...r1, ...l1);
    }
    return new Float32Array(out);
  }

  // ---------- yakın meşeler: shader beneklerinin üstünde 3D taç ----------
  _treeDensity(X, Z, h, n, vil) {
    const f = 0.65 * vn(X / 3000, Z / 3000, 601) + 0.35 * vn(X / 800, Z / 800, 602);
    let d = 0.08 + 0.62 * smooth(0.32, 0.72, f) + 0.22 * clamp(-n.z * 2.5, 0, 1);
    d -= 0.8 * Math.max(0, (1 - n.y) - 0.2);
    d *= 0.35 + 0.65 * smooth(20, 160, h);
    d *= smooth(500, 900, Math.hypot(X, Z));
    d *= 1 - 0.9 * vil;
    return clamp(d, 0, 0.8);
  }
  // 1000 ft'lik blok (20 × 20 hücre); sonuç deterministik (yakındaki köyler her zaman etkin listede) → önbellek
  _oakBlock(bx, bz) {
    const key = bx * 4194304 + bz; let rows = this.oakCache.get(key);
    if (rows) return rows;
    rows = [];
    const C = TREE_CELL, K = TREE_SNAP / C, n = this._n, mx = (bx + 0.5) * TREE_SNAP, mz = (bz + 0.5) * TREE_SNAP;
    const near = this.activeVillages.filter(v => Math.hypot(v.X - mx, v.Z - mz) < TREE_SNAP + v.r * 2.5);
    for (let a = bx * K; a < (bx + 1) * K; a++) for (let b = bz * K; b < (bz + 1) * K; b++) {
      const rnd = ihash(a, b, 701); if (rnd >= 0.8) continue;
      const X = (a + 0.5) * C, Z = (b + 0.5) * C, h = this.groundAt(X, Z); this.smoothNormalAt(X, Z, n);
      if (rnd >= this._treeDensity(X, Z, h, n, near.length ? this._villageMask(X, Z, near) : 0)) continue;
      const x = (a + 0.3 + 0.4 * ihash(a, b, 702)) * C, z = (b + 0.3 + 0.4 * ihash(a, b, 703)) * C, r = (0.18 + 0.12 * ihash(a, b, 704)) * C * 1.05;
      rows.push([x, this.groundAt(x, z) + r * 0.75 + 2, z, r, r * 0.8, r, ihash(a, b, 705) * 6.28, OAK_COLORS[ihashU(a, b, 706) & 3]]);
    }
    if (this.oakCache.size > 4000) this.oakCache.clear();
    this.oakCache.set(key, rows);
    return rows;
  }
  _buildOaks() {
    const { x: cx, z: cz } = this.treeCenter, B = TREE_SNAP, rows = [];
    for (let bx = Math.floor((cx - TREE_R) / B); bx <= Math.floor((cx + TREE_R) / B); bx++)
      for (let bz = Math.floor((cz - TREE_R) / B); bz <= Math.floor((cz + TREE_R) / B); bz++)
        if (Math.hypot((bx + 0.5) * B - cx, (bz + 0.5) * B - cz) < TREE_R) for (const r of this._oakBlock(bx, bz)) rows.push(r);
    this._fill("oaks", rows);
  }

  stats() {
    const tri = m => (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3;
    const out = { ground: tri(this.ground), roads: tri(this.roadMesh), villages: this.activeVillages.length };
    for (const [k, m] of Object.entries(this.inst)) out[k] = m.count;
    out.triangles = Math.round(out.ground + out.roads + Object.entries(this.inst).reduce((s, [, m]) => s + tri(m) * m.count, 0));
    return out;
  }
}
export const createTerrain = () => new Terrain();
