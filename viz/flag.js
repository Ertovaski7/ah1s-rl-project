// =====================================================================
// Pistin yanında Türk bayrağı: direk + dalgalanan bez (Three.js, birim ft)
// =====================================================================
// Bayrak ölçüleri Türk Bayrağı Kanunu'ndaki oranlarla (G = en): boy 1.5 G; dış hilal dairesinin merkezi gönderden G/2,
// çapı G/2; iç hilal dairesi merkezler arası G/16, çapı 0.4 G; yıldızı çevreleyen dairenin çapı G/4, iç hilal
// dairesinden uzaklığı G/3; yıldızın bir köşesi göndere bakar. Renk: al (#E30A17), beyaz.
//
// Bez kayıttaki rüzgârın estiği yöne dalgalanır (hız arttıkça yatay ve hızlı); rüzgâr yoksa direkte sarkar.
//
// Arayüz:
//   const flag = createPadFlag();  flag.group.position.set(x, zemin, z);  scene.add(flag.group);
//   flag.update(dt, windX, windZ)  dt: gerçek zaman (s); rüzgâr sahne çerçevesinde ft/s (X doğu, Z güney); NaN → sakin

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.169.0/+esm";

const RED = "#e30a17", WHITE = "#ffffff";

// bayrak dokusu: kanundaki oranlar, G = 60 000 birim (Vikipedi SVG'siyle aynı koordinatlar)
function flagTexture() {
  const W = 768, H = 512, k = H / 60000, cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), X = x => x * k, Y = y => (y + 30000) * k;
  g.fillStyle = RED; g.fillRect(0, 0, W, H);
  g.fillStyle = WHITE; g.beginPath(); g.arc(X(30000), Y(0), 15000 * k, 0, Math.PI * 2); g.fill();     // dış daire: merkez G/2, çap G/2
  g.fillStyle = RED; g.beginPath(); g.arc(X(33750), Y(0), 12000 * k, 0, Math.PI * 2); g.fill();       // iç daire: +G/16, çap 0.4 G
  const cx = 49250, R = 7500, r = R * Math.sin(Math.PI / 10) / Math.sin(7 * Math.PI / 10);           // yıldız: çevre çapı G/4
  g.fillStyle = WHITE; g.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = Math.PI + i * Math.PI / 5, rr = i % 2 ? r : R;                                           // ilk köşe göndere (sola) bakar
    g[i ? "lineTo" : "moveTo"](X(cx + rr * Math.cos(a)), Y(rr * Math.sin(a)));
  }
  g.closePath(); g.fill();
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

export class PadFlag {
  constructor({ poleHeight = 40, width = 8 } = {}) {
    this.G = width; this.L = 1.5 * width; this.top = poleHeight - 0.8;
    this.group = new THREE.Group(); this.group.name = "pad_flag";
    const metal = new THREE.MeshLambertMaterial({ color: 0xd9dcdc }), gold = new THREE.MeshLambertMaterial({ color: 0xc9a23a });
    const base = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 1.9, 1.2, 12), new THREE.MeshLambertMaterial({ color: 0xbdb7aa }));
    base.position.y = 0.4;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.24, poleHeight, 10), metal); pole.position.y = poleHeight / 2;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.45, 12, 8), gold); ball.position.y = poleHeight + 0.35;
    this.group.add(base, pole, ball);
    // bez: gönder kenarı direkte (x = 0), boy +x, en aşağı
    this.nx = 24; this.ny = 8;
    this.geo = new THREE.PlaneGeometry(this.L, this.G, this.nx, this.ny);
    this.cloth = new THREE.Mesh(this.geo, new THREE.MeshLambertMaterial({ map: flagTexture(), side: THREE.DoubleSide }));
    this.cloth.name = "pad_flag_cloth";
    this.pivot = new THREE.Group(); this.pivot.position.y = this.top; this.pivot.add(this.cloth); this.group.add(this.pivot);
    this.t = 0; this.wind = 0; this.yaw = 0;
    this.update(0, NaN, NaN);
  }
  update(dt, wx, wz) {
    const sp = Number.isFinite(wx) && Number.isFinite(wz) ? Math.hypot(wx, wz) : 0;
    const k = Math.min(1, dt * 1.5);
    this.wind += (Math.min(1, Math.max(0, (sp - 1) / 13)) - this.wind) * k;          // 1…14 ft/s (≈ 0.6…8 kt) → sarkık…yatay
    if (sp > 0.5) {                                                                   // bez rüzgârın estiği yöne döner
      const want = Math.atan2(-wz, wx); let d = want - this.yaw; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      this.yaw += d * k;
    }
    this.pivot.rotation.y = this.yaw;
    this.t += dt;
    const w = this.wind, droop = (1 - w) * 1.15 + 0.06, c = Math.cos(droop), s = Math.sin(droop);
    const amp = 0.3 + 1.1 * w, f = 0.5 + 1.3 * w, lam = 0.85 * this.L, P = this.geo.attributes.position, L = this.L, G = this.G;
    for (let j = 0; j <= this.ny; j++) for (let i = 0; i <= this.nx; i++) {
      const u = i / this.nx * L, v = j / this.ny * G, q = u / L;
      const z = amp * q * Math.sin(2 * Math.PI * (u / lam - f * this.t) + 0.5 * v / G) + 0.15 * amp * q * Math.sin(2 * Math.PI * (2.3 * u / lam - 1.7 * f * this.t));
      P.setXYZ(j * (this.nx + 1) + i, u * c, -v - u * s, z);
    }
    P.needsUpdate = true; this.geo.computeVertexNormals();
  }
}
export const createPadFlag = opts => new PadFlag(opts);
