"""Ekranlar: ana ekran (platformlar) → AH-1S (modüller) → Sürüş: başlangıç koşulları → simülasyon."""

from __future__ import annotations

import pygame

from . import ui
from .config import (AH1S_MODULES, COMING_SOON, LIMITS, LOCATIONS, PLATFORMS, SIM_GROUND_M, START_MODES, TURB_LEVELS,
                     StartConditions)
from .sim import SimProcess
from .turkey import ANATOLIA, CITIES, LAKES, THRACE, MapProjection

GAP = 10


class Scene:
    title = ""

    def __init__(self, app):
        self.app = app
        self.widgets: list[ui.Widget] = []

    def layout(self, size: tuple[int, int]):
        pass

    def handle(self, ev):
        for w in self.widgets:
            if w.handle(ev):
                break

    def update(self, dt: float):
        pass

    def draw(self, surf):
        surf.fill(ui.BG)
        for w in self.widgets:
            w.draw(surf)

    def back(self):
        self.app.pop()

    def on_exit(self):
        pass


def top_bar(surf, title: str, w: int, x: int = 124):
    pygame.draw.rect(surf, ui.SURFACE, (0, 0, w, 56))
    pygame.draw.line(surf, ui.LINE, (0, 56), (w, 56))
    ui.text(surf, title, (x, 28), 20, ui.TEXT, bold=True, anchor="midleft")


# ---------------------------------------------------------------------
# dikey panelli seçim ekranları
# ---------------------------------------------------------------------
class TileScene(Scene):
    tiles: tuple = ()
    with_back = False

    def __init__(self, app):
        super().__init__(app)
        self.tile_widgets = [ui.Tile(t, self.open, COMING_SOON) for t in self.tiles]
        self.back_btn = ui.Button("‹ Geri", self.back, kind="ghost") if self.with_back else None
        self.widgets = ([self.back_btn] if self.back_btn else []) + self.tile_widgets

    def layout(self, size):
        w, h = size
        n = len(self.tile_widgets)
        tw = (w - GAP * (n + 1)) / n
        for k, t in enumerate(self.tile_widgets):
            t.layout(pygame.Rect(int(GAP + k * (tw + GAP)), GAP, int(tw), h - 2 * GAP))
        if self.back_btn:
            self.back_btn.layout(pygame.Rect(GAP + 14, GAP + 14, 96, 38))

    def draw(self, surf):
        surf.fill(ui.BG)
        for t in self.tile_widgets:
            t.draw(surf)
        if self.back_btn:                                # panellerin üstünde
            self.back_btn.draw(surf)
            ui.text(surf, self.title, (self.back_btn.rect.right + 14, self.back_btn.rect.centery), 18, ui.TEXT, bold=True,
                    anchor="midleft")

    def open(self, key: str):
        pass


class HomeScene(TileScene):
    title = "Ana ekran"
    tiles = PLATFORMS

    def open(self, key):
        if key == "ah1s":
            self.app.push(AH1SMenuScene(self.app))

    def back(self):
        self.app.quit()


class AH1SMenuScene(TileScene):
    title = "AH-1S Cobra"
    tiles = AH1S_MODULES
    with_back = True

    def open(self, key):
        if key == "flight":
            self.app.push(FlightSetupScene(self.app))


# ---------------------------------------------------------------------
# Sürüş: başlangıç koşulları + Türkiye haritasında konum
# ---------------------------------------------------------------------
def _hdg(v: float) -> str:
    return f"{int(round(v)) % 360:03d}°"


def summary(s: StartConditions) -> str:
    where = {"ground": "yerde, rotor dönüyor", "hover": f"hover {s.alt_ft:.0f} ft",
             "cruise": f"ileri uçuş {s.speed_kt:.0f} kt, {s.alt_ft:.0f} ft"}[s.mode]
    wind = "rüzgâr yok" if s.wind_kt == 0 else f"rüzgâr {s.wind_kt:.0f} kt, {_hdg(s.wind_from_deg)}'den"
    turb = dict(TURB_LEVELS)[s.turb].lower()
    return (f"{s.location.name} · {where} · heading {_hdg(s.heading_deg)} · yakıt {s.fuel_total_lbs:.0f} lbs · "
            f"{wind} · türbülans {turb}{' + gust' if s.gusts else ''} · {s.temp_dc:+.0f} °C")


class FlightSetupScene(Scene):
    title = "AH-1S · Sürüş — Başlangıç koşulları"

    def __init__(self, app, sc: StartConditions | None = None):
        super().__init__(app)
        self.sc = sc or app.last_conditions or StartConditions()
        s = self.sc
        self.mode = ui.Segmented("Başlangıç durumu", START_MODES, s.mode, self._set("mode"))
        self.alt = ui.Slider("İrtifa (yerden, CG)", *LIMITS["alt_ft"], 10, s.alt_ft, lambda v: f"{v:.0f} ft", self._set("alt_ft"))
        self.speed = ui.Slider("Hava hızı", *LIMITS["speed_kt"], 5, s.speed_kt, lambda v: f"{v:.0f} kt", self._set("speed_kt"))
        self.heading = ui.Slider("Heading (burun yönü)", 0, 355, 5, s.heading_deg, _hdg, self._set("heading_deg"))
        self.fuel = ui.Slider("Yakıt (iki tanka eşit)", *LIMITS["fuel_total_lbs"], 20, s.fuel_total_lbs,
                              lambda v: f"{v:.0f} lbs", self._set("fuel_total_lbs"))
        self.wind = ui.Slider("Rüzgâr hızı", *LIMITS["wind_kt"], 1, s.wind_kt, lambda v: "sakin" if v == 0 else f"{v:.0f} kt",
                              self._set("wind_kt"))
        self.wind_dir = ui.Slider("Rüzgârın geldiği yön", 0, 350, 10, s.wind_from_deg, _hdg, self._set("wind_from_deg"))
        self.turb = ui.Segmented("Türbülans", TURB_LEVELS, s.turb, self._set("turb"))
        self.gusts = ui.Toggle("Gust (ani rüzgâr, 5–12 kt)", s.gusts, self._set("gusts"))
        self.temp = ui.Slider("Sıcaklık farkı", *LIMITS["temp_dc"], 1, s.temp_dc,
                              lambda v: f"{v:+.0f} °C (yerde {15 - 1.98 * SIM_GROUND_M * 3.28084 / 1000 + v:.0f} °C)",
                              self._set("temp_dc"))
        self.start_btn = ui.Button("Simülasyonu başlat", self.start)
        self.back_btn = ui.Button("‹ Geri", self.back, kind="ghost")
        self.col_a = [self.mode, self.alt, self.speed, self.heading, self.fuel]
        self.col_b = [self.wind, self.wind_dir, self.turb, self.gusts, self.temp]
        self.widgets = [self.back_btn, *self.col_a, *self.col_b, self.start_btn]
        self._sync()

    def _set(self, name):
        def f(v):
            setattr(self.sc, name, v)
            self._sync()
        return f

    def _sync(self):
        self.alt.enabled = self.sc.mode in ("hover", "cruise")
        self.speed.enabled = self.sc.mode == "cruise"
        self.wind_dir.enabled = self.sc.wind_kt > 0

    def layout(self, size):
        w, h = size
        self.back_btn.layout(pygame.Rect(GAP + 6, 9, 96, 38))
        left_w = int(w * 0.42)
        self.map_rect = pygame.Rect(GAP + 6, 56 + GAP + 6, left_w - 2 * GAP, int((h - 56) * 0.56))
        self.loc_rect = pygame.Rect(self.map_rect.left, self.map_rect.bottom + GAP, self.map_rect.width,
                                    h - self.map_rect.bottom - 2 * GAP - 6)
        fx = left_w + GAP
        fw = w - fx - GAP - 6
        colw = (fw - GAP) // 2
        for col, x in ((self.col_a, fx), (self.col_b, fx + colw + GAP)):
            y = 56 + GAP + 6
            for wd in col:
                wd.layout(pygame.Rect(x, y, colw, wd.HEIGHT))
                y += wd.HEIGHT + 8
        self.form_bottom = max(wd.rect.bottom for wd in self.col_a + self.col_b)
        self.start_btn.layout(pygame.Rect(w - GAP - 6 - 260, h - GAP - 6 - 48, 260, 48))
        self.info_rect = pygame.Rect(fx, self.form_bottom + 12, fw, self.start_btn.rect.top - self.form_bottom - 24)
        self.proj = MapProjection(self.map_rect, pad=18)

    def start(self):
        self.app.last_conditions = self.sc.copy()
        self.app.push(SimRunScene(self.app, self.sc.copy()))

    def draw(self, surf):
        w, h = surf.get_size()
        surf.fill(ui.BG)
        top_bar(surf, self.title, w)
        self._draw_map(surf)
        self._draw_location(surf)
        # bilgi: özet + eğitim dışı uyarılar
        r = self.info_rect
        y = ui.text(surf, summary(self.sc), (r.left + 8, r.top), 16, ui.TEXT, maxw=r.width - 16).bottom + 10
        for msg in self.sc.warnings():
            y = ui.text(surf, f"• {msg}", (r.left + 8, y), 15, ui.WARN, maxw=r.width - 16).bottom + 4
        if not self.sc.warnings():
            ui.text(surf, "Seçimler ajanın eğitildiği aralıkta.", (r.left + 8, y), 15, ui.OK)
        ui.text(surf, "3D görünüm ve komutlar tarayıcıda açılır.", (self.start_btn.rect.left - 16, self.start_btn.rect.centery),
                15, ui.MUTED, anchor="midright")
        for wd in self.widgets:
            wd.draw(surf)

    def _draw_map(self, surf):
        r, P = self.map_rect, self.proj
        pygame.draw.rect(surf, (24, 40, 52), r, border_radius=10)
        for poly in (ANATOLIA, THRACE):
            pts = [P(*p) for p in poly]
            pygame.draw.polygon(surf, (58, 72, 56), pts)
            pygame.draw.polygon(surf, (120, 132, 112), pts, 1)
        for lake in LAKES:
            pygame.draw.polygon(surf, (24, 40, 52), [P(*p) for p in lake])
        for name, lon, lat in CITIES:
            x, y = P(lon, lat)
            pygame.draw.circle(surf, ui.TEXT_2, (int(x), int(y)), 3)
            ui.text(surf, name, (x + 6, y), 13, ui.TEXT_2, anchor="midleft")
        loc = self.sc.location
        x, y = P(loc.lon, loc.lat)
        pygame.draw.circle(surf, ui.ERR, (int(x), int(y)), 14, 2)
        pygame.draw.circle(surf, ui.ERR, (int(x), int(y)), 5)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(surf, ui.ERR, (x + dx * 16, y + dy * 16), (x + dx * 24, y + dy * 24), 2)
        ui.text(surf, loc.name, (x, y - 28), 16, ui.TEXT, bold=True, anchor="midbottom")
        ui.text(surf, "Türkiye", (r.left + 14, r.top + 10), 15, ui.TEXT_2)

    def _draw_location(self, surf):
        r, loc = self.loc_rect, self.sc.location
        pygame.draw.rect(surf, ui.SURFACE, r, border_radius=10)
        x, y = r.left + 16, r.top + 12
        ui.text(surf, "BAŞLANGIÇ KONUMU" + (" (önceden seçili)" if len(LOCATIONS) == 1 else ""), (x, y), 13, ui.MUTED, bold=True)
        y = ui.text(surf, f"{loc.name} · {loc.region}", (x, y + 22), 20, ui.TEXT, bold=True).bottom + 6
        ns, ew = ("K" if loc.lat >= 0 else "G"), ("D" if loc.lon >= 0 else "B")
        y = ui.text(surf, f"{abs(loc.lat):.3f}° {ns}, {abs(loc.lon):.3f}° {ew} · yükseklik {loc.elev_m:.0f} m "
                          f"(simülasyon zemini {SIM_GROUND_M:.0f} m)", (x, y), 15, ui.TEXT_2, maxw=r.width - 32).bottom + 6
        ui.text(surf, loc.note + " 3D arazi bölgenin görünümünde, gerçek topografya değil.", (x, y), 14, ui.MUTED,
                maxw=r.width - 32)


# ---------------------------------------------------------------------
# simülasyon: süreç durumu, tarayıcı bağlantısı, canlı değerler
# ---------------------------------------------------------------------
class SimRunScene(Scene):
    title = "AH-1S · Sürüş — Simülasyon"
    STATE_TXT = {"starting": ("BAŞLATILIYOR", ui.WARN), "running": ("ÇALIŞIYOR", ui.OK), "done": ("UÇUŞ BİTTİ", ui.WARN),
                 "error": ("HATA", ui.ERR), "stopped": ("DURDURULDU", ui.MUTED)}

    def __init__(self, app, sc: StartConditions):
        super().__init__(app)
        self.sc = sc
        self.sim = SimProcess(sc, open_browser=app.open_browser).start()
        app.sims.append(self.sim)
        self.stop_btn = ui.Button("Durdur ve ayarlara dön", self.back, kind="ghost")
        self.open_btn = ui.Button("3D görünümü tarayıcıda aç", self.sim.open_page)
        self.widgets = [self.stop_btn, self.open_btn]

    def layout(self, size):
        w, h = size
        self.stop_btn.layout(pygame.Rect(GAP + 6, 9, 230, 38))
        self.card = pygame.Rect(GAP + 6, 56 + GAP + 6, w - 2 * GAP - 12, h - 56 - 2 * GAP - 12)
        self.open_btn.layout(pygame.Rect(self.card.right - 330, self.card.top + 18, 310, 44))

    def back(self):
        self.app.pop()

    def on_exit(self):
        self.sim.stop()
        if self.sim in self.app.sims:
            self.app.sims.remove(self.sim)

    def draw(self, surf):
        w, h = surf.get_size()
        surf.fill(ui.BG)
        top_bar(surf, self.title, w, x=self.stop_btn.rect.right + 18)
        snap = self.sim.snapshot()
        state = snap["state"]
        if state == "running" and snap["status"].get("state") == "done":
            state = "done"
        self.open_btn.enabled = state in ("running", "done")
        c = self.card
        pygame.draw.rect(surf, ui.SURFACE, c, border_radius=10)
        lab, col = self.STATE_TXT.get(state, (state.upper(), ui.TEXT))
        f = ui.font(15, True); img = f.render(lab, True, (20, 20, 20))
        pill = img.get_rect(topleft=(c.left + 22, c.top + 24)).inflate(24, 12)
        pygame.draw.rect(surf, col, pill, border_radius=pill.height // 2)
        surf.blit(img, img.get_rect(center=pill.center))
        msg = snap["message"] if state != "done" else (snap["status"].get("message") or "Uçuş bitti.")
        y = ui.text(surf, msg, (pill.right + 14, pill.centery), 17, ui.TEXT, anchor="midleft").bottom
        y = max(y, pill.bottom) + 14
        y = ui.text(surf, f"Başlangıç: {summary(self.sc)}", (c.left + 22, y), 15, ui.TEXT_2,
                    maxw=c.width - 360).bottom + 6
        y = ui.text(surf, f"3D sayfa: {snap['url']}  ·  komutlar (görev, Δhız / Δheading / Δirtifa) sayfadaki formdan",
                    (c.left + 22, y), 15, ui.MUTED, maxw=c.width - 44).bottom + 22
        # canlı değerler
        t = snap["telemetry"]
        cells = [("Süre", t.get("t"), "{:.0f} s"), ("İrtifa (AGL)", t.get("h"), "{:.0f} ft"),
                 ("Hava hızı", t.get("ua"), "{:.0f} kt"), ("Heading", t.get("psi"), "{:03.0f}°"),
                 ("Dikey hız", None if t.get("vs") is None else t["vs"] * 60, "{:+.0f} ft/dk"),
                 ("Yakıt", t.get("fuel"), "{:.0f} lbs"), ("Tork", t.get("tq"), "{:.1f} psi"), ("Rotor", t.get("rpm"), "{:.0f} rpm")]
        cols = 4
        cw = (c.width - 44 - (cols - 1) * GAP) // cols
        for k, (name, v, fm) in enumerate(cells):
            cell = pygame.Rect(c.left + 22 + (k % cols) * (cw + GAP), y + (k // cols) * 92, cw, 82)
            pygame.draw.rect(surf, ui.SURFACE_2, cell, border_radius=8)
            ui.text(surf, name.upper(), (cell.left + 14, cell.top + 12), 13, ui.MUTED, bold=True)
            val = fm.format(v) if isinstance(v, (int, float)) else "—"
            ui.text(surf, val, (cell.left + 14, cell.bottom - 12), 28, ui.TEXT, bold=True, anchor="bottomleft")
        y += 2 * 92 + 12
        if state in ("error", "starting") and snap["log"]:
            ui.text(surf, "Simülasyon çıktısı:", (c.left + 22, y), 14, ui.MUTED, bold=True)
            y += 22
            for line in snap["log"][-8:]:
                if y > c.bottom - 22:
                    break
                y = ui.text(surf, line[:160], (c.left + 22, y), 13, ui.ERR if state == "error" else ui.TEXT_2).bottom + 2
        for wd in self.widgets:
            wd.draw(surf)
