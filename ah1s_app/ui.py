"""pygame arayüz parçaları: renkler, yazı, düğme, kaydırıcı, seçim grubu, aç/kapa, yer tutucu panel.

Görsellik şimdilik yer tutucu (düz renkler); bileşenlerin davranışı ve yerleşimi kalıcı.
"""

from __future__ import annotations

from typing import Callable

import pygame

BG = (20, 23, 25)
SURFACE = (30, 35, 38)
SURFACE_2 = (40, 46, 50)
LINE = (66, 74, 79)
TEXT = (238, 236, 228)
TEXT_2 = (178, 184, 188)
MUTED = (124, 131, 136)
ACCENT = (214, 172, 82)
WARN = (236, 178, 64)
ERR = (226, 96, 84)
OK = (98, 190, 116)

_fonts: dict[tuple[int, bool], pygame.font.Font] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("dejavusans,liberationsans,notosans,arial,helvetica", size, bold=bold)
    return _fonts[key]


def wrap(s: str, f: pygame.font.Font, maxw: int) -> list[str]:
    lines = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split(" "):
            t = f"{cur} {word}".strip()
            if cur and f.size(t)[0] > maxw:
                lines.append(cur)
                cur = word
            else:
                cur = t
        lines.append(cur)
    return lines


def text(surf, s: str, pos, size: int = 18, color=TEXT, bold: bool = False, anchor: str = "topleft",
         maxw: int | None = None, line_gap: int = 4) -> pygame.Rect:
    """Yazı çizer; maxw verilirse kelime kaydırır. Kapladığı dikdörtgeni döndürür."""
    f = font(size, bold)
    lines = wrap(s, f, maxw) if maxw else [s]
    imgs = [f.render(line, True, color) for line in lines]
    w = max((i.get_width() for i in imgs), default=0)
    h = sum(i.get_height() for i in imgs) + line_gap * max(0, len(imgs) - 1)
    box = pygame.Rect(0, 0, w, h)
    setattr(box, anchor, pos)
    y = box.top
    for img in imgs:
        r = img.get_rect()
        if anchor in ("center", "midtop", "midbottom"):
            r.midtop = (box.centerx, y)
        elif anchor in ("topright", "midright", "bottomright"):
            r.topright = (box.right, y)
        else:
            r.topleft = (box.left, y)
        surf.blit(img, r)
        y += img.get_height() + line_gap
    return box


def fit(s: str, size: int, maxw: int, bold: bool = False, min_size: int = 12) -> int:
    """Yazının maxw'ye sığdığı en büyük punto (size'dan aşağı)."""
    while size > min_size and font(size, bold).size(s)[0] > maxw:
        size -= 1
    return size


def clip(s: str, f: pygame.font.Font, maxw: int) -> str:
    """Sığmazsa sonunu "…" ile kısaltır."""
    if f.size(s)[0] <= maxw:
        return s
    while s and f.size(s + "…")[0] > maxw:
        s = s[:-1]
    return s.rstrip() + "…"


def shade(c, k: float):
    return tuple(max(0, min(255, int(v * k))) for v in c)


class Widget:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.enabled = True
        self.hover = False

    def layout(self, rect: pygame.Rect):
        self.rect = pygame.Rect(rect)

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        return False

    def draw(self, surf):
        raise NotImplementedError


class Button(Widget):
    def __init__(self, label: str, on_click: Callable[[], None], kind: str = "primary"):
        super().__init__()
        self.label, self.on_click, self.kind = label, on_click, kind

    def handle(self, ev) -> bool:
        super().handle(ev)
        if self.enabled and ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.rect.collidepoint(ev.pos):
            self.on_click()
            return True
        return False

    def draw(self, surf):
        if self.kind == "primary":
            bg = ACCENT if self.enabled else SURFACE_2
            bg = shade(bg, 1.08) if self.hover and self.enabled else bg
            fg = (28, 24, 16) if self.enabled else MUTED
            pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        else:
            bg = SURFACE_2 if self.hover and self.enabled else SURFACE
            fg = TEXT if self.enabled else MUTED
            pygame.draw.rect(surf, bg, self.rect, border_radius=8)
            pygame.draw.rect(surf, LINE, self.rect, 1, border_radius=8)
        bold = self.kind == "primary"
        text(surf, self.label, self.rect.center, fit(self.label, 18, self.rect.width - 20, bold), fg, bold=bold, anchor="center")


class Tile(Widget):
    """Yer tutucu "resim": düz renk, başlık, alt başlık; hazır değilse "Coming soon"."""

    def __init__(self, tile, on_click: Callable[[str], None], soon_text: str):
        super().__init__()
        self.tile, self.on_click, self.soon = tile, on_click, soon_text
        self.enabled = tile.ready

    def handle(self, ev) -> bool:
        super().handle(ev)
        if self.enabled and ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.rect.collidepoint(ev.pos):
            self.on_click(self.tile.key)
            return True
        return False

    def draw(self, surf):
        r, c = self.rect, self.tile.color
        pygame.draw.rect(surf, shade(c, 1.12) if self.hover and self.enabled else c, r, border_radius=10)
        if not self.enabled:
            veil = pygame.Surface(r.size, pygame.SRCALPHA); veil.fill((0, 0, 0, 110))
            surf.blit(veil, r.topleft)
        pad = max(18, r.width // 14)
        size = max(22, min(44, r.width // 9))
        box = text(surf, self.tile.title, (r.left + pad, r.bottom - pad - (28 if self.tile.subtitle else 0)), size,
                   TEXT, bold=True, anchor="bottomleft", maxw=r.width - 2 * pad)
        if self.tile.subtitle:
            text(surf, self.tile.subtitle, (r.left + pad, box.bottom + 8), 17, TEXT_2, maxw=r.width - 2 * pad)
        if not self.enabled:
            f = font(max(18, min(26, r.width // 14)), True)
            img = f.render(self.soon.upper(), True, TEXT)
            pill = img.get_rect(center=r.center).inflate(32, 16)
            pygame.draw.rect(surf, (0, 0, 0), pill, border_radius=pill.height // 2)
            pygame.draw.rect(surf, TEXT_2, pill, 1, border_radius=pill.height // 2)
            surf.blit(img, img.get_rect(center=pill.center))
        if self.hover and self.enabled:
            pygame.draw.rect(surf, ACCENT, r, 3, border_radius=10)


class Slider(Widget):
    """Etiket + değer satırı ve altında ray. Sürükle, tıkla ya da üstündeyken tekerlekle adım adım değiştir."""

    HEIGHT = 58

    def __init__(self, label: str, vmin: float, vmax: float, step: float, value: float,
                 fmt: Callable[[float], str], on_change: Callable[[float], None]):
        super().__init__()
        self.label, self.vmin, self.vmax, self.step = label, vmin, vmax, step
        self.value, self.fmt, self.on_change = value, fmt, on_change
        self.drag = False

    @property
    def track(self) -> pygame.Rect:
        return pygame.Rect(self.rect.left + 8, self.rect.bottom - 16, self.rect.width - 16, 6)

    def set(self, v: float):
        v = round(round(v / self.step) * self.step, 6)            # adımın katlarına (sınırlar ayrıca kırpar)
        v = max(self.vmin, min(self.vmax, v))
        if v != self.value:
            self.value = v
            self.on_change(v)

    def _from_x(self, x: int):
        t = self.track
        self.set(self.vmin + (x - t.left) / max(1, t.width) * (self.vmax - self.vmin))

    def handle(self, ev) -> bool:
        super().handle(ev)
        if not self.enabled:
            self.drag = False
            return False
        hit = self.rect.collidepoint(ev.pos) if hasattr(ev, "pos") else False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hit:
            self.drag = True; self._from_x(ev.pos[0]); return True
        if ev.type == pygame.MOUSEMOTION and self.drag:
            self._from_x(ev.pos[0]); return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.drag:
            self.drag = False; return True
        if ev.type == pygame.MOUSEWHEEL and self.hover:
            self.set(self.value + ev.y * self.step); return True
        return False

    def draw(self, surf):
        col, vcol = (TEXT_2, TEXT) if self.enabled else (MUTED, MUTED)
        vbox = text(surf, self.fmt(self.value) if self.enabled else "—", (self.rect.right - 8, self.rect.top + 4), 16, vcol,
                    bold=True, anchor="topright")
        text(surf, clip(self.label, font(16), vbox.left - self.rect.left - 20), (self.rect.left + 8, self.rect.top + 4), 16, col)
        t = self.track
        pygame.draw.rect(surf, SURFACE_2, t, border_radius=3)
        k = (self.value - self.vmin) / (self.vmax - self.vmin)
        if self.enabled:
            pygame.draw.rect(surf, ACCENT, (t.left, t.top, int(t.width * k), t.height), border_radius=3)
            pygame.draw.circle(surf, TEXT if (self.hover or self.drag) else ACCENT, (t.left + int(t.width * k), t.centery), 9)


class Segmented(Widget):
    HEIGHT = 62

    def __init__(self, label: str, options: tuple[tuple[str, str], ...], value: str, on_change: Callable[[str], None]):
        super().__init__()
        self.label, self.options, self.value, self.on_change = label, options, value, on_change

    def _cells(self) -> list[pygame.Rect]:
        top = self.rect.top + 26
        w = (self.rect.width - 16) / len(self.options)
        return [pygame.Rect(int(self.rect.left + 8 + k * w), top, int(w) - 4, self.rect.bottom - top - 4)
                for k in range(len(self.options))]

    def handle(self, ev) -> bool:
        super().handle(ev)
        if self.enabled and ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            for (key, _), cell in zip(self.options, self._cells()):
                if cell.collidepoint(ev.pos):
                    if key != self.value:
                        self.value = key
                        self.on_change(key)
                    return True
        return False

    def draw(self, surf):
        text(surf, self.label, (self.rect.left + 8, self.rect.top + 4), 16, TEXT_2)
        for (key, lab), cell in zip(self.options, self._cells()):
            on = key == self.value
            pygame.draw.rect(surf, ACCENT if on else SURFACE_2, cell, border_radius=6)
            text(surf, lab, cell.center, fit(lab, 16, cell.width - 10, on), (28, 24, 16) if on else TEXT, bold=on, anchor="center")


class Toggle(Widget):
    HEIGHT = 34

    def __init__(self, label: str, value: bool, on_change: Callable[[bool], None]):
        super().__init__()
        self.label, self.value, self.on_change = label, value, on_change

    def handle(self, ev) -> bool:
        super().handle(ev)
        if self.enabled and ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.rect.collidepoint(ev.pos):
            self.value = not self.value
            self.on_change(self.value)
            return True
        return False

    def draw(self, surf):
        sw = pygame.Rect(self.rect.left + 8, self.rect.centery - 11, 42, 22)
        pygame.draw.rect(surf, ACCENT if self.value else SURFACE_2, sw, border_radius=11)
        pygame.draw.circle(surf, TEXT, (sw.right - 11 if self.value else sw.left + 11, sw.centery), 8)
        text(surf, clip(self.label, font(16), self.rect.right - sw.right - 20), (sw.right + 12, self.rect.centery), 16, TEXT_2,
             anchor="midleft")
