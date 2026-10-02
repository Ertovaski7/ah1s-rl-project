"""Uygulama döngüsü: pencere, ekran yığını, genel tuşlar.

Tuşlar: Esc = geri (ana ekranda çıkış), F11 = tam ekran. Pencere boyutlandırılabilir.
"""

from __future__ import annotations

import argparse

import pygame

from .scenes import HomeScene, Scene


class App:
    def __init__(self, size=(1280, 720), fullscreen: bool = False, open_browser: bool = True):
        pygame.display.init()                  # ses kullanılmıyor: mixer başlatılmaz (sessiz makinede ALSA uyarısı da çıkmaz)
        pygame.font.init()
        pygame.display.set_caption("AH-1S Simülatör")
        self.windowed_size = size
        self.fullscreen = fullscreen
        self.screen = self._make_window()
        self.open_browser = open_browser
        self.last_conditions = None           # Sürüş ekranı son seçimleri hatırlar
        self.sims = []                        # çalışan simülasyon süreçleri (çıkışta kapanır)
        self.stack: list[Scene] = []
        self.running = True
        self.push(HomeScene(self))

    def _make_window(self):
        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)

    # ---------------- ekran yığını ----------------
    @property
    def scene(self) -> Scene:
        return self.stack[-1]

    def push(self, scene: Scene):
        self.stack.append(scene)
        scene.layout(self.screen.get_size())

    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop().on_exit()
            self.scene.layout(self.screen.get_size())
        else:
            self.quit()

    def quit(self):
        self.running = False

    # ---------------- döngü ----------------
    def tick(self, events, dt: float):
        for ev in events:
            if ev.type == pygame.QUIT:
                self.quit()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.scene.back()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                self.fullscreen = not self.fullscreen
                self.screen = self._make_window()
                self.scene.layout(self.screen.get_size())
            elif ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                self.screen = pygame.display.get_surface()
                self.scene.layout(self.screen.get_size())
            else:
                self.scene.handle(ev)
        if self.running:
            self.scene.update(dt)
            self.scene.draw(self.screen)

    def run(self):
        clock = pygame.time.Clock()
        try:
            while self.running:
                dt = clock.tick(60) / 1000.0
                self.tick(pygame.event.get(), dt)
                pygame.display.flip()
        finally:
            self.shutdown()

    def shutdown(self):
        while len(self.stack) > 1:
            self.stack.pop().on_exit()
        for sim in list(self.sims):
            sim.stop()
        pygame.quit()


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ah1s_app", description="AH-1S simülatör uygulaması (pygame).")
    p.add_argument("--fullscreen", action="store_true", help="tam ekran başla (F11 ile değişir)")
    p.add_argument("--size", default="1280x720", help="pencere boyutu, ör. 1600x900")
    p.add_argument("--no-browser", action="store_true", help="simülasyon başlayınca 3D sayfayı tarayıcıda açma")
    a = p.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    App((w, h), fullscreen=a.fullscreen, open_browser=not a.no_browser).run()
