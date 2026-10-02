"""Canlı simülasyonu ayrı süreçte başlatır ve izler.

Simülasyon mevcut canlı sunucudur (command_viz.py: JSBSim + PPO uçuş ajanı + 3D sayfa). Uygulama onu seçilen
başlangıç koşullarıyla ayrı bir Python sürecinde açar, hazır olunca 3D sayfayı tarayıcıda açar ve HTTP API'sinden
(api/hello, api/state) son telemetri satırını okur. Ağır bağımlılıklar (jsbsim, torch) yalnızca o süreçte yüklenir;
menüler onlarsız da çalışır.
"""

from __future__ import annotations

import collections
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

from .config import FLIGHT_MODEL, LIVE_SERVER, REPO_ROOT, StartConditions


def free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("boş port bulunamadı")


def server_args(sc: StartConditions, port: int) -> list[str]:
    """Başlangıç koşulları → command_viz.py komut satırı."""
    a = ["--model", str(FLIGHT_MODEL), "--port", str(port), "--start", sc.mode,
         "--start-heading", f"{sc.heading_deg % 360:.0f}",
         "--fuel", f"{sc.fuel_tanks[0]:.1f}", f"{sc.fuel_tanks[1]:.1f}",
         "--wind-kt", f"{sc.wind_kt:.1f}", "--wind-dir", f"{sc.wind_dir_relative:.0f}",
         "--turb", sc.turb, "--temp-dc", f"{sc.temp_dc:.1f}",
         "--lat", f"{sc.location.lat:.5f}", "--lon", f"{sc.location.lon:.5f}", "--location-name", sc.location.name]
    if sc.mode in ("hover", "cruise"):
        a += ["--start-alt", f"{sc.alt_ft:.0f}"]
    if sc.mode == "cruise":
        a += ["--start-speed-kt", f"{sc.speed_kt:.0f}"]
    if sc.gusts:
        a.append("--gusts")
    return a


class SimProcess:
    """Durumlar: starting → running → (done | error | stopped)."""

    TELEMETRY = ("t", "h", "psi", "u", "vs", "ua", "fuel", "tq", "rpm", "wow")

    def __init__(self, sc: StartConditions, port: int | None = None, open_browser: bool = True):
        self.sc = sc
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.open_browser = open_browser
        self.state = "starting"
        self.message = "Simülasyon başlatılıyor (JSBSim + PPO ajanı yükleniyor)…"
        self.telemetry: dict[str, float] = {}
        self.sim_status: dict = {}
        self.log = collections.deque(maxlen=200)
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-u", str(LIVE_SERVER), *server_args(self.sc, self.port)]

    def start(self) -> "SimProcess":
        self.proc = subprocess.Popen(self.command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        with self._lock:
            if self.state in ("starting", "running"):
                self.state, self.message = "stopped", "Simülasyon durduruldu."

    def open_page(self):
        try:
            webbrowser.open(self.url)
        except Exception:                                   # noqa: BLE001
            pass

    # ---------------- iş parçacıkları ----------------
    def _read_output(self):
        for line in self.proc.stdout:
            self.log.append(line.rstrip())

    def _get(self, path: str, timeout: float = 2.0):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _watch(self):
        cols, flight, n = None, None, 0
        while not self._stop.is_set():
            code = self.proc.poll()
            if code is not None:
                with self._lock:
                    if self.state != "stopped":
                        self.state = "error"
                        self.message = f"Simülasyon süreci kapandı (çıkış kodu {code})."
                return
            try:
                if cols is None:
                    hello = self._get("api/hello")
                    cols = {c: i for i, c in enumerate(hello["cols"])}
                st = self._get(f"api/state?flight={flight if flight is not None else ''}&since={max(0, n - 1)}")
                if st["flight"] != flight:
                    flight = st["flight"]
                n = st["n"]
                row = st["rows"][-1] if st["rows"] else None
                status = st.get("status") or {}
                with self._lock:
                    self.sim_status = status
                    if row is not None:
                        self.telemetry = {k: row[cols[k]] for k in self.TELEMETRY if k in cols}
                    if status.get("state") in ("running", "done") and self.state == "starting":
                        self.state = "running"
                        if self.open_browser:
                            self.open_page()
                    if self.state == "running":
                        self.message = status.get("message") or ""
            except Exception:                               # noqa: BLE001 — sunucu daha açılmadı ya da meşgul
                pass
            time.sleep(0.5)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(state=self.state, message=self.message, telemetry=dict(self.telemetry),
                        status=dict(self.sim_status), log=list(self.log)[-12:], url=self.url)
