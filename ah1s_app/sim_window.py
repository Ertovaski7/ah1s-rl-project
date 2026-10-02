"""Simülasyonun 3D penceresi (masaüstü, pywebview; tarayıcı kullanılmaz).

Pencere hemen "hazırlanıyor" ekranıyla açılır; JSBSim ve PPO ajanı arka planda yüklenir, canlı sunucu hazır olunca
simülasyon aynı pencerede açılır. Pencere kapanınca simülasyon biter.

    python -m ah1s_app.sim_window <command_viz.py canlı seçenekleri>     (uygulama bunu kendisi çalıştırır)

Durum satırları (stdout; ah1s_app.sim okur): "3D_VIEW window" (pencere açıldı), "3D_VIEW ready <adres>",
"3D_VIEW closed" (kullanıcı kapattı), "3D_VIEW error <neden>".
Pencere: Windows WebView2, macOS WebKit, Linux Qt WebEngine (requirements-app.txt).
"""

from __future__ import annotations

import os
import sys
import threading
import traceback

LOADING = """<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>
html,body{height:100%;margin:0;background:#14181b;color:#eeece4;font:16px system-ui,-apple-system,"Segoe UI",sans-serif}
body{display:grid;place-items:center}.t{font-size:22px;font-weight:700}.s{margin-top:10px;color:#b2b8bc}
.bar{margin:22px auto 0;width:220px;height:4px;border-radius:2px;background:#2b3236;overflow:hidden}
.bar i{display:block;width:40%;height:100%;background:#d6ac52;animation:m 1.2s ease-in-out infinite alternate}
@keyframes m{from{transform:translateX(-10%)}to{transform:translateX(160%)}}
</style></head><body><div style="text-align:center"><div class="t">AH-1S Simülasyon</div>
<div class="s">JSBSim ve PPO uçuş ajanı yükleniyor…</div><div class="bar"><i></i></div></div></body></html>"""


def say(msg: str):
    print(f"3D_VIEW {msg}", flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        import webview
    except ImportError:
        say("error pywebview kurulu değil (pip install -r requirements-app.txt)")
        return 2
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        say("error ekran yok (DISPLAY / WAYLAND_DISPLAY)")            # Qt bu durumda süreci düşürür; önceden yakala
        return 2

    live: dict = {}
    window = webview.create_window("AH-1S Simülasyon — 3D", html=LOADING, width=1500, height=950, min_size=(960, 640),
                                   background_color="#14181B")

    def boot():                                    # pywebview bunu GUI açıldıktan sonra ayrı iş parçacığında çalıştırır
        try:
            from http.server import ThreadingHTTPServer

            from .config import REPO_ROOT          # repo kökü sys.path'te (command_viz ve env modülleri)
            os.chdir(REPO_ROOT)
            import command_viz as cv

            p = cv.live_parser()
            a = p.parse_args(argv)
            flight = cv.live_from_args(a)
            srv = ThreadingHTTPServer((a.host, a.port), cv.make_handler(flight))
            srv.daemon_threads = True
            live.update(flight=flight, srv=srv)
            threading.Thread(target=srv.serve_forever, kwargs=dict(poll_interval=0.25), daemon=True).start()
            flight.start()
            url = f"http://127.0.0.1:{a.port}/"
            say(f"ready {url}")
            window.load_url(url)
        except BaseException as exc:                # noqa: BLE001 — argparse SystemExit dahil
            traceback.print_exc()
            live["error"] = f"{type(exc).__name__}: {exc}"
            say(f"error {live['error']}")
            window.destroy()

    say("window")
    try:
        webview.start(boot)                         # pencere kapanınca döner
    except Exception as exc:                        # noqa: BLE001 — GUI arka ucu yok vb.
        say(f"error pencere açılamadı: {exc}")
        return 2
    finally:
        if "srv" in live:
            live["srv"].shutdown()
            live["srv"].server_close()
        if "flight" in live:
            live["flight"].stop()
    if "error" in live:
        return 1
    say("closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
