"""Headless smoke test of the viewer: loads a site folder, screenshots rest / mid-transition / overview.

  python tests/viewer_smoke.py site/synth out_dir [--three node_modules/three]
Serves the site locally and maps the jsDelivr three.js URLs onto a local copy (for offline sandboxes).
"""
import argparse
import functools
import http.server
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("site", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--three", type=Path, default=Path("/home/claude/node/node_modules/three"))
ap.add_argument("--size", default="1280x800")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
W, H = map(int, a.size.split("x"))

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


handler = functools.partial(Quiet, directory=str(a.site))
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{httpd.server_port}/index.html"


def route_three(route):
    rel = route.request.url.split("/npm/three@0.186.1/", 1)[1]
    route.fulfill(path=str(a.three / rel), content_type="application/javascript")


errors = []
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.route("**/cdn.jsdelivr.net/npm/three@0.186.1/**", route_three)
    pg.route("**/fonts.googleapis.com/**", lambda r: r.abort())
    pg.goto(url)
    pg.wait_for_function("window.mirante && window.mirante.cur", timeout=60000)
    time.sleep(2)
    pg.screenshot(path=str(a.out / "1_rest.png"))
    info = pg.evaluate("""() => { const c = window.mirante; const d = {};
        for (const k of ['left','right','fwd','back']) d[k] = c.pick(c.cur, k); d.cur = c.cur.i; return d; }""")
    print("start + neighbours:", info)
    target = info["right"] if info["right"] is not None else info["left"]
    pg.evaluate(f"window.mirante.goTo({target})")
    time.sleep(0.55)
    pg.screenshot(path=str(a.out / "2_mid_transition.png"))
    time.sleep(2.5)
    pg.screenshot(path=str(a.out / "3_after.png"))
    pg.keyboard.press("o")
    time.sleep(4)
    pg.screenshot(path=str(a.out / "4_overview.png"))
    pg.set_viewport_size({"width": 390, "height": 844})
    pg.keyboard.press("o")
    time.sleep(3)
    pg.screenshot(path=str(a.out / "5_phone.png"))
    b.close()
httpd.shutdown()
print("\n".join(errors) or "no console errors")
