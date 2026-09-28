"""Render a synthetic textured plaza with known cameras and write a harvester-style work folder.

  python tests/synth.py work/synth [--n-ring 28] [--size 1024x768]

Writes manifest.json (with noisy GPS, a few gross GPS outliers and a few un-located photos),
images/*.jpg, and truth.json (true camera centres/rotations in ENU) for evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirante.geo import enu_to_geodetic  # noqa: E402

LAT0, LON0, ALT0 = -19.91655, -43.93385, 852.0  # near Praça Rui Barbosa (Praça da Estação), BH
SUN = np.array([0.4, -0.3, 0.86]); SUN /= np.linalg.norm(SUN)


def texture(w, h, seed, base, kind):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), tuple(int(c) for c in base))
    d = ImageDraw.Draw(img)
    if kind == "paving":
        tile = 64
        for y in range(0, h, tile):
            off = (y // tile % 2) * tile // 2
            for x in range(-tile, w, tile):
                c = np.clip(np.array(base) + rng.normal(0, 18, 3), 0, 255).astype(int)
                d.rectangle([x + off, y, x + off + tile - 3, y + tile - 3], fill=tuple(c))
    if kind == "facade":
        for x in range(0, w, 90):
            for y in range(40, h - 40, 110):
                c = tuple(int(v) for v in rng.integers(20, 90, 3))
                d.rectangle([x + 15, y, x + 70, y + 70], fill=c, outline=(230, 230, 220), width=4)
    n = {"paving": 5000, "facade": 1500, "stone": 2500}[kind]
    for _ in range(n):  # random marks = features
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(3, 22))
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        else:
            d.rectangle([x - r, y - r // 2, x + r, y + r // 2], fill=c)
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(0.7)), np.float32)
    arr += rng.normal(0, 6, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def box(center, size, tex_seed, base):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, y0, z0 = cx - sx / 2, cy - sy / 2, cz
    faces = [
        ((x0, y0, z0), (sx, 0, 0), (0, 0, sz)),            # south
        ((x0 + sx, y0 + sy, z0), (-sx, 0, 0), (0, 0, sz)),  # north
        ((x0 + sx, y0, z0), (0, sy, 0), (0, 0, sz)),       # east
        ((x0, y0 + sy, z0), (0, -sy, 0), (0, 0, sz)),      # west
        ((x0, y0, z0 + sz), (sx, 0, 0), (0, sy, 0)),       # top
    ]
    out = []
    for k, (o, u, v) in enumerate(faces):
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        tex = texture(max(64, int(lu * 60)), max(64, int(lv * 60)), tex_seed * 10 + k, base, "stone")
        out.append((np.array(o, float), np.array(u, float), np.array(v, float), tex))
    return out


def build_scene():
    rects = []
    g = texture(4096, 4096, 1, (150, 140, 125), "paving")
    rects.append((np.array([-35., -35, 0]), np.array([70., 0, 0]), np.array([0., 70, 0]), g))
    fac = [((-30, -30, 0), (60, 0, 0)), ((30, 30, 0), (-60, 0, 0)), ((30, -30, 0), (0, 60, 0)), ((-30, 30, 0), (0, -60, 0))]
    for k, (o, u) in enumerate(fac):
        base = [(200, 170, 130), (180, 190, 200), (210, 200, 170), (170, 150, 150)][k]
        rects.append((np.array(o, float), np.array(u, float), np.array([0., 0, 16]), texture(3000, 800, 20 + k, base, "facade")))
    rects += box((0, 0, 0), (5, 5, 2.2), 3, (175, 165, 150))       # pedestal
    rects += box((0, 0, 2.2), (1.4, 1.4, 8), 4, (190, 185, 175))   # column
    rects += box((12, -8, 0), (3, 2, 2.8), 5, (120, 140, 120))     # kiosk
    rects += box((-10, 9, 0), (2, 4, 1.2), 6, (140, 120, 110))     # bench block
    return rects


def look_at(C, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - C; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)  # world_from_cam, COLMAP convention (x right, y down, z fwd)


def render(rects, C, R_wc, W, H, f, ss=2):
    w, h, fs = W * ss, H * ss, f * ss
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = np.stack([(u - w / 2) / fs, (v - h / 2) / fs, np.ones_like(u)], -1) @ R_wc.T
    d = d.reshape(-1, 3)
    best_t = np.full(len(d), np.inf)
    col = np.zeros((len(d), 3), np.float32)
    # sky
    sky_t = np.clip(d[:, 2] / np.linalg.norm(d, axis=1), 0, 1)[:, None]
    col[:] = (1 - sky_t) * np.array([200, 215, 235]) + sky_t * np.array([90, 140, 210])
    for O, U, V, tex in rects:
        N = np.cross(U, V); N /= np.linalg.norm(N)
        den = d @ N
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((O - C) @ N) / den
        ok = (t > 0.05) & (t < best_t)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        P = C + t[idx, None] * d[idx]
        a = (P - O) @ U / (U @ U)
        b = (P - O) @ V / (V @ V)
        inside = (a >= 0) & (a < 1) & (b >= 0) & (b < 1)
        idx, a, b = idx[inside], a[inside], b[inside]
        th, tw = tex.shape[:2]
        shade = 0.55 + 0.45 * max(0.0, abs(float(N @ SUN)))
        col[idx] = tex[((1 - b) * th).astype(int).clip(0, th - 1), (a * tw).astype(int).clip(0, tw - 1)] * shade
        best_t[idx] = t[idx]
    img = col.reshape(h, w, 3).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img).resize((W, H), Image.BOX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--n-ring", type=int, default=28)
    ap.add_argument("--size", default="1024x768")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    f = 0.85 * W
    rng = np.random.default_rng(a.seed)
    rects = build_scene()
    (a.workdir / "images").mkdir(parents=True, exist_ok=True)

    cams = []
    for k in range(a.n_ring):  # ring looking at the monument
        ang = 2 * np.pi * k / a.n_ring + rng.normal(0, 0.03)
        r = rng.uniform(11, 17)
        C = np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(1.5, 1.75)])
        tgt = np.array([0, 0, 4.0]) + rng.normal(0, 1.2, 3)
        cams.append((C, look_at(C, tgt)))
    for k in range(6):  # pairs looking at facades
        ang = 2 * np.pi * k / 6 + 0.3
        base = np.array([14 * np.cos(ang), 14 * np.sin(ang), 1.65])
        wall = base / np.linalg.norm(base[:2]) * 30; wall[2] = 5
        for dx in (-1.5, 1.5):
            side = np.cross([0, 0, 1], wall - base); side /= np.linalg.norm(side)
            C = base + side * dx
            cams.append((C, look_at(C, wall + side * dx * 2 + rng.normal(0, 1, 3))))

    items, truth = [], {}
    outliers = set(rng.choice(len(cams), 2, replace=False).tolist())
    unlocated = set(rng.choice([i for i in range(len(cams)) if i not in outliers], 2, replace=False).tolist())
    for i, (C, R_wc) in enumerate(cams):
        name = f"{1000 + i}_Synthetic_plaza_{i:02d}.jpg"
        render(rects, C, R_wc, W, H, f).save(a.workdir / "images" / name, quality=92)
        noise = rng.normal(0, [1.0, 1.0, 1.5])
        if i in outliers:
            noise[:2] += rng.choice([-1, 1], 2) * 30
        lat, lon, alt = enu_to_geodetic(*(C + noise), LAT0, LON0, ALT0)
        loc = None if i in unlocated else {"lat": float(lat), "lon": float(lon), "alt": float(alt),
                                           "heading": None, "source": "synthetic"}
        items.append({"pageid": 1000 + i, "title": f"File:Synthetic plaza {i:02d}.jpg", "file": name,
                      "file_page": None, "thumb_url": None, "thumb_size": [W, H], "original_size": [W, H],
                      "camera_location": loc, "object_location": None,
                      "focal_prior_px": f * rng.uniform(0.97, 1.03) if i % 2 == 0 else None,
                      "focal_prior_source": "synthetic" if i % 2 == 0 else None,
                      "author": "WPGT Mirante test renderer", "license": "CC0", "license_url": None,
                      "date": "2026-09-26", "description": "synthetic"})
        truth[name] = {"C": C.tolist(), "R_wc": R_wc.tolist(), "gps_outlier": i in outliers, "unlocated": i in unlocated}
        print(f"rendered {i + 1}/{len(cams)}", file=sys.stderr, flush=True)
    (a.workdir / "manifest.json").write_text(json.dumps(
        {"category": "Category:Synthetic test plaza", "items": items, "thumb_width": W}, indent=1))
    (a.workdir / "truth.json").write_text(json.dumps({"origin": [LAT0, LON0, ALT0], "f": f, "cams": truth}, indent=1))


if __name__ == "__main__":
    main()
