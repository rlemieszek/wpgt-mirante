"""Export a georeferenced reconstruction as a static WPGT Mirante viewer site.

Writes <site>/index.html, viewer.js, viewer.css, vendor/ (three.js + fonts; copied from viewer/),
       <site>/credits.html        every source photo with author and licence, single.html, and
       <site>/data/scene.json   cameras, intrinsics, attribution, neighbours
       <site>/data/points.bin   float32 xyz (ENU) * N  followed by uint8 rgb * N
       <site>/data/vis.bin      uint32 point indices per image (offsets in scene.json)
       <site>/data/img/         local copies of the photos (only with --local-images)

Without --local-images the viewer loads photos from upload.wikimedia.org at runtime.

Usage:
  python -m mirante.export work/rui-barbosa site/rui-barbosa [--local-images] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

from .geo import enu_to_geodetic
from .reconstruct import _cam_from_world

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
FLIP = np.diag([1.0, -1.0, -1.0])  # COLMAP camera (x right, y down, z fwd) -> three.js (y up, -z fwd)


def mat_to_quat(R):
    """3x3 rotation -> (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().tolist()


def export(workdir: Path, site: Path, local_images=False, title=None, max_reproj_px=2.0, local_width=1600,
           inline_bins=False):
    man = json.loads((workdir / "manifest.json").read_text())
    geo = json.loads((workdir / "georef.json").read_text())
    by_file = {it["file"]: it for it in man["items"]}
    anchors = {f for f, it in by_file.items() if it.get("anchor")} or set(by_file)  # flagged anchors, else all
    extra_man = workdir / "extra" / "manifest.json"
    if geo.get("extended") and extra_man.exists():
        by_file.update({it["file"]: it for it in json.loads(extra_man.read_text())["items"]})
    extra_offsets = (geo.get("extra") or {}).get("geotag_offset_m", {})
    extra_files = set(by_file) - set(it["file"] for it in man["items"])
    rec = pycolmap.Reconstruction(str(workdir / geo["model_dir"]))
    s = geo["sim3"]["scale"]
    R = np.array(geo["sim3"]["R"])
    t = np.array(geo["sim3"]["t"])

    # ---- points
    pids, xyz, rgb, err = [], [], [], []
    for pid, p in rec.points3D.items():
        pids.append(pid)
        xyz.append(p.xyz)
        rgb.append(p.color)
        err.append(p.error)
    xyz = s * np.asarray(xyz) @ R.T + t
    rgb = np.asarray(rgb, np.uint8)
    err = np.asarray(err)
    keep = err <= max_reproj_px
    # drop far-flung points (robust radius around the camera cluster)
    centers = np.array([s * R @ rec.images[i].projection_center() + t for i in rec.reg_image_ids()])
    ctr = np.median(centers, 0)
    d_cam = np.linalg.norm(centers - ctr, axis=1)
    d_pts = np.linalg.norm(xyz - ctr, axis=1)
    radius = max(np.percentile(d_pts[keep], 98) * 1.5, d_cam.max() * 3, 10.0)
    keep &= d_pts <= radius
    pid_index = {pid: i for i, pid in enumerate(np.asarray(pids)[keep])}
    xyz, rgb = xyz[keep].astype(np.float32), rgb[keep]
    print(f"points: {len(xyz)} kept of {len(pids)}", file=sys.stderr)

    # ---- cameras
    images, vis = [], []
    reg = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
    for iid in reg:
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        it = by_file.get(im.name, {})
        R_cw = _cam_from_world(im).rotation.matrix()
        C = s * R @ im.projection_center() + t
        R_wc = R @ R_cw.T
        f = cam.params[0]
        k1 = cam.params[3] if cam.model_name == "SIMPLE_RADIAL" else 0.0
        ids = sorted({pid_index[p.point3D_id] for p in im.points2D if p.has_point3D() and p.point3D_id in pid_index})
        vis.append(np.asarray(ids, np.uint32))
        loc = it.get("camera_location") or {}
        lat, lon, alt = enu_to_geodetic(*C, geo["origin"]["lat"], geo["origin"]["lon"], geo["origin"]["alt"]) \
            if geo["origin"] else (None, None, None)
        heading = float((np.degrees(np.arctan2(R_wc[0, 2], R_wc[1, 2])) + 360) % 360)  # optical axis azimuth
        pitch = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1, 1))))
        rec_img = {
            "file": im.name,
            "title": it.get("title", im.name),
            "file_page": it.get("file_page"),
            "src_remote": it.get("thumb_url"),
            "width": cam.width, "height": cam.height,
            "f": float(f), "cx": float(cam.params[1]), "cy": float(cam.params[2]), "k1": float(k1),
            "position": C.round(4).tolist(),
            "quaternion": mat_to_quat(R_wc @ FLIP),
            "sfm_lat": None if lat is None else round(float(lat), 7),
            "sfm_lon": None if lon is None else round(float(lon), 7),
            "sfm_alt": None if alt is None else round(float(alt), 2),
            "sfm_heading_deg": round(heading, 1), "sfm_pitch_deg": round(pitch, 1),
            "commons_heading_deg": loc.get("heading"),
            # anchor: geotag was a bundle-adjustment prior; photo: geotag only used for the georeferencing fit
            "role": ("anchor" if geo.get("used_priors_in_ba") and im.name in anchors else
                     "extra" if im.name in extra_files else "photo"),
            # anchors: residual of the georeferencing fit; extras: offset of their Commons geotag from SfM
            "gps_residual_m": (extra_offsets if im.name in extra_files else geo.get("per_image_residual_m", {})).get(im.name),
            "author": it.get("author"), "license": it.get("license"), "license_url": it.get("license_url"),
            "date": it.get("date"), "num_points": len(ids),
        }
        images.append(rec_img)

    # ---- co-visibility neighbours
    sets = [set(v.tolist()) for v in vis]
    for i, a in enumerate(sets):
        nb = sorted(((len(a & b), j) for j, b in enumerate(sets) if j != i and a & b), reverse=True)[:12]
        images[i]["neighbors"] = [[j, n] for n, j in nb]

    offsets, o = [], 0
    for v in vis:
        offsets.append([o, len(v)])
        o += len(v)
    for img, off in zip(images, offsets):
        img["vis"] = off

    data = site / "data"
    if site.exists():
        shutil.rmtree(site)
    data.mkdir(parents=True)
    copy_viewer(site)
    if local_images:
        (data / "img").mkdir()
        for img in images:
            src = workdir / "images" / img["file"]
            dst = data / "img" / (Path(img["file"]).stem + ".jpg")
            with Image.open(src) as pim:
                pim = pim.convert("RGB")
                if pim.width > local_width:
                    pim = pim.resize((local_width, round(pim.height * local_width / pim.width)), Image.LANCZOS)
                pim.save(dst, quality=85)
            img["src_local"] = f"data/img/{dst.name}"

    pts_bytes = xyz.tobytes() + rgb.tobytes()
    vis_bytes = np.concatenate(vis).astype(np.uint32).tobytes() if vis else b""
    if not inline_bins:
        (data / "points.bin").write_bytes(pts_bytes)
        (data / "vis.bin").write_bytes(vis_bytes)
    scene = {
        "version": 1,
        "title": title or man.get("category", "WPGT Mirante scene").removeprefix("Category:"),
        "category": man.get("category"),
        "origin": geo["origin"], "frame": geo["frame"], "aligned": geo.get("aligned", False),
        "georef": {k: geo.get(k) for k in ("num_inliers", "num_located_registered", "residual_m",
                                           "mean_camera_up_tilt_deg", "used_priors_in_ba")},
        "extra": {k: (geo.get("extra") or {}).get(k) for k in ("num_candidates", "num_registered", "anchor_drift_max_m")}
                 if geo.get("extended") else None,
        "num_images_total": geo.get("num_images_total"), "num_points": int(len(xyz)),
        "center": ctr.round(3).tolist(),
        "images": images,
    }
    if inline_bins:
        import base64
        scene["inline"] = {"points": base64.b64encode(pts_bytes).decode(), "vis": base64.b64encode(vis_bytes).decode()}
    (data / "scene.json").write_text(json.dumps(scene, ensure_ascii=False))
    write_single_page(site, scene["title"])
    write_credits(site, scene)
    print(f"site written to {site} ({len(images)} cameras)", file=sys.stderr)


def copy_viewer(site: Path):
    """Viewer files into the site; index.html references them with a content hash so browsers never run a
    stale cached viewer.js / viewer.css after an update."""
    import hashlib
    html = (VIEWER_DIR / "index.html").read_text()
    shutil.copytree(VIEWER_DIR / "vendor", site / "vendor", dirs_exist_ok=True)
    for fn in ("viewer.js", "viewer.css"):
        shutil.copy(VIEWER_DIR / fn, site / fn)
        v = hashlib.sha1((VIEWER_DIR / fn).read_bytes()).hexdigest()[:10]
        html = html.replace(f'"{fn}"', f'"{fn}?v={v}"')
    (site / "index.html").write_text(html)


def refresh_viewer(site: Path):
    """Copy the current viewer files into an existing site (data untouched); rebuild single.html and credits."""
    copy_viewer(site)
    scene = json.loads((site / "data" / "scene.json").read_text())
    write_single_page(site, scene["title"])
    write_credits(site, scene)
    print(f"viewer refreshed in {site}", file=sys.stderr)


def write_credits(site: Path, scene: dict):
    """credits.html: every photo the reconstruction is derived from, with author and licence (CC BY-SA credit)."""
    from html import escape
    roles = {"anchor": "survey anchor (RTK)", "extra": "matched to anchors", "photo": "placed by SfM"}
    rows = []
    for k, im in enumerate(sorted(scene["images"], key=lambda im: im["title"]), 1):
        title = escape(im["title"].removeprefix("File:"))
        link = f'<a href="{escape(im["file_page"])}">{title}</a>' if im.get("file_page") else title
        lic = escape(im.get("license") or "licence unknown")
        if (im.get("license_url") or "").startswith(("http://", "https://")):
            lic = f'<a href="{escape(im["license_url"])}">{lic}</a>'
        rows.append(f"<tr><td>{k}</td><td>{link}</td><td>{escape(im.get('author') or 'unknown')}</td>"
                    f"<td>{lic}</td><td>{roles.get(im.get('role'), '')}</td></tr>")
    t = escape(scene["title"])
    (site / "credits.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credits · {t} · WPGT Mirante</title>
<style>
:root {{ color-scheme: dark; }}
@font-face {{ font-family: "Archivo"; src: url("vendor/fonts/archivo-latin-wdth-normal.woff2") format("woff2"); font-weight: 100 900; }}
body {{ margin: 0 auto; max-width: 70rem; padding: 24px 16px 48px; background: #0e1113; color: #e7e5df;
  font: 14px/1.5 "Archivo", "Helvetica Neue", Arial, sans-serif; }}
a {{ color: #7cc4b8; }} h1 {{ margin: 0 0 4px; font-size: 22px; }} .sub {{ color: #8e979c; margin: 0 0 16px; }}
.wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid rgba(160,178,184,.18); vertical-align: top; }}
th {{ color: #8e979c; font-weight: 600; }} td:first-child {{ color: #8e979c; }}
</style></head><body>
<h1>Credits: {t}</h1>
<p class="sub"><a href="index.html">WPGT Mirante</a>, a
<a href="https://commons.wikimedia.org/wiki/Commons:WikiProject_GeoTwin">WikiProject GeoTwin</a> tool by Rafael Lemieszek</p>
<p>The 3D points and camera positions shown here are derived from the {len(rows)} photographs below, all from
Wikimedia Commons, and are released under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.
Each photograph remains under its own licence, listed here and shown with the photo in the viewer, and is loaded
directly from Commons.</p>
<div class="wrap"><table><thead><tr><th>#</th><th>Photo</th><th>Author</th><th>Licence</th><th>Role</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>
</body></html>
""")


def write_single_page(site: Path, title: str):
    """single.html: CSS and JS inlined (no doctype/head), for hosts that only accept one page + data files."""
    html = (VIEWER_DIR / "index.html").read_text()
    body = html.split("<!--VIEWER-BODY-->", 1)[1].split("<!--/VIEWER-BODY-->", 1)[0]
    body = body.replace('<script type="module" src="viewer.js"></script>', "")
    head = html.split("<head>", 1)[1].split("</head>", 1)[0]
    importmap = head[head.index('<script type="importmap">'):head.index("</script>", head.index("importmap")) + 9]
    icon = [ln.strip() for ln in head.splitlines() if 'rel="icon"' in ln][0]
    css = (VIEWER_DIR / "viewer.css").read_text()  # fonts and three.js resolve to vendor/ next to the page
    js = (VIEWER_DIR / "viewer.js").read_text()
    safe_title = title.replace("<", "").replace(">", "")
    (site / "single.html").write_text(
        f"<title>{safe_title}</title>\n{icon}\n<style>\n{css}\n</style>\n{importmap}\n{body}\n"
        f"<script type=\"module\">\n{js}\n</script>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("site", type=Path)
    ap.add_argument("--local-images", action="store_true", help="copy photos into the site instead of hotlinking Commons")
    ap.add_argument("--title")
    ap.add_argument("--inline-bins", action="store_true",
                    help="embed points/visibility as base64 in scene.json (for hosts that refuse .bin files)")
    ap.add_argument("--viewer-only", action="store_true", help="only refresh the viewer files of an existing site")
    a = ap.parse_args(argv)
    if a.viewer_only:
        return refresh_viewer(a.site)
    export(a.workdir, a.site, a.local_images, a.title, inline_bins=a.inline_bins)


if __name__ == "__main__":
    main()
