"""Merge the secondary SfM models of a work folder into its main (georeferenced) model.

When some photos do not connect to the main model (typically ground photos that share no features with a
high drone survey), incremental mapping leaves them in separate models under colmap/sparse/. Each such model is
georeferenced on its own from its photos' Commons geotags (same ENU origin as the main model, frame levelled
from the camera axes), mapped into the main model's coordinates and added to it, so one site shows everything.
No 3D points are shared across the models, so the viewer cannot fly between them, but the overview shows both
in place. Placement accuracy is that of the secondary photos' geotags (a few metres), and is reported.

Writes <workdir>/colmap/merged/ and updates georef.json (model_dir -> merged; the main georef kept under
"main"; per-model fits under "merged_models").

Usage:
  python -m mirante.merge work/boa-viagem [--min-images 10] [--inlier-m 20] [--icp]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

from .reconstruct import _cam_from_world, camera_positions_enu, georeference, umeyama


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _sim(g):
    return g["sim3"]["scale"], np.array(g["sim3"]["R"]), np.array(g["sim3"]["t"])


def icp_refine(src_pts, dst_pts, max_start=6.0, max_end=0.6, iters=60, max_scale_change=0.25, max_rot_deg=20.0,
               min_gain=0.3):
    """Trimmed similarity ICP of src points onto dst points (same frame). Returns (3x4 matrix, report) or
    (None, report) if the result is implausible or not clearly better: small groups tend to collapse onto a
    surface (large scale change), so scale and rotation are bounded and the median distance must drop."""
    from scipy.spatial import cKDTree
    c = np.median(src_pts, 0)
    dst = dst_pts[np.linalg.norm(dst_pts[:, :2] - c[:2], axis=1) < 120]
    if len(dst) < 500 or len(src_pts) < 200:
        return None, {"accepted": False, "reason": "too few points"}
    tree = cKDTree(dst)
    s, R, t, maxd = 1.0, np.eye(3), np.zeros(3), max_start
    d0 = np.median(tree.query(src_pts)[0])
    for _ in range(iters):
        X = s * src_pts @ R.T + t
        d, idx = tree.query(X)
        keep = (d < maxd) & (d <= np.percentile(d, 50))
        if keep.sum() < 50:
            break
        s, R, t = umeyama(src_pts[keep], dst[idx[keep]])
        maxd = max(max_end, maxd * 0.85)
    d1 = np.median(tree.query(s * src_pts @ R.T + t)[0])
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    rep = {"median_dist_before": round(float(d0), 2), "median_dist_after": round(float(d1), 2),
           "scale": round(float(s), 3), "rotation_deg": round(rot, 1),
           "shift_m": round(float(np.linalg.norm(s * c @ R.T + t - c)), 2)}
    ok = abs(s - 1) <= max_scale_change and rot <= max_rot_deg and d1 <= (1 - min_gain) * d0
    rep["accepted"] = bool(ok)
    return (np.hstack([s * R, t[:, None]]) if ok else None), rep


def merge(workdir: Path, min_images=10, inlier_m=20.0, icp=False):
    geo = json.loads((workdir / "georef.json").read_text())
    main_geo = geo.get("main", geo)
    items = json.loads((workdir / "manifest.json").read_text())["items"]
    origin, pos = camera_positions_enu(items)
    if origin != main_geo["origin"]:
        origin = main_geo["origin"]  # same manifest: identical in practice; keep the main frame authoritative
        from .geo import geodetic_to_enu
        pos = {it["file"]: (geodetic_to_enu(it["camera_location"]["lat"], it["camera_location"]["lon"],
                                            it["camera_location"].get("alt") or origin["alt"], origin["lat"],
                                            origin["lon"], origin["alt"]), it["camera_location"].get("alt") is not None)
               for it in items if it.get("camera_location")}
    main_dir = workdir / main_geo["model_dir"]
    main = pycolmap.Reconstruction(str(main_dir))
    sm, Rm, tm = _sim(main_geo)
    in_main = {main.images[i].name for i in main.reg_image_ids()}
    plain_items = [{k: v for k, v in it.items() if k != "anchor"} for it in items]  # secondaries fit on geotags

    merged_info, residuals = [], dict(main_geo.get("per_image_residual_m", {}))
    for d in sorted((workdir / "colmap" / "sparse").iterdir(), key=lambda p: p.name):
        if d.resolve() == main_dir.resolve() or not (d / "images.bin").exists():
            continue
        sec = pycolmap.Reconstruction(str(d))
        names = [sec.images[i].name for i in sec.reg_image_ids()]
        new = [n for n in names if n not in in_main]
        located = [n for n in new if n in pos]
        if len(new) < min_images or len(located) < 5:
            continue
        try:
            # horizontal fit only: phone altitudes are noisy and may not share the anchors' vertical datum
            g = georeference(workdir, sec, str(d.relative_to(workdir)), plain_items, pos, origin, False, inlier_m,
                             None, write=False, force_horizontal=True)
        except SystemExit as e:
            log(f"  model {d.name}: cannot georeference ({e})")
            continue
        n_in, n_loc = g.get("num_inliers", 0), g.get("num_located_registered", 0)
        if not g.get("aligned") or n_in < 5 or n_in < 0.4 * n_loc:
            log(f"  model {d.name}: geotags too inconsistent ({n_in}/{n_loc} within {inlier_m:g} m); skipped")
            continue
        if g.get("mean_camera_up_tilt_deg", 0) > 75:  # upright photos: image-up close to the sky
            log(f"  model {d.name}: frame not upright after levelling "
                f"(camera up tilt {g['mean_camera_up_tilt_deg']:.0f} deg); skipped")
            continue
        ss, Rs, ts = _sim(g)
        # height: put the group's ground level on the main model's ground level at the same place
        gp = np.array([ss * Rs @ p.xyz + ts for p in sec.points3D.values()])
        mp = np.array([sm * Rm @ p.xyz + tm for p in main.points3D.values()])
        near = mp[np.linalg.norm(mp[:, :2] - np.median(gp[:, :2], 0), axis=1) < 60]
        if len(near) > 100:
            ts = ts + np.array([0.0, 0.0, np.percentile(near[:, 2], 5) - np.percentile(gp[:, 2], 5)])
        # secondary model coords -> ENU -> main model coords:  X_m = (Rm^T (ss Rs X + ts - tm)) / sm
        M = np.hstack([Rm.T @ Rs * (ss / sm), (Rm.T @ (ts - tm) / sm)[:, None]])
        sec.transform(pycolmap.Sim3d(M))
        icp_rep = None
        if icp:  # refine against the main model's points (both now in main-model coordinates)
            src_pts = np.array([p.xyz for p in sec.points3D.values() if p.track.length() >= 3 and p.error < 2])
            dst_pts = np.array([p.xyz for p in main.points3D.values() if p.track.length() >= 3 and p.error < 1.5])
            T, icp_rep = icp_refine(src_pts, dst_pts) if len(src_pts) else (None, {"accepted": False})
            if T is not None:
                sec.transform(pycolmap.Sim3d(T))
            log(f"  model {d.name}: ICP {'accepted' if T is not None else 'rejected'} {icp_rep}")
        # drop degenerate registrations (e.g. telephoto shots SfM placed at "infinity"): far from the group's centre
        cen = {sec.images[i].name: sec.images[i].projection_center() for i in sec.reg_image_ids() if sec.images[i].name in new}
        ctr = np.median(np.array(list(cen.values())), 0)
        dist = {n: float(np.linalg.norm(c - ctr)) for n, c in cen.items()}
        lim = max(8 * float(np.median(list(dist.values()))), 300.0)
        far = sorted(n for n, dd in dist.items() if dd > lim)
        if far:
            log(f"  model {d.name}: dropped {len(far)} cameras placed > {lim:.0f} m from the group (degenerate)")
        added = _append(main, sec, set(new) - set(far))
        for n, r in (g.get("per_image_residual_m") or {}).items():
            if n in added:
                residuals[n] = r
        info = {"model": d.name, "images": len(added), "names": sorted(added), "dropped_far": far,
                "num_inliers": g["num_inliers"],
                "num_located": g["num_located_registered"], "residual_m": g["residual_m"],
                "levelled_from_camera_axes_deg": g.get("levelled_from_camera_axes_deg"), "icp": icp_rep}
        merged_info.append(info)
        in_main |= added
        log(f"  merged model {d.name}: {len(added)} images, georef {g['num_inliers']}/{g['num_located_registered']} "
            f"geotags, median residual {g['residual_m']['median']:.1f} m")

    if not merged_info:
        log("no secondary model to merge")
        return geo
    out_dir = workdir / "colmap" / "merged"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    main.write(str(out_dir))
    out = dict(main_geo)
    out.update({"main": main_geo, "model_dir": str(out_dir.relative_to(workdir)), "merged_models": merged_info,
                "num_registered": main.num_reg_images(), "per_image_residual_m": residuals})
    (workdir / "georef.json").write_text(json.dumps(out, indent=1))
    log(f"merged model: {main.num_reg_images()} images -> {out_dir}")
    return out


def _append(dst, src, names):
    """Add src's images (by name) with their cameras and 3D points to dst (ids renumbered as needed)."""
    added, id_map = set(), {}
    next_cam = max(dst.cameras) + 1 if dst.cameras else 1
    for iid in src.reg_image_ids():
        im = src.images[iid]
        if im.name not in names:
            continue
        cam = src.cameras[im.camera_id]
        new_cam = pycolmap.Camera(model=cam.model_name, width=cam.width, height=cam.height, params=cam.params)
        new_cam.camera_id = next_cam
        next_cam += 1
        dst.add_camera_with_trivial_rig(new_cam)
        new_id = iid if iid not in dst.images else max(dst.images) + 1
        img = pycolmap.Image(name=im.name, keypoints=np.array([p.xy for p in im.points2D], np.float64).reshape(-1, 2),
                             camera_id=new_cam.camera_id, image_id=new_id)
        dst.add_image_with_trivial_frame(img, _cam_from_world(im))
        id_map[iid] = new_id
        added.add(im.name)
    for p in src.points3D.values():
        els = [(id_map[e.image_id], e.point2D_idx) for e in p.track.elements if e.image_id in id_map]
        if len(els) < 2:
            continue
        pid = dst.add_point3D(p.xyz, pycolmap.Track(), p.color)
        for iid, k in els:
            dst.add_observation(pid, pycolmap.TrackElement(iid, k))
        dst.points3D[pid].error = p.error
    return added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--min-images", type=int, default=10, help="ignore secondary models smaller than this")
    ap.add_argument("--inlier-m", type=float, default=20.0, help="geotag RANSAC threshold for secondary models")
    ap.add_argument("--icp", action="store_true",
                    help="refine each group against the main point cloud (trimmed similarity ICP, bounded; kept only "
                         "if it clearly reduces the distance)")
    a = ap.parse_args(argv)
    merge(a.workdir, a.min_images, a.inlier_m, a.icp)


if __name__ == "__main__":
    main()
