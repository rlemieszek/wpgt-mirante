"""Register extra Commons photos of the same area into an existing (anchor) reconstruction.

The anchor model (from `reconstruct`) is kept fixed: its camera poses do not move, and the
georeferencing sim3 computed from the anchors' geotags is reused unchanged. Extra photos are
located only by image matching against the anchors; their own geotags (usually metres to tens of
metres off) are used to choose which anchors to match against and reported as an offset afterwards,
never as a constraint.

Candidates come from
  * categories (--category, recursed --depth levels; WPGT sets and names matching --exclude are skipped)
  * a GeoData grid search over the anchors' footprint (--geo-margin metres around it; --no-geo to skip)

Reads   <workdir>/manifest.json, georef.json, colmap/  (from reconstruct)
Writes  <workdir>/extra/manifest.json, <workdir>/images/<extra files>
        <workdir>/colmap/extended/{database.db, pairs.txt, model/}
        <workdir>/georef.json  (model_dir -> extended model; the anchor georef kept under "base")

Usage:
  python -m mirante.extend work/rui-barbosa --category "Category:Praça Rui Barbosa (Belo Horizonte)" --depth 2
  python -m mirante.export work/rui-barbosa site/rui-barbosa
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from . import harvest
from .geo import geodetic_to_enu
from .reconstruct import _cam_from_world, log, set_focal_priors

DEFAULT_EXCLUDE = (r"^Category:WPGT", r"Collections of", r"Interior", r"Exhibits? (in|of)")
MIN_LONG_SIDE = 800
MIN_INLIERS = 25


# --------------------------------------------------------------------------- discovery

def category_files(c: harvest.Commons, categories, depth, exclude) -> dict[int, str]:
    rx = [re.compile(p, re.I) for p in exclude]
    seen, files = set(), {}
    frontier = [(cat if cat.startswith("Category:") else "Category:" + cat, 0) for cat in categories]
    while frontier:
        cat, d = frontier.pop()
        if cat in seen or any(r.search(cat) for r in rx):
            continue
        seen.add(cat)
        for data in c.query_all(list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit="500"):
            for m in data["query"]["categorymembers"]:
                if m["ns"] == 6:
                    files[m["pageid"]] = m["title"]
                elif m["ns"] == 14 and d < depth:
                    frontier.append((m["title"], d + 1))
    log(f"categories: {len(files)} files in {len(seen)} categories")
    return files


def geo_files(c: harvest.Commons, lat_min, lat_max, lon_min, lon_max, step_m=100.0) -> dict[int, str]:
    hits = harvest.geosearch_grid(c, lat_min, lat_max, lon_min, lon_max, step_m)
    return {k: v[0] for k, v in hits.items()}


# --------------------------------------------------------------------------- pairing

def _enu(loc, origin):
    return np.asarray(geodetic_to_enu(loc["lat"], loc["lon"], origin["alt"], origin["lat"], origin["lon"], origin["alt"]))


def _farthest_point_sample(pts, k, seed_idx=0):
    k = min(k, len(pts))
    chosen = [seed_idx]
    d = np.linalg.norm(pts - pts[seed_idx], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return chosen


def view_geometry(rec, s, R, t):
    """Per registered image: ENU centre, 'footprint' (median ENU of the 3D points it observes),
    optical-axis azimuth and pitch (degrees)."""
    names, centers, feet, az, pitch = [], [], [], [], []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        xyz = [rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()]
        if not xyz:
            continue
        ax = R @ _cam_from_world(im).rotation.matrix()[2]  # camera z axis in ENU
        names.append(im.name)
        centers.append(s * R @ im.projection_center() + t)
        feet.append(s * R @ np.median(np.asarray(xyz), 0) + t)
        az.append(np.degrees(np.arctan2(ax[0], ax[1])) % 360)
        pitch.append(np.degrees(np.arcsin(np.clip(ax[2], -1, 1))))
    return names, np.asarray(centers), np.asarray(feet), np.asarray(az), np.asarray(pitch)


def anchor_geometry(rec, s, R, t):
    return view_geometry(rec, s, R, t)[:3]


def first_pass_pairs(extras, origin, a_names, a_centers, a_feet, n_near=30, n_spread=10, n_blind=60):
    pairs = set()
    spread = [a_names[i] for i in _farthest_point_sample(a_feet[:, :2], n_spread)]
    blind = [a_names[i] for i in _farthest_point_sample(a_feet[:, :2], n_blind)]
    ex_xy = {}
    for it in extras:
        cam, obj = it.get("camera_location"), it.get("object_location")
        if cam:
            p = _enu(cam, origin)
            ex_xy[it["file"]] = p[:2]
            target = p.copy()
            if cam.get("heading") is not None:  # look ~20 m ahead along the recorded heading
                h = np.radians(cam["heading"])
                target[:2] += 20.0 * np.array([np.sin(h), np.cos(h)])
            near_feet = np.argsort(np.linalg.norm(a_feet[:, :2] - target[:2], axis=1))[:n_near]
            near_cams = np.argsort(np.linalg.norm(a_centers[:, :2] - p[:2], axis=1))[:n_near // 3]
            cand = [a_names[i] for i in (*near_feet, *near_cams)] + spread
        elif obj:
            p = _enu(obj, origin)
            cand = [a_names[i] for i in np.argsort(np.linalg.norm(a_feet[:, :2] - p[:2], axis=1))[:n_near]] + spread
        else:
            cand = blind
        pairs.update((it["file"], a) for a in cand)
    # extras among themselves (geotagged only): lets a photo register via another extra
    names = list(ex_xy)
    if len(names) > 1:
        xy = np.asarray([ex_xy[n] for n in names])
        for i, n in enumerate(names):
            d = np.linalg.norm(xy - xy[i], axis=1)
            for j in np.argsort(d)[1:9]:
                if d[j] < 150:
                    pairs.add(tuple(sorted((n, names[j]))))
    return pairs


def verified_inliers(db_path) -> dict[str, dict[str, int]]:
    """Adjacency of geometrically verified pairs: name -> {other name: inlier count}."""
    con = sqlite3.connect(db_path)
    names = dict(con.execute("SELECT image_id, name FROM images"))
    out: dict[str, dict[str, int]] = {}
    for pair_id, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
        a, b = (names[i] for i in pycolmap.pair_id_to_image_pair(pair_id))
        out.setdefault(a, {})[b] = rows
        out.setdefault(b, {})[a] = rows
    con.close()
    return out


def second_pass_pairs(extras, inl, anchor_set, done, top=3, per=10):
    """Extras that matched some anchors: also try those anchors' strongest anchor neighbours."""
    def strongest(name, k, min_n=0):
        nb = inl.get(name, {})
        return sorted((b for b in nb if b in anchor_set and nb[b] >= min_n), key=nb.get, reverse=True)[:k]

    pairs = set()
    for it in extras:
        f = it["file"]
        for a in strongest(f, top, MIN_INLIERS):
            pairs.update((f, b) for b in strongest(a, per) if (f, b) not in done)
    return pairs


def match_pairs(db_path, pairs, pairs_path):
    pairs_path.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(pairs_path)
    pycolmap.match_image_pairs(db_path, pairing_options=po)


# --------------------------------------------------------------------------- main steps

def discover(workdir: Path, categories, depth, exclude, geo, geo_margin, width, geo_step,
             require_location=False, exclude_title=()):
    man = json.loads((workdir / "manifest.json").read_text())
    anchor_ids = {it["pageid"] for it in man["items"]}
    c = harvest.Commons()
    found = category_files(c, categories, depth, exclude) if categories else {}
    if geo:
        locs = [it["camera_location"] for it in man["items"] if it.get("camera_location")]
        lat = np.array([l["lat"] for l in locs])
        lon = np.array([l["lon"] for l in locs])
        mlat = geo_margin / 111_320.0
        mlon = geo_margin / (111_320.0 * np.cos(np.radians(lat.mean())))
        found.update(geo_files(c, lat.min() - mlat, lat.max() + mlat, lon.min() - mlon, lon.max() + mlon, geo_step))
    files = [{"pageid": k, "title": v} for k, v in found.items() if k not in anchor_ids]
    log(f"{len(files)} candidate files after removing anchors")
    items = harvest.fetch_details(c, files, width)
    items = [it for it in items if max(it["original_size"]) >= MIN_LONG_SIDE]
    if require_location:
        items = [it for it in items if it["camera_location"]]
    for rx in exclude_title:
        items = [it for it in items if not re.search(rx, it["title"])]
    log(f"{len(items)} candidates are images >= {MIN_LONG_SIDE}px, "
        f"{sum(1 for it in items if it['camera_location'])} with a camera location")
    (workdir / "extra").mkdir(exist_ok=True)
    (workdir / "extra" / "manifest.json").write_text(json.dumps(
        {"categories": categories, "geosearch": geo, "anchor_category": man.get("category"),
         "harvested": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "thumb_width": width, "items": items},
        ensure_ascii=False, indent=1))
    harvest.download(c, items, workdir / "images")
    return items


def register(workdir: Path, max_image_size=1600, max_features=8192):
    geo = json.loads((workdir / "georef.json").read_text())
    base = geo.get("base", geo)  # re-running extend starts again from the anchor model
    if not base.get("aligned"):
        raise SystemExit("anchor model is not georeferenced; run reconstruct first")
    img_dir = workdir / "images"
    extras = [it for it in json.loads((workdir / "extra" / "manifest.json").read_text())["items"]
              if (img_dir / it["file"]).exists()]
    base_dir = workdir / base["model_dir"]
    rec0 = pycolmap.Reconstruction(str(base_dir))
    s, R, t = base["sim3"]["scale"], np.array(base["sim3"]["R"]), np.array(base["sim3"]["t"])
    origin = base["origin"]

    xdir = workdir / "colmap" / "extended"
    if xdir.exists():
        shutil.rmtree(xdir)
    xdir.mkdir(parents=True)
    db_path = xdir / "database.db"
    shutil.copy(workdir / "colmap" / "database.db", db_path)  # anchor features + matches, same image ids

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    ext.sift.max_num_features = max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    log(f"extracting features for {len(extras)} extra images ...")
    pycolmap.extract_features(db_path, img_dir, image_names=[it["file"] for it in extras],
                              camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=reader, extraction_options=ext)
    with pycolmap.Database.open(db_path) as db:
        by_file = {it["file"]: it for it in extras}
        n_focal = set_focal_priors(db, by_file)
    log(f"focal priors on {n_focal} extra images")

    a_names, a_centers, a_feet = anchor_geometry(rec0, s, R, t)
    anchor_set = set(a_names)
    pairs = first_pass_pairs(extras, origin, a_names, a_centers, a_feet)
    log(f"matching pass 1: {len(pairs)} pairs ...")
    match_pairs(db_path, pairs, xdir / "pairs1.txt")
    inl = verified_inliers(db_path)
    done = pairs | {(b, a) for a, b in pairs}
    pairs2 = second_pass_pairs(extras, inl, anchor_set, done)
    if pairs2:
        log(f"matching pass 2: {len(pairs2)} pairs around matched anchors ...")
        match_pairs(db_path, pairs2, xdir / "pairs2.txt")
        inl = verified_inliers(db_path)
    linked = {it["file"] for it in extras
              if any(n >= MIN_INLIERS and b in anchor_set for b, n in inl.get(it["file"], {}).items())}
    log(f"{len(linked)}/{len(extras)} extras have >= {MIN_INLIERS} verified inliers with an anchor")

    log("registering extras into the fixed anchor model ...")
    rec = continue_mapping(db_path, img_dir, base_dir, xdir / "sparse", xdir / "model")
    return summarize(workdir, base, rec0, rec, extras, xdir / "model", {"num_linked": len(linked)})


def continue_mapping(db_path, img_dir, input_dir, sparse_dir, model_dir):
    """Incremental mapping that starts from `input_dir` and keeps all its frames fixed."""
    opts = pycolmap.IncrementalPipelineOptions()
    opts.fix_existing_frames = True      # anchors (and earlier registrations) keep their poses
    opts.multiple_models = False
    opts.use_prior_position = False      # extras' geotags are not trusted
    opts.ba_refine_focal_length = True
    maps = pycolmap.incremental_mapping(db_path, img_dir, sparse_dir, opts, input_path=str(input_dir))
    if not maps:
        raise SystemExit("registration failed")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    rec.write(str(model_dir))
    return rec


def summarize(workdir, base, rec0, rec, extras, model_dir, extra_fields, prev=None):
    """Check that anchors (and `prev` registrations) did not move; write georef.json."""
    s, R, t = base["sim3"]["scale"], np.array(base["sim3"]["R"]), np.array(base["sim3"]["t"])
    origin = base["origin"]
    fixed = {rec0.images[i].name: rec0.images[i].projection_center() for i in rec0.reg_image_ids()}
    if prev is not None:
        fixed.update({prev.images[i].name: prev.images[i].projection_center() for i in prev.reg_image_ids()})
    anchors = {rec0.images[i].name for i in rec0.reg_image_ids()}
    extra_files = {it["file"] for it in extras}
    moved, extra_names, late_anchors = [], [], []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        if im.name in fixed:
            moved.append(np.linalg.norm(s * (im.projection_center() - fixed[im.name])))
        if im.name in extra_files:
            extra_names.append(im.name)
        elif im.name not in anchors:  # anchor that failed in reconstruct, placed by matching only (geotag not used)
            late_anchors.append(im.name)
    moved = np.asarray(moved) if moved else np.zeros(1)
    log(f"  registered {len(extra_names)}/{len(extras)} extras; drift of fixed cameras max {moved.max():.4f} m"
        + (f"; {len(late_anchors)} previously unregistered anchors placed by matching" if late_anchors else ""))

    # extras: SfM position vs their own Commons geotag (horizontal; their altitudes are rarely meaningful)
    by_file = {it["file"]: it for it in extras}
    offsets = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        loc = by_file.get(im.name, {}).get("camera_location")
        if loc:
            C = s * R @ im.projection_center() + t
            offsets[im.name] = round(float(np.linalg.norm(C[:2] - _enu(loc, origin)[:2])), 2)
    if offsets:
        v = np.array(list(offsets.values()))
        log(f"  extras' geotag offset from SfM: median {np.median(v):.1f} m, p90 {np.percentile(v, 90):.1f} m")

    geo = json.loads((workdir / "georef.json").read_text())
    extra = dict(geo.get("extra") or {})
    extra.update(extra_fields)
    extra.update({"num_candidates": len(extras), "num_registered": len(extra_names),
                  "registered": sorted(extra_names), "late_anchors": sorted(late_anchors),
                  "anchor_drift_max_m": float(moved.max()), "geotag_offset_m": offsets})
    out = dict(base)
    out.update({"base": base, "extended": True, "model_dir": str(model_dir.relative_to(workdir)),
                "num_images_total": base["num_images_total"] + len(extras),
                "num_registered": rec.num_reg_images(), "extra": extra})
    (workdir / "georef.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- learned bridge (ALIKED + LightGlue)
#
# For photos SIFT cannot register (typically ground-level views against aerial anchors). Learned matches
# between a ground photo and a registered drone view rarely land on the drone view's existing SIFT 3D
# points (they tend to fall on distant, small-scale structure), so the drone-side keypoints are lifted to
# 3D instead: the drone view is matched with two co-visible registered views and those keypoints are
# triangulated from the fixed poses. The ground photo's pose then comes from PnP on these 2D-3D matches,
# and it is inserted into the model with the lifted points as shared tracks. Existing cameras never move.

def bridge_pairs(todo, origin, geom, n_oblique=12, n_nadir=3, radius=120.0, oblique_pitch=-65.0):
    """Registered views most likely to show what a located photo shows: oblique views whose footprint is
    near its geotag (within +-60 deg of its heading when known, else spread over viewing azimuths), plus
    a few nadir views."""
    names, _, feet, az, pitch = geom
    obl = np.flatnonzero(pitch > oblique_pitch)
    nad = np.flatnonzero(pitch <= oblique_pitch)
    pairs = set()
    for it in todo:
        cam = it["camera_location"]
        g = _enu(cam, origin)[:2]
        d = np.linalg.norm(feet[:, :2] - g, axis=1)
        o = obl[d[obl] < radius]
        o = o[np.argsort(d[o])]
        if cam.get("heading") is not None:
            dh = np.abs((az[o] - cam["heading"] + 180) % 360 - 180)
            chosen = list(o[dh < 60][:n_oblique])
        else:  # nearest view per 45-degree azimuth bin, then the nearest overall
            bins = (az[o] // 45).astype(int)
            chosen = [o[bins == b][0] for b in range(8) if (bins == b).any()]
            chosen += [i for i in o if i not in chosen][:max(0, n_oblique - len(chosen))]
        chosen += list(nad[np.argsort(d[nad])][:n_nadir])
        pairs.update((it["file"], names[i]) for i in chosen)
    return pairs


def helper_views(rec, image_id, k=2, min_baseline=3.0):
    """The k registered views sharing most 3D points with image_id (with some baseline)."""
    cnt = Counter()
    for p in rec.images[image_id].points2D:
        if p.has_point3D():
            for el in rec.points3D[p.point3D_id].track.elements:
                cnt[el.image_id] += 1
    cnt.pop(image_id, None)
    c0 = rec.images[image_id].projection_center()
    out = [i for i, _ in cnt.most_common(20) if np.linalg.norm(rec.images[i].projection_center() - c0) > min_baseline]
    return out[:k]


def _triangulate(P1, P2, x1, x2):
    """Linear triangulation of normalized image points x1, x2 (N, 2) with 3x4 cam_from_world P1, P2."""
    A = np.stack([x1[:, :1] * P1[2] - P1[0], x1[:, 1:] * P1[2] - P1[1],
                  x2[:, :1] * P2[2] - P2[0], x2[:, 1:] * P2[2] - P2[1]], axis=1)
    X = np.linalg.svd(A)[2][:, -1]
    return X[:, :3] / X[:, 3:]


def lift_keypoints(rec, t_id, helpers, pair_matches, kp, max_reproj=2.0, min_angle_deg=1.0):
    """3D positions (model frame) for learned keypoints of registered image t_id, from its matches with
    registered helper views. Returns {kp index in t: (xyz, helper id, helper kp index)}."""
    it = rec.images[t_id]
    ct = rec.cameras[it.camera_id]
    Pt = _cam_from_world(it).matrix()
    Ct = it.projection_center()
    out = {}
    for h in helpers:
        m = pair_matches.get(h)
        if m is None or len(m) < 8:
            continue
        ih = rec.images[h]
        ch = rec.cameras[ih.camera_id]
        Ph = _cam_from_world(ih).matrix()
        xt, xh = kp[it.name][m[:, 0]].astype(np.float64), kp[ih.name][m[:, 1]].astype(np.float64)
        X = _triangulate(Pt, Ph, ct.cam_from_img(xt), ch.cam_from_img(xh))
        ok = np.ones(len(X), bool)
        for P, cam, x in ((Pt, ct, xt), (Ph, ch, xh)):
            Xc = X @ P[:, :3].T + P[:, 3]
            ok &= Xc[:, 2] > 0
            proj = cam.img_from_cam(np.where(ok[:, None], Xc, 1.0))
            ok &= np.linalg.norm(proj - x, axis=1) < max_reproj
        r1, r2 = X - Ct, X - ih.projection_center()
        cosang = (r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
        ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > min_angle_deg
        for j in np.flatnonzero(ok):
            out.setdefault(int(m[j, 0]), (X[j], h, int(m[j, 1])))
    return out


def bridge(workdir: Path, resize=1280, max_keypoints=4096, features="aliked", only_title=None,
           min_inliers=30, min_views=2, max_geotag_offset_m=75.0, height_range_m=(-5.0, 60.0)):
    """Second chance for located extras SIFT could not register: ALIKED + LightGlue against registered
    views, with lifted 3D points and PnP (see the section comment)."""
    geo = json.loads((workdir / "georef.json").read_text())
    if not geo.get("extended"):
        raise SystemExit("run extend (SIFT registration) first")
    base = geo["base"]
    img_dir = workdir / "images"
    xdir = workdir / "colmap" / "extended"
    extras = [it for it in json.loads((workdir / "extra" / "manifest.json").read_text())["items"]
              if (img_dir / it["file"]).exists()]
    by_file = {it["file"]: it for it in extras}
    rec0 = pycolmap.Reconstruction(str(workdir / base["model_dir"]))
    prev_dir = workdir / geo["model_dir"]
    prev = pycolmap.Reconstruction(str(prev_dir))
    rec = pycolmap.Reconstruction(str(prev_dir))
    s, R, t = base["sim3"]["scale"], np.array(base["sim3"]["R"]), np.array(base["sim3"]["t"])
    origin = base["origin"]
    name_id = {rec.images[i].name: i for i in rec.reg_image_ids()}
    todo = [it for it in extras if it["file"] not in name_id and it.get("camera_location")
            and (only_title is None or re.search(only_title, it["title"], re.I))]
    geom = view_geometry(rec, s, R, t)
    ut_pairs = sorted(bridge_pairs(todo, origin, geom))
    targets = sorted({b for _, b in ut_pairs})
    helpers = {tn: [rec.images[h].name for h in helper_views(rec, name_id[tn])] for tn in targets}
    th_pairs = sorted({(tn, hn) for tn, hs in helpers.items() for hn in hs})
    pairs = ut_pairs + th_pairs
    lg_dir = xdir / "learned"
    lg_dir.mkdir(exist_ok=True)
    (lg_dir / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs))
    log(f"learned bridge: {len(todo)} unregistered located extras, {len(ut_pairs)} photo-view pairs, "
        f"{len(th_pairs)} view-helper pairs for lifting")
    subprocess.run([sys.executable, "-m", "mirante.learned", str(img_dir), str(lg_dir / "pairs.txt"),
                    str(lg_dir), "--features", features, "--resize", str(resize),
                    "--max-keypoints", str(max_keypoints)], check=True)

    z = np.load(lg_dir / "matches.npz")

    class _KP(dict):
        def __missing__(self, n):
            self[n] = np.load(lg_dir / "feats" / (n + ".npz"))["keypoints"]
            return self[n]
    kp = _KP()
    # local ground level for the height check: low percentile of model points near each photo's geotag
    enu_pts = np.array([s * R @ p.xyz + t for p in rec.points3D.values()])
    ground_tree = cKDTree(enu_pts[:, :2])

    def ground_z(xy, r=60.0):
        idx = ground_tree.query_ball_point(xy, r)
        return float(np.percentile(enu_pts[idx, 2], 5)) if len(idx) >= 20 else None

    m_of = {tuple(p): z[f"m{i}"] for i, p in enumerate(pairs)}
    lifted = {}
    for tn in targets:
        pm = {name_id[hn]: m_of[(tn, hn)] for hn in helpers[tn]}
        lifted[tn] = lift_keypoints(rec, name_id[tn], list(pm), pm, kp)
    log(f"  lifted {sum(map(len, lifted.values()))} keypoints on {len(targets)} registered views")

    with pycolmap.Database.open(xdir / "database.db") as db:
        db_ids = {im.name: im.image_id for im in db.read_all_images()}
    by_u = {}
    for u, tn in ut_pairs:
        by_u.setdefault(u, []).append(tn)
    est = pycolmap.AbsolutePoseEstimationOptions()
    ref = pycolmap.AbsolutePoseRefinementOptions()
    point_of = {}  # (target name, target kp) -> point3D id, shared between photos
    newly, stats, diag = [], Counter(), {}
    for u, tns in by_u.items():
        it = by_file[u]
        x2d, x3d, src = [], [], []
        seen = set()
        for tn in tns:
            m = m_of[(u, tn)]
            for ku, kt in m:
                hit = lifted[tn].get(int(kt))
                if hit is not None and int(ku) not in seen:
                    seen.add(int(ku))
                    x2d.append(kp[u][ku])
                    x3d.append(hit[0])
                    src.append((tn, int(kt), hit[1], hit[2]))
        diag[u] = {"lifted_matches": len(x2d)}
        if len(x2d) < min_inliers:
            stats["too few lifted matches"] += 1
            diag[u]["result"] = "too few lifted matches"
            continue
        w, h = it["thumb_size"]
        f = it.get("focal_prior_px") or 1.2 * max(w, h)
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h, params=[f, w / 2, h / 2, 0.0])
        est.estimate_focal_length = ref.refine_focal_length = not it.get("focal_prior_px")
        res = pycolmap.estimate_and_refine_absolute_pose(np.asarray(x2d, np.float64), np.asarray(x3d, np.float64),
                                                        cam, est, ref)
        if res is not None:
            diag[u]["pnp_inliers"] = int(res["num_inliers"])
        if res is None or res["num_inliers"] < min_inliers:
            stats["PnP failed"] += 1
            diag[u]["result"] = "PnP failed"
            continue
        inl = np.asarray(res["inlier_mask"], bool)
        diag[u].update(pnp_inliers=int(inl.sum()), views=len({src[k][0] for k in np.flatnonzero(inl)}))
        if len({src[k][0] for k in np.flatnonzero(inl)}) < min_views:
            stats["inliers from a single view"] += 1
            diag[u]["result"] = "inliers from a single view"
            continue
        if not 0.4 <= cam.params[0] / max(w, h) <= 3.0:
            stats["implausible focal length"] += 1
            diag[u]["result"] = "implausible focal length"
            continue
        pose = res["cam_from_world"]
        C = s * R @ (-pose.rotation.matrix().T @ pose.translation) + t
        g_enu = _enu(it["camera_location"], origin)
        offset = float(np.linalg.norm(C[:2] - g_enu[:2]))
        if offset > max_geotag_offset_m:
            stats["too far from geotag"] += 1
            diag[u]["result"] = "too far from geotag"
            continue
        gz = ground_z(g_enu[:2])
        if gz is not None and not height_range_m[0] <= C[2] - gz <= height_range_m[1]:
            stats["implausible height"] += 1
            diag[u]["result"] = "implausible height"
            continue
        stats["registered"] += 1
        diag[u].update(result="registered", geotag_offset_m=round(offset, 1))
        # insert: camera + image + observations of shared lifted points
        cam.camera_id = max(rec.cameras) + 1
        rec.add_camera_with_trivial_rig(cam)
        pts = np.asarray(x2d, np.float64)[inl]
        image = pycolmap.Image(name=u, keypoints=pts, camera_id=cam.camera_id, image_id=db_ids[u])
        rec.add_image_with_trivial_frame(image, pose)
        for k, (tn, kt, h, kh) in enumerate(np.asarray(src, object)[inl]):
            key = (tn, kt)
            if key not in point_of:
                xyz = np.asarray(x3d, np.float64)[inl][k]
                pid = rec.add_point3D(xyz, pycolmap.Track(), np.array([200, 200, 200], np.uint8))
                for img_name, kpi in ((tn, kt), (rec.images[h].name, kh)):
                    ti = rec.images[name_id[img_name]]
                    ti.points2D.append(pycolmap.Point2D(kp[img_name][kpi].astype(np.float64)))
                    rec.add_observation(pid, pycolmap.TrackElement(ti.image_id, len(ti.points2D) - 1))
                point_of[key] = pid
            rec.add_observation(point_of[key], pycolmap.TrackElement(db_ids[u], k))
        newly.append(u)
    log("  learned bridge: " + ", ".join(f"{k} {v}" for k, v in stats.most_common()))
    model_dir = xdir / "model_learned"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir()
    rec.write(str(model_dir))
    return summarize(workdir, base, rec0, rec, extras, model_dir,
                     {"learned": {"features": f"{features}+lightglue", "pairs": len(ut_pairs),
                                  "stats": dict(stats), "newly_registered": sorted(newly),
                                  "per_photo": diag}}, prev=prev)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--category", action="append", default=[], help="category to pull extras from (repeatable)")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDE),
                    help="regex; skip categories whose title matches (repeatable)")
    ap.add_argument("--no-geo", action="store_true", help="skip the GeoData search around the anchors")
    ap.add_argument("--geo-margin", type=float, default=50.0, help="metres around the anchors' extent")
    ap.add_argument("--geo-step", type=float, default=100.0)
    ap.add_argument("--width", type=int, default=1920, choices=harvest.STANDARD_WIDTHS)
    ap.add_argument("--skip-harvest", action="store_true", help="reuse <workdir>/extra/manifest.json")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--require-location", action="store_true", help="only extras with a camera location")
    ap.add_argument("--exclude-title", action="append", default=[], help="regex; drop extras whose title matches")
    ap.add_argument("--learned", action="store_true",
                    help="after SIFT, try ALIKED+LightGlue for the extras SIFT could not register")
    ap.add_argument("--learned-only", action="store_true", help="only run the learned bridge on the current model")
    ap.add_argument("--learned-resize", type=int, default=1280)
    ap.add_argument("--learned-max-keypoints", type=int, default=4096)
    ap.add_argument("--learned-only-title", metavar="REGEX",
                    help="limit the learned bridge to extras whose title matches (e.g. skip event photos)")
    a = ap.parse_args(argv)
    if not a.learned_only:
        if not a.skip_harvest:
            discover(a.workdir, a.category, a.depth, a.exclude, not a.no_geo, a.geo_margin, a.width, a.geo_step,
                     a.require_location, a.exclude_title)
        register(a.workdir, a.max_image_size)
    if a.learned or a.learned_only:
        bridge(a.workdir, a.learned_resize, a.learned_max_keypoints, only_title=a.learned_only_title)


if __name__ == "__main__":
    main()
