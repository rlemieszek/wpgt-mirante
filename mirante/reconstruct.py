"""Structure-from-motion on a harvested work folder, then georeference to local ENU.

Reads   <workdir>/manifest.json, <workdir>/images/
Writes  <workdir>/colmap/database.db, <workdir>/colmap/sparse/<k>/  (COLMAP models)
        <workdir>/georef.json  (ENU origin + similarity transform model->ENU + residuals)

Usage:
  python -m mirante.reconstruct work/rui-barbosa [--matcher auto|exhaustive|spatial]
         [--use-priors] [--inlier-m 5] [--max-image-size 1600]

--use-priors feeds the Commons camera positions into bundle adjustment as position priors.
Leave it off for ordinary Commons categories (geotags are often metres-to-tens-of-metres off);
turn it on for accurately positioned sets such as WPGT surveys.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

from .geo import geodetic_to_enu

EYE_HEIGHT_M = 1.6


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- geometry helpers

def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_sim2(src, dst, thresh, iters=2000, seed=0):
    """2D similarity (scale, rotation, translation) by RANSAC on 2-point samples + refinement."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 2, replace=False)
        if np.linalg.norm(src[idx[0]] - src[idx[1]]) < 1e-9:
            continue
        s, R, t = umeyama(src[idx], dst[idx])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        inl = res < thresh
        score = (inl.sum(), -np.median(res[inl]) if inl.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, inl)
    inl = best[1]
    for _ in range(5):
        s, R, t = umeyama(src[inl], dst[inl])
        new = np.linalg.norm(s * src @ R.T + t - dst, axis=1) < thresh
        if (new == inl).all() or new.sum() < 2:
            break
        inl = new
    s, R, t = umeyama(src[inl], dst[inl])
    return s, R, t, inl


def robust_sim3(src, dst, thresh, iters=3000, seed=0):
    """RANSAC + refinement. Returns (s, R, t, inlier_mask)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best = None
    for _ in range(iters if n > 3 else 1):
        idx = rng.choice(n, 3, replace=False) if n > 3 else np.arange(n)
        a = src[idx]
        if np.linalg.matrix_rank(a - a.mean(0), tol=1e-6) < 2:
            continue
        s, R, t = umeyama(a, dst[idx])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        inl = res < thresh
        score = (inl.sum(), -np.median(res[inl]) if inl.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, inl)
    if best is None:
        raise RuntimeError("degenerate camera layout (all collinear?)")
    inl = best[1]
    for _ in range(5):  # iterative refinement on inliers
        if inl.sum() < 3:
            break
        s, R, t = umeyama(src[inl], dst[inl])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        new = res < thresh
        if (new == inl).all():
            break
        inl = new
    s, R, t = umeyama(src[inl], dst[inl])
    return s, R, t, inl


def _rot_about(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def fix_roll_if_collinear(centers_enu, ups_enu):
    """If cameras lie almost on a line, rotation about that line is unconstrained by GPS.
    Choose the angle that makes the cameras' up-vectors most vertical. Returns 3x3 rotation (about centroid)."""
    c = centers_enu - centers_enu.mean(0)
    ev, evec = np.linalg.eigh(c.T @ c)
    if ev[-1] <= 0 or ev[-2] / ev[-1] > 0.02:
        return np.eye(3)
    axis = evec[:, -1]
    angs = np.radians(np.arange(-180, 180, 0.25))
    scores = [np.mean((ups_enu @ _rot_about(axis, a).T)[:, 2]) for a in angs]
    return _rot_about(axis, angs[int(np.argmax(scores))])


def level_from_camera_axes(rights, trim=0.2):
    """Handheld photos are mostly taken upright, so their x axes (right) are near horizontal; the vertical is
    the direction most orthogonal to all of them (trimmed to ignore tilted shots). Returns the rotation taking
    that direction to +z and its angle in degrees, or None when the x axes are nearly parallel (e.g. a
    drone grid flown at one heading): then they only constrain the vertical to a plane."""
    w = np.linalg.eigvalsh(rights.T @ rights)
    if w[1] < 0.1 * w[2]:
        return None
    keep = np.ones(len(rights), bool)
    for _ in range(3):
        v = np.linalg.eigh(rights[keep].T @ rights[keep])[1][:, 0]
        up = v * np.sign(v[2])
        dev = np.abs(rights @ up)
        keep = dev <= np.quantile(dev, 1 - trim)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    ang = np.arccos(np.clip(up[2], -1, 1))
    return (_rot_about(axis, ang) if np.linalg.norm(axis) > 1e-9 else np.eye(3)), float(np.degrees(ang))


# --------------------------------------------------------------------------- pipeline

def camera_positions_enu(items):
    located = [it for it in items if it.get("camera_location")]
    if not located:
        return None, {}
    ref = [it for it in located if it.get("anchor")] or located  # origin from the anchors when flagged
    lat0 = float(np.mean([it["camera_location"]["lat"] for it in ref]))
    lon0 = float(np.mean([it["camera_location"]["lon"] for it in ref]))
    alts = [it["camera_location"]["alt"] for it in ref if it["camera_location"].get("alt") is not None]
    alt0 = float(np.median(alts)) if alts else 0.0
    pos = {}
    for it in located:
        loc = it["camera_location"]
        has_alt = loc.get("alt") is not None
        alt = loc["alt"] if has_alt else alt0 + EYE_HEIGHT_M
        pos[it["file"]] = (geodetic_to_enu(loc["lat"], loc["lon"], alt, lat0, lon0, alt0), has_alt)
    return {"lat": lat0, "lon": lon0, "alt": alt0, "alt_known": bool(alts)}, pos


def set_focal_priors(db, by_file) -> int:
    """Write EXIF focal priors into the cameras of the images named in by_file. Returns the count."""
    n = 0
    for im in db.read_all_images():
        it = by_file.get(im.name)
        if it and it.get("focal_prior_px"):
            cam = db.read_camera(im.camera_id)
            scale = cam.width / it["thumb_size"][0]  # in case the file differs from thumb_size
            p = np.array(cam.params, float)
            p[0] = it["focal_prior_px"] * scale
            cam.params = p
            cam.has_prior_focal_length = True
            db.update_camera(cam)
            n += 1
    return n


def run(workdir: Path, matcher="auto", use_priors=False, inlier_m=None, max_image_size=1600,
        max_features=8192, prior_sigma_m=None, prior_sigma_z_m=None, reuse_matches=False, georef_only=False):
    man = json.loads((workdir / "manifest.json").read_text())
    img_dir = workdir / "images"
    items = [it for it in man["items"] if (img_dir / it["file"]).exists()]
    if len(items) < 3:
        raise SystemExit("need at least 3 downloaded images")
    origin, pos = camera_positions_enu(items)
    log(f"{len(items)} images, {len(pos)} with camera positions")
    if georef_only:  # keep the model, redo the georeferencing
        old = json.loads((workdir / "georef.json").read_text())
        rec = pycolmap.Reconstruction(str(workdir / old["model_dir"]))
        return georeference(workdir, rec, old["model_dir"], items, pos, origin, old.get("used_priors_in_ba", False),
                            inlier_m, old.get("prior_sigma_m"))

    cdir = workdir / "colmap"
    db_path = cdir / "database.db"
    reuse_matches = reuse_matches and db_path.exists()
    if reuse_matches:  # keep features + verified matches, redo priors and mapping only
        shutil.rmtree(cdir / "sparse", ignore_errors=True)
        shutil.rmtree(cdir / "extended", ignore_errors=True)
    elif cdir.exists():
        shutil.rmtree(cdir)
    (cdir / "sparse").mkdir(parents=True)

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    ext.sift.max_num_features = max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    if not reuse_matches:
        log("extracting features ...")
        pycolmap.extract_features(db_path, img_dir, image_names=[it["file"] for it in items],
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=reader, extraction_options=ext)

    by_file = {it["file"]: it for it in items}
    if prior_sigma_m is None:
        prior_sigma_m = 2.0 if use_priors else 10.0
    if prior_sigma_z_m is None:
        prior_sigma_z_m = prior_sigma_m
    anchors = {it["file"] for it in items if it.get("anchor")}  # if flagged, only these get priors
    with pycolmap.Database.open(db_path) as db:
        n_focal = set_focal_priors(db, by_file)
        db.clear_pose_priors()
        for im in db.read_all_images():
            if im.name in pos and (not anchors or im.name in anchors):
                xyz, has_alt = pos[im.name]
                cov = np.diag([prior_sigma_m ** 2, prior_sigma_m ** 2, (prior_sigma_z_m if has_alt else 50.0) ** 2])
                db.write_pose_prior(pycolmap.PosePrior(
                    corr_data_id=im.data_id, position=np.asarray(xyz, float), position_covariance=cov,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))
    log(f"focal priors on {n_focal} images; position priors sigma {prior_sigma_m} m (z {prior_sigma_z_m} m)")

    if reuse_matches:
        log("reusing features and matches from colmap/database.db")
    else:
        if matcher == "auto":
            matcher = "exhaustive" if len(items) <= 150 or len(pos) < len(items) * 0.8 else "spatial"
        log(f"matching ({matcher}) ...")
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(db_path)
        else:
            po = pycolmap.SpatialPairingOptions()
            po.ignore_z = not origin["alt_known"]
            po.max_num_neighbors = 40
            po.max_distance = 250.0
            pycolmap.match_spatial(db_path, pairing_options=po)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 3
    opts.multiple_models = True
    if use_priors and pos:
        opts.use_prior_position = True
        opts.use_robust_loss_on_prior_position = True
    log("mapping ...")
    maps = pycolmap.incremental_mapping(db_path, img_dir, cdir / "sparse", opts)
    if not maps:
        raise SystemExit("reconstruction failed: no model")
    ranked = sorted(maps.items(), key=lambda kv: -kv[1].num_reg_images())
    for k, r in ranked:
        log(f"  model {k}: {r.num_reg_images()} images, {r.num_points3D()} points")
    key, rec = ranked[0]
    return georeference(workdir, rec, f"colmap/sparse/{key}", items, pos, origin, bool(use_priors and pos), inlier_m,
                        {"horizontal": prior_sigma_m, "vertical": prior_sigma_z_m} if use_priors and pos else None)


def georeference(workdir, rec, model_dir, items, pos, origin, used_priors, inlier_m, prior_sigma):
    """Similarity model -> ENU from the geotags (RANSAC), plus gravity fixes; writes georef.json."""
    use_priors = used_priors
    georef = {"origin": origin, "model_dir": model_dir, "frame": "ENU metres (x=east, y=north, z=up)",
              "used_priors_in_ba": used_priors, "num_images_total": len(items), "prior_sigma_m": prior_sigma,
              "num_registered": rec.num_reg_images()}
    names, src, dst, ups, rights = [], [], [], [], []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        if im.name in pos:
            names.append(im.name)
            src.append(im.projection_center())
            dst.append(pos[im.name][0])
            R_cw = _cam_from_world(im).rotation.matrix()
            ups.append(-R_cw[1])  # camera -y axis in model coordinates
            rights.append(R_cw[0])
    if len(src) >= 3:
        src, dst, ups, rights = map(np.asarray, (src, dst, ups, rights))
        if inlier_m is None:
            inlier_m = 5.0 if use_priors else 20.0
        anchors = {it["file"] for it in items if it.get("anchor")}  # if flagged, only these drive the fit
        fit = np.array([not anchors or n in anchors for n in names])
        if fit.sum() < 3:
            raise SystemExit("fewer than 3 registered anchors: cannot georeference")
        horizontal = sum(1 for n, f in zip(names, fit) if f and pos[n][1]) < 0.5 * fit.sum()
        lvl = level_from_camera_axes(rights[fit]) if horizontal else None
        if horizontal and lvl is None:
            horizontal = False
            log("  anchors without altitude and camera x axes nearly parallel: 3D fit assuming one flight height")
        if horizontal:
            # The fit cameras have no altitudes (their assumed height would tilt a 3D fit): level from the camera
            # x axes, fit scale/heading/position horizontally, and put z relative to the local ground.
            R_lvl, ang = lvl
            lv = src @ R_lvl.T
            s, R2, t2, inl_fit = robust_sim2(lv[fit, :2], dst[fit, :2], inlier_m)
            R = np.eye(3)
            R[:2, :2] = R2
            R = R @ R_lvl
            pts = np.array([p.xyz for p in rec.points3D.values()]) @ R.T * s
            t = np.array([t2[0], t2[1], -np.percentile(pts[:, 2], 5)])  # ground ~ z = 0
            georef.update(horizontal_fit=True, levelled_from_camera_axes_deg=round(ang, 2),
                          note="anchors had no altitude: z is height above local ground, not ellipsoidal")
            log(f"  anchors without altitude: horizontal fit, levelled from camera x axes ({ang:.1f} deg)")
        else:
            s, R, t, inl_fit = robust_sim3(src[fit], dst[fit], inlier_m)
        inl = np.zeros(len(names), bool)
        inl[np.flatnonzero(fit)] = inl_fit
        centers = s * src @ R.T + t
        R_fix = np.eye(3) if horizontal else fix_roll_if_collinear(centers, ups @ R.T)
        if not np.allclose(R_fix, np.eye(3)):
            mu = centers[inl].mean(0)
            R, t = R_fix @ R, R_fix @ (t - mu) + mu
            centers = s * src @ R.T + t
            log("  cameras nearly collinear: roll fixed from gravity (camera up-vectors)")
        # Ordinary geotags (no --use-priors) barely constrain tilt: altitudes are missing or phone-GPS noisy
        # (metres to tens of metres) over camera spreads of tens of metres. Level the frame from the cameras'
        # x axes instead, which are near horizontal for upright handheld shots and gimbal-stabilised drones.
        lvl = level_from_camera_axes(rights @ R.T) if not use_priors and not horizontal and len(names) >= 10 else None
        if lvl is not None:
            R_lvl, ang = lvl
            mu = centers[inl].mean(0)
            R, t = R_lvl @ R, R_lvl @ (t - mu) + mu
            centers = s * src @ R.T + t
            georef["levelled_from_camera_axes_deg"] = round(ang, 2)
            log(f"  frame levelled from camera x axes ({ang:.1f} deg)")
        res = np.linalg.norm((centers - dst)[:, :2] if horizontal else centers - dst, axis=1)
        tilt = np.degrees(np.arccos(np.clip(np.mean((ups @ R.T)[:, 2]), -1, 1)))
        georef.update({
            "sim3": {"scale": float(s), "R": R.tolist(), "t": t.tolist()},
            "aligned": True, "inlier_threshold_m": inlier_m,
            "num_located_registered": len(src), "num_fit": int(fit.sum()), "num_inliers": int(inl.sum()),
            "residual_m": {"median": float(np.median(res[inl])), "p90": float(np.percentile(res[inl], 90)),
                           "max_inlier": float(res[inl].max())},
            "mean_camera_up_tilt_deg": float(tilt),
            "per_image_residual_m": {n: round(float(r), 3) for n, r in zip(names, res)},
            "outliers": [n for n, ok, f in zip(names, inl, fit) if f and not ok],
        })
        if anchors:
            other = res[~fit]
            georef["non_anchor_geotag_offset_m"] = {"median": float(np.median(other)) if len(other) else None,
                                                   "p90": float(np.percentile(other, 90)) if len(other) else None}
            log(f"  {int((~fit).sum())} located non-anchors: geotag offset median "
                f"{np.median(other) if len(other) else float('nan'):.1f} m")
        log(f"  georef: {inl.sum()}/{int(fit.sum())} inliers, median residual {np.median(res[inl]):.2f} m, "
            f"scale {s:.4f}, up-tilt {tilt:.1f} deg")
    else:
        georef.update({"aligned": False, "sim3": {"scale": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0]}})
        log("  fewer than 3 located images registered: exporting in arbitrary model frame")
    (workdir / "georef.json").write_text(json.dumps(georef, indent=1))
    return georef


def _cam_from_world(im):
    cfw = im.cam_from_world
    return cfw() if callable(cfw) else cfw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--matcher", default="auto", choices=["auto", "exhaustive", "spatial"])
    ap.add_argument("--use-priors", action="store_true", help="use camera positions as BA priors (accurate sets)")
    ap.add_argument("--inlier-m", type=float, help="georef RANSAC threshold in metres (default 5 with priors, 20 without)")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--prior-sigma", type=float, help="position prior std dev in metres (default 2 with --use-priors; "
                                                      "use ~0.05 for RTK-fixed sets)")
    ap.add_argument("--prior-sigma-z", type=float, help="vertical prior std dev (default: --prior-sigma)")
    ap.add_argument("--reuse-matches", action="store_true", help="keep colmap/database.db; redo priors and mapping only")
    ap.add_argument("--georef-only", action="store_true", help="keep the current model; redo georeferencing only")
    a = ap.parse_args(argv)
    run(a.workdir, a.matcher, a.use_priors, a.inlier_m, a.max_image_size, a.max_features,
        a.prior_sigma, a.prior_sigma_z, a.reuse_matches, a.georef_only)


if __name__ == "__main__":
    main()
