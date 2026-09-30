"""Register ground photos into an anchored (drone) model with RoMa dense matching.

Two steps, so the GPU-heavy middle part can run on another machine:

  prepare   for every photo not in the main model, choose the drone views most likely to see what it shows
            (from its merged pose, else from its geotag and heading, else the views that see the subject best),
            write each drone view's query points (its keypoints that have 3D points) and a job file.
            -> <workdir>/dense_job/ (+ a zip), runnable anywhere with `python run_dense.py jobs.json out`
  register  read the RoMa results (dense.npz), turn confident, cycle-consistent samples into 2D-3D matches,
            solve each photo's pose with PnP (LO-RANSAC + refinement) under strict acceptance checks, and add the
            accepted photos to the main model observing the drone model's existing 3D points (shared tracks, so
            the viewer can fly between ground and drone photos). Existing cameras never move.

Usage:
  python -m mirante.densereg prepare work/boa-viagem [--targets 8]
  (run the job: python -m mirante.dense work/boa-viagem/dense_job/jobs.json work/boa-viagem/dense_job/out)
  python -m mirante.densereg register work/boa-viagem [--dense work/boa-viagem/dense_job/out/dense.npz]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pycolmap

from .geo import geodetic_to_enu
from .reconstruct import _cam_from_world, camera_positions_enu

EYE = 1.6


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _sim(g):
    return g["sim3"]["scale"], np.array(g["sim3"]["R"]), np.array(g["sim3"]["t"])


def _load(workdir: Path):
    geo = json.loads((workdir / "georef.json").read_text())
    main_geo = geo.get("main", geo)
    items = json.loads((workdir / "manifest.json").read_text())["items"]
    main = pycolmap.Reconstruction(str(workdir / main_geo["model_dir"]))
    merged = pycolmap.Reconstruction(str(workdir / geo["model_dir"])) if geo.get("main") else None
    return geo, main_geo, items, main, merged


def prepare(workdir: Path, n_targets=8, n_blind=10, fov_deg=70.0, max_depth=120.0, subject=None):
    geo, main_geo, items, main, merged = _load(workdir)
    s, R, t = _sim(main_geo)
    to_enu = lambda X: s * X @ R.T + t
    origin = main_geo["origin"]
    pid_list = np.array(list(main.points3D.keys()))
    P = to_enu(np.array([main.points3D[p].xyz for p in pid_list]))
    row = {p: k for k, p in enumerate(pid_list)}
    aerial = [main.images[i] for i in main.reg_image_ids()]
    obs = {a.image_id: np.array([row[p.point3D_id] for p in a.points2D if p.has_point3D()]) for a in aerial}
    in_main = {a.name for a in aerial}
    by_file = {it["file"]: it for it in items}
    ctr = np.array(subject) if subject is not None else np.median(P, 0)  # what the ground photos look at

    def ground_z(xy, r=40.0):
        near = P[np.linalg.norm(P[:, :2] - xy, axis=1) < r]
        return float(np.percentile(near[:, 2], 5)) if len(near) > 50 else float(np.percentile(P[:, 2], 5))

    def score(C, fwd):
        v = P - C
        d = v @ fwd
        inside = (d > 2) & (d < max_depth) & (d > np.linalg.norm(v, axis=1) * np.cos(np.radians(fov_deg / 2)))
        return sorted(((int(inside[obs[a.image_id]].sum()), a) for a in aerial), key=lambda x: -x[0])

    merged_pose = {}
    if merged is not None:
        for i in merged.reg_image_ids():
            im = merged.images[i]
            if im.name not in in_main:
                merged_pose[im.name] = (to_enu(im.projection_center()), R @ _cam_from_world(im).rotation.matrix()[2])
    # blind targets: the views that see the subject best, spread over viewing azimuth
    near_subject = np.linalg.norm(P - ctr, axis=1) < 30
    vis = sorted(((int(near_subject[obs[a.image_id]].sum()), a) for a in aerial), key=lambda x: -x[0])
    blind, seen_az = [], []
    for n, a in vis:
        c = to_enu(a.projection_center())
        az = np.degrees(np.arctan2(*(ctr - c)[:2][::-1])) % 360 if np.linalg.norm((ctr - c)[:2]) > 1 else 0.0
        if all(abs((az - b + 180) % 360 - 180) > 25 for b in seen_az) or len(blind) >= n_blind // 2:
            blind.append(a)
            seen_az.append(az)
        if len(blind) >= n_blind:
            break

    jdir = workdir / "dense_job"
    if jdir.exists():
        shutil.rmtree(jdir)
    (jdir / "q").mkdir(parents=True)
    (jdir / "ids").mkdir()
    pairs, used, how = [], set(), {"merged pose": 0, "geotag": 0, "no location": 0}
    for it in items:
        n = it["file"]
        if n in in_main or it.get("anchor") or not (workdir / "images" / n).exists():
            continue
        if n in merged_pose:
            C, fwd = merged_pose[n]
            targets = [a for k, a in score(C, fwd)[:n_targets] if k > 0]
            how["merged pose"] += 1
        elif it.get("camera_location"):
            loc = it["camera_location"]
            g = np.asarray(geodetic_to_enu(loc["lat"], loc["lon"], origin["alt"], origin["lat"], origin["lon"], origin["alt"]))
            C = np.array([g[0], g[1], ground_z(g[:2]) + EYE])
            if loc.get("heading") is not None:
                h = np.radians(loc["heading"])
                fwd = np.array([np.sin(h), np.cos(h), 0.25])
            else:
                fwd = ctr + np.array([0, 0, 10.0]) - C
            fwd = fwd / np.linalg.norm(fwd)
            targets = [a for k, a in score(C, fwd)[:n_targets + 2] if k > 0]
            how["geotag"] += 1
        else:
            targets = blind
            how["no location"] += 1
        for a in targets:
            if a.image_id not in used:
                p2 = [(p.xy, p.point3D_id) for p in a.points2D if p.has_point3D()]
                np.save(jdir / "q" / f"{a.image_id}.npy", np.array([x for x, _ in p2], np.float32))
                np.save(jdir / "ids" / f"{a.image_id}.npy", np.array([i for _, i in p2], np.int64))
                used.add(a.image_id)
            pairs.append({"a": a.name, "b": n, "q": f"q/{a.image_id}.npy", "a_id": a.image_id})
    names = sorted({x for p in pairs for x in (p["a"], p["b"])})
    job = {"images": None, "urls": {x: by_file[x]["thumb_url"] for x in names}, "pairs": pairs,
           "made_by": "mirante.densereg prepare", "workdir": workdir.name}
    (jdir / "jobs.json").write_text(json.dumps(job, indent=1))
    shutil.copy(Path(__file__).with_name("dense.py"), jdir / "run_dense.py")
    (jdir / "requirements.txt").write_text("numpy\npillow\nrequests\nromatch @ git+https://github.com/Parskatt/RoMa.git\n"
                                           "# plus torch with CUDA: https://pytorch.org/get-started/locally/\n")
    (jdir / "README.txt").write_text(f"""WPGT Mirante: RoMa dense-matching job ({workdir.name}), {len(pairs)} pairs, {len(names)} images

Run on a machine with an NVIDIA GPU (about 0.5-1 s per pair):

  1. Python 3.10+ in a fresh virtual environment, then PyTorch with CUDA, e.g.
       pip install torch --index-url https://download.pytorch.org/whl/cu124
     (pick the command for your CUDA version at https://pytorch.org/get-started/locally/)
  2. pip install -r requirements.txt
  3. python run_dense.py jobs.json out
     The photos are downloaded from Wikimedia Commons into out/images/ on first run (~{len(names)} files).
     RoMa's weights (~1.7 GB) are downloaded on first use. Progress is printed every 50 pairs, and an
     interrupted run resumes where it stopped.
  4. Send back out/dense.npz (the only result file; out/images/ is not needed).
""")
    zp = workdir / f"dense_job_{workdir.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in jdir.rglob("*"):
            if f.is_file() and "out" not in f.relative_to(jdir).parts:
                z.write(f, Path("dense_job") / f.relative_to(jdir))
    log(f"{len(pairs)} pairs for {sum(how.values())} photos ({how}); {len(used)} drone views with queries; "
        f"{len(names)} images -> {jdir}, {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    return jdir


def register(workdir: Path, dense_path: Path | None = None, min_cert=0.55, max_cycle=4.0, ransac_px=6.0,
             min_inliers=40, min_views=2, max_geotag_offset_m=75.0, height_range_m=(-5.0, 60.0)):
    import sqlite3
    from .merge import _append
    from .reconstruct import robust_sim3
    geo, main_geo, items, main, merged = _load(workdir)
    s, R, t = _sim(main_geo)
    to_enu = lambda X: s * X @ R.T + t
    origin = main_geo["origin"]
    jdir = workdir / "dense_job"
    job = json.loads((jdir / "jobs.json").read_text())
    dense = np.load(dense_path or jdir / "out" / "dense.npz")
    by_file = {it["file"]: it for it in items}
    ids = {}
    P = {pid: p.xyz for pid, p in main.points3D.items()}
    Penu = to_enu(np.array(list(P.values())))

    def ground_z(xy, r=40.0):
        near = Penu[np.linalg.norm(Penu[:, :2] - xy, axis=1) < r]
        return float(np.percentile(near[:, 2], 5)) if len(near) > 50 else None

    cand = {}  # ground photo -> {point3D_id: (x, y, certainty, drone view)}
    for i, p in enumerate(job["pairs"]):
        r = dense.get(f"r{i}")
        if r is None:
            continue
        if p["a_id"] not in ids:
            ids[p["a_id"]] = np.load(jdir / "ids" / f"{p['a_id']}.npy")
        w, h = by_file[p["b"]]["thumb_size"]
        ok = (r[:, 2] > min_cert) & (r[:, 3] < max_cycle) & (r[:, 0] > 0) & (r[:, 0] < w) & (r[:, 1] > 0) & (r[:, 1] < h)
        c = cand.setdefault(p["b"], {})
        for (x, y, cert, _), pid in zip(r[ok], ids[p["a_id"]][ok]):
            if pid in P and (pid not in c or cert > c[pid][2]):
                c[int(pid)] = (float(x), float(y), float(cert), p["a_id"])

    con = sqlite3.connect(f"file:{workdir / 'colmap' / 'database.db'}?mode=ro", uri=True)
    db_ids = dict((n, i) for i, n in con.execute("SELECT image_id, name FROM images"))
    con.close()
    out = pycolmap.Reconstruction(str(workdir / main_geo["model_dir"]))
    est, ref = pycolmap.AbsolutePoseEstimationOptions(), pycolmap.AbsolutePoseRefinementOptions()
    est.ransac.max_error = ransac_px
    diag, poses = {}, {}
    for n, c in sorted(cand.items()):
        it = by_file[n]
        d = diag[n] = {"candidates": len(c)}
        if len(c) < min_inliers:
            d["result"] = "too few matches"
            continue
        pids = np.array(list(c))
        xy = np.array([c[k][:2] for k in pids], np.float64)
        X = np.array([P[k] for k in pids], np.float64)
        w, h = it["thumb_size"]
        f = it.get("focal_prior_px") or 1.2 * max(w, h)
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h, params=[f, w / 2, h / 2, 0.0])
        est.estimate_focal_length = ref.refine_focal_length = not it.get("focal_prior_px")
        res = pycolmap.estimate_and_refine_absolute_pose(xy, X, cam, est, ref)
        if res is None or res["num_inliers"] < min_inliers:
            d.update(result="PnP failed", pnp_inliers=int(res["num_inliers"]) if res else 0)
            continue
        inl = np.asarray(res["inlier_mask"], bool)
        views = len({c[k][3] for k in pids[inl]})
        d.update(pnp_inliers=int(inl.sum()), views=views)
        pose = res["cam_from_world"]
        C = to_enu(-pose.rotation.matrix().T @ pose.translation)
        gz = ground_z(C[:2])
        loc = it.get("camera_location")
        off = None
        if loc:
            gg = np.asarray(geodetic_to_enu(loc["lat"], loc["lon"], origin["alt"], origin["lat"], origin["lon"], origin["alt"]))
            off = float(np.linalg.norm(C[:2] - gg[:2]))
        bad = ("inliers from a single view" if views < min_views else
               "implausible focal length" if not 0.4 <= cam.params[0] / max(w, h) <= 3.0 else
               "too far from geotag" if off is not None and off > max_geotag_offset_m else
               "implausible height" if gz is not None and not height_range_m[0] <= C[2] - gz <= height_range_m[1] else None)
        if bad:
            d["result"] = bad
            continue
        d.update(result="registered", geotag_offset_m=None if off is None else round(off, 1),
                 height_above_ground_m=None if gz is None else round(float(C[2] - gz), 1))
        cam.camera_id = max(out.cameras) + 1
        out.add_camera_with_trivial_rig(cam)
        img = pycolmap.Image(name=n, keypoints=xy[inl], camera_id=cam.camera_id, image_id=db_ids[n])
        out.add_image_with_trivial_frame(img, pose)
        for k, pid in enumerate(pids[inl]):
            out.add_observation(int(pid), pycolmap.TrackElement(db_ids[n], k))
        poses[n] = -pose.rotation.matrix().T @ pose.translation  # centre, main-model coords
    reg = sorted(poses)
    log(f"RoMa registration: {len(reg)} of {len(cand)} photos with candidates; "
        + ", ".join(f"{k} {v}" for k, v in sorted(__import__('collections').Counter(x['result'] for x in diag.values()).items())))

    # separate ground groups: re-anchor on their RoMa-registered members, else keep their earlier placement
    groups_out, still_grouped = [], []
    for m in (geo.get("merged_models") or []):
        names = set(m["names"])
        hit = [n for n in reg if n in names]
        sec = pycolmap.Reconstruction(str(workdir / "colmap" / "sparse" / m["model"]))
        c_sec = {sec.images[i].name: sec.images[i].projection_center() for i in sec.reg_image_ids()}
        hit = [n for n in hit if n in c_sec]
        rest = names - set(reg)
        if len(hit) >= 3:
            try:
                ss, Rs, ts, inl = robust_sim3(np.array([c_sec[n] for n in hit]), np.array([poses[n] for n in hit]), 1.0 / s)
            except RuntimeError:
                inl = np.zeros(len(hit), bool)
            if inl.sum() >= 3:
                sec.transform(pycolmap.Sim3d(np.hstack([ss * Rs, ts[:, None]])))
                added = _append(out, sec, rest)
                groups_out.append({"model": m["model"], "anchored_on": int(inl.sum()), "added": len(added)})
                log(f"  group {m['model']}: re-anchored on {int(inl.sum())} RoMa-registered photos, {len(added)} more placed")
                continue
        if merged is not None and rest:  # keep the geotag(+ICP) placement for the others
            added = _append(out, merged, rest)
            still_grouped += sorted(added)
            groups_out.append({"model": m["model"], "anchored_on": 0, "kept_geotag_placement": len(added)})
    model_dir = workdir / "colmap" / "dense_registered"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    out.write(str(model_dir))
    new = dict(main_geo)
    new.update({"main": main_geo, "model_dir": str(model_dir.relative_to(workdir)), "num_registered": out.num_reg_images(),
                "merged_models": [{"model": "geotag-placed", "names": still_grouped}] if still_grouped else [],
                "dense": {"method": "RoMa (outdoor) samples at drone keypoints -> PnP", "registered": reg,
                          "groups": groups_out, "per_photo": diag}})
    (workdir / "georef.json").write_text(json.dumps(new, indent=1))
    log(f"model with {out.num_reg_images()} images -> {model_dir}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("workdir", type=Path)
    p1.add_argument("--targets", type=int, default=8)
    p2 = sub.add_parser("register")
    p2.add_argument("workdir", type=Path)
    p2.add_argument("--dense", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.workdir, a.targets)
    else:
        register(a.workdir, a.dense)


if __name__ == "__main__":
    main()
