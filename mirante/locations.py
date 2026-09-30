"""Camera locations recovered by SfM for photos whose Commons geotag is missing or looks wrong.

Writes a CSV with the SfM position, heading and pitch of each such photo, a quality indication, and a
{{Location}} template ready to paste on the file page. Nothing is written to Commons.

Usage:
  python -m mirante.locations work/sao-francisco [--out locations.csv] [--wrong-geotag-m 50]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

from .geo import enu_to_geodetic, geodetic_to_enu, geoid_undulation
from .reconstruct import _cam_from_world


def rows_for(workdir: Path, wrong_geotag_m: float = 50.0, include_grouped: bool = False):
    geo = json.loads((workdir / "georef.json").read_text())
    if not geo.get("aligned"):
        raise SystemExit("model is not georeferenced")
    items = json.loads((workdir / "manifest.json").read_text())["items"]
    if geo.get("extended") and (workdir / "extra" / "manifest.json").exists():
        items += json.loads((workdir / "extra" / "manifest.json").read_text())["items"]
    by_file = {it["file"]: it for it in items}
    rec = pycolmap.Reconstruction(str(workdir / geo["model_dir"]))
    s, R, t = geo["sim3"]["scale"], np.array(geo["sim3"]["R"]), np.array(geo["sim3"]["t"])
    o = geo["origin"]
    height_is_above_ground = bool(geo.get("horizontal_fit"))
    # the model's heights follow the anchors; if those are ellipsoidal (e.g. RTK), sea-level heights need N
    man = json.loads((workdir / "manifest.json").read_text())
    ellipsoidal = (man.get("reference") or {}).get("vertical_datum") == "WGS84 ellipsoid"
    N = geoid_undulation(geo["origin"]["lat"], geo["origin"]["lon"]) if ellipsoidal and not height_is_above_ground else 0.0
    # photos merged in from a separate SfM group (mirante.merge) are placed by their own geotags, not by the
    # anchors: not independent evidence for a location, so left out unless asked for
    grouped = {n for m in geo.get("merged_models") or [] for n in m.get("names", [])}
    rows = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        it = by_file.get(im.name)
        if it is None or it.get("anchor") or (im.name in grouped and not include_grouped):
            continue
        C = s * R @ im.projection_center() + t
        lat, lon, alt = enu_to_geodetic(*C, o["lat"], o["lon"], o["alt"])
        R_wc = R @ _cam_from_world(im).rotation.matrix().T
        heading = float((np.degrees(np.arctan2(R_wc[0, 2], R_wc[1, 2])) + 360) % 360)
        pitch = float(np.degrees(np.arcsin(np.clip(R_wc[2, 2], -1, 1))))
        loc = it.get("camera_location")
        if loc:
            g = geodetic_to_enu(loc["lat"], loc["lon"], o["alt"], o["lat"], o["lon"], o["alt"])
            offset = float(np.linalg.norm(C[:2] - g[:2]))
            if offset <= wrong_geotag_m:
                continue
            status = "geotag_off"
        else:
            offset, status = None, "no_geotag"
        obs = [p for p in im.points2D if p.has_point3D()]
        err = float(np.mean([rec.points3D[p.point3D_id].error for p in obs])) if obs else None
        rows.append({
            "title": it["title"], "file_page": it.get("file_page"), "status": status,
            "commons_lat": loc and loc["lat"], "commons_lon": loc and loc["lon"],
            "geotag_offset_m": None if offset is None else round(offset, 1),
            "sfm_lat": round(float(lat), 6), "sfm_lon": round(float(lon), 6),
            **({"height_above_ground_m": round(float(C[2]), 1)} if height_is_above_ground else
               {"alt_msl_m": round(float(alt) - N, 1),
                "alt_ellipsoidal_m": round(float(alt), 1) if ellipsoidal else None}),
            "heading_deg": round(heading), "pitch_deg": round(pitch),
            "num_3d_points": len(obs), "mean_reproj_error_px": None if err is None else round(err, 2),
            "confidence": "high" if len(obs) >= 150 else "medium" if len(obs) >= 50 else "low",
            "location_template": f"{{{{Location|{float(lat):.6f}|{float(lon):.6f}|heading:{round(heading)}}}}}",
        })
    rows.sort(key=lambda r: (r["status"], r["title"]))
    return rows, geo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--out", type=Path, help="CSV path (default: <workdir>/recovered_locations.csv)")
    ap.add_argument("--include-grouped", action="store_true",
                    help="also list photos from merged secondary groups (placed by their own geotags)")
    ap.add_argument("--wrong-geotag-m", type=float, default=50.0,
                    help="also list geotagged photos whose SfM position is farther than this from the geotag")
    a = ap.parse_args(argv)
    rows, geo = rows_for(a.workdir, a.wrong_geotag_m, a.include_grouped)
    out = a.out or a.workdir / "recovered_locations.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n_missing = sum(r["status"] == "no_geotag" for r in rows)
    print(f"{n_missing} photos without a geotag, {len(rows) - n_missing} with a geotag > {a.wrong_geotag_m:g} m off "
          f"-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
