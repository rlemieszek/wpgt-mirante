"""Compare a georeferenced reconstruction of the synthetic plaza with ground truth.

  python tests/evaluate.py work/synth
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirante.geo import enu_to_geodetic, geodetic_to_enu  # noqa: E402
from mirante.reconstruct import _cam_from_world  # noqa: E402

wd = Path(sys.argv[1])
truth = json.loads((wd / "truth.json").read_text())
geo = json.loads((wd / "georef.json").read_text())
o = geo["origin"]
lat0, lon0, alt0 = truth["origin"]
s, R, t = geo["sim3"]["scale"], np.array(geo["sim3"]["R"]), np.array(geo["sim3"]["t"])
rec = pycolmap.Reconstruction(str(wd / geo["model_dir"]))

pos_err, rot_err, f_err = [], [], []
for iid in rec.reg_image_ids():
    im = rec.images[iid]
    tr = truth["cams"][im.name]
    # truth ENU (synth origin) -> geodetic -> ENU (reconstruction origin)
    C_true = geodetic_to_enu(*enu_to_geodetic(*tr["C"], lat0, lon0, alt0), o["lat"], o["lon"], o["alt"])
    R_true = np.array(tr["R_wc"])  # the ENU frames differ by <1e-5 rad over metres, ignore
    C = s * R @ im.projection_center() + t
    R_wc = R @ _cam_from_world(im).rotation.matrix().T
    pos_err.append(np.linalg.norm(C - C_true))
    cosang = (np.trace(R_wc.T @ R_true) - 1) / 2
    rot_err.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    f_err.append(abs(rec.cameras[im.camera_id].params[0] - truth["f"]) / truth["f"] * 100)

pos_err, rot_err = np.array(pos_err), np.array(rot_err)
print(f"registered {rec.num_reg_images()}/{len(truth['cams'])}")
print(f"camera position error (m): median {np.median(pos_err):.3f}  p90 {np.percentile(pos_err, 90):.3f}  max {pos_err.max():.3f}")
print(f"camera rotation error (deg): median {np.median(rot_err):.3f}  max {rot_err.max():.3f}")
print(f"focal length error (%): median {np.median(f_err):.2f}  max {np.max(f_err):.2f}")
print("GPS outliers flagged:", geo.get("outliers"), " truth:",
      [n for n, c in truth["cams"].items() if c["gps_outlier"]])
