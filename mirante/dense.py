"""Dense matching worker (RoMa): where do given points of image A land in image B?

For registered photos whose SfM 3D points are known, sampling RoMa's dense A->B warp exactly at A's observed
keypoints turns every confident sample into a 2D-3D correspondence for B, with no need for matched keypoints
to coincide (the failure mode of sparse matchers across large viewpoint changes). Each sample is checked for
forward-backward consistency through the B->A half of the symmetric warp.

Runs in its own process: torch and pycolmap each bundle an OpenMP runtime and cannot be loaded together.

Reads   a job file {"images": DIR or null, "urls": {name: url}, "pairs": [{"a": name, "b": name, "q": path}, ...]}
        (queries: float32 (N, 2) pixel coordinates in A, COLMAP convention: pixel centres at +0.5; relative
        paths are resolved against the job file's folder). Images missing from DIR are downloaded from "urls"
        into <out>/images/, so a job can run on another machine (e.g. a CUDA PC) with just the job folder.
Writes  <out>/dense.npz   r<i> = float32 (N, 4) per query: x_B, y_B (pixels), certainty, cycle error (pixels in A)
        (checkpointed every 50 pairs; a restarted run skips pairs already done)

Usage:
  python -m mirante.dense JOBS.json OUT [--device auto|cuda|mps|cpu]
  python run_dense.py jobs.json out                 # the same file, standalone in a job bundle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.modules["pycolmap"] = None  # see module docstring

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample(field, xy_norm):
    """Bilinear sample of field (H, W, C) at normalised points (N, 2) -> (N, C)."""
    g = xy_norm.view(1, 1, -1, 2)
    return F.grid_sample(field.permute(2, 0, 1)[None], g, mode="bilinear", align_corners=False)[0, :, 0].T


def fetch_images(job, base: Path, out: Path):
    """Local path for every image the job needs, downloading from job["urls"] when missing."""
    import requests
    local = Path(job["images"]) if job.get("images") else None
    if local is not None and not local.is_absolute():
        local = base / local
    cache = out / "images"
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted({n for p in job["pairs"] for n in (p["a"], p["b"])})
    paths, s = {}, requests.Session()
    s.headers["User-Agent"] = "WPGT-Mirante/0.1 dense worker (https://github.com/rlemieszek/wpgt-mirante)"
    for k, n in enumerate(names, 1):
        if local is not None and (local / n).exists():
            paths[n] = local / n
            continue
        dst = cache / n
        if not (dst.exists() and dst.stat().st_size > 0):
            for attempt in range(6):
                r = s.get(job["urls"][n], timeout=120)
                if r.status_code == 429:
                    time.sleep(2 ** attempt * 2)
                    continue
                r.raise_for_status()
                dst.write_bytes(r.content)
                break
            time.sleep(0.2)
            if k % 50 == 0:
                log(f"  images {k}/{len(names)}")
        paths[n] = dst
    return paths


def run(jobs_file: Path, out: Path, model="outdoor", device="auto"):
    from romatch import roma_outdoor

    dev = _device(device)
    job = json.loads(jobs_file.read_text())
    base = jobs_file.resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    paths = fetch_images(job, base, out)
    matcher = roma_outdoor(device=dev)
    ck = out / "dense.npz"
    res = dict(np.load(ck)) if ck.exists() else {}
    log(f"RoMa outdoor on {dev}: {len(job['pairs'])} pairs ({len(res)} already done)")
    t0 = time.time()
    for i, p in enumerate(job["pairs"]):
        if f"r{i}" in res:
            continue
        pa, pb = paths[p["a"]], paths[p["b"]]
        with Image.open(pa) as ia, Image.open(pb) as ib:
            (wa, ha), (wb, hb) = ia.size, ib.size
        q = np.load(p["q"] if Path(p["q"]).is_absolute() else base / p["q"]).astype(np.float32)
        try:
            with torch.inference_mode():
                warp, cert = matcher.match(str(pa), str(pb), device=dev)
                if warp.dim() == 4:  # batched output: (1, H, 2W, 4) and (1, H, 2W)
                    warp, cert = warp[0], cert[0]
                W = warp.shape[1] // 2
                a2b, cert_ab = warp[:, :W, 2:4].float(), cert[:, :W, None].float()
                b2a = warp[:, W:, 0:2].float()
                qa = torch.from_numpy(np.stack([2 * q[:, 0] / wa - 1, 2 * q[:, 1] / ha - 1], 1)).to(dev)
                b = sample(a2b, qa)
                c = sample(cert_ab, qa)[:, 0]
                back = sample(b2a, b)
                cyc = torch.linalg.norm((back - qa) * torch.tensor([wa / 2, ha / 2], device=dev), dim=1)
                xb, yb = (b[:, 0] + 1) * wb / 2, (b[:, 1] + 1) * hb / 2
                res[f"r{i}"] = torch.stack([xb, yb, c, cyc], 1).cpu().numpy().astype(np.float32)
        except Exception as e:  # keep going; one unreadable image must not stop a long run
            log(f"  pair {i} failed: {e}")
            res[f"r{i}"] = np.full((len(q), 4), np.nan, np.float32)
        if (i + 1) % 50 == 0:
            np.savez(ck, **res)
            log(f"  {i + 1}/{len(job['pairs'])} pairs ({time.time() - t0:.0f} s)")
    np.savez(ck, **res)
    log(f"written {ck} ({time.time() - t0:.0f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run(a.jobs, a.out, device=a.device)


if __name__ == "__main__":
    main()
