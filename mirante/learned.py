"""Learned features + matching (ALIKED + LightGlue) for image pairs, as a standalone worker.

Runs in its own process: torch and pycolmap each bundle an OpenMP runtime and cannot be loaded
together, so this module never imports pycolmap (and hides it from lightglue, which would try).

Reads   a pairs file ("name1 name2" per line) and the images it names
Writes  <out>/feats/<name>.npz   keypoints (x, y in file pixels, COLMAP convention: pixel centres at +0.5)
        <out>/matches.npz        m<i> = (K, 2) int32 keypoint index pairs, s<i> = scores, for pair i of the file

Usage:
  python -m mirante.learned IMAGES PAIRS OUT [--features aliked] [--max-keypoints 4096] [--resize 1280]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.modules["pycolmap"] = None  # see module docstring

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402


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


def load_image(path: Path, resize: int):
    """RGB float tensor (3, H, W) at most `resize` on the long side, and the scale back to file pixels.
    EXIF orientation is ignored on purpose: COLMAP reads the stored pixel grid the same way."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, resize / max(w, h))
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        arr = np.asarray(im, np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1), 1.0 / s


def run(images: Path, pairs_file: Path, out: Path, features="aliked", max_keypoints=4096, resize=1280,
        device="auto", filter_threshold=0.1):
    from lightglue import ALIKED, DISK, LightGlue, SuperPoint

    dev = _device(device)
    pairs = [ln.split() for ln in pairs_file.read_text().splitlines() if ln.strip()]
    names = sorted({n for p in pairs for n in p})
    fdir = out / "feats"
    fdir.mkdir(parents=True, exist_ok=True)
    extractor = {"aliked": ALIKED, "disk": DISK, "superpoint": SuperPoint}[features](
        max_num_keypoints=max_keypoints).eval().to(dev)
    matcher = LightGlue(features=features, filter_threshold=filter_threshold).eval().to(dev)
    log(f"{features}+LightGlue on {dev}: {len(names)} images, {len(pairs)} pairs")

    t0 = time.time()
    feats = {}
    for k, n in enumerate(names, 1):
        cache = fdir / (n + ".npz")
        if cache.exists():
            z = np.load(cache)
            feats[n] = {key: z[key] for key in z.files}
        else:
            img, back = load_image(images / n, resize)
            with torch.inference_mode():
                f = extractor.extract(img.to(dev), resize=None)
            kp = f["keypoints"][0].cpu().numpy()
            feats[n] = {"kp_net": kp.astype(np.float32),  # network-input pixels (what LightGlue sees)
                        "keypoints": ((kp + 0.5) * back).astype(np.float32),  # file pixels, COLMAP convention
                        "scores": f["keypoint_scores"][0].cpu().numpy().astype(np.float32),
                        "descriptors": f["descriptors"][0].cpu().numpy().astype(np.float16),
                        "net_size": np.array(img.shape[1:][::-1], np.int32)}
            np.savez(cache, **feats[n])
        if k % 100 == 0:
            log(f"  features {k}/{len(names)}  ({time.time() - t0:.0f} s)")

    def as_input(n):
        f = feats[n]
        return {"keypoints": torch.from_numpy(f["kp_net"])[None].to(dev),
                "descriptors": torch.from_numpy(f["descriptors"].astype(np.float32))[None].to(dev),
                "image_size": torch.from_numpy(f["net_size"].astype(np.float32))[None].to(dev)}

    cached_pairs = out / "matches.pairs.txt"
    if (out / "matches.npz").exists() and cached_pairs.exists() and cached_pairs.read_text() == pairs_file.read_text():
        log(f"matches for these pairs already in {out / 'matches.npz'}")
        return
    t0 = time.time()
    res = {}
    for i, (a, b) in enumerate(pairs):
        with torch.inference_mode():
            m = matcher({"image0": as_input(a), "image1": as_input(b)})
        res[f"m{i}"] = m["matches"][0].cpu().numpy().astype(np.int32)
        res[f"s{i}"] = m["scores"][0].cpu().numpy().astype(np.float16)
        if (i + 1) % 500 == 0:
            log(f"  matched {i + 1}/{len(pairs)}  ({time.time() - t0:.0f} s)")
    np.savez(out / "matches.npz", **res)
    cached_pairs.write_text(pairs_file.read_text())
    log(f"matches written to {out / 'matches.npz'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("pairs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--features", default="aliked", choices=["aliked", "disk", "superpoint"])
    ap.add_argument("--max-keypoints", type=int, default=4096)
    ap.add_argument("--resize", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run(a.images, a.pairs, a.out, a.features, a.max_keypoints, a.resize, a.device)


if __name__ == "__main__":
    main()
