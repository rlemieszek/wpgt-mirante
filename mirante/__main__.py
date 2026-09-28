"""One-shot pipeline: harvest -> reconstruct -> export.

  python -m mirante "Category:..." work/NAME site/NAME [--use-priors] [--local-images] [--depth 0]
"""
import argparse
from pathlib import Path

from . import export, harvest, reconstruct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category")
    ap.add_argument("workdir", type=Path)
    ap.add_argument("site", type=Path)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--use-priors", action="store_true", help="accurately positioned set (e.g. WPGT): GPS as BA priors")
    ap.add_argument("--prior-sigma", type=float, help="prior std dev in metres (default 2; ~0.05 for RTK-fixed sets)")
    ap.add_argument("--prior-sigma-z", type=float)
    ap.add_argument("--inlier-m", type=float)
    ap.add_argument("--local-images", action="store_true")
    ap.add_argument("--skip-harvest", action="store_true", help="reuse an existing work folder")
    a = ap.parse_args()
    if not a.skip_harvest:
        harvest.main([a.category, str(a.workdir), "--depth", str(a.depth), "--width", str(a.width)])
    reconstruct.run(a.workdir, use_priors=a.use_priors, inlier_m=a.inlier_m,
                    prior_sigma_m=a.prior_sigma, prior_sigma_z_m=a.prior_sigma_z)
    export.export(a.workdir, a.site, local_images=a.local_images)
    print(f"\nDone. Preview with:  python -m http.server -d {a.site} 8000  ->  http://localhost:8000")


if __name__ == "__main__":
    main()
