# WPGT Mirante

A [WikiProject GeoTwin](https://commons.wikimedia.org/wiki/Commons:WikiProject_GeoTwin) tool by Rafael Lemieszek: 3D browsing of the
Wikimedia Commons photos of a place (*mirante* is Portuguese for a lookout). The photos are placed in 3D by
structure-from-motion, georeferenced with their Commons geotags, and shown in a browser
viewer that animates and cross-fades between neighbouring photos over a sparse point cloud.

```
Commons category ──harvest──▶ manifest.json + thumbnails
                 ──reconstruct──▶ COLMAP model + georef.json (model → local ENU, metres)
                 ──export──▶ static site: index.html + data/{scene.json, points.bin, vis.bin}
```

## Quick start (macOS / Linux, Python ≥ 3.10)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. find the exact category name
python -m mirante.harvest --search "WPGT Rui Barbosa"

# 2. run everything (WPGT sets are accurately positioned, so use them as BA priors)
python -m mirante "Category:WPGT - <exact name>" work/rui-barbosa site/rui-barbosa --use-priors

# 3. look at it
python -m http.server -d site/rui-barbosa 8000     # open http://localhost:8000
```

Each step can also run on its own:

| step | command | notes |
|---|---|---|
| harvest | `python -m mirante.harvest CATEGORY WORKDIR [--depth N] [--width 1920] [--require-location] [--within LAT LON METRES] [--exclude REGEX] [--anchor-category CAT]` | Camera position taken from SDC P1259, then EXIF GPS, then `{{Location}}`. `{{Object location}}` / P625 is kept apart and never used as a camera position. Focal-length prior from EXIF (35 mm equivalent, sensor table, or focal-plane resolution). |
| reconstruct | `python -m mirante.reconstruct WORKDIR [--use-priors] [--inlier-m 5] [--matcher auto\|exhaustive\|spatial]` | COLMAP incremental SfM via pycolmap. Exhaustive matching up to 150 images, then GPS-neighbour (spatial) matching. Georeferencing is a RANSAC similarity fit from SfM camera centres to geotags; outlier geotags are listed in `georef.json`. |
| extend | `python -m mirante.extend WORKDIR [--category CAT ...] [--depth 1] [--no-geo] [--geo-margin 50]` | Registers other Commons photos of the area into the reconstruction with the anchor poses held fixed (see below). |
| export | `python -m mirante.export WORKDIR SITE [--local-images] [--title ...]` | Without `--local-images` the viewer hotlinks the Commons thumbnails (smaller site, attribution stays with Commons). |

`--use-priors`: use for accurately positioned sets (WPGT). It also fixes scale and orientation during
bundle adjustment. Leave it off for ordinary Commons categories, whose geotags are often metres to
tens of metres off; those images are still reconstructed, and the georeferencing RANSAC rejects bad geotags
(default threshold 20 m without priors, 5 m with).

For RTK-fixed sets, tighten the priors to the fix accuracy so bundle adjustment keeps the cameras on
their geotags: `--use-priors --prior-sigma 0.05 --prior-sigma-z 0.10`. `--reuse-matches` keeps the
existing features and matches and only redoes priors, mapping and georeferencing.

## One joint model with anchors (`--anchor-category`)

When an accurately positioned set covers the same subject closely (e.g. a WPGT orbit of one building),
reconstruct everything together instead of extending:

```bash
python -m mirante.harvest "Category:Igreja de São Francisco de Assis (Belo Horizonte)" work/sao-francisco \
    --depth 2 --exclude Interior --exclude "^Category:WPGT" \
    --anchor-category "Category:WPGT - Igreja de São Francisco de Assis, Belo Horizonte, Brazil, September 2025"
python -m mirante.reconstruct work/sao-francisco --use-priors --matcher exhaustive
python -m mirante.export work/sao-francisco site/sao-francisco
```

Files from `--anchor-category` are flagged `"anchor": true`: only they get position priors and drive the
georeferencing; every other photo is placed by SfM alone and its geotag offset is reported.

Georeferencing details (`reconstruct --georef-only` redoes just this step on the current model):
- Without `--use-priors`, the frame is levelled from the cameras' x axes (near horizontal for upright
  handheld shots and gimbal drones): geotag altitudes are missing or phone-GPS noisy and would tilt the fit.
- Anchors without altitude (e.g. SDC P1259 without P2044) get a horizontal fit: level from the camera x axes,
  fit scale/heading/position on lat/lon, and z becomes height above local ground. If the x axes are nearly
  parallel (a grid flown at one heading) the vertical is undetermined and it falls back to a 3D fit that
  assumes one flight height.

## Adding other photos of the area

`extend` treats the reconstructed set as fixed anchors and registers other photos into it:

```bash
python -m mirante.extend work/rui-barbosa --category "Category:Praça Rui Barbosa (Belo Horizonte)" --depth 2
python -m mirante.export work/rui-barbosa site/rui-barbosa
```

- Candidates: the given categories (WPGT sets, museum collections and interiors skipped; `--exclude REGEX`)
  plus a GeoData grid search over the anchors' extent. Anchor files are removed.
- Pairing: each extra is matched against the anchors whose *view* (median of their observed 3D points) is
  nearest its geotag (or object location), plus a spread of anchors; un-located photos get a wider spread.
  A second pass adds the strongest neighbours of the anchors that matched.
- Registration: COLMAP incremental mapping continues the anchor model with `fix_existing_frames`, so
  anchor poses and the anchors' georeferencing are unchanged. Extras' geotags are never used as
  constraints; their offset from the SfM position is reported (`georef.json` → `extra.geotag_offset_m`)
  and shown as "geotag offset" in the viewer.
- Re-running `reconstruct` resets the extension; re-running `extend --skip-harvest` redoes registration only.
- `--require-location` keeps only geotagged extras; `--exclude-title REGEX` drops extras by file title.

### Ground photos against aerial anchors (`--learned`)

SIFT rarely links street-level photos to drone views. `extend --learned` (or `--learned-only` on an
existing extension) gives the photos SIFT could not register a second chance with ALIKED + LightGlue
(`pip install -r requirements-learned.txt`):

1. each located photo is matched against oblique registered views whose footprint is near its geotag
   (spread over viewing azimuths when its heading is unknown) and a few nadir views;
2. learned matches mostly fall on distant, small-scale structure where the drone view has no SIFT 3D
   points, so each drone view is also matched with two co-visible registered views and its keypoints are
   triangulated from the fixed poses ("lifted");
3. the photo's pose comes from PnP (LO-RANSAC + refinement, focal estimated when EXIF has none) on the
   lifted 2D-3D matches; it is inserted with the lifted points as shared tracks. Existing cameras never move.

`--learned-only-title REGEX` limits the pass to photos likely to show buildings (event and object photos
cannot match). The worker (`mirante.learned`) runs in its own process because torch and pycolmap each
bundle an OpenMP runtime and cannot share one.

## Recovered camera locations

```bash
python -m mirante.locations work/sao-francisco     # -> work/sao-francisco/recovered_locations.csv
```

Lists the placed photos that have no Commons geotag, plus those whose geotag is more than `--wrong-geotag-m`
(default 50) from their SfM position: SfM lat/lon, height (above local ground when the anchors had no
altitude), heading and pitch, number of 3D points with a high/medium/low confidence label, and a
`{{Location|lat|lon|heading:…}}` template. Nothing is written to Commons.

## Viewer

- **Photo mode**: each photo is projected with its recovered intrinsics (focal, principal point,
  radial distortion) onto a proxy plane. Going to another photo moves the virtual camera along the
  path between the two poses and cross-fades both projections on a plane through the points they share
  (Photo Tourism, Snavely et al. 2006).
- **Navigation**: ← → ↑ ↓ (or swipe) pick the neighbour that orbits/pans/moves in that direction.
  Clicking a spot jumps to the photo that shows it closest and most centred. The bottom strip orders
  photos by bearing around the scene.
- **Overview** (`O`): orbit the point cloud with every photo shown as a small translucent frustum with its
  thumbnail; click one to fly in. Frusta keep a constant on-screen size, capped in world size so they never
  cover the subject up close. `F` cycles thumbnails → wireframe → hidden.
- Memory stays bounded on long sessions: only the 16 most recently used full-resolution photos stay on the
  GPU, the overview uses 330 px thumbnails, and the page only redraws when something changes.
- Each photo's panel shows author, licence, Commons link, SfM position, heading, and the gap between
  the SfM position and the Commons geotag.
- Deep link to a photo with `#img-<index>`.

The site is plain static files (Toolforge, Netlify, GitHub Pages, any web server) with no third-party requests:
three.js 0.186.1 (MIT) and the Archivo / IBM Plex Mono fonts (SIL OFL 1.1) are bundled under `vendor/`, and photos
load from `upload.wikimedia.org`. `credits.html` lists every source photo with author and licence (the derived 3D
data is CC BY-SA 4.0). `single.html` is the same viewer with CSS/JS inlined, for hosts that take one page plus
data files (it still needs `vendor/` and `data/` next to it). `export --viewer-only` refreshes these files in an
existing site; asset links carry a content hash so browsers never run a stale cached viewer.

## Tests (synthetic plaza)

```bash
python tests/synth.py work/synth           # renders 40 photos of a textured plaza with known poses,
                                           # 1 m/1.5 m GPS noise, 2 gross GPS outliers, 2 un-geotagged photos
python -m mirante.reconstruct work/synth --max-image-size 1024 [--use-priors]
python tests/evaluate.py work/synth
python -m mirante.export work/synth site/synth --local-images
python tests/viewer_smoke.py site/synth shots/   # headless Chromium screenshots (needs playwright)
```

Result on the sandbox run (CPU only, 2 cores, ~2 min): 39/40 photos registered, both bad geotags rejected,
camera positions within 0.6 m median / 1.1 m max of truth (GPS noise was 1 m), focal lengths within 0.05 % median.

## Roadmap

- Learned features + matching (hloc: SuperPoint/ALIKED + LightGlue) for mixed Commons material.
- Image retrieval (NetVLAD/MegaLoc) for categories of thousands of photos.
- Merge secondary models; show fragments rather than discarding them.
- Better proxies: per-pair local planes or a coarse mesh; optional Gaussian splats (appearance-aware
  variants such as WildGaussians/Splatfacto-W) for photoreal in-between views.
- Write refined camera positions and headings back to `{{Location}}` via the geotagging bot
  (needs community discussion and bot approval).

## Publishing a site

A site is a static folder (`site/<name>/`, a few MB; photos load from Commons). No build step.

Unlisted preview on Netlify (the draft URL is unguessable, not password-protected):

```bash
npx netlify-cli login
npx netlify-cli deploy --dir site/sao-francisco          # draft deploy -> unique preview URL
npx netlify-cli deploy --dir site/sao-francisco --prod   # when it should be the site's main URL
```

Toolforge (after creating a tool, e.g. `wpgt-mirante`, in Toolforge's admin site):

```bash
rsync -a site/sao-francisco/ YOU@login.toolforge.org:sao-francisco/
ssh YOU@login.toolforge.org
become wpgt-mirante
mkdir -p ~/public_html && cp -r /home/YOU/sao-francisco ~/public_html/
webservice start          # the default web service serves ~/public_html
```

The site then lives at `https://wpgt-mirante.toolforge.org/sao-francisco/`. It makes no third-party requests,
as Toolforge requires.

## Licences and citing

- Code: GPL-3.0-or-later (`LICENSE`). Bundled in `viewer/vendor/`: three.js (MIT) and the Archivo and IBM Plex
  Mono fonts (SIL OFL 1.1), with their licence files.
- Derived 3D data (points, camera positions): CC BY-SA 4.0, crediting every source photo (`credits.html`).
- Photos: their own licences on Wikimedia Commons; the viewer shows author and licence with each photo.
- Citing: see `CITATION.cff`. Releases are archived on Zenodo with a DOI.

