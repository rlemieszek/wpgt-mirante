# Handoff: continuing WPGT Mirante on another machine

Written 2026-09-30 when work moved from a Mac (Apple GPU) to a PC with an NVIDIA GPU. It records the state,
the decisions behind it, and the next steps. Newest first where it matters.

## Setting up

The project folder (`~/work/commonssynth` on the Mac) is synced with Google Drive, so the PC opens the **same
folder**: code, `.git`, `work/` and `site/` are all there. Three rules:

1. **Don't use the Mac's `.venv`** (macOS arm64). Create a PC environment, ideally outside the synced folder so
   Drive doesn't sync ~1 GB of packages:
   ```bash
   python -m venv C:/venvs/mirante        # Python 3.10+; any path outside the Drive folder
   C:/venvs/mirante/Scripts/pip install -r requirements.txt
   # GPU parts: torch with CUDA first (https://pytorch.org/get-started/locally/), then
   C:/venvs/mirante/Scripts/pip install -r requirements-learned.txt    # LightGlue + RoMa (needs git on PATH)
   ```
2. **Make the folder available offline** in Google Drive for desktop. In "streaming" mode large files (e.g. the
   1.2 GB `work/boa-viagem/colmap/database.db`) are placeholders fetched on first read: slow, and risky for SQLite.
3. **One machine at a time.** Let Drive finish syncing before switching machines; git and the SQLite databases do
   not survive concurrent writes from both sides. Push to GitHub (branch `roma-dense-registration`) as the
   source of truth for code.

`torch` and `pycolmap` each bundle an OpenMP runtime and **cannot be imported in the same process**. That is
why `mirante.learned` and `mirante.dense` are separate worker processes that hide `pycolmap`; keep it that way.

`.claude/launch.json` (local viewer servers) points at the Mac's `.venv`; on the PC just run
`python -m http.server -d site/<name> 8004`.

## Current task: Catedral da Boa Viagem (Belo Horizonte), ground photos via RoMa

Work folder `work/boa-viagem`, site `site/boa-viagem`.

- Sources: `Category:Catedral da Boa Viagem (Belo Horizonte)` (depth 2, interior excluded) + photos geotagged
  within 150 m (`harvest --near -19.9285 -43.93502778 150`) + two WPGT drone surveys. 634 photos.
- **Anchors = the August 2026 WPGT survey only** (RTK fixed; the user's rule: an RTK-fixed set gives the true
  location for everything else). Its heights are **WGS84 ellipsoidal** (manifest `reference.vertical_datum`).
  Geoid undulation there N = -6.06 m (EGM2008): sea-level H = h + 6.06.
- The September 2024 WPGT survey is placed by SfM, not trusted: its altitudes are **+26.7 m** vs the 2026 RTK
  (constant, 0.9 m spread), i.e. about +20.6 m above true sea level; horizontally ~0.5 m off. Worth
  correcting on Commons.
- Main model `colmap/sparse/0`: 361 drone photos, 237k points; georef against 2026 only: 139/139 inliers,
  median 0.80 m. (`georef_main.json` = this georef.)
- SIFT found **zero** drone-ground pairs (drone 70-125 m up over dense trees; ground photos are facades/paths).
  Ground photos formed separate models; `mirante.merge --icp --min-images 40` added model 2 (93 photos; 4
  degenerate far cameras dropped), placed by its geotags (horizontal fit, ground level matched) + ICP (median
  distance 2.3 -> 1.6 m). Models 7 (fountain close-ups; ICP collapsed, rejected) and 1 (geotags 36 m off)
  are left out. Current `georef.json` points at `colmap/merged`; the site shows 454 photos.
- **RoMa job ready**: `work/boa-viagem/dense_job/` (also `dense_job_boa-viagem.zip`): 2,514 pairs for all 270
  ground photos (93 by merged pose, 144 by geotag/heading, 33 unlocated -> best general views), 154 drone
  views with queries. Validated on the Mac: on an overlapping drone-drone pair RoMa's samples land a median
  0.4 px from SfM's own points (so the sampling is right). One ground pair tried had no overlap (expected).
  On Apple MPS it ran ~8 s/pair; on CUDA expect ~0.5-1 s/pair.

### Next steps

1. Run RoMa (CUDA). `jobs.json` points at the synced photos (`"images": "../images"`); any missing one is
   downloaded from Commons:
   ```bash
   python -m mirante.dense work/boa-viagem/dense_job/jobs.json work/boa-viagem/dense_job/out
   ```
   Checkpoints every 50 pairs; rerun to resume.
2. Register. **Not yet exercised on real matches** (the Mac test was stopped): first look at the per-photo
   report it writes (`georef.json` -> `dense.per_photo`), and back up `georef.json` before running:
   ```bash
   cp work/boa-viagem/georef.json work/boa-viagem/georef_merged.json
   python -m mirante.densereg register work/boa-viagem
   ```
   Acceptance: >= 40 PnP inliers from >= 2 drone views, height -5..+60 m over local ground, <= 75 m from the
   geotag, plausible focal. Chance fits look like 20-80 inliers with absurd poses; real ones should be in the
   hundreds or more. It re-anchors the merged group on its RoMa-registered members (>= 3) and keeps the
   geotag placement otherwise.
3. **Verify visually** before trusting it: project the drone model's points into a registered ground photo
   from its new pose and overlay (this is how every earlier result was checked; a right pose puts spires,
   windows and roof lines on the photo's). Then export and look in the viewer (`python -m http.server -d
   site/boa-viagem 8004`); ground photos should now have drone neighbours.
   ```bash
   python -m mirante.export work/boa-viagem site/boa-viagem --title "Catedral da Boa Viagem, Belo Horizonte"
   ```
4. Locations for Commons: `python -m mirante.locations work/boa-viagem` (writes `alt_msl_m` and
   `alt_ellipsoidal_m`), then `python -m mirante.commons_edit work/boa-viagem` (preview first; `--apply`
   prompts for the user's bot password; the **user** runs the apply, never the assistant).

## Other sets

| set | work folder | state |
|---|---|---|
| São Francisco de Assis (BH) | `work/sao-francisco` | MVP. One joint model: 177 WPGT anchors (lat/lon only -> horizontal fit, heights above local ground) + 235 ground photos, median 0.43 m. `recovered_locations.csv` (156 missing + 13 wrong geotags) and a Commons edit plan (157 wikitext + 12 SDC, ground ASL 803.9 m) are prepared; the user runs `commons_edit --apply` themselves (check `commons_edits/applied.jsonl`). |
| Praça Rui Barbosa (BH) | `work/rui-barbosa` | 620 RTK anchors (EXIF GPS ~0.9 m body-frame offset from SfM, likely timing/lever arm) + 671 extras via `extend`; street-level photos did not register (SIFT or LightGlue). |
| Rosário, Ouro Preto | `work/ouro-preto-rosario`, `work/ouro-preto-ground` | Anchored drone model + 2 overlook photos; separate ground-only model (168 photos, levelled). |

## Decisions and conventions

- Name **WPGT Mirante**, "a WikiProject GeoTwin tool by Rafael Lemieszek"; package `mirante`; code
  GPL-3.0-or-later; derived 3D data CC BY-SA 4.0 (`credits.html`). GitHub + Zenodo (concept DOI
  10.5281/zenodo.23016972). No third-party requests in sites (three.js and fonts vendored).
- Commons `{{Location}}` follows the WPGT convention with heading added:
  `{{Location|lat|lon|alt:NNN_heading:NNN_source:WPGT}}`; SDC P1259 is corrected only where it exists
  (P7787 heading, P2044 elevation qualifiers). Edits need the user's explicit go-ahead and run under their
  account; a bot account would be needed for routine use.
- Never trust phone altitudes for placement; prefer RTK anchors; report, don't silently fix, survey
  inconsistencies.

## Pitfalls found so far

- Commons serves only standard thumbnail widths: 120, 250, 330, 500, 960, 1280, 1920, 3840 (320 and 640 -> HTTP 400).
- Levelling from camera x axes: the x axes fix the vertical line, the cameras' image-up vectors fix its sign
  (a geotag fit of a small group can come out upside down); a grid flown at one heading leaves it undetermined.
- ICP on small groups collapses them onto a surface (scale << 1): keep the bounds.
- A single badly placed camera once blanked the overview; the viewer now frames on the 95th percentile.
- RoMa's `tiny` model waits for a hidden torch.hub trust prompt (XFeat); use the outdoor model.
- `sample`/`py-spy` style checks help: a long GPU job with buffered logs looks stuck but isn't.
- Viewer asset links carry content hashes (`export --viewer-only` refreshes a site without re-exporting).

## Open items

- PR https://github.com/rlemieszek/wpgt-mirante/pull/1 (Commons write-back, `harvest --near`) is open.
- Zenodo dataset record for São Francisco (`work/zenodo/wpgt-mirante-sao-francisco-v0.1.0.zip`) to upload.
- Commons altitude corrections for the 2024 Boa Viagem survey (and check the 2026 set's `alt:` datum).
- Unregistered street-level photos elsewhere (Rui Barbosa, Ouro Preto) could also try the RoMa path.
