"""Harvest a Wikimedia Commons category into a local work folder.

Produces <workdir>/manifest.json and <workdir>/images/<file>.jpg (thumbnails).

Camera position priority (first found wins):
  1. Structured data P1259 "coordinates of the point of view" (+ P7787 heading, P2044 elevation)
  2. EXIF GPS (as parsed by MediaWiki)
  3. GeoData primary coordinate ({{Location}})
Object location ({{Object location}} / P625) is recorded separately and never used as a camera position.

Usage:
  python -m mirante.harvest "Category:WPGT - ..." work/rui-barbosa [--depth 0] [--width 1920]
  python -m mirante.harvest --search "WPGT Rui Barbosa"   # find category names
  python -m mirante.harvest --download work/rui-barbosa    # re-fetch a work folder's photos
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "WPGT-Mirante/0.1 (https://commons.wikimedia.org/wiki/Commons:WikiProject_GeoTwin; User:Rkieferbaum)"
# Wikimedia serves only a fixed set of thumbnail widths; others are refused with HTTP 400 (checked 2026-09).
STANDARD_WIDTHS = (120, 250, 330, 500, 960, 1280, 1920, 3840)
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

# Sensor widths (mm) for common drones/phones when EXIF lacks FocalLengthIn35mmFilm.
SENSOR_WIDTH_MM = {
    "FC3170": 6.3, "FC3411": 13.2, "FC3582": 9.6, "FC7303": 6.3, "FC8482": 9.6,  # DJI Mini/Air
    "FC6310": 13.2, "FC6310S": 13.2, "FC220": 6.17, "FC330": 6.17, "L1D-20c": 13.2,
    "L2D-20c": 17.3, "FC4170": 17.3, "M3E": 17.3, "ZH20T": 7.4,
}


class Commons:
    def __init__(self, sleep: float = 0.2):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.sleep = sleep

    def get(self, **params):
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(6):
            r = self.s.get(API, params=params, timeout=60)
            if r.status_code == 429 or "maxlag" in r.text[:200]:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(self.sleep)
            return r.json()
        raise RuntimeError(f"API kept refusing: {params}")

    def query_all(self, **params):
        cont = {}
        while True:
            data = self.get(action="query", **params, **cont)
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]


# --------------------------------------------------------------------------- helpers

def _num(v):
    """Parse EXIF-ish numbers: 24, '24', '24/1', '4.5', [a, b] -> float or None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list) and v:
        return _num(v[0])
    if isinstance(v, dict):
        return None
    try:
        return float(Fraction(str(v).strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _strip_html(s: str | None) -> str | None:
    if not s:
        return None
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip() or None


def _metadata_dict(meta) -> dict:
    out = {}
    for m in meta or []:
        name, value = m.get("name"), m.get("value")
        if isinstance(value, list) and value and isinstance(value[0], dict) and "name" in value[0]:
            value = _metadata_dict(value)
        out[name] = value
    return out


def _exif_gps(md: dict):
    lat, lon = _num(md.get("GPSLatitude")), _num(md.get("GPSLongitude"))
    if lat is None or lon is None or (lat == 0 and lon == 0):
        return None
    # MediaWiki normally stores signed decimals already; honour explicit refs if present.
    if str(md.get("GPSLatitudeRef", "")).upper().startswith("S") and lat > 0:
        lat = -lat
    if str(md.get("GPSLongitudeRef", "")).upper().startswith("W") and lon > 0:
        lon = -lon
    alt = _num(md.get("GPSAltitude"))
    if alt is not None and str(md.get("GPSAltitudeRef", "0")) in ("1", "b'\\x01'"):
        alt = -alt
    heading = _num(md.get("GPSImgDirection"))
    return {"lat": lat, "lon": lon, "alt": alt, "heading": heading, "source": "exif"}


def _focal_prior_px(md: dict, orig_w: int, orig_h: int, thumb_w: int, thumb_h: int):
    """Focal length prior in thumbnail pixels, or None."""
    long_thumb = max(thumb_w, thumb_h)
    f35 = _num(md.get("FocalLengthIn35mmFilm"))
    if f35 and f35 > 5:
        return f35 / 36.0 * long_thumb, "exif_35mm"
    f_mm = _num(md.get("FocalLength"))
    model = str(md.get("Model", "")).strip()
    sw = SENSOR_WIDTH_MM.get(model)
    if f_mm and sw:
        return f_mm / sw * long_thumb, f"sensor_db:{model}"
    # Focal plane resolution route
    fpx = _num(md.get("FocalPlaneXResolution"))
    unit = _num(md.get("FocalPlaneResolutionUnit"))
    if f_mm and fpx and unit in (2, 3):
        px_per_mm = fpx / (25.4 if unit == 2 else 10.0)
        return f_mm * px_per_mm * (long_thumb / max(orig_w, orig_h)), "exif_focal_plane"
    return None, None


def _sdc_location(entity: dict):
    """Camera location from structured data (P1259), object location (P625)."""
    stmts = (entity or {}).get("statements") or (entity or {}).get("claims") or {}

    def coord(pid):
        for st in stmts.get(pid, []):
            dv = st.get("mainsnak", {}).get("datavalue", {}).get("value")
            if not dv:
                continue
            quals = st.get("qualifiers", {})

            def q(p):
                for qq in quals.get(p, []):
                    v = qq.get("datavalue", {}).get("value")
                    if isinstance(v, dict) and "amount" in v:
                        return float(v["amount"])
                return None
            return {"lat": dv["latitude"], "lon": dv["longitude"], "alt": q("P2044"),
                    "heading": q("P7787"), "precision": dv.get("precision")}
        return None

    cam, obj = coord("P1259"), coord("P625")
    if cam:
        cam["source"] = "sdc_P1259"
    return cam, obj


def _ground_distance_m(loc: dict, lat0: float, lon0: float) -> float:
    """Horizontal distance (equirectangular; fine over a few km)."""
    import math
    dy = (loc["lat"] - lat0) * 111_320.0
    dx = (loc["lon"] - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


def _safe_name(pageid: int, title: str, ext: str = ".jpg") -> str:
    import unicodedata
    stem = unicodedata.normalize("NFKD", title.removeprefix("File:").rsplit(".", 1)[0]).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9\-]+", "_", stem).strip("_")[:80]
    return f"{pageid}_{stem}{ext}"


# --------------------------------------------------------------------------- harvesting

def list_files(c: Commons, category: str, depth: int = 0, exclude=(), excluded_ids: set | None = None) -> list[dict]:
    """Files in a category and its subcategories down to `depth`; subcategories whose title matches one of
    the `exclude` regexes are skipped (the top category never is). If `excluded_ids` is given, the page ids of
    the files directly in skipped subcategories are added to it (to keep them out of other sources too)."""
    if not category.startswith("Category:"):
        category = "Category:" + category
    rx = [re.compile(p, re.I) for p in exclude]
    seen_cats, files = set(), {}
    frontier = [(category, 0)]
    while frontier:
        cat, d = frontier.pop()
        if cat in seen_cats:
            continue
        if d > 0 and any(r.search(cat) for r in rx):
            seen_cats.add(cat)
            if excluded_ids is not None:
                for data in c.query_all(list="categorymembers", cmtitle=cat, cmtype="file", cmlimit="500"):
                    excluded_ids.update(m["pageid"] for m in data["query"]["categorymembers"])
            continue
        seen_cats.add(cat)
        for data in c.query_all(list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit="500"):
            for m in data["query"]["categorymembers"]:
                if m["ns"] == 6:
                    files[m["pageid"]] = m["title"]
                elif m["ns"] == 14 and d < depth:
                    frontier.append((m["title"], d + 1))
    print(f"{len(files)} files in {len(seen_cats)} categor{'y' if len(seen_cats) == 1 else 'ies'}", file=sys.stderr)
    return [{"pageid": k, "title": v} for k, v in files.items()]


def geosearch_grid(c: Commons, lat_min, lat_max, lon_min, lon_max, step_m=100.0) -> dict[int, tuple]:
    """GeoData search on a grid of circles (the API returns at most 500 hits per query, no continuation).
    -> {pageid: (title, lat, lon)} for files whose camera or object coordinates fall in the box."""
    import math
    files = {}
    dlat = step_m / 111_320.0
    dlon = step_m / (111_320.0 * math.cos(math.radians((lat_min + lat_max) / 2)))
    radius = step_m * 0.75  # > step/sqrt(2): circles overlap and cover the grid cells
    capped = 0
    lat = lat_min
    while lat <= lat_max + dlat:
        lon = lon_min
        while lon <= lon_max + dlon:
            for attempt in range(4):
                data = c.get(action="query", list="geosearch", gscoord=f"{lat:.6f}|{lon:.6f}",
                             gsradius=str(int(radius)), gsnamespace="6", gslimit="500", gsprimary="all")
                if "query" in data:
                    break
                print(f"  geosearch error at {lat:.6f},{lon:.6f}: {data.get('error', data)}", file=sys.stderr)
                time.sleep(2 ** attempt)
            else:
                lon += dlon
                continue
            hits = data["query"]["geosearch"]
            capped += len(hits) >= 500
            for h in hits:
                files[h["pageid"]] = (h["title"], h["lat"], h["lon"])
            lon += dlon
        lat += dlat
    print(f"geosearch: {len(files)} files" + (f" ({capped} cells hit the 500 cap; use a smaller step)" if capped else ""),
          file=sys.stderr)
    return files


def near_files(c: Commons, lat0: float, lon0: float, radius_m: float) -> list[dict]:
    """Files geotagged (camera or object location) within radius_m of lat0, lon0."""
    mlat = radius_m / 111_320.0
    mlon = radius_m / (111_320.0 * __import__("math").cos(__import__("math").radians(lat0)))
    step = min(100.0, max(30.0, radius_m / 3))
    hits = geosearch_grid(c, lat0 - mlat, lat0 + mlat, lon0 - mlon, lon0 + mlon, step)
    out = [{"pageid": k, "title": t} for k, (t, la, lo) in hits.items()
           if _ground_distance_m({"lat": la, "lon": lo}, lat0, lon0) <= radius_m]
    print(f"{len(out)} files geotagged within {radius_m:g} m", file=sys.stderr)
    return out


def fetch_details(c: Commons, files: list[dict], width: int) -> list[dict]:
    out = []
    for i in range(0, len(files), 50):
        batch = files[i:i + 50]
        ids = "|".join(str(f["pageid"]) for f in batch)
        data = c.get(action="query", pageids=ids, prop="imageinfo|coordinates",
                     iiprop="url|size|mime|extmetadata|metadata", iiurlwidth=str(width),
                     iiextmetadatafilter="Artist|LicenseShortName|LicenseUrl|ImageDescription|DateTimeOriginal|Credit|AttributionRequired",
                     coprop="type|name|dim|globe", coprimary="all", colimit="max")
        sdc = c.get(action="wbgetentities", ids="|".join(f"M{f['pageid']}" for f in batch)).get("entities", {})
        for page in data["query"]["pages"]:
            ii = (page.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in IMAGE_MIMES:
                continue
            md = _metadata_dict(ii.get("metadata"))
            em = {k: v.get("value") for k, v in (ii.get("extmetadata") or {}).items()}
            geo_primary = geo_object = None
            for co in page.get("coordinates", []):
                rec = {"lat": co["lat"], "lon": co["lon"], "source": "geodata"}
                if co.get("primary"):
                    geo_primary = rec
                else:
                    geo_object = rec
            sdc_cam, sdc_obj = _sdc_location(sdc.get(f"M{page['pageid']}"))
            exif = _exif_gps(md)
            camera = sdc_cam or exif or geo_primary
            if camera and camera.get("heading") is None:
                camera["heading"] = (exif or {}).get("heading")
            if camera and camera.get("alt") is None and exif:
                camera["alt"] = exif.get("alt")
            tw, th = ii.get("thumbwidth") or ii["width"], ii.get("thumbheight") or ii["height"]
            f_px, f_src = _focal_prior_px(md, ii["width"], ii["height"], tw, th)
            out.append({
                "pageid": page["pageid"],
                "title": page["title"],
                "file_page": ii.get("descriptionurl"),
                "original_url": ii.get("url"),
                "thumb_url": ii.get("thumburl") or ii.get("url"),
                "original_size": [ii["width"], ii["height"]],
                "thumb_size": [tw, th],
                "camera_location": camera,
                "object_location": sdc_obj or geo_object,
                "focal_prior_px": f_px,
                "focal_prior_source": f_src,
                "exif": {k: md.get(k) for k in ("Make", "Model", "FocalLength", "FocalLengthIn35mmFilm",
                                                "DateTimeOriginal", "Orientation") if md.get(k) is not None},
                "author": _strip_html(em.get("Artist")),
                "credit": _strip_html(em.get("Credit")),
                "license": em.get("LicenseShortName"),
                "license_url": em.get("LicenseUrl"),
                "description": _strip_html(em.get("ImageDescription")),
                "date": _strip_html(em.get("DateTimeOriginal")),
                "file": _safe_name(page["pageid"], page["title"]),
            })
        print(f"  details {min(i + 50, len(files))}/{len(files)}", file=sys.stderr)
    return out


def download(c: Commons, items: list[dict], img_dir: Path):
    img_dir.mkdir(parents=True, exist_ok=True)
    for n, it in enumerate(items, 1):
        dst = img_dir / it["file"]
        if dst.exists() and dst.stat().st_size > 0:
            continue
        for attempt in range(6):
            r = c.s.get(it["thumb_url"], timeout=120)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            dst.write_bytes(r.content)
            break
        else:
            print(f"  giving up on {it['title']}", file=sys.stderr)
        time.sleep(c.sleep)
        if n % 10 == 0:
            print(f"  downloaded {n}/{len(items)}", file=sys.stderr)


def search_categories(c: Commons, text: str):
    data = c.get(action="query", list="search", srsearch=f"intitle:{text}", srnamespace="14", srlimit="50")
    for r in data["query"]["search"]:
        print(r["title"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category", nargs="?")
    ap.add_argument("workdir", nargs="?")
    ap.add_argument("--depth", type=int, default=0, help="subcategory recursion depth")
    ap.add_argument("--width", type=int, default=1920, choices=STANDARD_WIDTHS, help="thumbnail width")
    ap.add_argument("--require-location", action="store_true", help="drop files without a camera location")
    ap.add_argument("--within", nargs=3, type=float, metavar=("LAT", "LON", "METRES"),
                    help="keep only files whose camera position is within METRES of LAT, LON")
    ap.add_argument("--exclude", action="append", default=[], metavar="REGEX",
                    help="skip subcategories whose title matches (repeatable), e.g. Interior")
    ap.add_argument("--near", nargs=3, type=float, metavar=("LAT", "LON", "METRES"),
                    help="also include files geotagged within METRES of LAT, LON (camera or object location)")
    ap.add_argument("--anchor-category", action="append", default=[], metavar="CAT",
                    help="also harvest this category and flag its files as anchors: accurately positioned photos "
                         "(e.g. a WPGT set) that alone get position priors and drive the georeferencing")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--search", metavar="TEXT", help="list categories whose title matches TEXT and exit")
    ap.add_argument("--download", metavar="WORKDIR",
                    help="(re)download the photos of an existing work folder (manifest.json and extra/manifest.json) "
                         "and exit; files already present are skipped")
    a = ap.parse_args(argv)
    c = Commons()
    if a.search:
        return search_categories(c, a.search)
    if a.download:
        wd = Path(a.download)
        items = []
        for m in (wd / "manifest.json", wd / "extra" / "manifest.json"):
            if m.exists():
                items += json.loads(m.read_text())["items"]
        todo = [it for it in {it["file"]: it for it in items}.values()
                if not ((wd / "images" / it["file"]).exists() and (wd / "images" / it["file"]).stat().st_size > 0)]
        print(f"{len(todo)} of {len(items)} photos to download into {wd / 'images'}", file=sys.stderr)
        return download(c, todo, wd / "images")
    if not (a.category and a.workdir):
        ap.error("category and workdir are required")
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    anchor_files = {f["pageid"]: f for cat in a.anchor_category for f in list_files(c, cat, 0)}
    excluded_ids: set = set()
    files = {f["pageid"]: f for f in list_files(c, a.category, a.depth, a.exclude, excluded_ids)}
    if a.near:
        extra = [f for f in near_files(c, *a.near) if f["pageid"] not in files]
        files.update({f["pageid"]: f for f in extra})
        print(f"  {len(extra)} of them not already in the category", file=sys.stderr)
    files = [f for k, f in files.items() if k not in anchor_files and k not in excluded_ids]
    if excluded_ids:
        print(f"  {len(excluded_ids)} files from excluded subcategories left out", file=sys.stderr)
    items = fetch_details(c, files + list(anchor_files.values()), a.width)
    for it in items:
        if it["pageid"] in anchor_files:
            it["anchor"] = True
    located = sum(1 for it in items if it["camera_location"])
    print(f"{len(items)} images, {located} with a camera location", file=sys.stderr)
    if a.require_location or a.within:
        items = [it for it in items if it["camera_location"]]
    if a.within:
        lat0, lon0, r = a.within
        items = [it for it in items if _ground_distance_m(it["camera_location"], lat0, lon0) <= r]
        print(f"{len(items)} images with a camera position within {r:g} m", file=sys.stderr)
    manifest = {"category": a.category, "within": a.within, "near": a.near, "anchor_categories": a.anchor_category, "harvested": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "thumb_width": a.width, "items": items}
    (wd / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    if not a.no_download:
        download(c, items, wd / "images")
    print(f"wrote {wd / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
