"""Write recovered camera locations (from `mirante.locations`) back to Wikimedia Commons.

For each row of <workdir>/recovered_locations.csv:
  * wikitext: the camera-location template ({{Location}}, {{Location dec}}, {{Camera location}}, ...) is replaced,
    or a new one is inserted after {{Information}}:  {{Location|lat|lon|alt:NNN_heading:NNN_source:WPGT}}
  * structured data: an existing P1259 (coordinates of the point of view) statement is corrected, with P7787
    (heading) and P2044 (elevation above sea level) qualifiers; files without P1259 are left alone.
Pages that are ambiguous (several camera-location templates, several P1259 statements, nowhere obvious to insert)
are skipped and listed for manual editing.

Nothing is written unless --apply is given. Without it, a plan (plan.json) and a preview (preview.html) are
written for review. --apply asks for your bot-password username (e.g. YourName@mirante, from
Special:BotPasswords) and password (not echoed), shows who is logged in and how many pages will be edited, and
waits for confirmation. It then re-checks that every page is unchanged since the preview, edits at a gentle
rate with maxlag, and logs each edit to applied.jsonl so an interrupted run resumes where it stopped.
(MIRANTE_BOT_USER / MIRANTE_BOT_PASSWORD in the environment, if set, skip the prompts.)

Usage:
  python -m mirante.commons_edit work/sao-francisco --ground-asl 803.9            # plan + preview
  python -m mirante.commons_edit work/sao-francisco --ground-asl 803.9 --apply --limit 3
  python -m mirante.commons_edit work/sao-francisco --ground-asl 803.9 --apply
"""
from __future__ import annotations

import argparse
import csv
import getpass
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from .harvest import API, USER_AGENT

LOCATION_TEMPLATES = {"location", "location dec", "location decimal", "camera location", "camera location dec"}
INFO_TEMPLATES = ("information", "photograph", "artwork", "art photo")
SUMMARY = {"insert": "Inserting camera location with coordinates extracted from WPGT Mirante "
                     "([[Commons:WikiProject GeoTwin|WikiProject GeoTwin]])",
           "update": "Updating camera location with coordinates extracted from WPGT Mirante "
                     "([[Commons:WikiProject GeoTwin|WikiProject GeoTwin]])"}
SDC_SUMMARY = ("Correcting coordinates of the point of view (P1259) with coordinates extracted from WPGT Mirante "
               "([[Commons:WikiProject GeoTwin|WikiProject GeoTwin]])")
Q_DEGREE, Q_METRE, Q_EARTH = "Q28390", "Q11573", "http://www.wikidata.org/entity/Q2"
CONFIDENCE = {"low": 0, "medium": 1, "high": 2}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- wikitext

def top_level_templates(text):
    """(start, end, name) of each top-level {{...}} in text (brace matching; ignores nested ones)."""
    out, depth, start = [], 0, None
    i = 0
    while i < len(text) - 1:
        two = text[i:i + 2]
        if two == "{{":
            if depth == 0:
                start = i
            depth += 1
            i += 2
            continue
        if two == "}}" and depth:
            depth -= 1
            i += 2
            if depth == 0:
                name = re.split(r"[|}\n]", text[start + 2:i], maxsplit=1)[0].strip()
                out.append((start, i, name))
            continue
        i += 1
    return out


def norm(name):
    name = name.replace("_", " ").strip()
    return (name[:1].lower() + name[1:]).lower() if name else name


def alt_msl(row, ground_asl):
    """Sea-level altitude: from the CSV when the model's heights are absolute, else ground level + height."""
    if row.get("alt_msl_m"):
        return round(float(row["alt_msl_m"]))
    if ground_asl is None:
        raise SystemExit("heights are relative to local ground: pass --ground-asl")
    return round(ground_asl + float(row["height_above_ground_m"]))


def location_template(row, ground_asl):
    alt = alt_msl(row, ground_asl)
    return f"{{{{Location|{float(row['sfm_lat']):.6f}|{float(row['sfm_lon']):.6f}|alt:{alt}_heading:{int(row['heading_deg'])}_source:WPGT}}}}"


def plan_wikitext(text, new_tpl):
    """-> (action, new_text, old_template) with action in insert/update/manual:<reason>."""
    tpls = top_level_templates(text)
    locs = [t for t in tpls if norm(t[2]) in LOCATION_TEMPLATES]
    if len(locs) > 1:
        return "manual:several camera-location templates", None, None
    if locs:
        s, e, _ = locs[0]
        return "update", text[:s] + new_tpl + text[e:], text[s:e]
    if re.search(r"\{\{\s*(Location|Camera location)", text, re.I):
        return "manual:camera-location template nested inside another template", None, None
    info = [t for t in tpls if norm(t[2]) in INFO_TEMPLATES]
    if not info:
        return "manual:no {{Information}} template to insert after", None, None
    e = info[0][1]
    return "insert", text[:e] + "\n" + new_tpl + text[e:], None


# --------------------------------------------------------------------------- structured data

def _quantity(v, unit):
    return {"amount": f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}", "unit": f"http://www.wikidata.org/entity/{unit}"}


def plan_sdc(entity, row, ground_asl):
    """-> (action, new_claim, old_value) with action in update/none/manual:<reason>."""
    claims = (entity or {}).get("statements") or (entity or {}).get("claims") or {}
    p1259 = claims.get("P1259", [])
    if not p1259:
        return "none", None, None
    if len(p1259) > 1:
        return "manual:several P1259 statements", None, None
    old = p1259[0]
    claim = json.loads(json.dumps(old))  # keep id, rank and references
    claim["mainsnak"] = {"snaktype": "value", "property": "P1259", "datatype": "globe-coordinate",
                         "datavalue": {"type": "globecoordinate", "value": {
                             "latitude": round(float(row["sfm_lat"]), 6), "longitude": round(float(row["sfm_lon"]), 6),
                             "altitude": None, "precision": 1e-6, "globe": Q_EARTH}}}
    alt = alt_msl(row, ground_asl)
    quals = {k: v for k, v in (old.get("qualifiers") or {}).items() if k not in ("P7787", "P2044")}
    quals["P7787"] = [{"snaktype": "value", "property": "P7787", "datatype": "quantity",
                       "datavalue": {"type": "quantity", "value": _quantity(int(row["heading_deg"]), Q_DEGREE)}}]
    quals["P2044"] = [{"snaktype": "value", "property": "P2044", "datatype": "quantity",
                       "datavalue": {"type": "quantity", "value": _quantity(alt, Q_METRE)}}]
    claim["qualifiers"] = quals
    claim["qualifiers-order"] = [k for k in (old.get("qualifiers-order") or []) if k in quals] + \
        [k for k in quals if k not in (old.get("qualifiers-order") or [])]
    return "update", claim, old["mainsnak"].get("datavalue", {}).get("value")


# --------------------------------------------------------------------------- plan

def build_plan(workdir: Path, ground_asl: float, min_confidence: str):
    rows = list(csv.DictReader(open(workdir / "recovered_locations.csv", encoding="utf-8")))
    keep = [r for r in rows if CONFIDENCE[r["confidence"]] >= CONFIDENCE[min_confidence]]
    skipped_conf = [r for r in rows if r not in keep]
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    plan = []
    for i in range(0, len(keep), 50):
        batch = keep[i:i + 50]
        q = s.get(API, params={"action": "query", "format": "json", "formatversion": "2",
                               "titles": "|".join(r["title"] for r in batch), "prop": "revisions",
                               "rvprop": "ids|timestamp|content", "rvslots": "main"}, timeout=60).json()
        pages = {p["title"]: p for p in q["query"]["pages"]}
        norm_map = {n["from"]: n["to"] for n in q["query"].get("normalized", [])}
        ids = [f"M{p['pageid']}" for p in pages.values() if "pageid" in p]
        ents = s.get(API, params={"action": "wbgetentities", "format": "json", "ids": "|".join(ids)},
                     timeout=60).json().get("entities", {})
        for r in batch:
            p = pages.get(norm_map.get(r["title"], r["title"]))
            if not p or "revisions" not in p:
                plan.append({"title": r["title"], "action": "manual:page not found"})
                continue
            rev = p["revisions"][0]
            text = rev["slots"]["main"]["content"]
            tpl = location_template(r, ground_asl)
            action, new_text, old_tpl = plan_wikitext(text, tpl)
            mid = f"M{p['pageid']}"
            sdc_action, claim, old_val = plan_sdc(ents.get(mid), r, ground_asl)
            plan.append({"title": p["title"], "pageid": p["pageid"], "mid": mid, "revid": rev["revid"],
                         "timestamp": rev["timestamp"], "status": r["status"], "confidence": r["confidence"],
                         "geotag_offset_m": r["geotag_offset_m"], "num_3d_points": r["num_3d_points"],
                         "action": action, "old_template": old_tpl, "new_template": tpl, "new_text": new_text,
                         "summary": SUMMARY.get(action), "sdc_action": sdc_action, "sdc_claim": claim,
                         "sdc_old": old_val})
        time.sleep(0.5)
    return plan, skipped_conf


def write_preview(out: Path, plan, skipped_conf, ground_asl):
    def esc(x):
        return html.escape(str(x)) if x is not None else "<i>none</i>"
    rows = []
    for p in plan:
        link = f'<a href="https://commons.wikimedia.org/wiki/{html.escape(p["title"].replace(" ", "_"))}">{esc(p["title"][5:])}</a>'
        sdc = ""
        if p.get("sdc_action") == "update":
            o = p["sdc_old"] or {}
            n = p["sdc_claim"]["mainsnak"]["datavalue"]["value"]
            sdc = f'{o.get("latitude")}, {o.get("longitude")} &rarr; {n["latitude"]}, {n["longitude"]} (+heading, elevation)'
        elif p.get("sdc_action"):
            sdc = esc(p["sdc_action"])
        rows.append(f'<tr class="{p["action"].split(":")[0]}"><td>{link}</td><td>{esc(p["action"])}</td>'
                    f'<td><code>{esc(p.get("old_template"))}</code></td><td><code>{esc(p.get("new_template"))}</code></td>'
                    f'<td>{sdc}</td><td>{esc(p.get("confidence"))}</td><td>{esc(p.get("geotag_offset_m"))}</td></tr>')
    counts = {}
    for p in plan:
        counts[p["action"].split(":")[0]] = counts.get(p["action"].split(":")[0], 0) + 1
    n_sdc = sum(1 for p in plan if p.get("sdc_action") == "update")
    skipped = "".join(f"<li>{esc(r['title'][5:])} ({r['num_3d_points']} points)</li>" for r in skipped_conf)
    (out / "preview.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Location edits preview · WPGT Mirante</title>
<style>body{{font:13px/1.45 system-ui,sans-serif;margin:16px;background:#fff;color:#111}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}} code{{font-size:11.5px;word-break:break-all}}
tr.manual td{{background:#fff3cd}} tr.insert td:nth-child(2){{color:#0a7}} tr.update td:nth-child(2){{color:#06c}}
@media (prefers-color-scheme: dark){{body{{background:#111;color:#eee}} td,th{{border-color:#333}} tr.manual td{{background:#3a3212}}}}</style>
</head><body><h1>Location edits preview</h1>
<p>{counts} wikitext edits; {n_sdc} structured-data (P1259) corrections. Altitude: {"sea-level height from the model (anchors' ellipsoidal heights corrected with EGM2008)" if ground_asl is None else f"{ground_asl:.1f} m (local ground, from the placed photos' elevations) + SfM height above ground"}. Summaries: <i>{html.escape(SUMMARY["insert"])}</i> /
<i>{html.escape(SUMMARY["update"])}</i>.</p>
<p>Excluded for low confidence (fewer than 50 supporting 3D points): <ul>{skipped or "<li>none</li>"}</ul></p>
<table><thead><tr><th>File</th><th>Action</th><th>Current template</th><th>New template</th><th>SDC P1259</th>
<th>Confidence</th><th>Old geotag offset (m)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></body></html>""")


# --------------------------------------------------------------------------- apply

class Wiki:
    def __init__(self):
        user = os.environ.get("MIRANTE_BOT_USER") or input("Bot-password username (e.g. YourName@mirante): ").strip()
        pw = os.environ.get("MIRANTE_BOT_PASSWORD") or getpass.getpass("Bot password (not shown): ")
        if not user or not pw:
            raise SystemExit("a username and password are needed (create them at Special:BotPasswords)")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        tok = self.get(action="query", meta="tokens", type="login")["query"]["tokens"]["logintoken"]
        r = self.post(action="login", lgname=user, lgpassword=pw, lgtoken=tok)
        if r.get("login", {}).get("result") != "Success":
            raise SystemExit(f"login failed: {r.get('login', {}).get('reason', r)}")
        self.user = r["login"]["lgusername"]
        del pw
        self.csrf = self.get(action="query", meta="tokens")["query"]["tokens"]["csrftoken"]

    def _call(self, method, **params):
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(8):
            r = (self.s.post(API, data=params, timeout=60) if method == "post"
                 else self.s.get(API, params=params, timeout=60))
            j = r.json()
            if j.get("error", {}).get("code") == "maxlag" or r.status_code in (429, 503):
                time.sleep(int(r.headers.get("Retry-After", 5 * (attempt + 1))))
                continue
            return j
        raise RuntimeError("server kept refusing (maxlag)")

    def get(self, **p):
        return self._call("get", **p)

    def post(self, **p):
        return self._call("post", **p)


def apply(out: Path, limit=None, delay=10.0):
    plan = json.loads((out / "plan.json").read_text())
    done_path = out / "applied.jsonl"
    done = {json.loads(l)["title"] for l in done_path.read_text().splitlines() if l.strip()} if done_path.exists() else set()
    todo = [p for p in plan if p["action"] in ("insert", "update") and p["title"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return log(f"nothing to do ({len(done)} pages already done)")
    w = Wiki()
    n_sdc = sum(1 for p in todo if p.get("sdc_action") == "update")
    log(f"logged in as {w.user}; {len(todo)} pages to edit ({n_sdc} with SDC corrections; {len(done)} already done)")
    if input("Make these edits now? [y/N] ").strip().lower() not in ("y", "yes"):
        return log("cancelled, nothing edited")
    for k, p in enumerate(todo, 1):
        cur = w.get(action="query", pageids=p["pageid"], prop="revisions", rvprop="ids")["query"]["pages"][0]
        if cur["revisions"][0]["revid"] != p["revid"]:
            log(f"  [{k}] SKIP (page changed since the preview): {p['title']}")
            with open(done_path, "a") as f:
                f.write(json.dumps({"title": p["title"], "result": "skipped: changed since preview"}) + "\n")
            continue
        r = w.post(action="edit", pageid=p["pageid"], text=p["new_text"], summary=p["summary"],
                   baserevid=p["revid"], nocreate="1", token=w.csrf)
        res = {"title": p["title"], "edit": r.get("edit", r.get("error"))}
        if p.get("sdc_action") == "update" and r.get("edit", {}).get("result") == "Success":
            time.sleep(delay / 2)
            rs = w.post(action="wbsetclaim", claim=json.dumps(p["sdc_claim"]), summary=SDC_SUMMARY,
                        baserevid=r["edit"]["newrevid"], token=w.csrf)
            res["sdc"] = "Success" if rs.get("success") else rs.get("error")
        with open(done_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        log(f"  [{k}/{len(todo)}] {p['action']}{' + SDC' if 'sdc' in res else ''}: {p['title']} -> "
            f"{res['edit'].get('result', res['edit']) if isinstance(res['edit'], dict) else res['edit']}"
            + (f" / SDC {res['sdc']}" if 'sdc' in res else ""))
        time.sleep(delay)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--ground-asl", type=float,
                    help="elevation of the local ground above sea level (m), for models whose heights are relative "
                         "to local ground (anchors without altitude); alt = this + SfM height above ground")
    ap.add_argument("--min-confidence", default="medium", choices=list(CONFIDENCE))
    ap.add_argument("--apply", action="store_true", help="make the edits (default: plan + preview only)")
    ap.add_argument("--limit", type=int, help="with --apply: only the next N pages (for a trial run)")
    ap.add_argument("--delay", type=float, default=10.0, help="seconds between edits")
    a = ap.parse_args(argv)
    out = a.workdir / "commons_edits"
    out.mkdir(exist_ok=True)
    if a.apply:
        if not (out / "plan.json").exists():
            raise SystemExit("run without --apply first and review preview.html")
        return apply(out, a.limit, a.delay)
    plan, skipped = build_plan(a.workdir, a.ground_asl, a.min_confidence)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    write_preview(out, plan, skipped, a.ground_asl)
    counts = {}
    for p in plan:
        counts[p["action"]] = counts.get(p["action"], 0) + 1
    log(f"plan: {counts}; SDC corrections: {sum(1 for p in plan if p.get('sdc_action') == 'update')}; "
        f"excluded (confidence < {a.min_confidence}): {len(skipped)} -> {out / 'preview.html'}")


if __name__ == "__main__":
    main()
