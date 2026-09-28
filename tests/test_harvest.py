"""Offline test of the harvester's parsing against canned API responses (run: python tests/test_harvest.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirante import harvest  # noqa: E402

PAGE = {
    "pageid": 123, "title": "File:Praça Rui Barbosa 01.jpg",
    "imageinfo": [{
        "mime": "image/jpeg", "width": 4000, "height": 3000, "thumbwidth": 1920, "thumbheight": 1440,
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/X.jpg",
        "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/X.jpg/1920px-X.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "metadata": [
            {"name": "Make", "value": "DJI"}, {"name": "Model", "value": "FC3582"},
            {"name": "FocalLength", "value": "672/100"}, {"name": "FocalLengthIn35mmFilm", "value": 24},
            {"name": "GPSLatitude", "value": -19.9166}, {"name": "GPSLongitude", "value": -43.9339},
            {"name": "GPSAltitude", "value": "8905/10"}, {"name": "GPSAltitudeRef", "value": "0"},
        ],
        "extmetadata": {"Artist": {"value": '<a href="//commons.wikimedia.org/wiki/User:Rkieferbaum">Rafael</a>'},
                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"}},
    }],
    "coordinates": [{"lat": -19.9167, "lon": -43.934, "primary": True, "globe": "earth"},
                    {"lat": -19.9170, "lon": -43.9345, "globe": "earth"}],
}
SDC = {"M123": {"statements": {"P1259": [{"mainsnak": {"datavalue": {"value": {"latitude": -19.91661, "longitude": -43.93391, "precision": 1e-6}}},
                                          "qualifiers": {"P7787": [{"datavalue": {"value": {"amount": "+135.5"}}}]}}]}}}


class FakeCommons(harvest.Commons):
    def __init__(self):
        pass

    def get(self, **params):
        if params.get("action") == "wbgetentities":
            return {"entities": SDC}
        return {"query": {"pages": [PAGE]}}


items = harvest.fetch_details(FakeCommons(), [{"pageid": 123, "title": PAGE["title"]}], 1920)
it = items[0]
assert it["camera_location"]["source"] == "sdc_P1259", it["camera_location"]
assert abs(it["camera_location"]["heading"] - 135.5) < 1e-9
assert abs(it["camera_location"]["alt"] - 890.5) < 1e-9, "altitude should fall back to EXIF"
assert abs(it["focal_prior_px"] - 24 / 36 * 1920) < 1e-6
assert it["author"] == "Rafael" and it["license"] == "CC BY-SA 4.0"
assert it["object_location"]["lat"] == -19.9170
assert it["file"].startswith("123_Pra") and it["file"].endswith(".jpg")
assert harvest._num("24/1") == 24 and harvest._num(None) is None
print("harvest parsing OK:", it["file"], it["camera_location"])
