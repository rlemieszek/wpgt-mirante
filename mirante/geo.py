"""WGS84 <-> local East-North-Up (ENU) conversions, pure numpy."""
from __future__ import annotations

import numpy as np

_A = 6378137.0  # WGS84 semi-major axis (m)
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _enu_rotation(lat0, lon0):
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    # rows: east, north, up expressed in ECEF
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    ecef = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float))
    ref = geodetic_to_ecef(lat0, lon0, alt0)
    return (ecef - ref) @ _enu_rotation(lat0, lon0).T


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """Inverse of geodetic_to_enu (iterative Bowring-free solution, mm accuracy)."""
    enu = np.stack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)], axis=-1)
    ecef = enu @ _enu_rotation(lat0, lon0) + geodetic_to_ecef(lat0, lon0, alt0)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(6):
        n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + alt)))
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt
