"""Precompute AACGM-v2 magnetic latitude on a global grid for the page (site/aacgm.json).

The auroral oval's latitude rules of thumb are stated in corrected geomagnetic latitude. A plain dipole puts
Europe several degrees too far poleward, so the page interpolates this grid instead.
"""
import datetime as dt
import json
from pathlib import Path

import aacgmv2
import numpy as np

STEP = 2
lats = np.arange(-90, 90 + STEP, STEP, dtype=float)
lons = np.arange(-180, 180 + STEP, STEP, dtype=float)
when = dt.datetime(2026, 7, 1)
grid = []
for la in lats:
    mlat, _, _ = aacgmv2.convert_latlon_arr(np.full(lons.shape, la), lons, 110.0, when, method_code="G2A")
    grid.append([None if not np.isfinite(v) else round(float(v), 1) for v in mlat])
out = {"step": STEP, "lat0": -90, "lon0": -180, "epoch": "2026-07-01", "alt_km": 110, "mlat": grid}
Path(__file__).resolve().parents[1].joinpath("site", "aacgm.json").write_text(json.dumps(out, separators=(",", ":")))
for name, la, lo in [("Edinburgh", 55.95, -3.19), ("London", 51.51, -0.13), ("Tromso", 69.65, 18.96),
                     ("Minneapolis", 44.98, -93.27), ("Orlando", 28.54, -81.38), ("Hobart", -42.88, 147.33), ("Sapporo", 43.06, 141.35)]:
    m, _, _ = aacgmv2.convert_latlon(la, lo, 110.0, when, method_code="G2A")
    print(f"{name:12s} AACGM {m:6.1f}")
