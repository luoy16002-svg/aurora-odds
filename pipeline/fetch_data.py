"""Download the history this project learns from and store it compactly.

- OMNI 5-minute solar wind, time-shifted to Earth's bow shock (NASA SPDF), 1998 onwards.
- GFZ Hp30, the half-hourly open-ended planetary geomagnetic index (GFZ Potsdam, CC BY 4.0).

Run: python pipeline/fetch_data.py  ->  data/omni5.parquet, data/hp30.parquet
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc"
HP30_URL = "https://kp.gfz.de/app/files/Hp30_ap30_complete_series.txt"

# 1-based column numbers in the OMNI high-resolution format (omni_5min_def.txt)
OMNI_COLS = {
    "year": 1, "doy": 2, "hour": 3, "minute": 4,
    "timeshift": 10,           # seconds from the spacecraft to the bow shock nose
    "bt": 14,                  # field magnitude average, nT
    "by": 18, "bz": 19,        # GSM components, nT
    "v": 22,                   # flow speed, km/s
    "n": 26,                   # proton density, n/cc
    "temp": 27,                # proton temperature, K
    "pdyn": 28,                # flow pressure, nPa
}
FILL = {"timeshift": 999999, "bt": 9999.99, "by": 9999.99, "bz": 9999.99, "v": 99999.9,
        "n": 999.99, "temp": 9999999.0, "pdyn": 99.99}


def get(url: str, tries: int = 6) -> requests.Response:
    for k in range(tries):
        try:
            r = requests.get(url, timeout=(30, 300))
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} after {type(e).__name__}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(url)


def omni_year(year: int) -> pd.DataFrame:
    cache = DATA / "omni" / f"{year}.parquet"
    if cache.exists() and year < pd.Timestamp.utcnow().year:
        return pd.read_parquet(cache)
    r = get(OMNI_URL.format(year=year))
    cols = [c - 1 for c in OMNI_COLS.values()]
    raw = np.loadtxt(io.StringIO(r.text), usecols=cols)
    df = pd.DataFrame(raw, columns=list(OMNI_COLS))
    df["time"] = (pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
                  + pd.to_timedelta(df["doy"] - 1, unit="D")
                  + pd.to_timedelta(df["hour"], unit="h") + pd.to_timedelta(df["minute"], unit="min"))
    for col, fill in FILL.items():
        df.loc[df[col] >= fill * 0.999, col] = np.nan
    out = df[["time", *FILL]].astype({c: "float32" for c in FILL})
    out = out[out["time"] <= pd.Timestamp.now(tz="UTC").tz_localize(None)].reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def hp30() -> pd.DataFrame:
    r = get(HP30_URL)
    rows = [line.split() for line in r.text.splitlines() if line and not line.startswith("#")]
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 7]]
    df.columns = ["y", "m", "d", "h", "hp30"]
    df = df.astype(float)
    df["time"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d)) + pd.to_timedelta(df.h, unit="h")
    df.loc[df.hp30 < 0, "hp30"] = np.nan
    return df[["time", "hp30"]].astype({"hp30": "float32"})


def main(first: int = 1998, last: int | None = None) -> None:
    DATA.mkdir(exist_ok=True)
    last = last or pd.Timestamp.utcnow().year
    parts = []
    for year in range(first, last + 1):
        part = omni_year(year)
        ok = part["bz"].notna().mean()
        print(f"OMNI {year}: {len(part):6d} rows, Bz present {ok:.0%}", flush=True)
        parts.append(part)
    omni = pd.concat(parts, ignore_index=True)
    omni.to_parquet(DATA / "omni5.parquet", index=False)
    h = hp30()
    h.to_parquet(DATA / "hp30.parquet", index=False)
    print(f"saved {len(omni)} OMNI rows ({omni.time.min()} .. {omni.time.max()}), "
          f"{len(h)} Hp30 rows ({h.time.min()} .. {h.time.max()})")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
