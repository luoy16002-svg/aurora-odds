"""Live nowcast: NOAA real-time solar wind from L1 -> TabPFN -> site/data/nowcast.json.

Runs every ten minutes in GitHub Actions. Uses the same feature code as training; the only new
step is time-shifting each L1 sample to its arrival at Earth's bow shock, which OMNI had done
for the history.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, features_at, prepare  # noqa: E402
from model import bar_probs, fit_tabpfn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RTSW = "https://services.swpc.noaa.gov/json/rtsw/rtsw_{}.json"
BSN_KM = 90_000          # typical distance of the bow shock nose from Earth's centre
LEVELS = np.round(np.arange(1, 13.01, 1 / 3), 3)  # Hp30 values the page can ask about
HISTORY_HOURS = 12


def active(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["active"]] if df["active"].any() else df[df["source"] == df["source"].iloc[0]]
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    return df.sort_values("time_tag")


def solar_wind() -> tuple[pd.DataFrame, dict]:
    get = lambda name: requests.get(RTSW.format(name), timeout=60).json()  # noqa: E731
    mag, wind, eph = active(get("mag_1m")), active(get("wind_1m")), active(get("ephemerides_1h"))
    m = mag.set_index("time_tag")[["bt", "by_gsm", "bz_gsm"]].rename(columns={"by_gsm": "by", "bz_gsm": "bz"})
    w = wind.set_index("time_tag")[["proton_speed", "proton_density"]].rename(columns={"proton_speed": "v", "proton_density": "n"})
    sw = m.join(w, how="outer").astype(float).resample("5min").mean()
    sw["pdyn"] = 2e-6 * sw["n"] * sw["v"] ** 2               # nPa, the OMNI high-resolution definition
    x_km = float(eph["x_gse"].dropna().iloc[-1]) if not eph.empty else 1.5e6
    v = sw["v"].interpolate(limit_direction="both").clip(lower=250)
    sw["shift"] = (x_km - BSN_KM) / v                         # seconds from L1 to the bow shock nose
    sw.index = sw.index + pd.to_timedelta(sw["shift"], unit="s")  # measurement time -> arrival time
    sw = sw.rename_axis("time").reset_index()
    src = str(mag["source"].iloc[-1]) if not mag.empty else "unknown"
    return sw, {"spacecraft": src, "l1_distance_km": round(x_km), "last_measurement": str(m.index.max())}


def main() -> None:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("5min")
    sw, meta = solar_wind()
    sw = prepare(sw)
    times = [now - pd.Timedelta(minutes=30 * i) for i in range(2 * HISTORY_HOURS, -1, -1)]
    rows, kept = [], []
    for t in times:
        f = features_at(sw, t)
        if f is not None:
            rows.append([f[c] for c in FEATURES])
            kept.append(t)
    if not rows:
        raise SystemExit("no usable solar wind data")
    ctx = pd.read_parquet(ROOT / "model" / "context.parquet")
    cfg = json.loads((ROOT / "model" / "context.json").read_text())
    model = fit_tabpfn(ctx, "cpu", cfg.get("n_estimators", 4))
    probs, borders = bar_probs(model, np.asarray(rows, dtype=np.float32), cfg.get("weights"))
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, axis=1)], axis=1)
    exceed = []
    for k in LEVELS:
        cut = k - 1 / 6
        j = np.clip(np.searchsorted(borders, cut) - 1, 0, len(borders) - 2)
        frac = np.clip((cut - borders[j]) / (borders[j + 1] - borders[j]), 0, 1)
        exceed.append(1 - (cdf[:, j] + probs[:, j] * frac))
    exceed = np.clip(np.column_stack(exceed), 0, 1)
    centers = (borders[:-1] + borders[1:]) / 2
    median = [float(np.interp(0.5, np.cumsum(p), centers)) for p in probs]
    latest = features_at(sw, kept[-1])
    recent = sw[(sw["time"] > now - pd.Timedelta(hours=3)) & (sw["measured"] <= now)]
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "TabPFN v2 regressor, in-context on %d labelled half-hours (1998-2019)" % len(ctx),
        "levels": LEVELS.tolist(),
        "source": meta,
        "now": {
            "time": kept[-1].isoformat() + "Z",
            "exceed": np.round(exceed[-1], 4).tolist(),
            "median": round(median[-1], 2),
            "ahead_minutes": round(float(latest["ahead_min"]), 1) if latest else 0,
            "bz": round(float(latest["now_bz_mean"]), 2) if latest else None,
            "bz_min_ahead": None if not latest or np.isnan(latest["ahead_bz_min"]) else round(float(latest["ahead_bz_min"]), 2),
            "speed": round(float(latest["now_v_mean"]), 0) if latest else None,
            "density": round(float(latest["now_n_mean"]), 2) if latest else None,
            "bt": round(float(latest["now_bt_mean"]), 2) if latest else None,
        },
        "history": [{"time": t.isoformat() + "Z", "median": round(mdn, 2), "exceed": np.round(e, 4).tolist()}
                    for t, mdn, e in zip(kept, median, exceed)],
        "solar_wind": [{"time": r.time.isoformat() + "Z", "bz": None if np.isnan(r.bz) else round(float(r.bz), 2),
                        "v": None if np.isnan(r.v) else round(float(r.v))} for r in recent.itertuples()],
    }
    dest = ROOT / "site" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "nowcast.json").write_text(json.dumps(out, separators=(",", ":")))
    p6 = out["now"]["exceed"][int(np.argmin(np.abs(LEVELS - 6)))]
    print(f"{out['now']['time']}: median Hp30 {out['now']['median']}, P(Hp30>=6) {p6:.1%}, "
          f"ahead {out['now']['ahead_minutes']} min, spacecraft {meta['spacecraft']}")


if __name__ == "__main__":
    main()
