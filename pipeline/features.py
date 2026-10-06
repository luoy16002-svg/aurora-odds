"""Turn solar wind measured at L1 into the features the model sees at one moment.

The same code builds the training table from OMNI history and the live features from NOAA's
real-time feed, so the model is only ever shown what would really be known at that moment.

Times are Earth arrival times (OMNI's time shifted to the bow shock nose). A sample measured at
L1 at wall-clock time m arrives at Earth at m + shift, so at wall-clock time t we already know
the solar wind that will hit Earth until about t + shift: the "ahead" window. That window is
why a nowcast from L1 can look 20-60 minutes into the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {  # name: (start, end) in minutes relative to t, on the arrival-time axis
    "now": (-30, 0),
    "recent": (-90, -30),
    "earlier": (-180, -90),
}
FEATURES = [
    "ahead_min", "ahead_bz_min", "ahead_bz_mean", "ahead_newell_max", "ahead_newell_mean", "ahead_v_max", "ahead_pdyn_max",
    "now_bz_mean", "now_bz_min", "now_bt_mean", "now_v_mean", "now_n_mean", "now_pdyn_mean", "now_newell_mean",
    "recent_bz_mean", "recent_newell_mean", "recent_v_mean",
    "earlier_bz_mean", "earlier_newell_mean",
    "newell_3h", "south_minutes_3h",
    "doy_sin", "doy_cos", "ut_sin", "ut_cos",
]


def newell(v, by, bz):
    """Newell et al. (2007) coupling function dPhi_MP/dt = v^4/3 Bt^2/3 sin^8/3(theta/2), scaled to ~0-10."""
    bt = np.hypot(by, bz)
    theta = np.arctan2(by, bz)
    return 1e-3 * np.power(np.abs(v), 4 / 3) * np.power(bt, 2 / 3) * np.power(np.abs(np.sin(theta / 2)), 8 / 3)


def prepare(sw: pd.DataFrame) -> pd.DataFrame:
    """sw: rows with arrival time `time`, `shift` (s), bt, by, bz, v, n, pdyn. Adds derived columns, sorted."""
    sw = sw.sort_values("time").reset_index(drop=True).copy()
    sw["newell"] = newell(sw["v"], sw["by"], sw["bz"])
    sw["measured"] = sw["time"] - pd.to_timedelta(sw["shift"].fillna(sw["shift"].median()), unit="s")
    return sw


def _agg(block: pd.DataFrame, prefix: str, cols: dict[str, str]) -> dict[str, float]:
    out = {}
    for col, how in cols.items():
        s = block[col.split("_")[0]] if col.split("_")[0] in block else block[col]
        out[f"{prefix}_{col}"] = getattr(s, how)() if len(s.dropna()) else np.nan
    return out


def features_at(sw: pd.DataFrame, t: pd.Timestamp) -> dict[str, float] | None:
    """Features for moment t using only samples measured at L1 by wall-clock time t."""
    known = sw[sw["measured"] <= t]
    if known.empty:
        return None
    rel = (known["time"] - t).dt.total_seconds() / 60
    ahead = known[rel > 0]
    row: dict[str, float] = {"ahead_min": float(rel.max()) if len(ahead) else 0.0}
    a = ahead.dropna(subset=["bz"])
    row.update({
        "ahead_bz_min": a["bz"].min() if len(a) else np.nan,
        "ahead_bz_mean": a["bz"].mean() if len(a) else np.nan,
        "ahead_newell_max": a["newell"].max() if len(a) else np.nan,
        "ahead_newell_mean": a["newell"].mean() if len(a) else np.nan,
        "ahead_v_max": a["v"].max() if len(a) else np.nan,
        "ahead_pdyn_max": a["pdyn"].max() if len(a) else np.nan,
    })
    blocks = {name: known[(rel > lo) & (rel <= hi)] for name, (lo, hi) in WINDOWS.items()}
    now, recent, earlier = blocks["now"], blocks["recent"], blocks["earlier"]
    if now["bz"].notna().sum() < 2:
        return None
    row.update({
        "now_bz_mean": now["bz"].mean(), "now_bz_min": now["bz"].min(), "now_bt_mean": now["bt"].mean(),
        "now_v_mean": now["v"].mean(), "now_n_mean": now["n"].mean(), "now_pdyn_mean": now["pdyn"].mean(),
        "now_newell_mean": now["newell"].mean(),
        "recent_bz_mean": recent["bz"].mean(), "recent_newell_mean": recent["newell"].mean(), "recent_v_mean": recent["v"].mean(),
        "earlier_bz_mean": earlier["bz"].mean(), "earlier_newell_mean": earlier["newell"].mean(),
    })
    past = known[(rel > -180) & (rel <= 0)]
    row["newell_3h"] = past["newell"].mean()
    row["south_minutes_3h"] = 5.0 * float((past["bz"] < -5).sum())
    doy = t.dayofyear + t.hour / 24
    row.update({"doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "ut_sin": np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24), "ut_cos": np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24)})
    return row


def table(sw: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorised version of features_at for many moments (history). Equivalent output, much faster."""
    sw = prepare(sw)
    tt = sw["time"].values.astype("datetime64[m]").astype(np.int64)      # arrival minutes
    mm = sw["measured"].values.astype("datetime64[m]").astype(np.int64)  # L1 measurement minutes
    cols = {c: sw[c].to_numpy(dtype=float) for c in ["bz", "bt", "v", "n", "pdyn", "newell"]}
    out = []
    starts = np.searchsorted(tt, times.values.astype("datetime64[m]").astype(np.int64) - 180, side="right")
    ends = np.searchsorted(tt, times.values.astype("datetime64[m]").astype(np.int64) + 120, side="right")
    for t, i0, i1 in zip(times, starts, ends):
        tm = np.int64(t.value // 60_000_000_000)
        sl = slice(i0, i1)
        known = mm[sl] <= tm
        rel = (tt[sl] - tm)[known]
        get = {k: v[sl][known] for k, v in cols.items()}
        if not len(rel):
            out.append(None)
            continue
        def pick(lo, hi):
            m = (rel > lo) & (rel <= hi)
            return {k: v[m] for k, v in get.items()}
        def f(arr, how):
            arr = arr[~np.isnan(arr)]
            return getattr(np, how)(arr) if len(arr) else np.nan
        ah = pick(0, 10_000)
        nw, rc, er, past = pick(-30, 0), pick(-90, -30), pick(-180, -90), pick(-180, 0)
        if np.sum(~np.isnan(nw["bz"])) < 2:
            out.append(None)
            continue
        doy = t.dayofyear + t.hour / 24
        hour = t.hour + t.minute / 60
        out.append({
            "ahead_min": float(rel.max()) if np.any(rel > 0) else 0.0,
            "ahead_bz_min": f(ah["bz"], "min"), "ahead_bz_mean": f(ah["bz"], "mean"),
            "ahead_newell_max": f(ah["newell"], "max"), "ahead_newell_mean": f(ah["newell"], "mean"),
            "ahead_v_max": f(ah["v"], "max"), "ahead_pdyn_max": f(ah["pdyn"], "max"),
            "now_bz_mean": f(nw["bz"], "mean"), "now_bz_min": f(nw["bz"], "min"), "now_bt_mean": f(nw["bt"], "mean"),
            "now_v_mean": f(nw["v"], "mean"), "now_n_mean": f(nw["n"], "mean"), "now_pdyn_mean": f(nw["pdyn"], "mean"),
            "now_newell_mean": f(nw["newell"], "mean"),
            "recent_bz_mean": f(rc["bz"], "mean"), "recent_newell_mean": f(rc["newell"], "mean"), "recent_v_mean": f(rc["v"], "mean"),
            "earlier_bz_mean": f(er["bz"], "mean"), "earlier_newell_mean": f(er["newell"], "mean"),
            "newell_3h": f(past["newell"], "mean"),
            "south_minutes_3h": 5.0 * float(np.sum(past["bz"] < -5)),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "ut_sin": np.sin(2 * np.pi * hour / 24), "ut_cos": np.cos(2 * np.pi * hour / 24),
        })
    keep = [i for i, r in enumerate(out) if r is not None]
    df = pd.DataFrame([out[i] for i in keep], columns=FEATURES)
    df.insert(0, "time", times[keep])
    return df
