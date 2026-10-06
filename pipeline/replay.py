"""Replays: the nowcast exactly as the page would have shown it at a past moment (site/data/replay-<name>.json).

Uses the OMNI history (already shifted to the bow shock, with its own propagation times), the same feature
code and the same TabPFN context as the live job, and keeps only what had been measured at L1 by that moment.

Run: python pipeline/replay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, features_at, prepare  # noqa: E402
from model import bar_probs, fit_tabpfn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.round(np.arange(1, 13.01, 1 / 3), 3)
REPLAYS = {
    "may2024": ("2024-05-10 22:00", "the Gannon superstorm", "London"),
    "oct2024": ("2024-10-11 02:00", "the October 2024 storm", "Chicago"),
}


def exceed_table(probs: np.ndarray, borders: np.ndarray) -> np.ndarray:
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, axis=1)], axis=1)
    cols = []
    for k in LEVELS:
        cut = k - 1 / 6
        j = np.clip(np.searchsorted(borders, cut) - 1, 0, len(borders) - 2)
        frac = np.clip((cut - borders[j]) / (borders[j + 1] - borders[j]), 0, 1)
        cols.append(1 - (cdf[:, j] + probs[:, j] * frac))
    return np.clip(np.column_stack(cols), 0, 1)


def main() -> None:
    omni = pd.read_parquet(ROOT / "data" / "omni5.parquet").rename(columns={"timeshift": "shift"})
    hp = pd.read_parquet(ROOT / "data" / "hp30.parquet").set_index("time")["hp30"]
    ctx = pd.read_parquet(ROOT / "model" / "context.parquet")
    cfg = json.loads((ROOT / "model" / "context.json").read_text())
    model = fit_tabpfn(ctx, "cuda" if torch.cuda.is_available() else "cpu", cfg.get("n_estimators", 4))
    for name, (when, label, place) in REPLAYS.items():
        now = pd.Timestamp(when)
        sw = prepare(omni[(omni.time > now - pd.Timedelta(hours=16)) & (omni.time < now + pd.Timedelta(hours=3))])
        times = [now - pd.Timedelta(minutes=30 * i) for i in range(24, -1, -1)]
        rows, kept = [], []
        for t in times:
            f = features_at(sw, t)
            if f is not None:
                rows.append([f[c] for c in FEATURES]); kept.append(t)
        probs, borders = bar_probs(model, np.asarray(rows, dtype=np.float32), cfg.get("weights"))
        exceed = exceed_table(probs, borders)
        centers = (borders[:-1] + borders[1:]) / 2
        median = [float(np.interp(0.5, np.cumsum(p), centers)) for p in probs]
        latest = features_at(sw, kept[-1])
        recent = sw[(sw["time"] > now - pd.Timedelta(hours=3)) & (sw["measured"] <= now)]
        actual = float(np.nanmax([hp.get(now, np.nan), hp.get(now + pd.Timedelta(minutes=30), np.nan)]))
        out = {
            "generated": now.isoformat() + "+00:00",
            "model": f"TabPFN v2 regressor, in-context on {len(ctx)} labelled half-hours (1998-2019)",
            "levels": LEVELS.tolist(),
            "source": {"spacecraft": "ACE/Wind (NASA OMNI)", "l1_distance_km": 1500000, "last_measurement": str(now)},
            "replay": {"name": name, "label": label, "time": now.isoformat() + "Z", "place": place, "actual_hp30": actual},
            "now": {
                "time": kept[-1].isoformat() + "Z",
                "exceed": np.round(exceed[-1], 4).tolist(),
                "median": round(median[-1], 2),
                "ahead_minutes": round(float(latest["ahead_min"]), 1),
                "bz": round(float(latest["now_bz_mean"]), 2),
                "bz_min_ahead": None if np.isnan(latest["ahead_bz_min"]) else round(float(latest["ahead_bz_min"]), 2),
                "speed": round(float(latest["now_v_mean"])),
                "density": round(float(latest["now_n_mean"]), 2),
                "bt": round(float(latest["now_bt_mean"]), 2),
            },
            "history": [{"time": t.isoformat() + "Z", "median": round(m, 2), "exceed": np.round(e, 4).tolist()}
                        for t, m, e in zip(kept, median, exceed)],
            "solar_wind": [{"time": r.time.isoformat() + "Z", "bz": None if np.isnan(r.bz) else round(float(r.bz), 2),
                            "v": None if np.isnan(r.v) else round(float(r.v))} for r in recent.itertuples()],
        }
        (ROOT / "site" / "data" / f"replay-{name}.json").write_text(json.dumps(out, separators=(",", ":")))
        print(f"{name} {now}: median {median[-1]:.2f}, actual next-hour max Hp30 {actual:.2f}, "
              f"P(>=7) {exceed[-1][int(np.argmin(np.abs(LEVELS - 7)))]:.2f}")


if __name__ == "__main__":
    main()
