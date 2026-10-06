"""Build the training/evaluation table: one row per half hour, features as known at that moment.

Target: the highest Hp30 in the coming hour (the two half-hour intervals starting at t).
Also stores `hp30_last`, the Hp30 of the half hour that just ended, for the persistence baseline.

Run: python pipeline/build_table.py  ->  data/table.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import table  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    omni = pd.read_parquet(DATA / "omni5.parquet").rename(columns={"timeshift": "shift"})
    hp = pd.read_parquet(DATA / "hp30.parquet").set_index("time")["hp30"]
    start = max(omni.time.min(), hp.index.min()) + pd.Timedelta(hours=3)
    end = min(omni.time.max(), hp.dropna().index.max()) - pd.Timedelta(minutes=30)
    times = pd.date_range(start.ceil("30min"), end.floor("30min"), freq="30min")
    t0 = time.time()
    tab = table(omni, times)
    print(f"features for {len(tab)}/{len(times)} moments in {time.time() - t0:.0f} s", flush=True)
    nxt = hp.reindex(tab.time).to_numpy()
    nxt2 = hp.reindex(tab.time + pd.Timedelta(minutes=30)).to_numpy()
    tab["y"] = np.fmax(nxt, nxt2)
    tab["hp30_last"] = hp.reindex(tab.time - pd.Timedelta(minutes=30)).to_numpy()
    tab = tab.dropna(subset=["y"]).reset_index(drop=True)
    tab.to_parquet(DATA / "table.parquet", index=False)
    share = {k: float((tab.y >= k).mean()) for k in (4, 5, 6, 7, 8)}
    print(f"saved {len(tab)} rows; share with next-hour Hp30 >= k: " + ", ".join(f"{k}: {v:.2%}" for k, v in share.items()))


if __name__ == "__main__":
    main()
