"""Choose the TabPFN context set, test it on solar cycle 25, compare with baselines.

Splits by time: context drawn from 1998-2019, choices made on 2020-2022, final test 2023 onwards
(the solar maximum, including the May 2024 and October 2024 superstorms).

Run: python pipeline/evaluate.py  ->  model/context.parquet, out/metrics.json, out/test_predictions.parquet
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES  # noqa: E402
from model import exceedance, fit_tabpfn, sample_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, MODEL = ROOT / "data", ROOT / "out", ROOT / "model"
THRESHOLDS = [4, 5, 6, 7, 8]
TRAIN_END, VALID_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")


def scores(y: np.ndarray, p: np.ndarray, base: float) -> dict[str, float]:
    bs = brier_score_loss(y, p)
    if y.sum() == 0:
        return {"events": 0, "brier": bs, "bss": None, "auc": None}
    bs_clim = brier_score_loss(y, np.full_like(p, base))
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return {"events": int(y.sum()), "brier": bs, "bss": 1 - bs / bs_clim, "auc": auc}


def predict_tabpfn(model, X: np.ndarray, weights, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        out.append(exceedance(model, X[i:i + batch], THRESHOLDS, weights))
    return np.vstack(out)


def lightgbm_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    cols = []
    for k in THRESHOLDS:
        clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        clf.fit(train[FEATURES], (train.y >= k - 1 / 6).astype(int))
        cols.append(clf.predict_proba(test[FEATURES])[:, 1])
    return np.column_stack(cols)


def report(name: str, y: np.ndarray, P: np.ndarray, base: dict[int, float]) -> dict:
    res = {f"hp{k}": scores((y >= k - 1 / 6).astype(int), P[:, i], base[k]) for i, k in enumerate(THRESHOLDS)}
    fmt = lambda r: "no events" if r["bss"] is None else f"BSS {r['bss']:+.3f} AUC {r['auc']:.3f}"  # noqa: E731
    line = "  ".join(f"Hp>={k}: {fmt(res[f'hp{k}'])}" for k in THRESHOLDS)
    print(f"{name:28s} {line}", flush=True)
    return res


def main(n_context: int = 2000, n_estimators: int = 4, modes: str = "stratified") -> None:
    tag = "" if n_context == 2000 else f"-{n_context}"
    OUT.mkdir(exist_ok=True)
    MODEL.mkdir(exist_ok=True)
    tab = pd.read_parquet(DATA / "table.parquet")
    tab = tab.dropna(subset=["now_bz_mean", "now_v_mean"]).reset_index(drop=True)
    train = tab[tab.time < TRAIN_END]
    valid = tab[(tab.time >= TRAIN_END) & (tab.time < VALID_END)]
    test = tab[tab.time >= VALID_END].reset_index(drop=True)
    base = {k: float((train.y >= k - 1 / 6).mean()) for k in THRESHOLDS}
    print(f"train {len(train)}  valid {len(valid)}  test {len(test)}  base rates {base}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(7)
    vsub = valid.sample(min(len(valid), 10000), random_state=1)

    results, chosen = {}, None
    for mode in modes.split(","):
        ctx, weights = sample_context(train, n_context, mode, rng)
        t0 = time.time()
        model = fit_tabpfn(ctx, device, n_estimators)
        Pv = predict_tabpfn(model, vsub[FEATURES].to_numpy(np.float32), weights)
        res = report(f"TabPFN {mode} (valid)", vsub.y.to_numpy(), Pv, base)
        res["seconds"] = time.time() - t0
        results[f"tabpfn_{mode}_valid"] = res
        score = np.mean([res[f"hp{k}"]["bss"] for k in (5, 6, 7)])
        if chosen is None or score > chosen[0]:
            chosen = (score, mode, ctx, weights)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    _, mode, ctx, weights = chosen
    print(f"chosen context: {mode}", flush=True)
    model = fit_tabpfn(ctx, device, n_estimators)
    ctx.to_parquet(MODEL / f"context{tag}.parquet", index=False)
    (MODEL / f"context{tag}.json").write_text(json.dumps({"mode": mode, "weights": weights, "thresholds": THRESHOLDS, "n_estimators": n_estimators,
                                                    "features": FEATURES, "train_end": str(TRAIN_END.date())}, indent=1))

    y = test.y.to_numpy()
    t0 = time.time()
    P_tab = predict_tabpfn(model, test[FEATURES].to_numpy(np.float32), weights)
    print(f"TabPFN test prediction: {len(test)} rows in {time.time() - t0:.0f} s")
    results["tabpfn_test"] = report("TabPFN (test)", y, P_tab, base)
    P_lgb = lightgbm_probs(train, test)
    results["lightgbm_test"] = report("LightGBM, all train rows", y, P_lgb, base)
    ok = test.hp30_last.notna().to_numpy()
    P_per = np.column_stack([(test.hp30_last >= k - 1 / 6).astype(float) for k in THRESHOLDS])
    results["persistence_test"] = report("Persistence (oracle)", y[ok], P_per[ok], base)
    results["meta"] = {"n_context": len(ctx), "n_estimators": n_estimators, "context_mode": mode, "n_train_rows": len(train), "test_from": str(VALID_END.date()),
                       "test_to": str(test.time.max()), "base_rates": base}
    (OUT / f"metrics{tag}.json").write_text(json.dumps(results, indent=1, default=float))
    keep = test[["time", "y", "hp30_last"]].copy()
    for i, k in enumerate(THRESHOLDS):
        keep[f"tabpfn_p{k}"] = P_tab[:, i]
        keep[f"lgb_p{k}"] = P_lgb[:, i]
    keep.to_parquet(OUT / f"test_predictions{tag}.parquet", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a[:2]), *a[2:3])
