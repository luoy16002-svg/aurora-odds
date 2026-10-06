"""TabPFN as an aurora nowcaster: choose the context rows, fit, and read exceedance probabilities.

TabPFN does not train on our data. It reads a few thousand labelled examples (the context) in a
single forward pass and returns, for each new moment, a full probability distribution over the
next hour's Hp30. One distribution answers every latitude at once: P(Hp30 >= k) for any k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURES

# Strata on the target and how many context rows to draw from each (scaled to n_context).
STRATA = [0, 2, 3, 4, 5, 6, 7, 8, 99]
STRATA_SHARE = [0.16, 0.15, 0.15, 0.15, 0.13, 0.11, 0.08, 0.07]


def sample_context(train: pd.DataFrame, n: int, mode: str, rng: np.random.Generator):
    """Return (context rows, label-shift weights per stratum or None)."""
    if mode == "uniform":
        idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
        return train.iloc[np.sort(idx)].reset_index(drop=True), None
    bins = np.digitize(train.y.to_numpy(), STRATA[1:-1])
    parts, weights = [], []
    for s, share in enumerate(STRATA_SHARE):
        pool = np.flatnonzero(bins == s)
        take = min(len(pool), int(round(share * n)))
        parts.append(rng.choice(pool, size=take, replace=False) if take else np.array([], int))
    ctx_idx = np.sort(np.concatenate(parts))
    ctx_bins = bins[ctx_idx]
    for s in range(len(STRATA_SHARE)):
        p_true = (bins == s).mean()
        p_ctx = (ctx_bins == s).mean()
        weights.append(float(p_true / p_ctx) if p_ctx > 0 else 0.0)
    return train.iloc[ctx_idx].reset_index(drop=True), weights


def fit_tabpfn(ctx: pd.DataFrame, device: str = "cpu", n_estimators: int = 8, cache: bool = True):
    """With cache=True the context is encoded once and kept, so later predictions are cheap."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device=device, n_estimators=n_estimators, ignore_pretraining_limits=True, random_state=0,
        fit_mode="fit_with_cache" if cache else "fit_preprocessors", memory_saving_mode=True)
    model.fit(ctx[FEATURES].to_numpy(np.float32), ctx.y.to_numpy(np.float32))
    return model


def bar_probs(model, X: np.ndarray, weights=None):
    """Probability of each bar of TabPFN's predictive histogram, optionally corrected for label shift."""
    import torch
    full = model.predict(X, output_type="full")
    crit, logits = full["criterion"], full["logits"]
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1).cpu().numpy()
    borders = crit.borders.detach().cpu().numpy().astype(np.float64)
    if weights is not None:
        centers = (borders[:-1] + borders[1:]) / 2
        w = np.asarray(weights)[np.clip(np.digitize(centers, STRATA[1:-1]), 0, len(weights) - 1)]
        probs = probs * w
        probs /= probs.sum(axis=1, keepdims=True)
    return probs, borders


def exceedance(model, X: np.ndarray, thresholds, weights=None) -> np.ndarray:
    """P(next-hour Hp30 >= k) for each k. Hp30 comes in thirds, so 'k' means at least the value k itself."""
    probs, borders = bar_probs(model, X, weights)
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, axis=1)], axis=1)  # cdf at each border
    out = []
    for k in thresholds:
        cut = k - 1 / 6
        j = np.clip(np.searchsorted(borders, cut) - 1, 0, len(borders) - 2)
        frac = np.clip((cut - borders[j]) / (borders[j + 1] - borders[j]), 0, 1)
        below = cdf[:, j] + probs[:, j] * frac
        out.append(1 - below)
    return np.clip(np.column_stack(out), 0, 1)
