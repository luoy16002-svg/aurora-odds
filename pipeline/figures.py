"""Figures for the write-up: storm replays, reliability and skill. Output: out/fig_*.png

Run after evaluate.py: python pipeline/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES  # noqa: E402
from model import bar_probs, fit_tabpfn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, MODEL = ROOT / "data", ROOT / "out", ROOT / "model"
BG, PANEL, LINE, TEXT, MUTED, ACCENT, WARM = "#070b12", "#0e1520", "#1b2533", "#e7edf4", "#8b98a9", "#7fe0ab", "#e9c46a"
STORMS = {
    "may2024": ("2024-05-10 09:00", "2024-05-12 09:00", "The May 2024 superstorm (10-11 May)"),
    "oct2024": ("2024-10-10 06:00", "2024-10-11 18:00", "The October 2024 storm (10-11 October)"),
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8, "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": ["Segoe UI", "DejaVu Sans"],
    })


def quantiles(probs: np.ndarray, borders: np.ndarray, qs) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    out = np.empty((len(probs), len(qs)))
    for i, q in enumerate(qs):
        j = np.argmax(cdf >= q, axis=1)
        prev = np.where(j > 0, cdf[np.arange(len(cdf)), j - 1], 0.0)
        frac = (q - prev) / np.maximum(probs[np.arange(len(probs)), j], 1e-12)
        out[:, i] = borders[j] + np.clip(frac, 0, 1) * (borders[j + 1] - borders[j])
    return out


def storm_replays(model, weights, tab: pd.DataFrame) -> None:
    for key, (a, b, title) in STORMS.items():
        part = tab[(tab.time >= a) & (tab.time <= b)].reset_index(drop=True)
        if part.empty:
            continue
        probs, borders = bar_probs(model, part[FEATURES].to_numpy(np.float32), weights)
        q = quantiles(probs, borders, [0.1, 0.5, 0.9])
        fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
        ax.fill_between(part.time, q[:, 0], q[:, 2], color=ACCENT, alpha=0.18, linewidth=0, label="TabPFN 10-90% range")
        ax.plot(part.time, q[:, 1], color=ACCENT, linewidth=2.2, label="TabPFN median")
        ax.step(part.time, part.y, where="post", color=TEXT, linewidth=1.4, alpha=0.9, label="What happened (next-hour max Hp30)")
        for level, name in ((4.85, "Edinburgh, by eye"), (7.5, "London, by eye")):
            ax.axhline(level, color=WARM, linewidth=0.9, linestyle=(0, (4, 4)), alpha=0.7)
            ax.text(part.time.iloc[0], level + 0.15, name, color=WARM, fontsize=10, alpha=0.9)
        ax.set_ylim(0, max(10, float(np.nanmax(part.y)) + 1))
        ax.set_ylabel("Hp30 (Kp scale)")
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=12)
        ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=MUTED)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / f"fig_storm_{key}.png")
        plt.close(fig)


def reliability(pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150, sharey=True)
    for ax, k in zip(axes, (5, 6, 7)):
        y = (pred.y >= k - 1 / 6).to_numpy()
        for col, color, name in ((f"tabpfn_p{k}", ACCENT, "TabPFN"), (f"lgb_p{k}", WARM, "LightGBM")):
            p = pred[col].to_numpy()
            bins = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0001])
            idx = np.digitize(p, bins) - 1
            xs, ys, ns = [], [], []
            for i in range(len(bins) - 1):
                m = idx == i
                if m.sum() >= 15:
                    xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
            ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=4, label=name)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(f"Hp30 ≥ {k}", loc="left", fontsize=13, color=TEXT)
        ax.set_xlabel("forecast probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(frameon=False, labelcolor=MUTED)
    fig.suptitle("Reliability on 2023-2026 (unseen years)", x=0.01, ha="left", fontsize=15, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability.png")
    plt.close(fig)


def skill(metrics: dict, big: dict | None) -> None:
    ks = [4, 5, 6, 7, 8]
    rows = []
    if big:
        rows.append((big["tabpfn_test"], "TabPFN, 8,000 rows in context", ACCENT, 1.0))
    rows += [(metrics["tabpfn_test"], "TabPFN, 2,000 rows (the live page)", ACCENT, 0.55),
             (metrics["lightgbm_test"], "LightGBM, trained on 370,000 rows", WARM, 0.95),
             (metrics["persistence_test"], "Last half-hour's index (oracle)", MUTED, 0.6)]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    w = 0.8 / len(rows)
    for i, (res, name, color, alpha) in enumerate(rows):
        vals = [res[f"hp{k}"]["bss"] for k in ks]
        ax.bar(np.arange(len(ks)) + (i - (len(rows) - 1) / 2) * w, vals, width=w * 0.92, color=color, alpha=alpha, label=name)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(np.arange(len(ks)), [f"Hp30 ≥ {k}" for k in ks])
    ax.set_ylabel("Brier skill score vs climatology")
    ax.set_title("Skill on 2023-2026, the years none of the models saw", loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=12)
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left", ncol=2, fontsize=10)
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_skill.png")
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(exist_ok=True)
    tab = pd.read_parquet(DATA / "table.parquet")
    cfg = json.loads((MODEL / "context.json").read_text())
    ctx = pd.read_parquet(MODEL / "context.parquet")
    model = fit_tabpfn(ctx, "cuda" if torch.cuda.is_available() else "cpu", cfg.get("n_estimators", 4))
    storm_replays(model, cfg.get("weights"), tab)
    pred = pd.read_parquet(OUT / "test_predictions.parquet")
    reliability(pred)
    big = OUT / "metrics-8000.json"
    skill(json.loads((OUT / "metrics.json").read_text()), json.loads(big.read_text()) if big.exists() else None)
    print("figures:", sorted(p.name for p in OUT.glob("fig_*.png")))


if __name__ == "__main__":
    main()
