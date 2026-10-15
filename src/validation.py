from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .clean import suspect_rate_mask
from .config import ROLLING_HOLDOUT_MONTHS


def rolling_splits(df: pd.DataFrame, gap_months: int = 0) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yields (month, train, holdout): holdout is the month, train is every row ending `gap_months` before it.

    gap_months=1 mimics the final task, where December is two months past the last training date.
    """
    for month in ROLLING_HOLDOUT_MONTHS:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)
        train_end = start - pd.offsets.MonthBegin(gap_months) if gap_months else start
        yield month, df[df["date"] < train_end], df[(df["date"] >= start) & (df["date"] < end)]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def evaluate(make_model: Callable[[], object], df: pd.DataFrame, gap_months: int = 0) -> pd.DataFrame:
    """Fits a fresh model per fold and scores in dollars, on all holdout rows and on rows with plausible rates."""
    rows = []
    for month, train, holdout in rolling_splits(df, gap_months):
        pred = make_model().fit(train).predict(holdout)
        y = holdout["posted_rate"].to_numpy()
        clean = ~suspect_rate_mask(holdout).to_numpy()
        all_m, clean_m = _metrics(y, pred), _metrics(y[clean], pred[clean])
        rows.append({
            "holdout": month,
            "n_train": len(train),
            "n_holdout": len(holdout),
            "mae_all": all_m["mae"],
            "rmse_all": all_m["rmse"],
            "mae_clean": clean_m["mae"],
            "rmse_clean": clean_m["rmse"],
        })
    return pd.DataFrame(rows)


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per experiment: mean over folds plus the October fold."""
    metrics = ["mae_all", "rmse_all", "mae_clean", "rmse_clean"]
    rows = {}
    for name, folds in results.items():
        mean = folds[metrics].mean().add_prefix("mean_")
        october = folds.loc[folds["holdout"] == ROLLING_HOLDOUT_MONTHS[-1], metrics].iloc[0].add_prefix("oct_")
        rows[name] = pd.concat([mean, october])
    return pd.DataFrame(rows).T
