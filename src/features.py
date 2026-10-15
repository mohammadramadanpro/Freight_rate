from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import DISTANCE_FLAG_RATIO, RANDOM_STATE

EARTH_RADIUS_MILES = 3958.8
TREND_ORIGIN = pd.Timestamp("2025-01-01")


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that depend only on the row itself (no fitting required)."""
    out = df.copy()
    out["haversine"] = haversine_miles(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    out["distance_ratio"] = out["distance"] / out["haversine"].clip(lower=1.0)
    out["distance_flag"] = (out["distance_ratio"] > DISTANCE_FLAG_RATIO).astype(int)
    out["log_distance"] = np.log(out["distance"])
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
    out["day_of_week"] = out["date"].dt.dayofweek
    out["days_since_start"] = (out["date"] - TREND_ORIGIN).dt.days.astype(float)
    out["lane"] = out["pickup"] + "|" + out["delivery"]
    return out


class LaneEncoder:
    """Smoothed target encoding (mean log rate per mile) and frequency counts for lane and cities.

    Training rows are encoded out-of-fold so a row never sees its own target.
    Unseen categories fall back to the global mean and a count of zero.
    """

    KEYS = ("lane", "pickup", "delivery")

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5):
        self.smoothing = smoothing
        self.n_splits = n_splits

    @staticmethod
    def _log_rpm(df: pd.DataFrame) -> pd.Series:
        return np.log(df["posted_rate"] / df["distance"])

    def _stats(self, df: pd.DataFrame, target: pd.Series, key: str) -> tuple[pd.Series, float]:
        prior = float(target.mean())
        grouped = target.groupby(df[key]).agg(["sum", "count"])
        encoded = (grouped["sum"] + self.smoothing * prior) / (grouped["count"] + self.smoothing)
        return encoded, prior

    def fit(self, df: pd.DataFrame) -> "LaneEncoder":
        target = self._log_rpm(df)
        self.encodings_ = {key: self._stats(df, target, key) for key in self.KEYS}
        self.counts_ = {key: df[key].value_counts() for key in self.KEYS}
        return self

    def _apply(self, df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for key, (encoded, prior) in encodings.items():
            out[f"{key}_te"] = df[key].map(encoded).astype(float).fillna(prior)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._apply(df, self.encodings_)
        for key in self.KEYS:
            out[f"{key}_count"] = df[key].map(self.counts_[key]).fillna(0).astype(float)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        target = self._log_rpm(df)
        oof = pd.DataFrame(index=df.index, columns=[f"{k}_te" for k in self.KEYS], dtype=float)
        folds = KFold(self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        for fit_idx, enc_idx in folds.split(df):
            fit_rows, enc_rows = df.iloc[fit_idx], df.iloc[enc_idx]
            encodings = {key: self._stats(fit_rows, target.iloc[fit_idx], key) for key in self.KEYS}
            oof.iloc[enc_idx] = self._apply(enc_rows, encodings).to_numpy()
        for key in self.KEYS:
            oof[f"{key}_count"] = df[key].map(self.counts_[key]).astype(float)
        return oof
