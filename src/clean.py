from __future__ import annotations

import pandas as pd

from .config import DATA_DIR, DISTANCE_FLAG_RATIO, SUSPECT_RPM_HIGH, SUSPECT_RPM_LOW, WEIGHT_CAP
from .features import haversine_miles


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed, stateless cleaning rules. Never drops rows."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out["weight_was_missing"] = out["weight"].isna().astype(int)
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_is_capped"] = (out["weight"] == WEIGHT_CAP).astype(int)

    out["market_index_was_missing"] = out["market_index"].isna().astype(int)
    same_date_median = out.groupby("date")["market_index"].transform("median")
    out["market_index"] = out["market_index"].fillna(same_date_median)
    return out


class WeightImputer:
    """Fills missing weight with the equipment median learned from training rows."""

    def fit(self, df: pd.DataFrame) -> "WeightImputer":
        self.medians_ = df.groupby("equipment")["weight"].median()
        self.overall_ = float(df["weight"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        fill = out["equipment"].map(self.medians_).fillna(self.overall_)
        out["weight"] = out["weight"].fillna(fill)
        return out


def suspect_rate_mask(df: pd.DataFrame) -> pd.Series:
    rpm = df["posted_rate"] / df["distance"]
    return (rpm < SUSPECT_RPM_LOW) | (rpm > SUSPECT_RPM_HIGH)


def city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for df in frames:
        for side in ("pickup", "delivery"):
            parts.append(df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def prepare_december(december: pd.DataFrame, validation_clean: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Adds the inputs the chart file lacks: city coordinates and same-date market medians from validation."""
    out = december.drop(columns="predicted_rate").copy()
    out["date"] = pd.to_datetime(out["date"])
    out["distance"] = out["distance"].astype(float)
    out["weight"] = out["weight"].astype(float)
    for side in ("pickup", "delivery"):
        coords = lookup.loc[out[side]]
        out[f"{side}_lat"] = coords["lat"].to_numpy()
        out[f"{side}_lon"] = coords["lon"].to_numpy()

    daily = validation_clean.groupby("date")[["market_index", "quote_signal"]].median()
    out = out.join(daily, on="date")
    missing_dates = out.loc[out["market_index"].isna(), "date"]
    if not missing_dates.empty:
        raise ValueError(f"No validation rows for December dates: {sorted(missing_dates.dt.date)}")
    return clean(out)


def quality_report(raw: pd.DataFrame) -> pd.Series:
    """Counts of each data-quality issue in a raw (uncleaned) frame."""
    ratio = raw["distance"] / haversine_miles(raw["pickup_lat"], raw["pickup_lon"], raw["delivery_lat"], raw["delivery_lon"])
    report = {
        "rows": len(raw),
        "missing_weight": int(raw["weight"].isna().sum()),
        "negative_weight": int((raw["weight"] < 0).sum()),
        "capped_weight": int((raw["weight"].abs() == WEIGHT_CAP).sum()),
        "missing_market_index": int(raw["market_index"].isna().sum()),
        f"distance_over_{DISTANCE_FLAG_RATIO}x_haversine": int((ratio > DISTANCE_FLAG_RATIO).sum()),
    }
    if "posted_rate" in raw:
        report["suspect_rate"] = int(suspect_rate_mask(raw).sum())
    return pd.Series(report)
