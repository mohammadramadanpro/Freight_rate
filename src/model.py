from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .clean import WeightImputer, suspect_rate_mask
from .config import RANDOM_STATE
from .features import LaneEncoder, add_row_features

CATEGORICAL = ["equipment", "pickup", "delivery"]
NUMERIC = [
    "distance", "log_distance", "haversine", "distance_ratio", "distance_flag",
    "weight", "weight_is_capped", "weight_was_missing", "weight_was_negative",
    "market_index", "market_index_was_missing", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon",
    "day_of_week",
]

# Huber transition point, in the units of each target.
HUBER_DELTA = {"log": 0.1, "raw": 150.0, "rpm": 0.25}

# "lgbm" and "catboost" are pure tree ensembles; "lgbm_linear" fits linear models in the leaves;
# "linear" is a robust linear model on log rate; "hybrid" is that linear model plus LightGBM on its residuals.
LEARNERS = ("lgbm", "lgbm_linear", "catboost", "linear", "hybrid")

# Chosen on rolling validation (one and two months ahead); used by predict.py.
FINAL_CONFIG = dict(learner="hybrid", splines=True, trend_damping=0.0)


@dataclass
class ModelConfig:
    learner: str = "lgbm"
    target: str = "log"         # "log" | "raw" | "rpm"
    outliers: str = "all"       # "all" | "winsorize" | "exclude"
    objective: str = "huber"    # LightGBM objective; CatBoost and linear learners are always robust
    lane_encoding: bool = True
    city_categories: bool = True
    quote_signal: bool = True
    # Linear and hybrid learners only:
    splines: bool = False        # smooth spline curves for distance, weight and market signals
    trend_damping: float = 1.0   # 1 extends the trend past the training data, 0 freezes it at the last training day
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.learner not in LEARNERS:
            raise ValueError(f"unknown learner {self.learner!r}")
        if self.learner in ("linear", "hybrid") and self.target != "log":
            raise ValueError("linear and hybrid learners model log rate only")
        if not 0.0 <= self.trend_damping <= 1.0:
            raise ValueError("trend_damping must be between 0 and 1")


class MedianBaseline:
    """Median rate per mile by equipment and distance band, scaled by distance."""

    BINS = [0, 250, 500, 750, 1000, 1500, 2000, 2500, np.inf]

    def _band(self, df):
        return pd.cut(df["distance"], self.BINS)

    def fit(self, df: pd.DataFrame) -> "MedianBaseline":
        rpm = df["posted_rate"] / df["distance"]
        self.table_ = rpm.groupby([df["equipment"], self._band(df)], observed=True).median()
        self.fallback_ = float(rpm.median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([df["equipment"], self._band(df)])
        rpm = self.table_.reindex(keys).fillna(self.fallback_).to_numpy()
        return rpm * df["distance"].to_numpy()


class FreightRateModel:
    """Cleaned frame in, dollar predictions out. All fitted state comes from the frame passed to fit()."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    # ----- targets -------------------------------------------------------------------------------

    def _to_target(self, df: pd.DataFrame) -> pd.Series:
        rate = df["posted_rate"]
        return {"log": np.log(rate), "raw": rate, "rpm": rate / df["distance"]}[self.config.target]

    def _from_target(self, y: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return {"log": lambda: np.exp(y), "raw": lambda: y, "rpm": lambda: y * df["distance"].to_numpy()}[self.config.target]()

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        rpm = df["posted_rate"] / df["distance"]
        bounds = rpm.groupby(df["equipment"]).quantile([0.01, 0.99]).unstack()
        low = df["equipment"].map(bounds[0.01])
        high = df["equipment"].map(bounds[0.99])
        out = df.copy()
        out["posted_rate"] = rpm.clip(low, high) * df["distance"]
        return out

    # ----- design matrices -----------------------------------------------------------------------

    def _tree_design(self, df: pd.DataFrame, lane_features: pd.DataFrame | None) -> pd.DataFrame:
        cfg = self.config
        cols = [c for c in NUMERIC if cfg.quote_signal or c != "quote_signal"]
        if cfg.learner == "lgbm_linear":
            cols.append("days_since_start")
        X = df[cols].copy()
        cats = CATEGORICAL if cfg.city_categories else ["equipment"]
        for col in cats:
            if cfg.learner == "catboost":
                X[col] = df[col].astype(str)
            else:
                X[col] = pd.Categorical(df[col], categories=self.categories_[col])
        if cfg.learner == "catboost" and cfg.city_categories:
            X["lane"] = df["lane"].astype(str)
        if lane_features is not None:
            X = X.join(lane_features)
        return X

    def _linear_design(self, df: pd.DataFrame, lane_features: pd.DataFrame | None) -> pd.DataFrame:
        cfg = self.config
        X = pd.DataFrame(index=df.index)
        X["log_distance"] = df["log_distance"]
        for equipment in self.categories_["equipment"][1:]:
            is_eq = (df["equipment"] == equipment).astype(float)
            X[f"is_{equipment}"] = is_eq
            X[f"is_{equipment}_x_log_distance"] = is_eq * df["log_distance"]

        curve_cols = self._curve_columns()
        if cfg.splines:
            basis = self.splines_.transform(df[curve_cols])
            for i in range(basis.shape[1]):
                X[f"spline_{i}"] = basis[:, i]
        else:
            X["log_distance_sq"] = df["log_distance"] ** 2
            for col in curve_cols[1:]:
                X[col] = df[col]

        for col in ["weight_is_capped", "weight_was_missing", "market_index_was_missing", "days_since_start",
                    "distance_flag", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon"]:
            X[col] = df[col]
        X["distance_ratio"] = df["distance_ratio"].clip(upper=3.0)
        for day in range(1, 7):
            X[f"dow_{day}"] = (df["day_of_week"] == day).astype(float)
        if lane_features is not None:
            for col in lane_features:
                X[col] = np.log1p(lane_features[col]) if col.endswith("_count") else lane_features[col]
        return X

    # ----- learners ------------------------------------------------------------------------------

    def _lgbm(self) -> lgb.LGBMRegressor:
        cfg = self.config
        params = {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "objective": cfg.objective,
        }
        if cfg.objective == "huber":
            params["alpha"] = HUBER_DELTA[cfg.target]
        if cfg.learner == "lgbm_linear":
            params.update(linear_tree=True, linear_lambda=1.0)
        params.update(cfg.params)
        return lgb.LGBMRegressor(**params)

    def _catboost(self, cat_features: list[str]) -> CatBoostRegressor:
        params = {
            "iterations": 1500,
            "learning_rate": 0.06,
            "depth": 6,
            "loss_function": f"Huber:delta={HUBER_DELTA[self.config.target]}",
            # Newton leaf steps divide by the Huber second derivative, which is zero beyond delta and diverges
            "leaf_estimation_method": "Gradient",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
            "cat_features": cat_features,
        }
        params.update(self.config.params)
        return CatBoostRegressor(**params)

    @staticmethod
    def _linear():
        return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000))

    def _curve_columns(self) -> list[str]:
        cols = ["log_distance", "weight", "market_index"]
        return cols + ["quote_signal"] if self.config.quote_signal else cols

    def _damp_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slows the trend for dates after the training data; the tree stage never sees the date."""
        damping = self.config.trend_damping
        if damping == 1.0:
            return df
        out = df.copy()
        beyond = out["days_since_start"] > self.last_day_
        out.loc[beyond, "days_since_start"] = self.last_day_ + damping * (out.loc[beyond, "days_since_start"] - self.last_day_)
        return out

    # ----- fit / predict -------------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        df = add_row_features(self.imputer_.transform(df))
        lane_features = self.lane_encoder_.transform(df) if self.config.lane_encoding else None
        return df, lane_features

    def fit(self, df: pd.DataFrame) -> "FreightRateModel":
        cfg = self.config
        if cfg.outliers == "exclude":
            df = df[~suspect_rate_mask(df)]
        elif cfg.outliers == "winsorize":
            df = self._winsorize(df)

        self.imputer_ = WeightImputer().fit(df)
        df = add_row_features(self.imputer_.transform(df))
        self.categories_ = {col: sorted(df[col].unique()) for col in CATEGORICAL}
        self.last_day_ = float(df["days_since_start"].max())
        if cfg.splines:
            self.splines_ = SplineTransformer(n_knots=6, degree=3, extrapolation="linear", include_bias=False)
            self.splines_.fit(df[self._curve_columns()])

        lane_features = None
        if cfg.lane_encoding:
            self.lane_encoder_ = LaneEncoder()
            lane_features = self.lane_encoder_.fit_transform(df)

        y = self._to_target(df)

        if cfg.learner in ("linear", "hybrid"):
            self.linear_ = self._linear().fit(self._linear_design(df, lane_features), y)
            if cfg.learner == "linear":
                return self
            y = y - self.linear_.predict(self._linear_design(df, lane_features))

        X = self._tree_design(df, lane_features)
        if cfg.learner == "catboost":
            cat_features = [c for c in X.columns if X[c].dtype == object]
            self.tree_ = self._catboost(cat_features).fit(X, y)
        else:
            self.tree_ = self._lgbm().fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        df, lane_features = self._prepare(df)
        y = np.zeros(len(df))
        if cfg.learner in ("linear", "hybrid"):
            y += self.linear_.predict(self._linear_design(self._damp_trend(df), lane_features))
        if cfg.learner != "linear":
            y += self.tree_.predict(self._tree_design(df, lane_features))
        return self._from_target(y, df)


class BlendModel:
    """Weighted average of several models' dollar predictions."""

    def __init__(self, models: list, weights: list[float]):
        if len(models) != len(weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("need one weight per model, summing to 1")
        self.models = models
        self.weights = weights

    def fit(self, df: pd.DataFrame) -> "BlendModel":
        for model in self.models:
            model.fit(df)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return sum(w * m.predict(df) for m, w in zip(self.models, self.weights))
