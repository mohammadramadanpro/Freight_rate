from __future__ import annotations

import argparse
import time

import pandas as pd

from src.clean import clean, load, quality_report
from src.config import REPORTS_DIR
from src.model import FINAL_CONFIG, BlendModel, FreightRateModel, MedianBaseline, ModelConfig
from src.validation import evaluate, summarize

def best(**changes) -> FreightRateModel:
    return FreightRateModel(ModelConfig(**{**FINAL_CONFIG, **changes}))


# "best_*" experiments change one thing relative to FINAL_CONFIG (= spline_hybrid_trend_frozen);
# "target_*", "outliers_*", "loss_*" and "no_*" change one thing relative to lgbm_base.
EXPERIMENTS = {
    "median_baseline": lambda: MedianBaseline(),
    "lgbm_base": lambda: FreightRateModel(ModelConfig()),
    "linear": lambda: FreightRateModel(ModelConfig(learner="linear")),
    "hybrid": lambda: FreightRateModel(ModelConfig(learner="hybrid")),
    "lgbm_linear_tree": lambda: FreightRateModel(ModelConfig(learner="lgbm_linear")),
    "catboost": lambda: FreightRateModel(ModelConfig(learner="catboost")),
    "hybrid_trend_half": lambda: FreightRateModel(ModelConfig(learner="hybrid", trend_damping=0.5)),
    "hybrid_trend_frozen": lambda: FreightRateModel(ModelConfig(learner="hybrid", trend_damping=0.0)),
    "spline_linear": lambda: FreightRateModel(ModelConfig(learner="linear", splines=True)),
    "spline_hybrid": lambda: FreightRateModel(ModelConfig(learner="hybrid", splines=True)),
    "spline_hybrid_trend_half": lambda: FreightRateModel(ModelConfig(learner="hybrid", splines=True, trend_damping=0.5)),
    "spline_hybrid_trend_frozen": lambda: FreightRateModel(ModelConfig(learner="hybrid", splines=True, trend_damping=0.0)),
    "blend_hybrid50_lgbm50": lambda: BlendModel(
        [FreightRateModel(ModelConfig(learner="hybrid")), FreightRateModel(ModelConfig())], [0.5, 0.5]),
    "blend_hybrid70_lgbm30": lambda: BlendModel(
        [FreightRateModel(ModelConfig(learner="hybrid")), FreightRateModel(ModelConfig())], [0.7, 0.3]),
    "target_raw": lambda: FreightRateModel(ModelConfig(target="raw")),
    "target_rpm": lambda: FreightRateModel(ModelConfig(target="rpm")),
    "outliers_winsorize": lambda: FreightRateModel(ModelConfig(outliers="winsorize")),
    "outliers_exclude": lambda: FreightRateModel(ModelConfig(outliers="exclude")),
    "loss_l2": lambda: FreightRateModel(ModelConfig(objective="regression")),
    "loss_l1": lambda: FreightRateModel(ModelConfig(objective="regression_l1")),
    "no_lane_encoding": lambda: FreightRateModel(ModelConfig(lane_encoding=False)),
    "no_city_categories": lambda: FreightRateModel(ModelConfig(city_categories=False)),
    "no_quote_signal": lambda: FreightRateModel(ModelConfig(quote_signal=False)),
    "best_outliers_winsorize": lambda: best(outliers="winsorize"),
    "best_outliers_exclude": lambda: best(outliers="exclude"),
    "best_tree_loss_l2": lambda: best(objective="regression"),
    "best_tree_loss_l1": lambda: best(objective="regression_l1"),
    "best_no_lane_encoding": lambda: best(lane_encoding=False),
    "best_no_city_categories": lambda: best(city_categories=False),
    "best_no_quote_signal": lambda: best(quote_signal=False),
}
BEST_VARIANTS = [name for name in EXPERIMENTS if name.startswith("best_")]
DEFAULT = ["median_baseline", "lgbm_base", "linear", "hybrid", "hybrid_trend_half", "hybrid_trend_frozen",
           "spline_linear", "spline_hybrid", "spline_hybrid_trend_half", "spline_hybrid_trend_frozen",
           "blend_hybrid50_lgbm50", "blend_hybrid70_lgbm30"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling monthly validation of freight rate models.")
    parser.add_argument("experiments", nargs="*", default=DEFAULT, help=f"any of: {', '.join(EXPERIMENTS)}, or the groups 'all' / 'best_variants'")
    parser.add_argument("--gap", type=int, default=0, help="months left out between training data and holdout month")
    args = parser.parse_args()
    groups = {"all": list(EXPERIMENTS), "best_variants": BEST_VARIANTS}
    names = [n for arg in args.experiments for n in groups.get(arg, [arg])]
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments: {sorted(unknown)}")

    raw = load("train_test")
    print("Data-quality report (raw train):")
    print(quality_report(raw).to_string(), "\n")
    train = clean(raw)

    results = {}
    for name in names:
        start = time.perf_counter()
        results[name] = evaluate(EXPERIMENTS[name], train, args.gap)
        print(f"{name}: done in {time.perf_counter() - start:.0f}s")
        print(results[name].round(1).to_string(index=False), "\n")

    # Merge with earlier runs so experiments can be re-run individually
    suffix = f"_gap{args.gap}" if args.gap else ""
    folds_path = REPORTS_DIR / f"cv_folds{suffix}.csv"
    if folds_path.is_file():
        previous = pd.read_csv(folds_path)
        for name, folds in previous[~previous["experiment"].isin(names)].groupby("experiment", sort=False):
            results.setdefault(name, folds.drop(columns="experiment").reset_index(drop=True))

    summary = summarize(results).round(1)
    print("Summary across all recorded experiments (dollars):")
    print(summary.to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    pd.concat(results, names=["experiment"]).reset_index(level=0).to_csv(folds_path, index=False)
    summary.to_csv(REPORTS_DIR / f"cv_summary{suffix}.csv", index_label="experiment")


if __name__ == "__main__":
    main()
