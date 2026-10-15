"""Trains the final model on all labeled data and writes both submission files."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.clean import city_lookup, clean, load, prepare_december
from src.config import DATA_DIR, ROOT
from src.model import FINAL_CONFIG, FreightRateModel, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-out", default=ROOT / "validation_predictions.csv")
    parser.add_argument("--december-out", default=DATA_DIR / "december_chart_inputs.csv",
                        help="defaults to filling the provided chart file in place, as the README asks")
    args = parser.parse_args()

    raw_train, raw_valid = load("train_test"), load("validation")
    template, december = load("validation_predictions_template"), load("december_chart_inputs")
    train, valid = clean(raw_train), clean(raw_valid)

    print(f"Training {FINAL_CONFIG} on {len(train):,} loads ({train['date'].min().date()} to {train['date'].max().date()})")
    model = FreightRateModel(ModelConfig(**FINAL_CONFIG)).fit(train)

    predicted = pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": model.predict(valid).round(2)})
    submission = template[["load_id"]].merge(predicted, on="load_id", how="left", validate="one_to_one")
    if len(submission) != len(template) or submission["predicted_rate"].isna().any():
        raise SystemExit("validation predictions do not cover every template load_id")
    if not (np.isfinite(submission["predicted_rate"]) & (submission["predicted_rate"] > 0)).all():
        raise SystemExit("validation predictions contain non-positive or non-finite rates")
    submission.to_csv(args.predictions_out, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.predictions_out}")
    print(submission["predicted_rate"].describe().round(2).to_string(), "\n")

    december_features = prepare_december(december, valid, city_lookup(raw_train, raw_valid))
    chart = december.copy()
    chart["predicted_rate"] = model.predict(december_features).round(2)
    chart.to_csv(args.december_out, index=False)
    print(f"Wrote December chart inputs to {args.december_out}")
    print(chart[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
