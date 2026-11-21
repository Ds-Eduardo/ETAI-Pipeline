"""
Predicts new people with the model saved by main.py: no training, just the saved pipeline.

    python predict.py data/compas_new_people.csv predictions.csv
"""
import argparse
import os
import sys

import joblib
import pandas as pd
import yaml

from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """
    Returns the id, the probability of the positive class and the prediction (at 0.5) for every row.
    """
    clean = clean_dataset(df, config["diagnostics"])
    if config["preprocessing"].get("engineer_features", False):      # COMPAS-specific columns
        clean = add_engineered_features(clean)
    X, _no_target, _extras = split_features_target(
        clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", [])
    )
    probability = pipeline.predict_proba(X)[:, 1]
    id_column = config["diagnostics"].get("id_column")
    ids = df[id_column].to_numpy() if id_column else df.index.to_numpy()
    predictions = pd.DataFrame({id_column or "row": ids, "probability": probability,
                                "prediction": (probability >= 0.5).astype(int)})
    if len(predictions) != len(df):
        raise RuntimeError(f"{len(df)} rows in but {len(predictions)} rows out.")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    if not os.path.exists(model_path):
        sys.exit(f"No saved model at {model_path}: run `python main.py {args.config}` first.")
    pipeline = joblib.load(model_path)
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")