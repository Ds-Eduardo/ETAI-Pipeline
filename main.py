"""
Run the configured model on the development set: default hyperparameters, the no-model floor,
Optuna tuning with nested CV (if it is on and the model has a search space), the reports, and
one new row on the scoreboard. The locked test set is never evaluated.

`python main.py` runs the model in config.yaml. `run(config)` does the same for any config
dict, which is how `run_all.py` runs every model.
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import joblib
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from predict import predict
from src.data import load_data
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.model import build_pipeline
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_development_data(config: dict):
    """Cleaning and splitting. The locked test set is split off here and then left alone."""
    diagnostics = config["diagnostics"]
    df = load_data(config["data"]["path"])
    df = clean_dataset(df, diagnostics)
    if config["preprocessing"].get("engineer_features", False):     # COMPAS-specific columns
        df = add_engineered_features(df)
    df = drop_duplicate_rows(df, diagnostics.get("id_column"))   # training data only

    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"].get("mnar_indicator_sources", [])
    )
    test_set = config.get("test_set", config.get("split", {}))
    X_dev, X_test, y_dev, _y_test, extras_dev, _extras_test = split_dev_test(
        X, y, extras,
        test_size=test_set.get("test_size", test_set.get("size")),
        random_state=test_set["random_state"],
    )
    return X_dev, y_dev, extras_dev, len(X_test)


def predict_new_data(config: dict, pipeline, output_path: str) -> int:
    """Predicts the rows of `data.new_path` (no label) with `predict.predict` and saves them."""
    predictions = predict(load_data(config["data"]["new_path"]), pipeline, config)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return len(predictions)


def run(config: dict) -> dict:
    """
    Runs the model in `config["model"]`, saves its report and scoreboard row, returns the row.
    The pipeline caches its fitted preprocessing and selection steps in a temporary folder
    (a new one per run, removed at the end), so tuning the model doesn't refit them every trial.
    """
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")
    try:
        return _run(config, cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run(config: dict, cache_dir: str) -> dict:
    """The body of `run`; `cache_dir` is where the Pipeline caches its fitted steps."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    X_dev, y_dev, extras_dev, n_test = prepare_development_data(config)

    # From here on, every decision uses the development set only; the test set stays locked.
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    if selection_metric not in metrics:
        raise ValueError(f"cv.selection_metric '{selection_metric}' must be one of cv.metrics {metrics}")
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    selection_config = config.get("feature_selection") or {}
    selection_on = bool(selection_config.get("enabled"))
    imputer = config["preprocessing"].get("imputation", {}).get("numeric_strategy", "median")
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV   |   feature selection: {'on' if selection_on else 'off'}   |   imputer: {imputer}")
    print(f"Data:  {len(X_dev)} development rows   |   {n_test} test rows locked away (not used)")

    # No-model floor and default hyperparameters, on the same folds
    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config, memory=cache_dir)
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    # Tuning, only if it is on and this model has a search space
    tuning_config = config.get("tuning", {})
    model_space = (tuning_config.get("search_spaces") or {}).get(model_type)
    selection_space = (selection_config.get("search_space") or {}) if selection_on else {}
    search_space = {**model_space, **selection_space} if model_space else None   # tuned together
    tuned = bool(tuning_config.get("enabled")) and bool(search_space)
    if tuning_config.get("enabled") and not search_space:
        print(f"\nNo tuning.search_spaces entry for '{model_type}': default hyperparameters only.")

    study = None
    if tuned:
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold probabilities feed the reports.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<28s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<28s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuned else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    all_names = [name.split("__", 1)[-1] for name in pipeline[:1].get_feature_names_out()]
    if selection_on:
        kept_names = [name.split("__", 1)[-1] for name in pipeline[:-1].get_feature_names_out()]
        min_votes = pipeline.named_steps["select"].min_votes
        selection_label = f"vote(min_votes={min_votes})"
        kept_line = (f"Features kept: {', '.join(kept_names)}   "
                     f"({len(kept_names)} of {len(all_names)}, min_votes={min_votes})")
    else:
        kept_names, selection_label = all_names, "off"
        kept_line = f"Features kept: all {len(all_names)} (feature selection off)"
    report += "\n" + refit + "\n" + kept_line + "\n"

    # Rows without a label (if the config has some): predicted by the refit pipeline
    pred_line = ""
    predictions_path = config.get("output", {}).get("predictions")
    if predictions_path and config["data"].get("new_path"):
        n_new = predict_new_data(config, pipeline, predictions_path)
        pred_line = f"Predictions for {config['data']['new_path']}: {n_new} rows saved to {predictions_path}"
        report += pred_line + "\n"

    # The final fitted pipeline: what predict.py loads to predict new data.
    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    pipeline.set_params(memory=None)        # the cache folder is deleted at the end of the run; the saved model doesn't need it
    joblib.dump(pipeline, model_path)
    model_line = f"Model saved to {model_path}"
    report += model_line + "\n"

    # One new row on the scoreboard
    row = {"run": run_id, "data": config["data"]["path"], "model": model_type, "tuned": tuned,
           "feature_selection": selection_label, "imputer": imputer, "features_kept": ", ".join(kept_names),
           "encoder": config["preprocessing"].get("encoder", "onehot"),
           "scaler": config["preprocessing"].get("scaler", "standard"),
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuned else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           "best_trial": study.best_value if tuned else np.nan,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "feature_selection", "imputer", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    run_config = {**config, "tuning": {**tuning_config, "enabled": tuned}}
    path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, run_config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    if pred_line:
        print(pred_line)
    print(model_line)
    print(kept_line)
    return row


def main():
    # `python main.py` runs config.yaml (COMPAS); `python main.py config_adult.yaml` another problem
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))


if __name__ == "__main__":
    main()