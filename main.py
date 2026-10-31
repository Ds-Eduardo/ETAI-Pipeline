"""
Run the configured model on the development set: default hyperparameters, the no-model floor,
Optuna tuning with nested CV (if it is on and the model has a search space), the reports, and
one new row on the scoreboard. The locked test set is never evaluated.

`python main.py` runs the model in config.yaml. `run(config)` does the same for any config
dict, which is how `run_all.py` runs every model.
"""
from datetime import datetime

import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

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


def run(config: dict) -> dict:
    """Runs the model in `config["model"]`, saves its report and scoreboard row, returns the row."""
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
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {n_test} test rows locked away (not used)")

    # No-model floor and default hyperparameters, on the same folds
    pipeline = build_pipeline(config["preprocessing"], config["model"])
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
    search_space = (tuning_config.get("search_spaces") or {}).get(model_type)
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
    report += "\n" + refit + "\n"

    # One new row on the scoreboard
    row = {"run": run_id, "data": config["data"]["path"], "model": model_type, "tuned": tuned,
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
             [["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    run_config = {**config, "tuning": {**tuning_config, "enabled": tuned}}
    path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, run_config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    return row


def main():
    run(load_config())


if __name__ == "__main__":
    main()