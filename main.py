"""Run the nested-CV feature comparison on the development set."""
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold


from src.data import load_data
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.evaluate import cv_report, evaluate, fairness_report
from src.model import build_pipeline
from src.results import save_run
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report

MODEL_NAMES = {
    "dummy": "Dummy",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "oblique_random_forest": "Oblique random forest",
    "adaboost": "AdaBoost",
}

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cv_cell(scores: pd.DataFrame) -> str:
    """'mean ± std (gap)' of the validation scores, as in the README tables."""
    validation = scores["validation"]
    return f"{validation.mean():.3f} ± {validation.std(ddof=1):.3f} ({scores['gap'].mean():+.3f})"
 
 
def format_params(params: dict) -> str:
    """Chosen hyperparameters without the Pipeline prefix: `model__max_depth` -> `max_depth`."""
    items = []
    for name, value in params.items():
        short = name.split("__", 1)[1] if "__" in name else name
        items.append(f"{short}={value:.4g}" if isinstance(value, float) else f"{short}={value}")
    return ", ".join(items)
 
 
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
    X_dev, _X_test, y_dev, _y_test, extras_dev, _extras_test = split_dev_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    return X_dev, y_dev, extras_dev
 
 
def main():
    config = load_config()
    cv_cfg = config["cv"]
    tuning_cfg = config.get("tuning", {})
    model_type = config["model"]["type"]
    scoring = cv_cfg["scoring"]
 
    sections = []
 
    def emit(text: str):
        print(text)
        sections.append(text)
 
    X_dev, y_dev, extras_dev = prepare_development_data(config)
    pipeline = build_pipeline(config["preprocessing"], config["model"])
    emit(f"Development set: {len(X_dev)} rows, {X_dev.shape[1]} columns before preprocessing\n")
 
    # 1. default hyperparameters
    emit("=" * 60 + "\nDEFAULT HYPERPARAMETERS\n" + "=" * 60)
    default = evaluate(pipeline, X_dev, y_dev, n_splits=cv_cfg["n_splits"],
                       random_state=cv_cfg["random_state"], scoring=scoring,
                       n_jobs=cv_cfg.get("n_jobs", 1), verbose=True)
    sections.append(default["report"])
    y_oof = default["y_oof"]
 
    # 2. tuning + nested CV, only if it is on and this model has a search space
    search_space = tuning_cfg.get("search_spaces", {}).get(model_type)
    tuned = bool(tuning_cfg.get("enabled")) and bool(search_space)
    if tuning_cfg.get("enabled") and not search_space:
        emit(f"\nNo search space for '{model_type}' in config.yaml: default CV only.")
 
    if tuned:
        emit("\n" + "=" * 60 + "\nTUNED (Optuna) -- NESTED CROSS-VALIDATION\n" + "=" * 60)
        outer_cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=True,
                                   random_state=cv_cfg["random_state"])
        inner_cv = StratifiedKFold(n_splits=tuning_cfg["n_splits"], shuffle=True,
                                   random_state=tuning_cfg["random_state"])
        common = dict(scoring=scoring, search_space=search_space, n_trials=tuning_cfg["n_trials"],
                      random_state=tuning_cfg["random_state"], n_jobs=tuning_cfg.get("n_jobs", 1))
 
        nested_scores, y_oof = nested_cross_validate(pipeline, X_dev, y_dev, outer_cv, inner_cv, **common)
        sections.append(cv_report(nested_scores[["fold", "train", "validation", "gap", "inner_best"]],
                                  scoring, verbose=True))
 
        # the final tuning on the whole development set gives the best trial and the
        # hyperparameters to report; the honest score is the nested one above
        _best_pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, **common)
        sections.append(tuning_report(study, nested_scores, scoring))
 
    # 3. fairness of the out-of-fold predictions (nested ones if the model was tuned)
    emit("")
    sections.append(fairness_report(y_dev, y_oof, extras_dev,
                                    sensitive_attr=config["data"]["sensitive_attr"]))
 
    # 4. the row for the README tables
    name = MODEL_NAMES.get(model_type, model_type)
    if tuned:
        row = (f"| {name} | {cv_cell(default['fold_scores'])} | {cv_cell(nested_scores)} | "
               f"{study.best_value:.3f} | {format_params(study.best_params)} |")
    else:
        row = f"| {name} | {cv_cell(default['fold_scores'])} | — | — | — |"
    emit("README row (Model | Default | Tuned nested | Best trial | Hyperparameters):\n" + row)
 
    # the run header describes tuning only if it really ran
    run_config = {**config, "tuning": {**tuning_cfg, "enabled": tuned}}
    path = save_run(config.get("output", {}).get("results_dir", "results"), run_config, "\n\n".join(sections))
    print(f"\nFull results saved to {path}")
 
 
if __name__ == "__main__":
    main()
