"""Saving each run's report to disk."""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<timestamp>.txt` (the
    folder is created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    cv = config.get("cv", {})
    tuning = config.get("tuning", {})
    test_set = config.get("test_set", config.get("split", {}))
    if tuning.get("enabled"):
        tuning_line = (
            f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
            f"random_state={tuning['random_state']}"
        )
    elif cv.get("inner_n_splits"):
        tuning_line = f"GridSearchCV, {cv['inner_n_splits']} inner folds, nested"
    else:
        tuning_line = "off"
    header = (
        f"Run: {timestamp}\n"
        f"Configured model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing'].get('encoder', 'onehot')}  "
        f"scaler={config['preprocessing'].get('scaler', 'standard')}\n"
        f"CV: {cv.get('n_splits', 'unspecified')} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  random_state={cv.get('random_state')}  "
        f"metric={cv.get('scoring', 'accuracy')}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={test_set.get('test_size', test_set.get('size', 'unspecified'))}  "
        f"random_state={test_set.get('random_state')}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)
    return path