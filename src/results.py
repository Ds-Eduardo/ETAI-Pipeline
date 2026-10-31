"""Saving each run's report to disk."""
import os
import pandas as pd



def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<run>.txt` (the folder is
    created if needed) and returns the path. `run` is the run id built in `main.py`.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")
 
    cv = config.get("cv", {})
    tuning = config.get("tuning", {})
    test_set = config.get("test_set", config.get("split", {}))
    if tuning.get("enabled"):
        tuning_line = (
            f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
            f"random_state={tuning['random_state']}"
        )
    else:
        tuning_line = "off"
    header = (
        f"Run: {run}\n"
        f"Configured model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing'].get('encoder', 'onehot')}  "
        f"scaler={config['preprocessing'].get('scaler', 'standard')}\n"
        f"CV: {cv.get('n_splits', 'unspecified')} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  random_state={cv.get('random_state')}\n"
        f"Selection metric: {cv.get('selection_metric')}   metrics: {cv.get('metrics')}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={test_set.get('test_size', test_set.get('size', 'unspecified'))}  "
        f"random_state={test_set.get('random_state')}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )
 
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)
    return path
 