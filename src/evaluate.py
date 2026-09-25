"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline,
out-of-fold reports, a fairness check and a paired comparison of two models. The locked test
set is never used here.

Quick use:   results = evaluate(pipeline, X_dev, y_dev)
Fine-grained use: cross_validate_pipeline -> cv_report / oof_classification_report /
fairness_report / corrected_paired_test.

`pipeline` must be the complete Pipeline (preprocessing included), never already-transformed
data, otherwise imputation/encoding/scaling would be learned on the validation folds.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate


def _take(data, idx):
    """Rows by position, for DataFrames/Series and for arrays."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold with the train score, the validation score and their gap;
      - y_oof: out-of-fold predictions, each row predicted by the fold model that did not
        train on it, taken from the same fits as the scores.
    `cv` must be a partition (each row validated exactly once); otherwise a ValueError is raised.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)

    times_validated = np.zeros(len(X), dtype=int)
    for val_idx in scores["indices"]["test"]:
        times_validated[val_idx] += 1
    if not (times_validated == 1).all():
        raise ValueError("`cv` must validate every row exactly once to build out-of-fold predictions.")

    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(_take(X, val_idx))
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy", verbose: bool = True) -> str:
    """Per-fold table with mean and std (ddof=1), as text (printed if `verbose`, and returned)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    if verbose:
        print(text)
    return text


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float | None = None) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold).

    `test_train_ratio` defaults to 1 / (k - 1), correct for plain k-fold. For repeated CV, pass
    the ratio explicitly and give the scores of all repeats (k = number of scores).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    if k < 2:
        raise ValueError("Need at least two folds to compare models.")
    if test_train_ratio is None:
        test_train_ratio = 1 / (k - 1)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, y_pred, verbose: bool = True) -> str:
    """Classification report on the out-of-fold predictions, as text (printed if `verbose`, and returned)."""
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    if verbose:
        print(text)
    return text


def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race",
                    compas_column: str = "score_text", compas_low_label: str = "Low",
                    verbose: bool = True) -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's predictions and, if `compas_column` is in
    `extras`, for COMPAS's own score (anything other than `compas_low_label` counts as
    predicted to reoffend), on the same rows. `y_true` and `y_pred` are row-aligned with
    `extras` (Series or arrays). Rows with an unknown group are reported as "unknown".
    A deliberately simple check, not a substitute for a full audit: it shows that
    "accurate" and "fair" are not the same thing.
    """
    if not (len(y_true) == len(y_pred) == len(extras)):
        raise ValueError("y_true, y_pred and extras must have the same number of rows.")

    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = np.asarray(y_pred)
    predictors = [("Our model", "y_pred_model")]
    if compas_column in df.columns:
        df["y_pred_compas"] = (df[compas_column] != compas_low_label).astype(int)
        predictors.append(("COMPAS's own score", "y_pred_compas"))

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in predictors:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    if verbose:
        print(text)
    return text


def evaluate(pipeline, X, y, n_splits: int = 5, random_state: int = 42,
             scoring: str = "accuracy", n_jobs: int = 1, verbose: bool = True) -> dict:
    """
    One-call evaluation: stratified k-fold CV of the full pipeline, the per-fold table and the
    out-of-fold classification report.

    Returns a dict with the numbers, so they can be reused (e.g. in `corrected_paired_test`):
      - "fold_scores": DataFrame with train / validation / gap per fold;
      - "y_oof": out-of-fold predictions;
      - "report": the full text report.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X, y, cv, scoring=scoring, n_jobs=n_jobs)
    report = "\n\n".join([
        cv_report(fold_scores, scoring, verbose=False),
        oof_classification_report(y, y_oof, verbose=False),
    ])
    if verbose:
        print(report)
    return {"fold_scores": fold_scores, "y_oof": y_oof, "report": report}