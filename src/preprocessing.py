"""
Preprocessing: rule-based cleaning, training-only de-duplication, the feature/target split,
the leak-safe preprocessor and the development/test split.
  
`clean_dataset` and `split_features_target` learn nothing from the data and never remove rows,
so they are safe on any rows, including label-free data to predict. Everything learned from
data (imputation, encoding, scaling) lives in `build_preprocessor`, inside the model Pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)
from category_encoders import CountEncoder


def _normalize_tokens(tokens) -> set:
    return {str(t).strip().lower() for t in tokens}
 
 
def _placeholder_mask(series: pd.Series, lowered_tokens: set) -> pd.Series:
    """True where a non-null value, stripped and lower-cased, is a placeholder token."""
    as_text = series.astype(str).str.strip().str.lower()
    return series.notna() & as_text.isin(lowered_tokens)
 
 
def flag_invalid_values(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN. Works on a copy. Returns (cleaned_df, report) where the report has the
    number of violations per column.
    """
    out = df.copy()
    report_rows = []
    for column, bounds in rules.items():
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        ok = pd.Series(True, index=numeric.index)
        if "min" in bounds:
            ok &= numeric >= bounds["min"]
        if "max" in bounds:
            ok &= numeric <= bounds["max"]
        violations = numeric.notna() & ~ok
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        out.loc[violations, column] = np.nan
    return out, pd.DataFrame(report_rows, columns=["column", "rule", "violations"])
 
 
def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """
    One spelling per category: strips, maps case-insensitively through `columns_and_maps`
    ({column: {lowercase_variant: canonical}}), and turns placeholders into NaN.
    Real missing values stay NaN (they are not turned into the string "nan").
    """
    out = df.copy()
    lowered_tokens = _normalize_tokens(placeholder_tokens)
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        mapping = {str(k).strip().lower(): v for k, v in mapping.items()}
        series = out[col]
        was_missing = series.isna()
 
        cleaned = series.astype(str).str.strip()
        mapped = cleaned.str.lower().map(mapping)
        result = mapped.where(mapped.notna(), cleaned)
 
        is_placeholder = result.astype(str).str.strip().str.lower().isin(lowered_tokens)
        out[col] = result.mask(is_placeholder | was_missing, np.nan)
    return out
 
 
def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the cleaning rules in `diagnostics_config`:
      1. placeholder tokens ("-", "?", "n/a", ...) in any text column become NaN
         (case-insensitive, whitespace-tolerant);
      2. columns that should be numeric but loaded as text are coerced;
      3. category spellings are canonicalized;
      4. domain-rule violations become NaN;
      5. redundant columns are dropped.
    Row-preserving (the index is kept, so predictions line up with the input rows) and
    target-agnostic, so it runs unchanged on data to predict. Duplicates are NOT handled
    here: see `drop_duplicate_rows`.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))
    lowered_tokens = _normalize_tokens(placeholder_tokens)
    validity_rules = diagnostics_config.get("validity_rules", {})
    canonical_maps = diagnostics_config.get("canonical_categories", {})
    redundant_columns = diagnostics_config.get("redundant_columns", [])
    numeric_text_columns = diagnostics_config.get("numeric_text_columns", list(validity_rules.keys()))
 
    # 1. placeholders -> NaN in every text column
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out.loc[_placeholder_mask(out[column], lowered_tokens), column] = np.nan
 
    # 2. text columns that should be numeric
    for column in numeric_text_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
 
    # 3. category hygiene
    out = canonicalize_categories(out, canonical_maps, placeholder_tokens)
 
    # 4. impossible-but-present values -> NaN
    out, _invalid_report = flag_invalid_values(out, validity_rules)
 
    # 5. redundant columns (see correlation heatmap + VIF in EDA)
    return out.drop(columns=[c for c in redundant_columns if c in out.columns])
 



ENGINEERED_NUMERIC = ["priors_per_adult_year", "juv_severity"]
ENGINEERED_CATEGORICAL = []
 
 
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise features that use only the row's own values and constants fixed in advance, so
    nothing is learned from the data: safe before the split and on data to predict.
    Call it after `clean_dataset` and before `split_features_target`, and list the new columns
    in `preprocessing.numeric_features` / `categorical_features`.
 
    NOT allowed here (they learn from the data -> they belong inside the Pipeline):
    quantile bins (qcut), z-scores, "above the median" flags, group means, frequency/target
    encodings.
    """
    out = df.copy()
    cols = set(out.columns)
 
    # ratio: priors per year of adult life (the denominator is floored at 1 to avoid /0)
    if {"priors_count", "age"} <= cols:
        adult_years = (out["age"] - 18).clip(lower=1)
        out["priors_per_adult_year"] = out["priors_count"] / adult_years
 
    # weighted sum with fixed, hand-chosen weights (a felony weighs more than a misdemeanour)
    juv_weights = {"juv_fel_count": 3, "juv_misd_count": 2, "juv_other_count": 1}
    present = [c for c in juv_weights if c in cols]
    if present:
        out["juv_severity"] = sum(out[c] * juv_weights[c] for c in present)
 
    # an inf from a ratio would break the imputer: make it a missing value instead
    return out.replace([np.inf, -np.inf], np.nan)
 


def find_duplicate_mask(df: pd.DataFrame, id_column: str | None = None) -> dict:
    """Counts and a combined mask of exact duplicate rows and repeated ids (first is kept)."""
    exact_mask = df.duplicated(keep="first")
    if id_column and id_column in df.columns:
        id_mask = df[id_column].duplicated(keep="first")
    else:
        id_mask = pd.Series(False, index=df.index)
    return {
        "exact_duplicate_count": int(exact_mask.sum()),
        "id_duplicate_count": int(id_mask.sum()),
        "combined_mask": exact_mask | id_mask,
    }
 
 
def drop_duplicate_rows(df: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    """
    Training data only, before `split_dev_test`: drops exact duplicate rows and repeated ids
    (keeping the first) and resets the index. Never call it on data to predict: every row
    needs a prediction.
    """
    mask = find_duplicate_mask(df, id_column)["combined_mask"]
    return df.loc[~mask].reset_index(drop=True)
 
  
def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each listed column, before it is imputed."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out
 
 
def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras).
      - `extras` holds the columns kept out of the features for auditing: the sensitive
        attribute and the `extra_columns` (default: COMPAS's own `score_text`). It is None if
        none of them are present.
      - `y` is None on label-free data.
      - Target, sensitive attribute, extra columns and `drop_columns` never reach X.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])
    extra_columns = data_config.get("extra_columns", ["score_text"])
 
    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None
 
    extras_cols = [c for c in dict.fromkeys([sensitive_attr] + list(extra_columns)) if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None
 
    excluded = set(drop_columns) | set(extra_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in excluded]
    return df[feature_cols], y, extras
 
 
_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# sklearn's TargetEncoder cross-fits: a training row's encoding never uses its own label.
# (shuffle / random_state as constructor arguments need scikit-learn >= 1.4.)
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}
 
 
def _lookup(options: dict, key: str, kind: str):
    try:
        return options[key]
    except KeyError:
        raise ValueError(f"Unknown {kind} '{key}'. Choose one of: {sorted(options)}") from None
 
 
def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: imputation and
    scaling for numeric columns, imputation and encoding for categorical ones, and the
    missingness flags passed through. Every encoder tolerates unseen categories.
 
    Config keys: numeric_features, categorical_features, mnar_indicator_sources,
    scaler ("none" | "standard" | "minmax" | "robust", default "standard"),
    encoder ("onehot" | "ordinal" | "count" | "target", default "onehot"),
    imputation {"numeric_strategy", "categorical_strategy"}, random_state.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _lookup(_SCALERS, preprocessing_config.get("scaler", "standard"), "scaler")
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder_factory = _lookup(_ENCODERS, preprocessing_config.get("encoder", "onehot"), "encoder")
    encoder = encoder_factory(preprocessing_config.get("random_state"))
 
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]
 
    transformers = [
        ("numeric", numeric_pipeline, preprocessing_config.get("numeric_features", [])),
        ("categorical", categorical_pipeline, preprocessing_config.get("categorical_features", [])),
    ]
    if indicator_cols:
        transformers.append(("indicators", "passthrough", indicator_cols))
    return ColumnTransformer(transformers, remainder="drop")

def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split into a development set, used for every decision, and a locked test set,
    used only for the final assessment. Returns X_dev, X_test, y_dev, y_test, extras_dev,
    extras_test, row-aligned.
    """
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)