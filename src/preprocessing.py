"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def flag_invalid_values(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    report_rows = []
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")

        def is_valid(value, rule=rule, col=col):
            if pd.isna(value):
                return True
            return bool(eval(rule, {"__builtins__":{}}, {col: value}))

        valid_mask = numeric.map(is_valid)
        violations = df[col].notna() & ~valid_mask
        report_rows.append({"column": col, "rule": rule, "violations": int(violations.sum())})
        df.loc[violations, col] = np.nan
    return df, pd.DataFrame(report_rows)

def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set ) -> pd.DataFrame:
    df = df.copy()
    lowered_tokens = {t.lower() for t in placeholder_tokens}
    for col, mapping in columns_and_maps.items():
        if col not in df.columns:
            continue
        series = df[col]
        was_missing = series.isna()

        cleaned = series.astype(str).str.strip()
        lowered = cleaned.str.lower()
        mapped = lowered.map(mapping)
        result = mapped.where(mapped.notna(), cleaned)
 
        is_placeholder = result.astype(str).str.strip().str.lower().isin(lowered_tokens)
        result = result.mask(is_placeholder | was_missing, np.nan)
        df[col] = result
    return df

def find_duplicate_mask(df: pd.DataFrame, id_column: str | None) -> dict:
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

def clean_data(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    df = df.copy()
 
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))
    canonical_maps = diagnostics_config.get("canonical_categories", [])
    validity_rules = diagnostics_config.get("validity_rules", {})
    id_column = diagnostics_config.get("id_column")
    redundant_columns = diagnostics_config.get("redundant_columns", [])
    numeric_columns_to_coerce = diagnostics_config.get(
        "numeric_columns_to_coerce", list(validity_rules.keys())
    )
 
    # 1. placeholder tokens ("-", "?", "n/a", ...) -> real NaN
    for column in df.columns:
        if df[column].dtype == object:
            is_placeholder = df[column].astype(str).str.strip().isin(placeholder_tokens)
            df.loc[is_placeholder, column] = np.nan
 
    # 2. columns that should be numeric but loaded as text because of the
    # placeholder tokens above (e.g. priors_count, prior_offenses)
    for column in numeric_columns_to_coerce:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
 
    # 3. category hygiene: same category, one spelling
    df = canonicalize_categories(df, canonical_maps, placeholder_tokens)
 
    # 4. domain rules: impossible-but-present values -> NaN
    df, _invalid_report = flag_invalid_values(df, validity_rules)
 
    # 5. duplicates: exact rows and repeated ids, keep first occurrence
    dup_info = find_duplicate_mask(df, id_column)
    df = df.loc[~dup_info["combined_mask"]]
 
    # 6. multicollinearity: drop columns that carry no information beyond
    # what other columns already carry (see correlation heatmap + VIF in EDA)
    df = df.drop(columns=[c for c in redundant_columns if c in df.columns])
 
    return df.reset_index(drop=True)

def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    df = df.copy()
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])
    extra_columns = data_config.get("extra_columns", ["score_text"])
 
    for column in mnar_indicator_sources:
        if column in df.columns:
            df[f"{column}_was_missing"] = df[column].isna().astype(int)
 
    y = df[target]
 
    extras_columns = list(dict.fromkeys([sensitive_attr] + extra_columns))
    extras = df[[c for c in extras_columns if c in df.columns]].copy()
 
    columns_to_exclude = list(dict.fromkeys(
        [target, sensitive_attr] + extra_columns + [c for c in drop_columns if c in df.columns]
    ))
    X = df.drop(columns=[c for c in columns_to_exclude if c in df.columns])
 
    return X, y, extras

def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    numeric_features = preprocessing_config.get("numeric_features", [])
    categorical_features = preprocessing_config.get("categorical_features", [])
    indicator_features = [
        f"{column}_was_missing" for column in preprocessing_config.get("mnar_indicator_sources", [])
    ]
 
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
 
    transformers = [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
    if indicator_features:
        transformers.append(("indicator", "passthrough", indicator_features))
 
    return ColumnTransformer(transformers, remainder="drop")

def split_train_test(X, y, extras, test_size: float, random_state: int):
    return train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
