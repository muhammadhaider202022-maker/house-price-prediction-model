import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price"


def load_data(csv_path: str, target: str = TARGET_COLUMN) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset")
    if df.shape[0] == 0:
        raise ValueError("The dataset is empty")

    return df


def infer_feature_columns(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Tuple[List[str], List[str]]:
    features = [col for col in df.columns if col != target]
    if len(features) == 0:
        raise ValueError("No feature columns were found in the dataset")

    numeric_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[features].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature column is required")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(
    X: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_feature_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
    return pipeline


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> Pipeline:
    pipeline = build_pipeline(X, random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_model(model: Pipeline, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def load_model(model_path: str) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(path)


def normalize_sample(sample: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
    normalized = {}
    for feature in feature_columns:
        if feature in sample:
            normalized[feature] = sample[feature]
    return normalized


def predict_samples(model: Pipeline, samples: List[Dict[str, Any]]) -> List[float]:
    if len(samples) == 0:
        raise ValueError("At least one sample is required for prediction")

    input_df = pd.DataFrame(samples)
    if hasattr(model, "feature_names_in_"):
        required_cols = list(model.feature_names_in_)
        input_df = input_df.reindex(columns=required_cols)
    return model.predict(input_df).tolist()


def load_json_sample(sample_json: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(sample_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON sample: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The sample JSON must represent a single object")
    return parsed
