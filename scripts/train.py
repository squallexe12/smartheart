"""Train LogisticRegressionScratch on the UCI Cleveland dataset.

Exports:
- models/scratch_model.pkl  — pickled Python model + scaler metadata
- web/public/model.json     — browser-consumable artifact
- data/processed/*.csv      — split + encoded splits for reproducibility
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smartheart.metrics import accuracy, auc, confusion_matrix, f1_score, precision, recall, roc_curve
from smartheart.model import LogisticRegressionScratch
from smartheart.preprocessing import (
    one_hot_encode,
    standardize_apply,
    standardize_fit,
    train_test_split_manual,
)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = {
    "cp": [0, 1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

FEATURE_UI_METADATA = {
    "age": {"kind": "numeric", "label": "Age (years)", "min": 29, "max": 77, "step": 1},
    "sex": {"kind": "categorical", "label": "Sex", "options": [
        {"value": 0, "label": "Female"}, {"value": 1, "label": "Male"},
    ]},
    "cp": {"kind": "categorical", "label": "Chest pain type", "options": [
        {"value": 1, "label": "Typical angina"},
        {"value": 2, "label": "Atypical angina"},
        {"value": 3, "label": "Non-anginal pain"},
        {"value": 4, "label": "Asymptomatic"},
    ]},
    "trestbps": {"kind": "numeric", "label": "Resting BP (mm Hg)", "min": 90, "max": 200, "step": 1},
    "chol": {"kind": "numeric", "label": "Cholesterol (mg/dl)", "min": 120, "max": 565, "step": 1},
    "fbs": {"kind": "categorical", "label": "Fasting blood sugar > 120 mg/dl", "options": [
        {"value": 0, "label": "No"}, {"value": 1, "label": "Yes"},
    ]},
    "restecg": {"kind": "categorical", "label": "Resting ECG", "options": [
        {"value": 0, "label": "Normal"},
        {"value": 1, "label": "ST-T abnormality"},
        {"value": 2, "label": "LV hypertrophy"},
    ]},
    "thalach": {"kind": "numeric", "label": "Max heart rate", "min": 70, "max": 210, "step": 1},
    "exang": {"kind": "categorical", "label": "Exercise-induced angina", "options": [
        {"value": 0, "label": "No"}, {"value": 1, "label": "Yes"},
    ]},
    "oldpeak": {"kind": "numeric", "label": "ST depression (oldpeak)", "min": 0.0, "max": 6.5, "step": 0.1},
    "slope": {"kind": "categorical", "label": "Peak exercise ST slope", "options": [
        {"value": 1, "label": "Upsloping"},
        {"value": 2, "label": "Flat"},
        {"value": 3, "label": "Downsloping"},
    ]},
    "ca": {"kind": "numeric", "label": "Major vessels colored (0–3)", "min": 0, "max": 3, "step": 1},
    "thal": {"kind": "categorical", "label": "Thalassemia", "options": [
        {"value": 3, "label": "Normal"},
        {"value": 6, "label": "Fixed defect"},
        {"value": 7, "label": "Reversible defect"},
    ]},
}


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return encoded matrix (numeric || binary || one-hot dropped-first categoricals) and column names."""
    numeric_block = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    binary_block = df[BINARY_FEATURES].to_numpy(dtype=np.float64)
    cat_blocks = []
    cat_columns: list[str] = []
    for col, cats in CATEGORICAL_FEATURES.items():
        # Only include categories that actually appear in the cleaned dataset
        effective_cats = [c for c in cats if c in df[col].unique()]
        encoded, names = one_hot_encode(df[col].to_numpy(), effective_cats, drop_first=True, prefix=col)
        cat_blocks.append(encoded)
        cat_columns.extend(names)

    matrix = np.hstack([numeric_block, binary_block, *cat_blocks])
    columns = NUMERIC_FEATURES + BINARY_FEATURES + cat_columns
    return matrix, columns


def main() -> None:
    raw_path = Path("data/raw/heart.csv")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")

    X_full, encoded_columns = build_feature_matrix(df)
    y_full = df["target"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X_full, y_full, test_size=0.2, random_state=42, stratify=True
    )

    # Standardize numeric columns only (first len(NUMERIC_FEATURES) columns)
    n_numeric = len(NUMERIC_FEATURES)
    mean_num, std_num = standardize_fit(X_train[:, :n_numeric])
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()
    X_train_std[:, :n_numeric] = standardize_apply(X_train[:, :n_numeric], mean_num, std_num)
    X_test_std[:, :n_numeric] = standardize_apply(X_test[:, :n_numeric], mean_num, std_num)

    # Persist processed splits
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_std, columns=encoded_columns).to_csv(processed_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test_std, columns=encoded_columns).to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series(y_train, name="target").to_csv(processed_dir / "y_train.csv", index=False)
    pd.Series(y_test, name="target").to_csv(processed_dir / "y_test.csv", index=False)

    # Train
    model = LogisticRegressionScratch(learning_rate=0.1, n_iters=2000, random_state=42)
    model.fit(X_train_std, y_train)

    # Evaluate
    y_pred = model.predict(X_test_std)
    y_proba = model.predict_proba(X_test_std)
    fpr, tpr = roc_curve(y_test, y_proba)

    metrics = {
        "test_accuracy": accuracy(y_test, y_pred),
        "test_precision": precision(y_test, y_pred),
        "test_recall": recall(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_auc": auc(fpr, tpr),
        "final_cost": model.cost_history[-1],
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:>16s}  {v:.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    # Pickle Python artifact
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoded_columns": encoded_columns,
            "numeric_mean": mean_num,
            "numeric_std": std_num,
            "metrics": metrics,
        },
        models_dir / "scratch_model.pkl",
    )

    # Export browser artifact
    artifact = {
        "version": "1.0",
        "model_type": "logistic_regression",
        "trained_on": "UCI Heart Disease (Cleveland)",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "features": [
            {"name": name, **FEATURE_UI_METADATA[name],
             "default": float(df[name].median()) if FEATURE_UI_METADATA[name]["kind"] == "numeric" else int(df[name].mode().iat[0])}
            for name in [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
            ]
        ],
        "preprocessing": {
            "numeric_features": NUMERIC_FEATURES,
            "binary_features": BINARY_FEATURES,
            "categorical_features": {
                col: [c for c in cats if c in df[col].unique()]
                for col, cats in CATEGORICAL_FEATURES.items()
            },
            "encoded_feature_order": encoded_columns,
            "numeric_mean": mean_num.tolist(),
            "numeric_std": std_num.tolist(),
        },
        "weights": model.weights.tolist(),
        "bias": model.bias,
    }
    web_public = Path("web/public")
    web_public.mkdir(parents=True, exist_ok=True)
    with open(web_public / "model.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote browser artifact to {web_public / 'model.json'}")


if __name__ == "__main__":
    main()
