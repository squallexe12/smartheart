"""Download the UCI Heart Disease (Cleveland) dataset and save as CSV.

Source: UCI ML Repository.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "heart.csv"

    print(f"Downloading from {UCI_URL} ...")
    tmp_path = raw_dir / "processed.cleveland.data"
    urlretrieve(UCI_URL, tmp_path)

    df = pd.read_csv(tmp_path, header=None, names=COLUMN_NAMES, na_values="?")
    # Drop rows with missing values (small number; preserves clean Cleveland subset)
    df = df.dropna().reset_index(drop=True)
    # Binarize target: UCI codes 1-4 all mean "disease present"
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(raw_path, index=False)
    tmp_path.unlink()

    print(f"Saved {len(df)} rows to {raw_path}")
    print(df["target"].value_counts().to_dict())


if __name__ == "__main__":
    main()
