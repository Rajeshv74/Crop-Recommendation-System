"""
train_model.py
----------------
Trains a Random Forest classifier on the crop recommendation dataset
and saves the trained model + label encoder to disk for the Flask API to use.

Expected CSV columns (8 total):
N, P, K, temperature, humidity, ph, rainfall, label

Usage:
    python train_model.py --data path/to/Crop_recommendation.csv
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names in case of stray whitespace/case differences
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def train(csv_path: str, n_estimators: int = 200, random_state: int = 42):
    df = load_data(csv_path)
    print(f"Loaded {len(df)} records with {df[TARGET_COLUMN].nunique()} crop classes.")

    X = df[FEATURE_COLUMNS]
    y_raw = df[TARGET_COLUMN]

    # Encode crop labels (strings) -> integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5-fold CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Feature importance (useful to show on frontend/insights)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))

    # Save artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"\nModel + encoder saved to: {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop recommendation Random Forest model")
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV dataset")
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()

    train(args.data, n_estimators=args.n_estimators)
