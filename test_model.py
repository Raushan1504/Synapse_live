"""
test_model.py
-------------
Evaluates the trained Bowling and Batting strategy models using
match-level train/test split (same logic as tactical_model_trainer.py)
to ensure test accuracy reflects unseen matches — NOT random row leakage.

Usage:
    python test_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from feature_extractor import (
    extract_features,
    features_to_array,
    derive_bowling_strategy,
    derive_batting_strategy,
)
from player_stats import compute_player_stats


# ── Paths (must match tactical_model_trainer.py) ─────────────────────────────
MODEL_DIR          = "models"
BOWLING_MODEL_PATH = os.path.join(MODEL_DIR, "bowling_strategy_model.pkl")
BATTING_MODEL_PATH = os.path.join(MODEL_DIR, "batting_strategy_model.pkl")
BOWLING_ENC_PATH   = os.path.join(MODEL_DIR, "bowling_label_encoder.pkl")
BATTING_ENC_PATH   = os.path.join(MODEL_DIR, "batting_label_encoder.pkl")


def match_level_split(df, test_fraction=0.20, random_state=42):
    """Split by match_id — identical logic to the trainer."""
    rng = np.random.RandomState(random_state)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)
    n_test = max(1, int(len(match_ids) * test_fraction))
    test_matches = set(match_ids[:n_test])
    is_test  = df["match_id"].isin(test_matches).values
    is_train = ~is_test
    return is_train, is_test


def main():
    print("=" * 60)
    print("  Synapse Live -- Model Evaluation")
    print("=" * 60)

    # 1. Load dataset
    DATA_PATH = "clean_ipl_dataset.csv"
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: {DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print(f"\n  Dataset: {len(df):,} rows | {df['match_id'].nunique()} matches")

    # 2. Compute player stats
    batsman_stats, bowler_stats = compute_player_stats(df)

    # 3. Build features + labels
    print("  Building features (this takes ~30s)...")
    X_rows, y_bowl, y_bat = [], [], []

    for _, row in df.iterrows():
        row = row.copy()
        row["strike_rate"]    = batsman_stats.get(row.get("batsman"), 120.0)
        row["bowler_economy"] = bowler_stats.get(row.get("bowler"),   7.5)

        feats = extract_features(row)
        X_rows.append(features_to_array(feats))
        y_bowl.append(derive_bowling_strategy(row))
        y_bat.append(derive_batting_strategy(row))

    X      = np.array(X_rows, dtype=np.float32)
    y_bowl = np.array(y_bowl)
    y_bat  = np.array(y_bat)

    # 4. Match-level split (same as trainer — no data leakage)
    train_mask, test_mask = match_level_split(df)
    print(f"  Train: {train_mask.sum():,} rows | Test: {test_mask.sum():,} rows")
    print(f"  Test matches: {df.loc[test_mask, 'match_id'].nunique()} (completely unseen)")

    X_test     = X[test_mask]
    y_bowl_test = y_bowl[test_mask]
    y_bat_test  = y_bat[test_mask]

    # 5. Load trained models + encoders
    print("\n  Loading models...")
    for path in [BOWLING_MODEL_PATH, BATTING_MODEL_PATH, BOWLING_ENC_PATH, BATTING_ENC_PATH]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found. Run: python tactical_model_trainer.py")

    bowl_model = joblib.load(BOWLING_MODEL_PATH)
    bat_model  = joblib.load(BATTING_MODEL_PATH)
    bowl_enc   = joblib.load(BOWLING_ENC_PATH)
    bat_enc    = joblib.load(BATTING_ENC_PATH)

    # 6. Evaluate BOWLING model
    print("\n" + "-" * 50)
    print("  BOWLING STRATEGY MODEL")
    print("-" * 50)

    y_bowl_true_enc = bowl_enc.transform(y_bowl_test)
    y_bowl_pred     = bowl_model.predict(X_test)
    bowl_acc        = accuracy_score(y_bowl_true_enc, y_bowl_pred)

    print(f"\n  Accuracy: {bowl_acc:.4f}")
    print("\n" + classification_report(
        y_bowl_true_enc, y_bowl_pred, target_names=bowl_enc.classes_,
    ))

    # 7. Evaluate BATTING model
    print("-" * 50)
    print("  BATTING STRATEGY MODEL")
    print("-" * 50)

    y_bat_true_enc = bat_enc.transform(y_bat_test)
    
    # Drop features for batting: powerplay=8, wicket=12, phase=15
    keep_cols_bat = [i for i in range(X_test.shape[1]) if i not in [8, 12, 15]]
    X_test_bat = X_test[:, keep_cols_bat]
    
    y_bat_pred     = bat_model.predict(X_test_bat)
    bat_acc        = accuracy_score(y_bat_true_enc, y_bat_pred)

    print(f"\n  Accuracy: {bat_acc:.4f}")
    print("\n" + classification_report(
        y_bat_true_enc, y_bat_pred, target_names=bat_enc.classes_,
    ))

    # 8. Summary
    print("=" * 60)
    print(f"  BOWLING Accuracy: {bowl_acc:.4f}")
    print(f"  BATTING Accuracy: {bat_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()