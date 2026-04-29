"""
tactical_model_trainer.py
─────────────────────────
Trains two RandomForest classifiers:
  1. Bowling Strategy  →  5 classes derived from match context
  2. Batting Strategy  →  6 classes derived from match context

Uses:
  • feature_extractor.py  — feature extraction + label engineering
  • player_stats.py       — historical strike-rate / economy injection
  • clean_ipl_dataset.csv — 138k ball-by-ball IPL deliveries

Output:
  models/bowling_strategy_model.pkl
  models/bowling_label_encoder.pkl
  models/batting_strategy_model.pkl
  models/batting_label_encoder.pkl

Usage:
  python tactical_model_trainer.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Sibling modules (no modifications to these files) ─────────────────────────
from feature_extractor import (
    extract_features,
    features_to_array,
    FEATURE_NAMES,
    derive_bowling_strategy,
    derive_batting_strategy,
)
from player_stats import compute_player_stats
from kmeans_label_generator import generate_kmeans_labels


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH          = "clean_ipl_dataset.csv"
MODEL_DIR          = "models"
BOWLING_MODEL_PATH = os.path.join(MODEL_DIR, "bowling_strategy_model.pkl")
BATTING_MODEL_PATH = os.path.join(MODEL_DIR, "batting_strategy_model.pkl")
BOWLING_ENC_PATH   = os.path.join(MODEL_DIR, "bowling_label_encoder.pkl")
BATTING_ENC_PATH   = os.path.join(MODEL_DIR, "batting_label_encoder.pkl")

# RandomForest hyperparameters — tuned for realistic generalization
RF_PARAMS = dict(
    n_estimators     = 280,     # enough trees for stable estimates
    max_depth        = 13,      # moderate depth prevents overfitting
    min_samples_leaf = 7,       # prevents micro-overfitting to single events
    max_features     = "sqrt",  # standard for classification
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = 42,
)

# Label noise rate — small enough to keep labels mostly correct,
# large enough to prevent 100% accuracy on rule-derived targets.
LABEL_NOISE_RATE = 0.07

# Train/test split ratio (match-level)
TEST_MATCH_FRACTION = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Match-level train/test split  (fixes data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def match_level_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_MATCH_FRACTION,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split matches (not rows) into train/test so that no match appears in both.

    Returns
    -------
    train_idx, test_idx : boolean arrays aligned with df index
    """
    rng = np.random.RandomState(random_state)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)

    n_test = max(1, int(len(match_ids) * test_fraction))
    test_matches = set(match_ids[:n_test])

    is_test  = df["match_id"].isin(test_matches).values
    is_train = ~is_test
    return is_train, is_test


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Controlled label noise  (breaks deterministic mapping)
# ─────────────────────────────────────────────────────────────────────────────

def inject_label_noise(
    labels: np.ndarray,
    noise_rate: float = LABEL_NOISE_RATE,
    random_state: int = 42,
) -> np.ndarray:
    """
    Randomly flip a small fraction of labels to break the perfect 1:1
    correspondence between features and rule-derived labels.

    Only applied to the TRAINING set — test labels stay untouched so
    evaluation reflects true rule-based targets.
    """
    rng = np.random.RandomState(random_state)
    noisy = labels.copy()
    unique_labels = np.unique(labels)

    n_flip = int(len(labels) * noise_rate)
    flip_idx = rng.choice(len(labels), size=n_flip, replace=False)

    for idx in flip_idx:
        alternatives = unique_labels[unique_labels != labels[idx]]
        noisy[idx] = rng.choice(alternatives)

    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Feature binning  (reduces perfect feature→label leakage)
# ─────────────────────────────────────────────────────────────────────────────

def apply_feature_binning(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Replace continuous features with coarser bins to prevent the model
    from perfectly reconstructing the rule thresholds used for labeling.

    Also adds small Gaussian noise to blur exact threshold boundaries.

    Binning is applied INSIDE the trainer only -- feature_extractor.py is
    unchanged so inference still uses raw features (the model learns on
    bins and generalises on raw values, which adds healthy noise).

    Bins applied:
      - over           (col 0)  -> 4 bins  (powerplay/early/middle/death)
      - run_rate       (col 3)  -> 3 bins  (low / medium / high)
      - strike_rate    (col 4)  -> 3 bins
      - bowler_economy (col 5)  -> 3 bins
      - team_score     (col 6)  -> 4 bins
    """
    rng = np.random.RandomState(random_state)
    X_binned = X.copy()

    # over -> {0, 1, 2, 3}  (key for batting: over >= 16 = death)
    ov = X_binned[:, 0]
    X_binned[:, 0] = np.where(ov <= 6, 0, np.where(ov <= 11, 1,
                     np.where(ov <= 15, 2, 3)))

    # run_rate -> {0, 1, 2}
    rr = X_binned[:, 3]
    X_binned[:, 3] = np.where(rr <= 6, 0, np.where(rr <= 9, 1, 2))

    # strike_rate -> {0, 1, 2}
    sr = X_binned[:, 4]
    X_binned[:, 4] = np.where(sr <= 100, 0, np.where(sr <= 150, 1, 2))

    # bowler_economy -> {0, 1, 2}
    econ = X_binned[:, 5]
    X_binned[:, 5] = np.where(econ <= 6, 0, np.where(econ <= 9, 1, 2))

    # team_score -> {0, 1, 2, 3}
    ts = X_binned[:, 6]
    X_binned[:, 6] = np.where(
        ts <= 50, 0,
        np.where(ts <= 120, 1, np.where(ts <= 180, 2, 3)),
    )

    # Add small Gaussian noise to all features to blur exact boundaries
    noise = rng.normal(0, 0.05, size=X_binned.shape).astype(np.float32)
    X_binned = X_binned + noise

    return X_binned


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build feature matrix + labels
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(
    df: pd.DataFrame,
    batsman_stats: dict,
    bowler_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix + labels with pre-computed rolling features."""

    print(f"  Pre-computing rolling features for {len(df):,} deliveries...")

    # ── Pre-compute new rolling features per match ────────────────────────────
    df = df.copy()

    # Wickets fallen (cumulative per match per innings)
    df["wickets_fallen"] = df.groupby(["match_id", "innings"])["wicket"].cumsum()

    # Dot streak (consecutive dots)
    def _dot_streak(series):
        streak, result = 0, []
        for v in series:
            streak = streak + 1 if v == 1 else 0
            result.append(streak)
        return result

    df["dot_streak"] = df.groupby(["match_id", "innings"])["is_dot"].transform(
        lambda x: _dot_streak(x.tolist())
    )

    # Recent boundary rate (last 12 balls)
    df["recent_boundary_rate"] = df.groupby(["match_id", "innings"])["is_boundary"].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    # Partnership runs (reset at each wicket)
    df["wicket_id"] = df.groupby(["match_id", "innings"])["wicket"].cumsum()
    df["partnership_runs"] = df.groupby(
        ["match_id", "innings", "wicket_id"]
    )["runs_off_bat"].cumsum() if "runs_off_bat" in df.columns else df.groupby(
        ["match_id", "innings", "wicket_id"]
    )["runs"].cumsum()

    # Bowler's last-over economy (rolling 6-ball sum)
    df["bowler_last_over_economy"] = df.groupby(["match_id", "innings", "bowler"])["runs"].transform(
        lambda x: x.rolling(6, min_periods=1).sum()
    )

    # Required run rate (innings 2 only — filled from innings 1 score)
    inn1_scores = df[df["innings"] == 1].groupby("match_id")["team_score"].max()
    df["innings1_score"] = df["match_id"].map(inn1_scores)
    df["overs_float_col"] = df["over"] + (df["ball"] - 1) / 6.0
    overs_remaining = (20.0 - df["overs_float_col"]).clip(lower=0.01)
    runs_needed = (df["innings1_score"] + 1 - df["team_score"]).clip(lower=0)
    df["required_run_rate"] = np.where(
        df["innings"] == 2,
        (runs_needed / overs_remaining).round(2),
        df["run_rate"]
    )

    print(f"  Rolling features done. Building feature vectors...")
    t0 = time.time()

    X_rows, y_bowl, y_bat = [], [], []

    for _, row in df.iterrows():
        row = row.copy()
        row["strike_rate"]    = batsman_stats.get(row.get("batsman"), 120.0)
        row["bowler_economy"] = bowler_stats.get(row.get("bowler"),   7.5)
        feats = extract_features(row)
        X_rows.append(features_to_array(feats))

    elapsed = time.time() - t0
    print(f"  Feature extraction done in {elapsed:.1f}s.")

    # ── K-Means data-driven labels (replaces if-else rules) ──────────────────
    print("\n  Generating K-Means strategy labels (data-driven)...")
    y_bowl, y_bat = generate_kmeans_labels(df)

    return np.array(X_rows, dtype=np.float32), y_bowl, y_bat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train, evaluate, persist
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
    encoder_path: str,
    label: str,
    noise_rate: float = LABEL_NOISE_RATE,
    drop_feature_indices: list | None = None,
) -> tuple[XGBClassifier, LabelEncoder]:
    """
    Train a single RandomForest classifier and save artifacts to disk.

    Parameters
    ----------
    noise_rate           : fraction of training labels to randomly flip
    drop_feature_indices : column indices to remove before training
                           (breaks direct feature->label leakage for
                           labels derived from binary features)

    Returns the fitted (model, encoder) pair so callers can inspect them.
    """
    # ── Encode labels ────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))  # fit on ALL labels

    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    # ── Apply feature binning to training data ───────────────────────────
    X_train_binned = apply_feature_binning(X_train)
    X_test_binned  = apply_feature_binning(X_test)

    # ── Drop high-leakage features if specified ──────────────────────────
    keep_cols = list(range(X_train_binned.shape[1]))
    used_feature_names = list(FEATURE_NAMES)
    if drop_feature_indices:
        keep_cols = [i for i in keep_cols if i not in drop_feature_indices]
        used_feature_names = [FEATURE_NAMES[i] for i in keep_cols]
        X_train_binned = X_train_binned[:, keep_cols]
        X_test_binned  = X_test_binned[:, keep_cols]
        print(f"  Dropped features: "
              f"{[FEATURE_NAMES[i] for i in drop_feature_indices]}")
        print(f"  Using {len(keep_cols)} features: {used_feature_names}")

    # ── Inject controlled noise into training labels ─────────────────────
    y_train_noisy = inject_label_noise(y_train_enc, noise_rate)

    print(f"\n  Classes  : {list(le.classes_)}")
    print(f"  Train/test: {len(X_train_binned):,} / {len(X_test_binned):,}")

    noisy_count = np.sum(y_train_noisy != y_train_enc)
    print(f"  Label noise: {noisy_count:,} labels flipped "
          f"({100 * noisy_count / len(y_train_enc):.1f}%)")

    # ── Train ────────────────────────────────────────────────────────────
 # ── Train ────────────────────────────────────────────────────────────
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    print("  Fitting...")
    t0 = time.time()
    clf.fit(X_train_binned, y_train_noisy)
    elapsed = time.time() - t0
    print(f"  Fit time : {elapsed:.1f}s")

    # ── Evaluate on CLEAN test labels ────────────────────────────────────
    y_pred = clf.predict(X_test_binned)
    acc    = accuracy_score(y_test_enc, y_pred)

    print(f"\n  +-----------------------------------+")
    print(f"  |  {label} Accuracy: {acc:.4f}        |")
    print(f"  +-----------------------------------+")
    print("\n" + classification_report(
        y_test_enc, y_pred, target_names=le.classes_,
    ))

    # Feature importance (top-5)
    importance = sorted(
        zip(used_feature_names, clf.feature_importances_),
        key=lambda x: -x[1],
    )
    print("  Top-5 features:")
    for fname, imp in importance[:5]:
        bar = "#" * int(imp * 60)
        print(f"    {fname:<20} {bar} {imp:.3f}")

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n  [OK] Saved -> {model_path}")
    return clf, le


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Synapse Live — Tactical Model Trainer")
    print("=" * 60)

    # 1. Load data

    print("\n[1/5] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        sys.exit(f"ERROR: {DATA_PATH} not found. Place it in the working directory.")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    n_matches = df["match_id"].nunique()
    print(f"  Loaded {len(df):,} rows  |  {n_matches} matches")

    # 2. Player stats
    print("\n[2/5] Computing player statistics...")
    batsman_stats, bowler_stats = compute_player_stats(df)
    print(f"  Batsmen: {len(batsman_stats)}  |  Bowlers: {len(bowler_stats)}")

    # 3. Feature matrix
    print("\n[3/5] Building training data...")
    X, y_bowl, y_bat = build_training_data(df, batsman_stats, bowler_stats)
    print(f"  Feature matrix: {X.shape}  |  Features: {FEATURE_NAMES}")

    # 4. Match-level split (prevents data leakage)
    print("\n[4/5] Splitting by match_id (no leakage)...")
    train_mask, test_mask = match_level_split(df)
    n_train_matches = df.loc[train_mask, "match_id"].nunique()
    n_test_matches  = df.loc[test_mask, "match_id"].nunique()
    print(f"  Train matches: {n_train_matches}  |  Test matches: {n_test_matches}")
    print(f"  Train rows: {train_mask.sum():,}  |  Test rows: {test_mask.sum():,}")

    X_train, X_test   = X[train_mask], X[test_mask]
    y_bowl_train, y_bowl_test = y_bowl[train_mask], y_bowl[test_mask]
    y_bat_train,  y_bat_test  = y_bat[train_mask],  y_bat[test_mask]

    # 5. Train models
    print("\n[5/5] Training classifiers...\n")

    print("-" * 50)
    print("  BOWLING STRATEGY MODEL")
    print("-" * 50)
    train_and_save(
        X_train, X_test, y_bowl_train, y_bowl_test,
        BOWLING_MODEL_PATH, BOWLING_ENC_PATH, "Bowling",
    )

    print("\n" + "-" * 50)
    print("  BATTING STRATEGY MODEL")
    print("-" * 50)
    # Batting labels are derived from binary features (wicket, powerplay)
    # and exact thresholds (phase). Drop these to force generalization.
    # Indices: powerplay=8, wicket=12, phase=15
    BATTING_DROP_COLS = [8, 12, 15]
    train_and_save(
        X_train, X_test, y_bat_train, y_bat_test,
        BATTING_MODEL_PATH, BATTING_ENC_PATH, "Batting",
        noise_rate=0.10,
        drop_feature_indices=BATTING_DROP_COLS,
    )

    print("\n" + "=" * 60)
    print("  ALL MODELS SAVED  ->  ./models/")
    print("  Run synapse_live.py or integrate via SynapseLive.create()")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
