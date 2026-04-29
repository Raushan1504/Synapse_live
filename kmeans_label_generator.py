"""
kmeans_label_generator.py
─────────────────────────
Replaces rule-based derive_bowling_strategy() and derive_batting_strategy()
with data-driven K-Means cluster labels.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── Bowling cluster → strategy name mapping ───────────────────────────────────
# After fit, inspect profiles and assign names. These defaults work for IPL data.
BOWLING_CLUSTER_NAMES = {
    0: "Powerplay Attack",
    1: "Containment",
    2: "Middle Overs Build",
    3: "Variation & Pressure",
    4: "Death Bowling",
}

BATTING_CLUSTER_NAMES = {
    0: "Accumulate & Rotate",
    1: "Powerplay Build",
    2: "Chase Aggression",
    3: "Recovery Mode",
    4: "Death Hitting",
    5: "Steady Chase",
}

BOWL_FEATURES = [
    "over", "run_rate", "bowler_economy",
    "powerplay", "is_dot", "is_boundary", "overs_float",
]

BAT_FEATURES = [
    "over", "run_rate", "innings",
    "is_boundary", "is_dot", "overs_float", "team_score",
]


def _auto_name_bowling_clusters(profiles: pd.DataFrame) -> dict:
    """
    Auto-assign strategy names by reading cluster profile means.
    Overrides the default BOWLING_CLUSTER_NAMES with data-driven naming.
    """
    mapping = {}
    used = set()

    for cluster_id, row in profiles.iterrows():
        over = row["over"]
        pp   = row["powerplay"]
        econ = row["bowler_economy"]
        dots = row["is_dot"]

        if pp > 0.5:
            name = "Powerplay Attack"
        elif over >= 16:
            name = "Death Bowling"
        elif econ > 8.5 or (dots < 0.35 and over > 6):
            name = "Variation & Pressure"
        elif dots > 0.45:
            name = "Containment"
        else:
            name = "Middle Overs Build"

        # Avoid duplicate names
        if name in used:
            name = name + f" ({cluster_id})"
        used.add(name)
        mapping[cluster_id] = name

    return mapping


def _auto_name_batting_clusters(profiles: pd.DataFrame) -> dict:
    mapping = {}
    used = set()

    for cluster_id, row in profiles.iterrows():
        over    = row["over"]
        rr      = row["run_rate"]
        innings = row["innings"]
        pp      = row.get("powerplay", 0)
        dots    = row["is_dot"]

        if pp > 0.5:
            name = "Powerplay Build"
        elif over >= 16:
            name = "Death Hitting"
        elif innings == 2 and rr > 9:
            name = "Chase Aggression"
        elif innings == 2 and rr < 6:
            name = "Steady Chase"
        elif dots > 0.45:
            name = "Recovery Mode"
        else:
            name = "Accumulate & Rotate"

        if name in used:
            name = name + f" ({cluster_id})"
        used.add(name)
        mapping[cluster_id] = name

    return mapping


def generate_kmeans_labels(df: pd.DataFrame):
    """
    Fit K-Means on bowling and batting features.
    Returns (bowling_labels, batting_labels) as numpy string arrays.
    Prints silhouette scores and cluster profiles.
    """
    df = df.copy().fillna(0)

    # ── BOWLING ───────────────────────────────────────────────────────────────
    print("  Fitting bowling K-Means (k=5)...")
    X_bowl = df[BOWL_FEATURES].values
    scaler_bowl = StandardScaler()
    X_bowl_scaled = scaler_bowl.fit_transform(X_bowl)

    km_bowl = KMeans(n_clusters=5, random_state=42, n_init=10)
    bowl_cluster_ids = km_bowl.fit_predict(X_bowl_scaled)

    sil_bowl = silhouette_score(X_bowl_scaled, bowl_cluster_ids, sample_size=10000)
    print(f"  Bowling silhouette score: {sil_bowl:.3f}")

    bowl_profiles = df.copy()
    bowl_profiles["_cluster"] = bowl_cluster_ids
    bowl_profile_means = bowl_profiles.groupby("_cluster")[BOWL_FEATURES].mean()
    print("\n  Bowling cluster profiles:")
    print(bowl_profile_means.round(2))

    bowl_name_map = {0:"Powerplay Attack", 1:"Containment", 2:"Middle Overs Build", 3:"Variation & Pressure", 4:"Death Bowling"}
    print(f"\n  Bowling cluster profiles:\n{bowl_profile_means.round(2)}")
    bowling_labels = np.array([bowl_name_map[c] for c in bowl_cluster_ids])

    # ── BATTING ───────────────────────────────────────────────────────────────
    print("\n  Fitting batting K-Means (k=6)...")
    X_bat = df[BAT_FEATURES].values
    scaler_bat = StandardScaler()
    X_bat_scaled = scaler_bat.fit_transform(X_bat)

    km_bat = KMeans(n_clusters=6, random_state=42, n_init=10)
    bat_cluster_ids = km_bat.fit_predict(X_bat_scaled)

    sil_bat = silhouette_score(X_bat_scaled, bat_cluster_ids, sample_size=10000)
    print(f"  Batting silhouette score: {sil_bat:.3f}")

    bat_profiles = df.copy()
    bat_profiles["_cluster"] = bat_cluster_ids
    bat_profile_means = bat_profiles.groupby("_cluster")[BAT_FEATURES].mean()
    print("\n  Batting cluster profiles:")
    print(bat_profile_means.round(2))

    bat_name_map = {0:"Accumulate & Rotate", 1:"Powerplay Build", 2:"Chase Aggression", 3:"Recovery Mode", 4:"Death Hitting", 5:"Steady Chase"}
    print(f"\n  Batting cluster profiles:\n{bat_profile_means.round(2)}")
    batting_labels = np.array([bat_name_map[c] for c in bat_cluster_ids])at_cluster_ids])

    return bowling_labels, batting_labels