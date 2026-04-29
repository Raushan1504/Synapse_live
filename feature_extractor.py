"""
feature_extractor.py
────────────────────
Converts a single ball row (pd.Series or dict) into a numeric feature
vector that the ML models can consume.

Rules:
  • No side effects — pure function, stateless.
  • Works identically for training (full dataset loop) and live simulation.
  • All features are numeric — tree models don't need scaling, but the
    vector shape must never change between training and inference.
"""

import pandas as pd
import numpy as np


# ── Feature names (ordered) ───────────────────────────────────────────────────
# Keeping these as a module-level constant ensures training and inference
# always produce the same column order.
FEATURE_NAMES = [
    "over",
    "ball",
    "innings",
    "run_rate",
    "strike_rate",
    "bowler_economy",
    "team_score",
    "overs_float",
    "powerplay",
    "is_boundary",
    "is_dot",
    "is_six",
    "wicket",
    # Engineered features
    "balls_remaining",
    "pressure_index",
    "phase",
    "batsman_role_code",
    "bowler_type_code",
    # NEW Day 1 features
    "rrr_crr_delta",
    "wickets_in_hand",
    "dot_streak",
    "recent_boundary_rate",
    "resource_index",
    "partnership_runs",
    "match_stage",
    "bowler_last_over_economy",
]


def _safe_get(row, key: str, default: float = 0.0) -> float:
    """Return row[key] if it exists and is not NaN, otherwise default."""
    try:
        val = row[key] if isinstance(row, dict) else row.get(key, default)
        return default if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
    except (KeyError, TypeError, ValueError):
        return default


def extract_features(row) -> dict:
    over         = _safe_get(row, "over",           0.0)
    ball         = _safe_get(row, "ball",           0.0)
    innings      = _safe_get(row, "innings",        1.0)
    run_rate     = _safe_get(row, "run_rate",       7.0)
    strike_rate  = _safe_get(row, "strike_rate",  100.0)
    economy      = _safe_get(row, "bowler_economy", 7.5)
    team_score   = _safe_get(row, "team_score",     0.0)
    overs_float  = over + (ball - 1) / 6.0
    powerplay    = _safe_get(row, "powerplay",       0.0)
    is_boundary  = _safe_get(row, "is_boundary",    0.0)
    is_dot       = _safe_get(row, "is_dot",         0.0)
    is_six       = _safe_get(row, "is_six",         0.0)
    wicket       = _safe_get(row, "wicket",         0.0)

    # Original engineered features
    balls_remaining = max(0.0, (20.0 - overs_float) * 6.0)
    pressure_index  = run_rate / max(economy, 1.0)

    if powerplay:
        phase = 0.0
    elif over >= 16:
        phase = 2.0
    else:
        phase = 1.0

    role = str(row.get("batsman_role", "middle")).lower() if isinstance(row, dict) else str(getattr(row, "batsman_role", "middle")).lower()
    if "opener" in role:
        batsman_role_code = 0.0
    elif "finisher" in role:
        batsman_role_code = 2.0
    else:
        batsman_role_code = 1.0

    bowler_type = str(row.get("bowler_type", "pace")).lower() if isinstance(row, dict) else str(getattr(row, "bowler_type", "pace")).lower()
    bowler_type_code = 1.0 if "spin" in bowler_type else 0.0

    # ── NEW Day 1 features ────────────────────────────────────────────────────
    # RRR vs CRR delta (how far behind/ahead of required rate)
    required_run_rate = _safe_get(row, "required_run_rate", run_rate)
    rrr_crr_delta = required_run_rate - run_rate

    # Wickets in hand (pre-computed in training loop; fallback to 10)
    wickets_fallen   = _safe_get(row, "wickets_fallen", 0.0)
    wickets_in_hand  = max(0.0, 10.0 - wickets_fallen)

    # Dot streak (pre-computed in training loop; fallback 0)
    dot_streak = _safe_get(row, "dot_streak", 0.0)

    # Recent boundary rate (pre-computed in training loop; fallback 0)
    recent_boundary_rate = _safe_get(row, "recent_boundary_rate", 0.0)

    # Resource index (DLS approximation)
    resource_index = (balls_remaining / 120.0) * (wickets_in_hand / 10.0)

    # Partnership runs (pre-computed in training loop; fallback 0)
    partnership_runs = _safe_get(row, "partnership_runs", 0.0)

    # Match stage (0=powerplay, 1=early middle, 2=middle, 3=late middle, 4=death)
    if over <= 6:
        match_stage = 0.0
    elif over <= 10:
        match_stage = 1.0
    elif over <= 15:
        match_stage = 2.0
    elif over <= 18:
        match_stage = 3.0
    else:
        match_stage = 4.0

    # Bowler's runs in last over (pre-computed in training loop; fallback economy)
    bowler_last_over_economy = _safe_get(row, "bowler_last_over_economy", economy)

    return {
        "over":                    over,
        "ball":                    ball,
        "innings":                 innings,
        "run_rate":                run_rate,
        "strike_rate":             strike_rate,
        "bowler_economy":          economy,
        "team_score":              team_score,
        "overs_float":             overs_float,
        "powerplay":               powerplay,
        "is_boundary":             is_boundary,
        "is_dot":                  is_dot,
        "is_six":                  is_six,
        "wicket":                  wicket,
        "balls_remaining":         balls_remaining,
        "pressure_index":          pressure_index,
        "phase":                   phase,
        "batsman_role_code":       batsman_role_code,
        "bowler_type_code":        bowler_type_code,
        # New
        "rrr_crr_delta":           rrr_crr_delta,
        "wickets_in_hand":         wickets_in_hand,
        "dot_streak":              dot_streak,
        "recent_boundary_rate":    recent_boundary_rate,
        "resource_index":          resource_index,
        "partnership_runs":        partnership_runs,
        "match_stage":             match_stage,
        "bowler_last_over_economy": bowler_last_over_economy,
    }


def features_to_array(feature_dict: dict) -> list:
    """
    Convert a feature dict to an ordered list aligned with FEATURE_NAMES.
    Use this when calling model.predict() to guarantee column order.
    """
    return [feature_dict[k] for k in FEATURE_NAMES]


# ── Label engineering (used only during training) ─────────────────────────────

def derive_bowling_strategy(row) -> str:
    """
    Engineer a bowling strategy label from match context.
    These labels become the target classes for the bowling classifier.
    """
    over        = _safe_get(row, "over",           0.0)
    run_rate    = _safe_get(row, "run_rate",        7.0)
    economy     = _safe_get(row, "bowler_economy",  7.5)
    powerplay   = _safe_get(row, "powerplay",       0.0)
    strike_rate = _safe_get(row, "strike_rate",   100.0)

    if over >= 16:
        return "Death Bowling"
    if powerplay:
        return "Powerplay Attack"
    if economy > 9 or (strike_rate > 150 and run_rate > 9):
        return "Variation & Pressure"
    if run_rate <= 7 and economy < 7:
        return "Containment"
    return "Middle Overs Build"


def derive_batting_strategy(row) -> str:
    """
    Engineer a batting strategy label from match context.
    These labels become the target classes for the batting classifier.
    """
    innings    = _safe_get(row, "innings",   1.0)
    over       = _safe_get(row, "over",      0.0)
    run_rate   = _safe_get(row, "run_rate",  7.0)
    powerplay  = _safe_get(row, "powerplay", 0.0)
    wicket     = _safe_get(row, "wicket",    0.0)

    if wicket:
        return "Recovery Mode"
    if powerplay:
        return "Powerplay Build"
    if over >= 16:
        return "Death Hitting"
    if innings == 2 and run_rate > 9:
        return "Chase Aggression"
    if innings == 2 and run_rate < 6:
        return "Steady Chase"
    return "Accumulate & Rotate"