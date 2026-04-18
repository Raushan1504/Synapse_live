# player_stats.py

import pandas as pd


def compute_player_stats(df: pd.DataFrame):
    """
    Compute:
    - batsman average strike rate
    - bowler average economy
    """

    # --- Batsman Strike Rate ---
    batsman_stats = (
        df.groupby("batsman")
        .agg(
            total_runs=("runs", "sum"),
            balls=("ball", "count")
        )
        .reset_index()
    )

    batsman_stats["strike_rate"] = (
        batsman_stats["total_runs"] / batsman_stats["balls"]
    ) * 100

    batsman_dict = dict(
        zip(batsman_stats["batsman"], batsman_stats["strike_rate"])
    )

    # --- Bowler Economy ---
    bowler_stats = (
        df.groupby("bowler")
        .agg(
            runs_conceded=("runs", "sum"),
            balls=("ball", "count")
        )
        .reset_index()
    )

    bowler_stats["economy"] = (
        bowler_stats["runs_conceded"] / bowler_stats["balls"]
    ) * 6

    bowler_dict = dict(
        zip(bowler_stats["bowler"], bowler_stats["economy"])
    )

    return batsman_dict, bowler_dict