"""
test_strategy.py
----------------
Quick integration test: loads the ML engine via SynapseLive.create()
and prints strategy output for a few representative ball scenarios.

Usage:
    python test_strategy.py
"""

import pandas as pd
from synapse_live import SynapseLive
from player_stats import compute_player_stats

# Load data
print("Loading dataset...")
df = pd.read_csv("clean_ipl_dataset.csv")
df.columns = df.columns.str.strip()

# Stats
batsman_stats, bowler_stats = compute_player_stats(df)

# Engine
synapse = SynapseLive.create(batsman_stats, bowler_stats)
print(f"Engine type: {synapse.engine_type}\n")

# Pick representative rows: powerplay, middle overs, death overs
test_cases = [
    ("Powerplay ball",    df[(df["over"] <= 6) & (df["innings"] == 1)].iloc[10]),
    ("Middle-over ball",  df[(df["over"] >= 8) & (df["over"] <= 14)].iloc[50]),
    ("Death-over ball",   df[df["over"] >= 17].iloc[20]),
]

for label, row in test_cases:
    # Inject player stats (same as simulation loop)
    row = row.copy()
    row["strike_rate"]    = batsman_stats.get(row.get("batsman"), 120.0)
    row["bowler_economy"] = bowler_stats.get(row.get("bowler"),   7.5)
    row["wickets_fallen"] = 0

    print(f"===== {label} (Over {row['over']}.{row['ball']}, Innings {row['innings']}) =====")

    strategy = synapse.generate_strategy(row)
    print("\nStrategy:")
    print(strategy)

    json_out = synapse.get_tactical_json(row)
    print("\nML Predictions:")
    print(f"  Batting:  {json_out['batting_team']['ml_strategy']}"
          f" (conf: {json_out['batting_team']['ml_confidence']})")
    print(f"  Bowling:  {json_out['bowling_team']['ml_strategy']}"
          f" (conf: {json_out['bowling_team']['ml_confidence']})")
    print()