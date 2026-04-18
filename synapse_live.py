"""
synapse_live.py
───────────────
Drop-in integration adapter for Synapse Live.

Provides
────────
  SynapseLive          — engine factory; auto-selects ML or rule-based engine
  generate_strategy()  — replaces generate_strategy() in simulation_code.py
                         with ZERO changes to that file
  get_tactical_json()  — full structured dict for API / frontend consumption

Integration (two-line change to simulation_code.py's main())
─────────────────────────────────────────────────────────────

  # At the top of main():
  from synapse_live import SynapseLive
  synapse = SynapseLive.create(batsman_stats, bowler_stats)

  # In simulate_match(), replace the generate_strategy() call:
  strategy = synapse.generate_strategy(row_ext)   # ← drop-in replacement

  # Optional — store the full JSON for API or logging:
  tactical_json = synapse.get_tactical_json(row_ext)

End-of-innings 2 required rate support
──────────────────────────────────────
  # At the start of innings 2 (inside simulate_match when innings changes):
  if current_innings == 2:
      inn1_score = df_match[df_match["innings"]==1]["team_score"].max()
      synapse.set_innings1_score(inn1_score)
"""

import os
import json
from typing import Dict, Optional

from tactical_decision_engine import (
    TacticalDecisionEngine,
    RuleBasedFallbackEngine,
    BOWLING_MODEL_PATH,
    BATTING_MODEL_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPSE LIVE — UNIFIED INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class SynapseLive:
    """
    Unified Synapse Live interface.

    Auto-selects:
      • TacticalDecisionEngine (ML) when models exist in ./models/
      • RuleBasedFallbackEngine    when models are absent (no training needed)

    All public methods have the same signature regardless of which engine
    is running under the hood — callers never need to branch.
    """

    def __init__(self, engine):
        self._engine = engine
        self._engine_type = type(engine).__name__

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        batsman_stats: Dict,
        bowler_stats: Dict,
        force_rules: bool = False,
    ) -> "SynapseLive":
        """
        Factory method — returns a ready SynapseLive instance.

        Parameters
        ----------
        batsman_stats : dict  {batsman_name → historical strike_rate}
        bowler_stats  : dict  {bowler_name  → historical economy}
        force_rules   : bool  If True, skip ML even if models exist.
        """
        models_ready = (
            os.path.exists(BOWLING_MODEL_PATH)
            and os.path.exists(BATTING_MODEL_PATH)
        )

        if models_ready and not force_rules:
            engine = TacticalDecisionEngine(batsman_stats, bowler_stats)
            print("[OK] Synapse Live: ML Tactical Engine active.")
        else:
            engine = RuleBasedFallbackEngine(batsman_stats, bowler_stats)
            if force_rules:
                print("[INFO] Synapse Live: Rule-based engine (forced).")
            else:
                print("[WARN] Synapse Live: ML models not found -> rule-based fallback.")
                print("    Train models: python tactical_model_trainer.py")

        return cls(engine)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_tactical_json(self, row) -> Dict:
        """
        Full structured tactical decision — Python dict, JSON-serialisable.
        Call this for API responses, logging, or frontend delivery.
        """
        return self._engine.decide(row)

    def get_tactical_json_str(self, row, indent: int = 2) -> str:
        """Full tactical decision as a formatted JSON string."""
        return self._engine.decide_json(row, indent)

    def generate_strategy(self, row) -> str:
        """
        DROP-IN REPLACEMENT for generate_strategy() in simulation_code.py.

        Returns a concise, richly formatted one-liner for terminal display.
        Same signature: takes `row` (pd.Series | dict), returns str.
        No changes to simulation_code.py are required.

        Example output
        ──────────────
        [DEATH] BAT: Death Hitting — MS Dhoni (Finisher, hist SR 141.6) — go for
        broke. Pre-meditate, use the crease … | BOWL: Death Bowling — JJ Bumrah
        (2 overs left, eco 6.80) — yorker-to-yorker plan …
        [END OF OVER 18] → Next: JJ Bumrah | eco 6.80 | Reason: …
        """
        decision = self._engine.decide(row)
        over      = row.get("over", "?")
        ball      = int(row.get("ball", 0))
        phase     = decision.get("phase", "unknown").upper()

        bat  = decision["batting_team"]
        bowl = decision["bowling_team"]

        bat_strat  = bat.get("ml_strategy", "")
        bowl_strat = bowl.get("ml_strategy", "")
        bat_note   = bat.get("captain_instruction", "")
        bowl_note  = bowl.get("captain_instruction", "")

        # Confidence badge (only for ML engine)
        bat_conf  = bat.get("ml_confidence")
        bowl_conf = bowl.get("ml_confidence")
        conf_str  = (
            f" [bat {bat_conf:.0%} | bowl {bowl_conf:.0%}]"
            if bat_conf is not None else ""
        )

        # End-of-over recommendation
        eoo_str = ""
        if ball == 6 and "next_over_recommendation" in bowl:
            rec = bowl["next_over_recommendation"]
            nb  = rec["recommended_bowler"]
            eco = rec.get("economy", "?")
            why = rec["reason"]
            reserves = rec.get("death_over_reserves", [])
            res_str  = f" | Death reserve: {', '.join(reserves)}" if reserves else ""
            eoo_str  = (
                f"\n  +-- END OF OVER {over} -----------------------------------\n"
                f"  | Next bowler: {nb} (eco {eco}){res_str}\n"
                f"  | Reason: {why}\n"
                f"  +--------------------------------------------------------"
            )

        return (
            f"[{phase}]{conf_str} "
            f"BAT: {bat_strat} - {bat_note} "
            f"| BOWL: {bowl_strat} - {bowl_note}"
            f"{eoo_str}"
        )

    def set_innings1_score(self, score: int) -> None:
        """
        Supply innings-1 final score so the engine can compute required rate.
        Call this at the start of innings 2 inside simulate_match.
        """
        self._engine.set_innings1_score(score)

    @property
    def engine_type(self) -> str:
        return self._engine_type


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST  (run directly to verify integration without simulation)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Runs a 5-ball sequence through the engine and prints JSON output.
    Uses the real dataset and real player stats — no mocks.
    """
    import sys
    import pandas as pd
    from player_stats import compute_player_stats

    DATA_PATH = "clean_ipl_dataset.csv"
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    batsman_stats, bowler_stats = compute_player_stats(df)
    engine = SynapseLive.create(batsman_stats, bowler_stats)

    # Pick a death-over sequence for a good demo
    sample = (
        df[(df["over"] >= 17) & (df["innings"] == 2)]
        .head(6)
        .copy()
    )

    print("\n" + "=" * 70)
    print("  SYNAPSE LIVE — SMOKE TEST  (6 death-over balls, innings 2)")
    print("=" * 70)

    inn1_score = df[df["match_id"] == sample["match_id"].iloc[0]].query("innings==1")["team_score"].max()
    engine.set_innings1_score(int(inn1_score))

    for _, row in sample.iterrows():
        row = row.copy()
        row["strike_rate"]    = batsman_stats.get(row.get("batsman"), 120.0)
        row["bowler_economy"] = bowler_stats.get(row.get("bowler"),   7.5)
        row["wickets_fallen"] = 0   # placeholder for smoke test

        print(f"\n── Ball {row['over']}.{row['ball']} ──")
        print(engine.generate_strategy(row))

        if int(row["ball"]) == 6:
            print("\n  Full JSON (last ball of over):")
            print(engine.get_tactical_json_str(row, indent=2))


if __name__ == "__main__":
    _smoke_test()
