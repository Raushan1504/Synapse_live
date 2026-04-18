"""
tactical_decision_engine.py
────────────────────────────
Core ML-powered tactical decision system for Synapse Live.

Modules inside this file
────────────────────────
  BowlerConstraintTracker  — enforces max-4-over rule, tracks overs per bowler,
                             recommends next bowler using phase-aware scoring
  MatchStateAnalyzer       — derives phase, pressure level, required run rate
  TacticalDecisionEngine   — loads trained models, generates dual-team JSON output
  RuleBasedFallbackEngine  — identical output schema, no models required (warm-up)

Key contracts
─────────────
  • NEVER modifies simulation_code.py or feature_extractor.py
  • Accepts the same `row` dict/Series that extract_features() accepts
  • Always returns a fully structured dict — every key is always present
  • `next_over_recommendation` key is only added when ball == 6
"""

import os
import json
import joblib
from typing import Dict, List, Optional, Tuple


import numpy as np

from feature_extractor import (
    extract_features,
    features_to_array,
    derive_bowling_strategy,
    derive_batting_strategy,
)


# ── Paths (must match tactical_model_trainer.py) ──────────────────────────────
MODEL_DIR           = "models"
BOWLING_MODEL_PATH  = os.path.join(MODEL_DIR, "bowling_strategy_model.pkl")
BATTING_MODEL_PATH  = os.path.join(MODEL_DIR, "batting_strategy_model.pkl")
BOWLING_ENC_PATH    = os.path.join(MODEL_DIR, "bowling_label_encoder.pkl")
BATTING_ENC_PATH    = os.path.join(MODEL_DIR, "batting_label_encoder.pkl")

MAX_OVERS_PER_BOWLER = 4
TOTAL_OVERS          = 20

PHASE_LABELS = {0: "powerplay", 1: "middle", 2: "death"}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: BOWLER CONSTRAINT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class BowlerConstraintTracker:
    """
    Tracks, per innings, how many overs each bowler has bowled.
    Enforces the T20 max-4-overs rule.
    Recommends the next bowler using phase-weighted economy scoring.

    State update: call register_ball() on every delivery.
    Overs increment on ball == 6 (last delivery of the over).
    """

    def __init__(self, bowler_economy_map: Dict[str, float]):
        # Historical economy for every bowler in the dataset
        self.economy_map: Dict[str, float] = bowler_economy_map
        # Overs completed this innings per bowler
        self.overs_done: Dict[str, int] = {}
        self.current_bowler: Optional[str] = None
        self.last_over_bowler: Optional[str] = None

    def register_ball(self, bowler: str, ball: int) -> None:
        """Update tracker state for a single delivery."""
        self.current_bowler = bowler
        if bowler not in self.overs_done:
            self.overs_done[bowler] = 0
        if ball == 6:  # Last ball of over → credit the over
            self.overs_done[bowler] += 1
            self.last_over_bowler = bowler

    def overs_remaining(self, bowler: str) -> int:
        return max(0, MAX_OVERS_PER_BOWLER - self.overs_done.get(bowler, 0))

    def available_bowlers(self, exclude_last_over: bool = False) -> List[Dict]:
        """
        Return bowlers with quota remaining, sorted by economy (ascending).
        exclude_last_over: skip the bowler who just finished — same bowler can't
        bowl consecutive overs in most interpretations.
        """
        result = []
        for bowler, done in self.overs_done.items():
            remaining = MAX_OVERS_PER_BOWLER - done
            if remaining <= 0:
                continue
            if exclude_last_over and bowler == self.last_over_bowler:
                continue
            result.append({
                "name": bowler,
                "overs_bowled": done,
                "overs_remaining": remaining,
                "economy": round(self.economy_map.get(bowler, 7.5), 2),
            })
        return sorted(result, key=lambda b: b["economy"])

    def recommend_next_bowler(self, current_over: int, phase_code: int) -> Dict:
        """
        Phase-aware next-bowler recommendation.

        Death phase  : use the lowest-economy available bowler
                       unless we're before over 18 — then save the single
                       best finisher for the last 2 overs.
        Powerplay    : prefer sub-7.5 economy bowlers (swing / seam).
        Middle overs : if ≤5 overs remain, don't burn the best finisher yet.
        """
        pool = self.available_bowlers(exclude_last_over=True)
        if not pool:
            # Edge case: everyone used up or only the last-over bowler left
            pool = self.available_bowlers(exclude_last_over=False)
        if not pool:
            return {
                "recommended_bowler": "N/A",
                "reason": "All bowling quotas exhausted",
                "overs_remaining_for_bowler": 0,
                "economy": None,
                "death_over_reserves": [],
                "all_available": [],
            }

        overs_left_in_match = TOTAL_OVERS - current_over

        # ── Death phase ────────────────────────────────────────────────────
        if phase_code == 2 or current_over >= 16:
            if current_over < 18:
                # Don't waste the ace finisher here — pick 2nd best
                candidates = [b for b in pool if b["overs_remaining"] >= 2]
                pick = min(candidates or pool, key=lambda b: b["economy"])
                reason = (
                    f"Over {current_over + 1}: Holding death specialist — "
                    f"deploying reliable setup bowler"
                )
            else:
                pick = min(pool, key=lambda b: b["economy"])
                reason = (
                    f"Over {current_over + 1}: Best economy finisher for "
                    f"final crunch over"
                )

        # ── Powerplay ──────────────────────────────────────────────────────
        elif phase_code == 0:
            elite = [b for b in pool if b["economy"] < 7.5]
            pick  = min(elite or pool, key=lambda b: b["economy"])
            reason = (
                f"Over {current_over + 1}: Powerplay — swing/seam threat with "
                f"economy control"
            )

        # ── Middle overs ───────────────────────────────────────────────────
        else:
            if overs_left_in_match <= 5:
                # Save best for death; use next-best now
                candidates = [b for b in pool if b["overs_remaining"] >= 2]
                pick = min(candidates or pool, key=lambda b: b["economy"])
                reason = (
                    f"Over {current_over + 1}: Preserving death specialists — "
                    f"using efficient middle-over option"
                )
            else:
                pick = min(pool, key=lambda b: b["economy"])
                reason = (
                    f"Over {current_over + 1}: Best available economy bowler "
                    f"for middle-over containment"
                )

        # Death reserves = top-2 economy bowlers with ≥2 overs left, not the pick
        reserves = [
            b["name"] for b in pool
            if b["economy"] < 8.0
            and b["overs_remaining"] >= 2
            and b["name"] != pick["name"]
        ][:2]

        return {
            "recommended_bowler": pick["name"],
            "reason": reason,
            "overs_remaining_for_bowler": pick["overs_remaining"],
            "economy": pick["economy"],
            "death_over_reserves": reserves,
            "all_available": [
                f"{b['name']} ({b['overs_remaining']} left | eco {b['economy']})"
                for b in pool
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: MATCH STATE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class MatchStateAnalyzer:
    """
    Derives high-level match context — phase, pressure, required run rate.
    Stateful: call set_innings1_score() at the start of innings 2.
    """

    def __init__(self):
        self._inn1_score: Optional[int] = None

    def set_innings1_score(self, score: int) -> None:
        self._inn1_score = score

    def analyze(self, row) -> Dict:
        over        = int(row.get("over", 1))
        ball        = int(row.get("ball", 1))
        innings     = int(row.get("innings", 1))
        team_score  = int(row.get("team_score", 0))
        wickets     = int(row.get("wickets_fallen", 0))
        run_rate    = float(row.get("run_rate", 0.0))
        powerplay   = bool(row.get("powerplay", 0))
        overs_float = float(row.get("overs_float", 0.0))

        overs_remaining = max(0.0, TOTAL_OVERS - overs_float)
        balls_remaining = int(overs_remaining * 6)

        phase_code = (
            0 if powerplay else
            2 if over >= 16 else
            1
        )

        # Required rate for chasing innings
        required_rate = None
        target        = None
        if innings == 2 and self._inn1_score is not None:
            target      = self._inn1_score + 1
            runs_needed = target - team_score
            required_rate = (
                round(runs_needed / overs_remaining, 2)
                if overs_remaining > 0 else float("inf")
            )

        collapse    = wickets >= 4 and over <= 14
        recent_wkt  = bool(row.get("wicket", 0)) or bool(row.get("wicket_prev", 0))
        pressure    = self._pressure_level(
            run_rate, required_rate, wickets, phase_code, innings
        )

        return {
            "over": over,
            "ball": ball,
            "label": f"{over}.{ball}",
            "innings": innings,
            "score": f"{team_score}/{wickets}",
            "team_score": team_score,
            "wickets": wickets,
            "run_rate": round(run_rate, 2),
            "required_rate": required_rate,
            "target": target,
            "overs_remaining": round(overs_remaining, 1),
            "balls_remaining": balls_remaining,
            "phase": PHASE_LABELS[phase_code],
            "phase_code": phase_code,
            "is_powerplay": powerplay,
            "is_collapse": collapse,
            "recent_wicket": recent_wkt,
            "pressure_level": pressure,
        }

    @staticmethod
    def _pressure_level(
        run_rate: float,
        required_rate: Optional[float],
        wickets: int,
        phase_code: int,
        innings: int,
    ) -> str:
        if innings == 2 and required_rate is not None:
            gap = required_rate - run_rate
            if gap > 4 or (wickets >= 7 and phase_code == 2):
                return "critical"
            if gap > 2 or wickets >= 5:
                return "high"
            if gap > 0.5:
                return "medium"
            return "low"
        # Innings 1 pressure
        if wickets >= 5 and phase_code >= 1:
            return "high"
        if run_rate < 5 and phase_code == 0:
            return "medium"
        return "low"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: TACTICAL PLAN LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────

# ── Bowling plans keyed by ML strategy label ──────────────────────────────────
BOWLING_PLAN_MAP: Dict[str, Dict] = {
    "Death Bowling": {
        "recommended_length": "Yorker / Low full toss",
        "recommended_line":   "Toe-crushers / Wide outside off",
        "recommended_variation": "Knuckle ball yorker / Slow bouncer",
        "field_setting": (
            "Fine leg, Deep square leg, Long-on, Long-off, Deep mid-wicket, "
            "Third man, Sweeper cover — 7 on the fence"
        ),
        "over_intent": (
            "Strangle every run. No free hits in the arc. Force mistimed lofts "
            "into the gaps. Accept the wide-off side single; deny the boundary."
        ),
    },
    "Powerplay Attack": {
        "recommended_length": "Full length / Outswing length",
        "recommended_line":   "Off stump / Just outside off",
        "recommended_variation": "Outswing / Seam movement / Late in-ducker",
        "field_setting": (
            "Slip, Gully, Mid-off (straight), Mid-on (straight), "
            "Square leg, Fine leg — attack with 2 inside ring"
        ),
        "over_intent": (
            "Use new ball movement. Bowl in pairs — one swings out, next shapes in. "
            "Attack outside off to induce edges. Don't be defensive in the powerplay."
        ),
    },
    "Variation & Pressure": {
        "recommended_length": "Back of a length / Good length",
        "recommended_line":   "Off stump / Body line",
        "recommended_variation": "Off-cutter / Slow ball / Carrom ball",
        "field_setting": (
            "Mid-wicket, Cover, Long-on, Long-off, Deep square, "
            "Mid-off, Fine leg — orthodox 5-3 containment"
        ),
        "over_intent": (
            "Disrupt batting rhythm — no two consecutive balls the same pace. "
            "Cutter into the body, slower ball outside off. Let the pitch do work."
        ),
    },
    "Containment": {
        "recommended_length": "Good length / Just short of a length",
        "recommended_line":   "Off stump / Outside off",
        "recommended_variation": "Seam movement / Leg-cutter",
        "field_setting": (
            "Mid-off, Mid-on, Cover, Square leg, Fine leg, Third man — "
            "standard containment with mid-off and mid-on up"
        ),
        "over_intent": (
            "Dry up the run rate. Hit the channel consistently. "
            "Invite singles, block boundaries. One dot ball creates the next."
        ),
    },
    "Middle Overs Build": {
        "recommended_length": "Good length",
        "recommended_line":   "Off stump",
        "recommended_variation": "Leg-cutter / Off-cutter / Arm ball",
        "field_setting": (
            "Mid-off, Mid-on, Deep cover, Deep mid-wicket, "
            "Long-on, Fine leg, Square leg — pressure build"
        ),
        "over_intent": (
            "Build dot-ball pressure. Target off stump, beat outside edge, "
            "invite false shots. A wicket here resets the innings."
        ),
    },
}

# ── Batting plans keyed by ML strategy label ──────────────────────────────────
BATTING_PLAN_MAP: Dict[str, Dict] = {
    "Recovery Mode": {
        "shot_zones":   ["Mid-on gap", "Fine leg", "Square leg", "Backward point"],
        "intent":        "Occupation over accumulation — block, leave, re-anchor",
        "priority": (
            "Block the hat-trick ball. No shots outside off for first 6 balls. "
            "Let the partner take control while the new batsman settles."
        ),
        "footwork":     "Front foot — play everything directly under the eyes",
    },
    "Powerplay Build": {
        "shot_zones":   ["Cover point", "Straight", "Midwicket", "Fine leg"],
        "intent":        "Exploit field restrictions — hit gaps, take on powerplay field",
        "priority": (
            "Drive through covers while mid-off is up. Hit straight for mid-on gaps. "
            "Run hard for 2s — the powerplay field concedes them."
        ),
        "footwork":     "Balanced — front-foot drives, back-foot pulls",
    },
    "Death Hitting": {
        "shot_zones":   ["Long-on", "Long-off", "Deep mid-wicket", "Fine leg ramp"],
        "intent":        "Unconditional aggression — every ball is a scoring opportunity",
        "priority": (
            "Pre-meditate based on bowling plan. Inside the line → slog sweep. "
            "Outside off → ramp / reverse scoop. Full on stumps → lofted straight. "
            "Accept wicket risk — dot balls are the enemy now."
        ),
        "footwork":     "Pre-position — get inside or outside the line to create angles",
    },
    "Chase Aggression": {
        "shot_zones":   ["Long-on", "Long-off", "Deep cover", "Sweeper cover"],
        "intent":        "Chase acceleration — required rate must fall every over",
        "priority": (
            "One boundary minimum every other over. Find gaps, not the fielders. "
            "Turn 1s to 2s aggressively. Don't wait for the perfect ball."
        ),
        "footwork":     "Down the track against spinners; back-foot punch against pace",
    },
    "Steady Chase": {
        "shot_zones":   ["Mid-on", "Mid-wicket", "Square leg", "Third man"],
        "intent":        "Manageable run rate — grind, don't gamble",
        "priority": (
            "Let the bowling attack deteriorate. Singles and twos win this. "
            "Absorb pressure, wait for the bad ball, and punish it."
        ),
        "footwork":     "Still — play late, use pace of ball off the pitch",
    },
    "Accumulate & Rotate": {
        "shot_zones":   ["Mid-on gap", "Cover point", "Third man", "Square leg"],
        "intent":        "Tick the scoreboard — deny dot balls, build partnership",
        "priority": (
            "Convert every 0 to a 1 and every 1 to a 2. "
            "Put bowlers under pressure by making them bowl a 6th ball. "
            "Rotate strike — don't let one batsman get bogged down."
        ),
        "footwork":     "Balanced — read line/length early, minimal foot movement",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: PLAYER CONTEXT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _batsman_context(row, batsman_stats: Dict) -> Dict:
    name = row.get("batsman", "Unknown")
    hist_sr = batsman_stats.get(name, float(row.get("strike_rate", 100.0)))
    curr_sr = float(row.get("strike_rate", hist_sr))
    role = (
        "Finisher / Aggressor"   if hist_sr > 150 else
        "Middle-order striker"   if hist_sr > 135 else
        "Accumulator"            if hist_sr > 115 else
        "Anchor"
    )
    return {
        "name":                  name,
        "historical_strike_rate": round(hist_sr, 1),
        "current_match_sr":       round(curr_sr, 1),
        "role":                   role,
    }


def _bowler_context(row, bowler_stats: Dict, tracker: BowlerConstraintTracker) -> Dict:
    name    = row.get("bowler", "Unknown")
    hist_ec = bowler_stats.get(name, float(row.get("bowler_economy", 7.5)))
    curr_ec = float(row.get("bowler_economy", hist_ec))
    return {
        "name":                     name,
        "historical_economy":       round(hist_ec, 2),
        "current_match_economy":    round(curr_ec, 2),
        "overs_bowled_this_innings": tracker.overs_done.get(name, 0),
        "overs_remaining":          tracker.overs_remaining(name),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: CAPTAIN INSTRUCTION GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _batting_captain_note(strategy: str, state: Dict, batsman: Dict) -> str:
    """
    One-liner, actionable captain instruction for the batting team.
    Names the batsman, gives a concrete tactical directive.
    """
    name = batsman["name"]
    sr   = batsman["historical_strike_rate"]
    role = batsman["role"]
    rrr  = state.get("required_rate")

    if strategy == "Death Hitting":
        return (
            f"{name} ({role}, hist SR {sr}) — go for broke. "
            "Pre-meditate, use the crease, target long-on and wide long-off. "
            "A mis-timed six is better than a dot ball."
        )
    if strategy == "Chase Aggression":
        return (
            f"RRR is {rrr} — {name} must accelerate NOW. "
            "Hit every 2 into a 3. One boundary per 2 overs minimum. "
            "Don't respect any bowler."
        )
    if strategy == "Recovery Mode":
        return (
            f"Wicket down — {name} must re-anchor. "
            "No shots outside off for 6 balls. Build a partnership first, "
            "then accelerate once set."
        )
    if strategy == "Powerplay Build":
        return (
            f"Field restrictions live — {name} must exploit gaps. "
            "Drive through covers (mid-off is up). Hit straight for singles "
            "into mid-on area. Run hard."
        )
    if strategy == "Steady Chase":
        return (
            f"Run rate manageable — {name} should grind, not gamble. "
            "Absorb pressure, rotate strike, wait for bad balls. "
            "Let the bowler make mistakes."
        )
    return (
        f"{name} — rotate strike, deny dots, hit the bad ball hard. "
        "One big over can change the momentum."
    )


def _bowling_captain_note(strategy: str, state: Dict, bowler: Dict) -> str:
    """
    One-liner, actionable captain instruction for the bowling team.
    Names the bowler, gives a concrete field/plan directive.
    """
    name = bowler["name"]
    eco  = bowler["current_match_economy"]
    left = bowler["overs_remaining"]

    if strategy == "Death Bowling":
        return (
            f"{name} ({left} overs left, eco {eco}) — yorker-to-yorker plan. "
            "No free hits in the arc. Fine leg and deep square cut off the single. "
            "Bowl wide off if batsman is a leg-side hitter."
        )
    if strategy == "Powerplay Attack":
        return (
            f"{name} — new ball, attack with full length. "
            "One slip, one gully. Outswing first ball every over. "
            "Aim to bowl the batsman on the drive."
        )
    if strategy == "Variation & Pressure":
        return (
            f"{name} (eco {eco}) — change of pace every 2 balls. "
            "No two consecutive balls the same length. Use the crease, "
            "bowl wider of a length to a back-foot player."
        )
    if strategy == "Containment":
        return (
            f"{name} — strangle with channel bowling. "
            "Off stump line, good length, mid-off and mid-on up. "
            "Don't give width. Let the batsman get frustrated."
        )
    return (
        f"{name} — build dot-ball pressure, target off stump, vary pace. "
        "A wicket here opens the match."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6: ML TACTICAL DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TacticalDecisionEngine:
    """
    Primary ML-based engine. Requires trained model files in ./models/.

    Usage
    ─────
        engine = TacticalDecisionEngine(batsman_stats, bowler_stats)
        decision_dict = engine.decide(row)
        decision_json = engine.decide_json(row)
    """

    def __init__(self, batsman_stats: Dict, bowler_stats: Dict):
        self.batsman_stats = batsman_stats
        self.bowler_stats  = bowler_stats

        self.bowl_model, self.bowl_enc = self._load(BOWLING_MODEL_PATH, BOWLING_ENC_PATH)
        self.bat_model,  self.bat_enc  = self._load(BATTING_MODEL_PATH,  BATTING_ENC_PATH)

        self._trackers:  Dict[int, BowlerConstraintTracker] = {}
        self._analyzer   = MatchStateAnalyzer()
        self._cur_innings: Optional[int] = None

    @staticmethod
    def _load(model_path, enc_path):
       model = joblib.load(model_path)
       encoder = joblib.load(enc_path)
       return model, encoder

    def _tracker(self, innings: int) -> BowlerConstraintTracker:
        if innings not in self._trackers:
            self._trackers[innings] = BowlerConstraintTracker(self.bowler_stats)
        return self._trackers[innings]

    def _predict(self, row, model, encoder, drop_feature_indices=None) -> Tuple[str, float]:
        feats = extract_features(row)
        X     = np.array([features_to_array(feats)])
        if drop_feature_indices:
            keep_cols = [i for i in range(X.shape[1]) if i not in drop_feature_indices]
            X = X[:, keep_cols]
            
        probs = model.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        label = encoder.inverse_transform([idx])[0]
        conf  = round(float(probs[idx]), 3)
        return label, conf

    def decide(self, row) -> Dict:
        """
        Generate a full dual-perspective tactical decision for one delivery.

        Parameters
        ----------
        row : pd.Series | dict
            Ball row with injected `strike_rate` and `bowler_economy`
            (as done in simulation_code.py's simulate_match loop).
            Optional: `innings1_score` (int) in innings-2 rows enables RRR.

        Returns
        -------
        dict — structured, JSON-serialisable tactical output.
        """
        innings = int(row.get("innings", 1))
        over    = int(row.get("over", 1))
        ball    = int(row.get("ball", 1))

        # Innings transition
        if innings != self._cur_innings:
            self._cur_innings = innings
            if innings == 2:
                score1 = row.get("innings1_score")
                if score1 is not None:
                    self._analyzer.set_innings1_score(int(score1))

        # Tracker update
        tracker = self._tracker(innings)
        tracker.register_ball(str(row.get("bowler", "Unknown")), ball)

        # Match context
        state      = self._analyzer.analyze(row)
        phase_code = state["phase_code"]

        # ML predictions
        BATTING_DROP_COLS = [8, 12, 15]
        bowl_strat, bowl_conf = self._predict(row, self.bowl_model, self.bowl_enc)
        bat_strat,  bat_conf  = self._predict(row, self.bat_model,  self.bat_enc, drop_feature_indices=BATTING_DROP_COLS)

        # Tactical plans
        bowl_plan = dict(BOWLING_PLAN_MAP.get(bowl_strat, BOWLING_PLAN_MAP["Middle Overs Build"]))
        bat_plan  = dict(BATTING_PLAN_MAP.get(bat_strat,  BATTING_PLAN_MAP["Accumulate & Rotate"]))

        # Player contexts
        bat_ctx  = _batsman_context(row, self.batsman_stats)
        bowl_ctx = _bowler_context(row, self.bowler_stats, tracker)

        # End-of-over next bowler recommendation
        eoo_rec = (
            tracker.recommend_next_bowler(over, phase_code)
            if ball == 6 else None
        )

        batting_team = str(row.get("batting_team", "Batting Team"))
        bowling_team = str(row.get("bowling_team", "Bowling Team"))

        decision = {
            "engine":    "ml",
            "ball":      f"{over}.{ball}",
            "phase":     state["phase"],
            "match_state": state,
            "batting_team": {
                "team":               batting_team,
                "ml_strategy":        bat_strat,
                "ml_confidence":      bat_conf,
                "batting_plan":       bat_plan,
                "batsman_context":    bat_ctx,
                "captain_instruction": _batting_captain_note(bat_strat, state, bat_ctx),
            },
            "bowling_team": {
                "team":               bowling_team,
                "ml_strategy":        bowl_strat,
                "ml_confidence":      bowl_conf,
                "bowling_plan":       bowl_plan,
                "bowler_context":     bowl_ctx,
                "captain_instruction": _bowling_captain_note(bowl_strat, state, bowl_ctx),
                "bowler_rotation": {
                    "current_bowler": bowl_ctx["name"],
                    "available_bowlers": tracker.available_bowlers(exclude_last_over=False),
                },
                **({"next_over_recommendation": eoo_rec} if eoo_rec else {}),
            },
        }
        return decision

    def decide_json(self, row, indent: int = 2) -> str:
        return json.dumps(self.decide(row), indent=indent, default=str)

    def set_innings1_score(self, score: int) -> None:
        """Convenience wrapper — call at start of innings 2."""
        self._analyzer.set_innings1_score(score)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7: RULE-BASED FALLBACK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedFallbackEngine:
    """
    Identical output schema as TacticalDecisionEngine.
    Uses the deterministic label functions from feature_extractor.py
    instead of ML models. Requires no training.

    Used automatically by SynapseLive.create() when models are absent.
    """

    def __init__(self, batsman_stats: Dict, bowler_stats: Dict):
        self.batsman_stats = batsman_stats
        self.bowler_stats  = bowler_stats
        self._trackers:    Dict[int, BowlerConstraintTracker] = {}
        self._analyzer     = MatchStateAnalyzer()
        self._cur_innings: Optional[int] = None

    def _tracker(self, innings: int) -> BowlerConstraintTracker:
        if innings not in self._trackers:
            self._trackers[innings] = BowlerConstraintTracker(self.bowler_stats)
        return self._trackers[innings]

    def decide(self, row) -> Dict:
        innings = int(row.get("innings", 1))
        over    = int(row.get("over", 1))
        ball    = int(row.get("ball", 1))

        if innings != self._cur_innings:
            self._cur_innings = innings
            if innings == 2:
                score1 = row.get("innings1_score")
                if score1 is not None:
                    self._analyzer.set_innings1_score(int(score1))

        tracker = self._tracker(innings)
        tracker.register_ball(str(row.get("bowler", "Unknown")), ball)

        state      = self._analyzer.analyze(row)
        phase_code = state["phase_code"]

        bowl_strat = derive_bowling_strategy(row)
        bat_strat  = derive_batting_strategy(row)

        bowl_plan = dict(BOWLING_PLAN_MAP.get(bowl_strat, BOWLING_PLAN_MAP["Middle Overs Build"]))
        bat_plan  = dict(BATTING_PLAN_MAP.get(bat_strat,  BATTING_PLAN_MAP["Accumulate & Rotate"]))

        bat_ctx  = _batsman_context(row, self.batsman_stats)
        bowl_ctx = _bowler_context(row, self.bowler_stats, tracker)

        eoo_rec = (
            tracker.recommend_next_bowler(over, phase_code)
            if ball == 6 else None
        )

        decision = {
            "engine":    "rule_based",
            "ball":      f"{over}.{ball}",
            "phase":     state["phase"],
            "match_state": state,
            "batting_team": {
                "team":               str(row.get("batting_team", "Batting Team")),
                "ml_strategy":        bat_strat,
                "ml_confidence":      None,
                "batting_plan":       bat_plan,
                "batsman_context":    bat_ctx,
                "captain_instruction": _batting_captain_note(bat_strat, state, bat_ctx),
            },
            "bowling_team": {
                "team":               str(row.get("bowling_team", "Bowling Team")),
                "ml_strategy":        bowl_strat,
                "ml_confidence":      None,
                "bowling_plan":       bowl_plan,
                "bowler_context":     bowl_ctx,
                "captain_instruction": _bowling_captain_note(bowl_strat, state, bowl_ctx),
                "bowler_rotation": {
                    "current_bowler": bowl_ctx["name"],
                    "available_bowlers": tracker.available_bowlers(exclude_last_over=False),
                },
                **({"next_over_recommendation": eoo_rec} if eoo_rec else {}),
            },
        }
        return decision

    def decide_json(self, row, indent: int = 2) -> str:
        return json.dumps(self.decide(row), indent=indent, default=str)

    def set_innings1_score(self, score: int) -> None:
        self._analyzer.set_innings1_score(score)
