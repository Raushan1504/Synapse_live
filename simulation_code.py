"""
=============================================================================
 IPL BALL-BY-BALL MATCH SIMULATION ENGINE
 Author  : Cricket Simulation System
 Dataset : final_ipl_dataset.csv (ball-by-ball IPL data)
 Purpose : Simulate a real IPL match ball-by-ball with:
             - Realistic bowling attributes (length, line, variation)
             - Batting behavior (shot type, intent, footwork)
             - Tactical strategy engine
             - Rich, live-match style output
 Design  : Fully modular — every logical concern lives in its own function.
           Random choices are seeded by match context so the same match
           always produces the same "simulation layer" on top of real data.
=============================================================================
"""
from feature_extractor import extract_features, features_to_array
from player_stats import compute_player_stats
from synapse_live import SynapseLive
import pandas as pd
import random
import sys
import time


# SECTION 0 ─ CONSTANTS & ATTRIBUTE POOLS


# Bowling attribute pools
LENGTHS   = ["Yorker", "Full Length", "Good Length", "Short Ball", "Bouncer"]
LINES     = ["Off stump", "Middle stump", "Leg stump", "Wide outside off", "Body line"]
VARIATIONS = ["Seam movement", "Outswing", "Inswing", "Off-cutter",
              "Leg-cutter", "Slower ball", "Bouncer variation",
              "Carrom ball", "Googly", "Top spin"]

# Batting attribute pools
SHOT_TYPES = ["Defensive block", "Cover Drive", "On Drive", "Pull Shot",
              "Cut Shot", "Lofted Drive", "Sweep Shot", "Ramp/Scoop",
              "Reverse Sweep", "Flick off pads", "Straight Drive", "Slog Sweep"]
INTENTS    = ["Purely defensive", "Watchful & controlled", "Positive intent",
              "Full aggression"]
FOOTWORK   = ["Front foot", "Back foot", "Standing tall", "Dancing down the track"]

# ANSI colour codes for terminal output
CLR = {
    "reset"  : "\033[0m",
    "header" : "\033[1;36m",      # bold cyan
    "team"   : "\033[1;33m",      # bold yellow
    "ball"   : "\033[1;37m",      # bold white
    "bowl"   : "\033[1;34m",      # bold blue
    "bat"    : "\033[1;32m",      # bold green
    "score"  : "\033[1;35m",      # bold magenta
    "runs"   : "\033[1;31m",      # bold red
    "strategy": "\033[3;33m",     # italic yellow
    "sep"    : "\033[90m",        # dark grey
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ DATA LOADING & MATCH SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("👉 Make sure you uploaded the file in Colab")
        return None

def list_available_matches(df: pd.DataFrame, n: int = 10) -> None:
    """
    Print a summary table of available matches so the user can choose.
    """
    summary = (
        df.groupby("match_id")
          .agg(
              Year=("year",          "first"),
              Batting=("batting_team", "first"),
              Bowling=("bowling_team", "first"),
              Balls=("ball",           "count"),
          )
          .reset_index()
          .sort_values(["Year", "match_id"])  # MATCH LISTING FIX: Sort chronologically
          .head(n)
    )
    print(f"{CLR['header']}{'─'*60}")
    print(f"  AVAILABLE MATCHES (first {n} shown)")
    print(f"{'─'*60}{CLR['reset']}")
    for _, row in summary.iterrows():
        print(f"  ID {row['match_id']:>8}  |  {row['Year']}  |  "
              f"{row['Batting'][:20]:<20} vs {row['Bowling'][:20]:<20}  "
              f"|  {row['Balls']} balls")
    print(f"{CLR['sep']}{'─'*60}{CLR['reset']}\n")


def select_match(df: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Filter the dataset for a single match and sort ball-by-ball.
    Adds a running 'wickets_fallen' column used by the strategy engine.
    """
    match_df = df[df["match_id"] == match_id].copy()
    if match_df.empty:
        sys.exit(f"[ERROR] match_id {match_id} not found in dataset.")

    match_df = match_df.sort_values(["innings", "over", "ball"]).reset_index(drop=True)

    # Cumulative wickets per innings (needed for strategy logic)
    match_df["wickets_fallen"] = (
        match_df.groupby("innings")["wicket"].cumsum()
    )

    return match_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ BOWLING DETAIL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _seed_random(over: int, ball: int, innings: int, extra: int = 0) -> None:
    """Seed random with ball context so results are reproducible per ball."""
    random.seed(innings * 10_000 + over * 100 + ball + extra)


def generate_bowling_details(row: pd.Series) -> dict:
    """
    Derive realistic bowling attributes from match context.

    Decision logic:
      • High run_rate (>9) or powerplay              → Yorker / Full + wide line
      • Aggressive batsman (strike_rate >150)        → Bouncer / Short ball
      • Dot ball pressure (bowler_economy <6.5)      → Good length, stump line
      • Batsman just in (low balls faced implied)    → Full ball to induce edge
      • Default                                      → Good length, off stump
    """
    _seed_random(row["over"], row["ball"], row["innings"], extra=1)

    run_rate      = row.get("run_rate", 7.0)
    strike_rate   = row.get("strike_rate", 100.0)
    economy       = row.get("bowler_economy", 7.5)
    is_powerplay  = row.get("powerplay", 0)
    wicket        = row.get("wicket", 0)

    # ── Length determination ──────────────────────────────────────────────
    if run_rate > 10 or (is_powerplay and run_rate > 8):
        # Death/pressure → squeeze with yorkers
        length = random.choices(LENGTHS, weights=[50, 30, 15, 3, 2])[0]
    elif strike_rate > 150:
        # Very attacking batsman → test with short stuff
        length = random.choices(LENGTHS, weights=[10, 15, 20, 35, 20])[0]
    elif strike_rate > 120 and run_rate > 9:
        length = random.choices(LENGTHS, weights=[30, 25, 25, 15, 5])[0]
    elif wicket == 1:
        # New batsman → full, testing their technique
        length = random.choices(LENGTHS, weights=[20, 45, 25, 5, 5])[0]
    elif economy < 6.5:
        # Bowling well → maintain good length discipline
        length = random.choices(LENGTHS, weights=[10, 20, 55, 10, 5])[0]
    else:
        length = random.choices(LENGTHS, weights=[10, 25, 45, 15, 5])[0]

    # ── Line determination ────────────────────────────────────────────────
    if run_rate > 9:
        # Attack outside off to prevent big swings
        line = random.choices(LINES, weights=[10, 5, 5, 60, 20])[0]
    elif strike_rate > 140:
        # Bowler targets body to cramp the batsman
        line = random.choices(LINES, weights=[20, 15, 30, 15, 20])[0]
    elif is_powerplay:
        line = random.choices(LINES, weights=[40, 20, 10, 25, 5])[0]
    else:
        line = random.choices(LINES, weights=[35, 25, 15, 20, 5])[0]

    # ── Variation determination ───────────────────────────────────────────
    if run_rate > 9:
        variation = random.choices(VARIATIONS, weights=[5, 5, 5, 15, 10, 40, 10, 4, 3, 3])[0]
    elif length in ["Bouncer", "Short Ball"]:
        variation = random.choices(VARIATIONS, weights=[10, 5, 5, 10, 10, 20, 30, 0, 5, 5])[0]
    elif economy > 9:
        # Expensive → try something different
        variation = random.choices(VARIATIONS, weights=[5, 10, 10, 15, 10, 25, 5, 8, 7, 5])[0]
    else:
        variation = random.choices(VARIATIONS, weights=[25, 15, 15, 10, 10, 10, 5, 3, 4, 3])[0]

    return {"length": length, "line": line, "variation": variation}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ BATTING DETAIL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_batting_details(row: pd.Series, bowling: dict) -> dict:
    """
    Derive realistic batting attributes from ball outcome and context.

    Decision logic:
      • Boundary (4 or 6)    → aggressive shot, positive intent
      • Dot ball              → defensive/blocked
      • Wicket                → (edge / mistimed) lofted or defensive attempt
      • Run scored (1–3)      → controlled, rotating strike
    """
    _seed_random(row["over"], row["ball"], row["innings"], extra=2)

    is_boundary   = row.get("is_boundary", 0)
    is_dot        = row.get("is_dot", 0)
    wicket        = row.get("wicket", 0)
    runs          = row.get("runs", 0)
    strike_rate   = row.get("strike_rate", 100.0)
    bowl_length   = bowling["length"]
    is_six        = row.get("is_six", 0)

    # ── Shot selection ────────────────────────────────────────────────────
    if wicket:
        # Dismissed — most likely an edge, mistimed or bowled
        shot = random.choices(
            SHOT_TYPES,
            weights=[20, 15, 10, 10, 10, 15, 5, 5, 2, 5, 3, 0]
        )[0]
        intent    = "Positive intent"           # batsman was going for it
        footwork  = random.choices(FOOTWORK, weights=[30, 40, 10, 20])[0]

    elif is_boundary and is_six:
        shot = random.choices(
            SHOT_TYPES,
            weights=[0, 5, 10, 15, 5, 25, 10, 10, 5, 5, 5, 5]
        )[0]
        intent    = "Full aggression"
        footwork  = random.choices(FOOTWORK, weights=[20, 15, 5, 60])[0]

    elif is_boundary:
        # Boundary 4 — attacking but controlled
        if bowl_length in ["Short Ball", "Bouncer"]:
            shot = random.choices(SHOT_TYPES,
                                  weights=[0, 5, 30, 20, 15, 5, 5, 5, 3, 5, 2, 5])[0]
            footwork = random.choices(FOOTWORK, weights=[5, 55, 5, 35])[0]
        else:
            shot = random.choices(SHOT_TYPES,
                                  weights=[0, 25, 10, 5, 20, 20, 5, 5, 5, 5, 5, 0])[0]
            footwork = random.choices(FOOTWORK, weights=[55, 15, 10, 20])[0]
        intent = "Full aggression"

    elif is_dot:
        # Dot ball — beaten or blocked
        shot     = random.choices(SHOT_TYPES,
                                  weights=[45, 10, 5, 5, 10, 5, 5, 3, 2, 5, 5, 0])[0]
        intent   = "Purely defensive"
        footwork = random.choices(FOOTWORK, weights=[40, 40, 15, 5])[0]

    else:
        # 1, 2 or 3 runs — rotating strike
        shot     = random.choices(SHOT_TYPES,
                                  weights=[10, 20, 12, 10, 15, 8, 8, 3, 3, 7, 4, 0])[0]
        intent   = random.choices(INTENTS, weights=[5, 40, 40, 15])[0]
        footwork = random.choices(FOOTWORK, weights=[40, 30, 20, 10])[0]

    return {"shot": shot, "intent": intent, "footwork": footwork}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ TACTICAL STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Reusable strategy templates parameterised by context
_STRATEGY_TEMPLATES = {
    # key : (condition_fn, template strings list)
    "death_high_rr": (
        lambda r: r.get("run_rate", 0) > 9 and r.get("over", 0) >= 16,
        [
            "Wide yorkers outside off stump — set deep cover, long-off, and fine-leg; deny boundaries at all cost.",
            "Full-pitched deliveries on off stump to minimise bat-swing room; extra fielder at deep mid-wicket.",
            "Bowling in the block-hole with mid-off up — dare them to loft into long-on and long-off.",
        ]
    ),
    "powerplay_attack": (
        lambda r: r.get("powerplay", 0) == 1,
        [
            "Swing the new ball early — two slips and a gully in the cordon; bowl a full, swinging length.",
            "Use outswing to shape the ball away from the right-hander; gully and third slip in catching position.",
            "Bowl a tight off-stump channel to build pressure; attack with one set fielder at leg-slip.",
        ]
    ),
    "new_batsman": (
        lambda r: r.get("wicket_prev", 0) == 1,
        [
            "Attack the new batsman with full deliveries — two slips, gully; don't let them settle.",
            "New batsman in — pitch it up, swing it both ways; keep backward point and third slip.",
            "Incoming batsman: bowl straight and full; place mid-on up and trust swing or seam.",
        ]
    ),
    "aggressive_striker": (
        lambda r: r.get("strike_rate", 0) > 150,
        [
            "Bowl short into the body — keep deep square leg and fine-leg; cramp the pull shot.",
            "Alternate yorker-bouncer combo; position two men on the leg-side boundary.",
            "Bowl a wide half-tracker outside off — field deep point and deep cover to cut off the cut shot.",
        ]
    ),
    "economy_pressure": (
        lambda r: r.get("bowler_economy", 99) < 6 and r.get("run_rate", 99) < 7,
        [
            "Bowler in control — maintain off-stump channel, keep mid-off up; force the batsman to take risk.",
            "Build dot-ball pressure; bowl straight with mid-on saving one; wait for the defensive error.",
            "Good economy — keep three men inside the circle on the off-side; choke the singles.",
        ]
    ),
    "expensive_bowler": (
        lambda r: r.get("bowler_economy", 0) > 10,
        [
            "Bowler leaking runs — switch to variation (slower-ball cutters); deep fielders at both boundaries.",
            "Expensive spell: try cross-seam deliveries; add an extra man at deep mid-wicket.",
            "Change of plan: bowl wide of off stump with point, cover, and deep cover in the ring.",
        ]
    ),
    "middle_overs_spin": (
        lambda r: 7 <= r.get("over", 0) <= 15,
        [
            "Middle overs — toss it up on off stump; mid-on and mid-off saving one; invite the lofted shot.",
            "Spin in the middle: vary pace and flight; set a sweeper cover and a long-on for the big shot.",
            "Flight it on a good-length off-stump channel; backward square leg and mid-wicket inside the ring.",
        ]
    ),
    "default": (
        lambda r: True,
        [
            "Maintain a tight off-stump line; keep mid-off up and mid-on saving one; build over-by-over pressure.",
            "Bowl full and straight, rotate through variations; trust the field to apply boundary restriction.",
            "Standard field — two saving ones on the off-side, mid-wicket; probe the corridor outside off stump.",
        ]
    ),
}


def generate_strategy(row: pd.Series) -> str:
    """
    Select the most context-relevant strategy from the template bank.
    Tries each strategy in priority order; falls back to 'default'.
    """
    _seed_random(row["over"], row["ball"], row["innings"], extra=3)

    priority = [
        "death_high_rr",
        "powerplay_attack",
        "new_batsman",
        "aggressive_striker",
        "expensive_bowler",
        "economy_pressure",
        "middle_overs_spin",
        "default",
    ]

    for key in priority:
        condition_fn, templates = _STRATEGY_TEMPLATES[key]
        try:
            if condition_fn(row):
                return random.choice(templates)
        except Exception:
            continue

    return random.choice(_STRATEGY_TEMPLATES["default"][1])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def _runs_display(runs: int, is_wicket: bool, is_six: bool, is_four: bool) -> str:
    """Return a coloured, annotated runs string."""
    if is_wicket:
        return f"{CLR['runs']}WICKET! ({runs} runs){CLR['reset']}"
    elif is_six:
        return f"{CLR['runs']}SIX! ⭐ ({runs} runs){CLR['reset']}"
    elif is_four:
        return f"{CLR['runs']}FOUR! ({runs} runs){CLR['reset']}"
    elif runs == 0:
        return f"{CLR['sep']}· Dot ball{CLR['reset']}"
    else:
        return f"{CLR['ball']}{runs} run{'s' if runs > 1 else ''}{CLR['reset']}"


def print_ball(row: pd.Series, bowling: dict, batting: dict, strategy: str) -> None:
    """
    Pretty-print a single ball's full simulation output in a live-match style.
    """
    sep = f"{CLR['sep']}{'─' * 66}{CLR['reset']}"

    over_label   = f"{int(row['over'])}.{int(row['ball'])}"
    batting_team = row["batting_team"]
    bowling_team = row["bowling_team"]
    
    # OUTPUT CONSISTENCY & ROBUSTNESS FIX
    score_str    = f"{int(row.get('team_score', 0))}/{int(row.get('wickets_fallen', 0))}"
    
    runs_str     = _runs_display(
        int(row["runs"]),
        bool(row["wicket"]),
        bool(row.get("is_six", 0)),
        bool(row.get("is_four", 0)),
    )

    print(sep)
    print(
        f"  {CLR['ball']}Over {over_label:<6}{CLR['reset']}  "
        f"{CLR['team']}{batting_team}{CLR['reset']} vs "
        f"{CLR['team']}{bowling_team}{CLR['reset']}   "
        f"{CLR['score']}Score: {score_str}{CLR['reset']}"
        f"   Run Rate: {row.get('run_rate', 0.0):.2f}"
    )
    print(  
    f"  {CLR['bowl']}🎳 Bowler : {row['bowler']:<22}{CLR['reset']} "
    f"(Economy: {row.get('bowler_economy', 7.5):.2f}) "
    f"Length: {bowling['length']:<14} "
    f"Line: {bowling['line']:<20} "
    f"Variation: {bowling['variation']}"
)
    
    print(  
    f"  {CLR['bat']}🏏 Batsman: {row['batsman']:<22}{CLR['reset']} "
    f"(SR: {row.get('strike_rate', 100.0):.1f}) "
    f"Shot: {batting['shot']:<20} "
    f"Intent: {batting['intent']:<24} "
    f"Footwork: {batting['footwork']}"
)
    
      
    
    
    print(f"  Result → {runs_str}")
    print(
        f"  {CLR['strategy']}⚙  Strategy: {strategy}{CLR['reset']}"
    )


def print_innings_header(innings: int, batting_team: str, bowling_team: str) -> None:
    """Print a decorative header at the start of each innings."""
    print(f"\n{CLR['header']}{'═' * 66}")
    print(f"  INNINGS {innings}:  {batting_team.upper()}  batting  vs  {bowling_team.upper()}")
    print(f"{'═' * 66}{CLR['reset']}\n")


def print_match_header(match_id: int, df_match: pd.DataFrame) -> None:
    """Print a one-time header for the selected match."""
    inn1 = df_match[df_match["innings"] == 1].iloc[0]
    year = df_match["year"].iloc[0]
    total_balls = len(df_match)

    print(f"\n{CLR['header']}{'╔' + '═'*64 + '╗'}")
    print(f"  🏆  IPL {year} — Match ID: {match_id}")
    print(f"  {inn1['batting_team']}  vs  {inn1['bowling_team']}")
    print(f"  Total deliveries in dataset: {total_balls}")
    print(f"{'╚' + '═'*64 + '╝'}{CLR['reset']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def simulate_match(df_match, match_id, batsman_stats, bowler_stats, synapse, delay=0.0, max_balls=None):
    """
         Iterate over every ball in the match and produce a live simulation.

        Parameters
        ----------
         df_match  : Filtered, sorted DataFrame for one match (from select_match).
         match_id  : The match ID (for display purposes).
         delay     : Seconds to pause between balls (0 = instant, 0.3 = live feel).
          max_balls : Cap on balls to simulate (None = all balls).
      """
     
    print_match_header(match_id, df_match)

    current_innings = None
    prev_wicket     = 0           # track previous ball's wicket flag for "new batsman" strategy

    for idx, row in df_match.iterrows():

        # ── Innings header ────────────────────────────────────────────────
        if row["innings"] != current_innings:
            current_innings = row["innings"]
            print_innings_header(
                current_innings,
                row["batting_team"],
                row["bowling_team"],
            )
            prev_wicket = 0       # FLOW FIX: Reset prev_wicket at start of each innings

        # Inject previous wicket context for strategy engine
        row_ext = row.copy()
        row_ext["wicket_prev"] = prev_wicket

        # ?? SAFE INJECTION (DO NOT TOUCH ANYTHING ELSE)
        batsman = row.get("batsman")
        bowler  = row.get("bowler")

        row_ext["strike_rate"] = batsman_stats.get(batsman, 120.0)
        row_ext["bowler_economy"] = bowler_stats.get(bowler, 7.5)
        # ── Core simulation steps ─────────────────────────────────────────
        bowling  = generate_bowling_details(row_ext)
        batting  = generate_batting_details(row_ext, bowling)
        strategy = synapse.generate_strategy(row_ext)
        tactical_json = synapse.get_tactical_json(row_ext)
        
        features = extract_features(row_ext)
        X_input = [features_to_array(features)]
        # ── Output ────────────────────────────────────────────────────────
        print_ball(row_ext, bowling, batting, strategy)

        # OVER SUMMARY FIX: Print after the 6th ball of the over
        if int(row.get("ball", 0)) == 6:
            score = int(row.get("team_score", 0))
            wickets = int(row.get("wickets_fallen", 0))
            print(f"  {CLR['header']}{'-'*20} End of Over {int(row['over'])}: Score {score}/{wickets} {'-'*20}{CLR['reset']}\n")

        # Update state
        prev_wicket = int(row.get("wicket", 0))

        # Optional live-match pacing
        if delay > 0:
            time.sleep(delay)

        # Optional early stop for demos
        if max_balls is not None:
            max_balls -= 1
            if max_balls <= 0:
                print(f"\n{CLR['sep']}[Simulation capped — increase max_balls to see more.]{CLR['reset']}\n")
                return

    # ── Match summary ─────────────────────────────────────────────────────
    print(f"\n{CLR['header']}{'═'*66}")
    print("  ✅  MATCH SIMULATION COMPLETE")

    for inn in sorted(df_match["innings"].unique()):
        inn_df  = df_match[df_match["innings"] == inn]
        bt      = inn_df["batting_team"].iloc[0]
        final   = inn_df.iloc[-1]
        wickets = int(final["wickets_fallen"])
        runs    = int(final["team_score"])
        overs   = f"{int(final['over'])}.{int(final['ball'])}"
        print(f"  Innings {inn}: {bt:<30} {runs}/{wickets}  ({overs} overs)")

    print(f"{'═'*66}{CLR['reset']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main entry point.
    Adjust MATCH_ID and MAX_BALLS_TO_SHOW to taste.

    ML-INTEGRATION HOOKS (future):
      • generate_bowling_details  → replace weighted random with an ML model
                                    trained on bowler × batsman matchups.
      • generate_batting_details  → replace with an RL agent or shot-classifier.
      • generate_strategy         → replace with a tactic recommendation model.
    """

    # ── Config ────────────────────────────────────────────────────────────
    DATA_PATH       = "clean_ipl_dataset.csv"
    MATCH_ID        = 1082591          # ← Change to any match_id in the dataset
    DELAY_SECONDS   =0.5         # 0.0 = instant;  0.3 = live fee
    MAX_BALLS_TO_SHOW = 240       # None = simulate entire match; int = demo cap
   

    # ── Run ───────────────────────────────────────────────────────────────
    df = load_dataset(DATA_PATH)

    batsman_stats, bowler_stats = compute_player_stats(df)
    synapse = SynapseLive.create(batsman_stats, bowler_stats)

     #list_available_matches(df, n=10)

    df_match  = select_match(df, MATCH_ID)
    #simulate_match(df_match, MATCH_ID, batsman_stats, bowler_stats, delay=DELAY_SECONDS, max_balls=MAX_BALLS_TO_SHOW)
    simulate_match(df_match,  MATCH_ID, batsman_stats, bowler_stats, synapse, delay= DELAY_SECONDS, max_balls=MAX_BALLS_TO_SHOW)


if __name__ == "__main__":
    main()