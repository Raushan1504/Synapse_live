# 🧠🏏 Synapse Live — AI Tactical Cricket Engine

> A real-time AI decision engine that analyzes ball-by-ball IPL data and generates optimal tactical strategies for cricket captains.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Clone & Setup](#4-clone--setup)
5. [Dataset Setup](#5-dataset-setup)
6. [Running the Project](#6-running-the-project)
7. [File-by-File Explanation](#7-file-by-file-explanation)
8. [System Architecture](#8-system-architecture)
9. [Development Phases — Execution Order](#9-development-phases--execution-order)
10. [Team Responsibilities](#10-team-responsibilities)
11. [Contribution Guide](#11-contribution-guide)
12. [Important Rules](#12-important-rules)

---

## 1. Project Overview

**Synapse Live** is an AI-powered tactical engine built for real-time cricket decision-making. It processes IPL match data and simulates live match conditions to generate actionable strategies for batting, bowling, and field placement.

### What it does

- Ingests and processes **ball-by-ball IPL data** via `clean_ipl_dataset.csv`
- Simulates **real-time match scenarios** through `simulation_code.py`
- Pulls **player performance profiles** from `player_stats.py`
- Generates tactical recommendations through `tactical_decision_engine.py`
- Trains and validates ML models via `tactical_model_trainer.py`

### Design Philosophy

- Acts as a **Virtual Intelligent Captain** — not just stats, but actionable over-by-over decisions
- Built around **real-time tactical adaptation** to changing match conditions
- Modular file structure so Model, Backend, and Frontend teams can work independently without conflicts

---

## 2. Repository Structure

```
Synapse_live/
│
├── models/                       # Saved trained model files (.pkl, .h5, etc.)
│
├── .gitignore                    # Files Git should ignore (datasets, __pycache__, etc.)
├── README.md                     # This file
│
├── clean_ipl_dataset.csv         # Preprocessed IPL match dataset (ball-by-ball features)
│
├── feature_extractor.py          # Converts raw match state into ML-ready feature vectors
├── player_stats.py               # Fetches and computes player performance statistics
├── simulation_code.py            # Simulates live match conditions ball-by-ball
├── synapse_live.py               # Main entry point — connects all modules together
├── tactical_decision_engine.py   # Translates model output into captain instructions
├── tactical_model_trainer.py     # Trains, evaluates, and saves the ML model
│
├── test_model.py                 # Validates the trained model loads and predicts correctly
└── test_strategy.py              # Full pipeline test: simulation → features → model → strategy
```

> 📁 The `models/` folder is auto-populated by `tactical_model_trainer.py`. Do **not** manually edit files inside it.

---

## 3. Prerequisites

Ensure the following are installed before setting up the project:

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 or 3.11 | Core runtime |
| pip | Latest | Package management |
| Git | Any recent | Version control |

> ⚠️ **Python 3.12+ is NOT recommended** — TensorFlow and some other ML libraries may not be fully compatible yet.

---

## 4. Clone & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Raushan1504/Synapse_live.git
cd Synapse_live
```

### Step 2 — Create a Virtual Environment (Strongly Recommended)

A virtual environment prevents dependency conflicts with other Python projects on your machine.

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt when active.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

If a `requirements.txt` does not yet exist in the repo, create one with the following:

```
numpy
pandas
scikit-learn==1.3.2
joblib
xgboost
lightgbm
tensorflow
fastapi
uvicorn
streamlit
matplotlib
seaborn
```

> ⚠️ `scikit-learn` is pinned to `1.3.2` for model compatibility. Do **not** upgrade it without team discussion.

---

## 5. Dataset Setup

### What IS already in the repo

`clean_ipl_dataset.csv` — a preprocessed, ML-ready version of IPL ball-by-ball data. This is your primary input for model training and should be present in the repo root.

### If you need the raw IPL JSON data (for re-engineering features from scratch)

Download the full dataset from Google Drive:

```
👉 [PASTE YOUR GOOGLE DRIVE LINK HERE]
```

After downloading, place the extracted folder as follows:

```
Synapse_live/
    └── ipl_json/
            ├── 2008/
            ├── 2009/
            ├── ...
            └── 2024/
```

> ❌ Do **not** rename the folder. Raw data parsing scripts expect the folder to be called `ipl_json`.  
> ❌ Do **not** commit this folder to Git. It is already covered by `.gitignore`.

---

## 6. Running the Project

Run these scripts in order. Do not skip steps.

### Step 1 — Train the Model

```bash
python tactical_model_trainer.py
```

This reads `clean_ipl_dataset.csv`, trains the ML model, and saves the output into `models/`. You must do this before running any other script.

### Step 2 — Validate the Model

```bash
python test_model.py
```

Confirms the saved model in `models/` loads correctly and can generate predictions from a sample input.

### Step 3 — Run the Full Strategy Pipeline

```bash
python test_strategy.py
```

Runs the complete end-to-end flow:

```
clean_ipl_dataset.csv
    → feature_extractor.py         (builds feature vector)
    → models/ (trained model)      (generates prediction)
    → tactical_decision_engine.py  (produces strategy text)
    → Console output
```

### Step 4 — Launch the Main Application

```bash
python synapse_live.py
```

This is the main entry point that wires all modules together and runs Synapse Live as a complete system.

> ✅ If all four steps complete without errors, your local setup is fully working.

---

## 7. File-by-File Explanation

Read through each file before making changes to it.

| File | What It Does | Who Owns It |
|------|-------------|------------|
| `clean_ipl_dataset.csv` | Preprocessed IPL data. Primary input for training. Do not modify unless re-cleaning raw data. | Abhinaya |
| `feature_extractor.py` | Reads current match state and outputs a structured feature vector for the model. | Abhinaya |
| `player_stats.py` | Returns batting/bowling stats (averages, SR, economy) for specific players. Called during feature extraction. | Abhinaya |
| `simulation_code.py` | Simulates a match ball-by-ball by generating synthetic match states. Feeds into the feature extractor. | Abhinaya |
| `tactical_model_trainer.py` | Trains the ML classifier on `clean_ipl_dataset.csv` and saves the model to `models/`. | Abhinaya |
| `tactical_decision_engine.py` | Takes model output (probabilities/classes) and converts them into readable captain instructions. | Abhinaya / Anjali |
| `synapse_live.py` | Main entry point. Connects simulation → feature extraction → model prediction → strategy output. | All Teams |
| `test_model.py` | Quick sanity check. Run this after every model change. | All Teams |
| `test_strategy.py` | Full pipeline smoke test. Must pass before any Pull Request is opened. | All Teams |
| `models/` | Auto-generated folder. Stores `.pkl` / `.h5` model files. Never edit manually. | Generated by Abhinaya |

---

## 8. System Architecture

```
                                        ┌─────────────────────────┐
                                        │   Player Stats Engine   │
                                        │     (player_stats.py)   │
                                        └────────────┬────────────┘
                                                     │ injects player
                                                     │ profiles
                                                     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  IPL Ball-by-   │    │  Data Cleaning  │    │ Match Simulation│    │Feature Extraction│
│  Ball Dataset   │───▶│       &         │───▶│    Engine       │───▶│    Module        │
│  (JSON / CSV)   │    │ Preprocessing   │    │(simulation_code │    │(feature_extractor│
└─────────────────┘    └─────────────────┘    │    .py)         │    │    .py)          │
                                              └─────────────────┘    └────────┬─────────┘
                                                                              │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │    ML Models    │
                                                                    │ Batting Strategy│
                                                                    │ Bowling Strategy│
                                                                    └────────┬────────┘
                                                                             │
                                                                             ▼
                                                                    ┌─────────────────────────────┐
                                                                    │   Tactical Decision Engine  │
                                                                    │  • Match State Analyzer     │
                                                                    │  • Bowler Constraint Tracker│
                                                                    │  • Strategy Generator       │
                                                                    └────────────┬────────────────┘
                                                                                 │
                                                                                 ▼
                                                                    ┌────────────────────────┐
                                                                    │  Synapse Live Interface │
                                                                    │    (synapse_live.py)    │
                                                                    └────────────┬────────────┘
                                                                                 │
                                                                                 ▼
                                                                    ┌────────────────────────┐
                                                                    │  Tactical Strategy     │
                                                                    │  Output (Text / JSON)  │
                                                                    └────────────────────────┘
```

### Component Breakdown

| Stage | File | What It Does |
|-------|------|-------------|
| IPL Dataset | `clean_ipl_dataset.csv` / `ipl_json/` | Raw ball-by-ball input data |
| Data Cleaning & Preprocessing | *(preprocessing logic)* | Cleans, normalizes, and structures raw IPL data |
| Match Simulation Engine | `simulation_code.py` | Recreates live match conditions ball-by-ball |
| Player Stats Engine | `player_stats.py` | Injects real player profiles (form, SR, economy) into the feature pipeline |
| Feature Extraction Module | `feature_extractor.py` | Combines match state + player stats into an ML-ready feature vector |
| ML Models | `tactical_model_trainer.py` + `models/` | Predicts optimal batting and bowling strategies |
| Tactical Decision Engine | `tactical_decision_engine.py` | Runs Match State Analyzer, Bowler Constraint Tracker, and Strategy Generator |
| Synapse Live Interface | `synapse_live.py` | Main entry point — orchestrates the full pipeline end-to-end |
| Output | Text / JSON | Human-readable captain instructions, also served via FastAPI for the Streamlit UI |

---

## 9. Development Phases — Execution Order

> 🚨 **CRITICAL: Follow this phase order strictly. Teams must not skip ahead.**  
> Backend depends on a stable model. Frontend depends on a working backend.

---

### 🔹 Phase 1 — Orientation (ALL TEAMS — Do This First)

**Goal:** Every contributor understands the full system before writing new code.

1. Clone the repo and complete setup (Section 4).
2. Confirm `clean_ipl_dataset.csv` is present in the repo root.
3. Run the full sequence:
   ```bash
   python tactical_model_trainer.py
   python test_model.py
   python test_strategy.py
   python synapse_live.py
   ```
4. Read files in this order:
   - `feature_extractor.py` — what features are being used?
   - `player_stats.py` — what player data is available?
   - `tactical_decision_engine.py` — how are strategies formed?
   - `synapse_live.py` — how does everything connect?

**Expected result:** You can explain what happens between step 1 (`clean_ipl_dataset.csv`) and final output (printed strategy) without looking at the code.

---

### 🔹 Phase 2 — Model Improvement (ABHINAYA — Complete Before Phase 3)

**Goal:** Improve prediction accuracy and enrich the feature set before the backend is built on top.

> ⚠️ Anjali will load whatever is saved in `models/`. Finalize the model and document its input/output schema before Phase 3 begins.

**2a. Feature Engineering** — Extend `feature_extractor.py`:

| New Feature | Description |
|-------------|-------------|
| `bowler_type` | Pace vs Spin |
| `bowling_style` | Right-arm fast, Left-arm orthodox, Leg-break, etc. |
| `batsman_role` | Opener, Middle-order, Finisher |
| `venue` | Stadium name |
| `pressure_index` | Custom metric combining run-rate deficit, wickets in hand, overs remaining |

**2b. Improve `player_stats.py`:**
- Add recent form (last 5 matches)
- Head-to-head stats (specific batsman vs bowler)
- Venue-specific performance data

**2c. Add New Models to `tactical_model_trainer.py`:**
- **XGBoost** — strong baseline for tabular cricket data
- **Logistic Regression** — for interpretability and fast inference
- *(Advanced)* **LSTM** — for sequential, over-by-over pattern modelling using `simulation_code.py`

**2d. Evaluate and Document Results** — Create `models/RESULTS.md`:

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|---------------|
| Baseline | ? | ? | ? |
| XGBoost | ? | ? | ? |
| Logistic Regression | ? | ? | ? |

**2e. No Data Leakage Rule**  
Only use features that are known **at the exact moment a ball is bowled**. Never use final match result, total score, or any future-state variable as an input.

**Deliverable Checklist:**
- [ ] `feature_extractor.py` updated with new features
- [ ] `player_stats.py` updated with richer stats
- [ ] `tactical_model_trainer.py` updated with new models
- [ ] New model files saved in `models/`
- [ ] `models/RESULTS.md` created with comparison table
- [ ] Input/output feature schema documented and shared with Anjali

---

### 🔹 Phase 3 — Backend API (ANJALI — Start After Phase 2)

**Goal:** Wrap `tactical_decision_engine.py` in a FastAPI server so the frontend can query strategies over HTTP.

> ⚠️ Get the finalized feature schema from Abhinaya before writing any endpoint code.

**3a. Create `api/main.py`:**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

model = joblib.load("models/your_model.pkl")  # load once at startup, not per request
```

**3b. Implement Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_strategy` | POST | Accepts match state, returns batting + bowling strategy |
| `/simulate_match` | POST | Runs `simulation_code.py` and returns over-by-over output |
| `/player_stats` | GET | Returns stats for a given player name |
| `/health` | GET | Returns `{"status": "ok"}` — used to verify server is alive |

**3c. JSON Contract** — Define and share with Ved:

```json
// POST /predict_strategy — Input
{
  "batting_team": "CSK",
  "bowling_team": "MI",
  "over": 14,
  "ball": 3,
  "score": 112,
  "wickets": 3,
  "venue": "Wankhede Stadium",
  "target": null
}

// Response
{
  "batting_strategy": "Push for boundaries. SR target: 160+",
  "bowling_strategy": "Yorker-heavy. Bring back your main pacer.",
  "confidence_score": 0.84
}
```

**3d. Run locally:**

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for auto-generated interactive API docs.

**Deliverable Checklist:**
- [ ] `api/main.py` with all endpoints implemented
- [ ] Model loads at startup from `models/`
- [ ] All endpoints tested with sample inputs
- [ ] JSON schema documented and shared with Ved
- [ ] `requirements.txt` updated if new packages added

---

### 🔹 Phase 4 — Frontend UI (VED — Start After Phase 3)

**Goal:** Build a Streamlit captain dashboard that calls the backend and displays tactical recommendations.

> ⚠️ Do not build API calls until Anjali's server is running and the JSON schema is confirmed.

**4a. Create `ui/app.py` with:**

- Batting Team + Bowling Team dropdowns (all 10 IPL franchises)
- Inputs: Over (1–20), Ball (1–6), Score, Wickets (0–10), Target (optional)
- Venue dropdown (Wankhede, Chepauk, Eden Gardens, Narendra Modi, etc.)
- **"Get Strategy"** button

**4b. API Call:**

```python
import requests, streamlit as st

if st.button("Get Strategy"):
    response = requests.post(
        "http://127.0.0.1:8000/predict_strategy",
        json={"batting_team": batting_team, "bowling_team": bowling_team,
              "over": over, "ball": ball, "score": score,
              "wickets": wickets, "venue": venue, "target": target or None}
    )
    if response.status_code == 200:
        data = response.json()
        st.success(data["batting_strategy"])   # primary styling — your team
        st.info(data["bowling_strategy"])      # secondary styling — opponent
        st.metric("Confidence", f"{data['confidence_score']*100:.1f}%")
    else:
        st.error("Backend unreachable. Run: uvicorn api.main:app --reload")
```

**4c. Run the UI:**

```bash
streamlit run ui/app.py
```

**Deliverable Checklist:**
- [ ] `ui/app.py` runs without errors
- [ ] All input fields functional
- [ ] API call works and strategy displays correctly
- [ ] Team color styling (primary vs secondary) implemented
- [ ] Error shown when backend is unreachable

---

### 🔹 Phase 5 — Integration Testing (ALL TEAMS)

1. Abhinaya confirms `models/` has the final model
2. Anjali runs: `uvicorn api.main:app --reload`
3. Ved runs: `streamlit run ui/app.py`
4. Test at least 3 different match scenarios together

**Common blockers and fixes:**

| Issue | Fix |
|-------|-----|
| Frontend sends wrong field names | Cross-check with Anjali's JSON schema doc |
| Model file not found | Confirm path in `api/main.py` matches actual file in `models/` |
| CORS error | Add `CORSMiddleware` in `api/main.py` (see Phase 3) |
| Strategy output garbled | Abhinaya and Anjali debug `tactical_decision_engine.py` output format together |

---

### 🔹 Phase 6 — Final Polish (ALL TEAMS)

- Add plain-English reasoning alongside each strategy (e.g., *"Why: High pressure index. Opponent has 2 death-over specialists remaining."*)
- Improve Streamlit layout using `st.columns()` and `st.metric()`
- Run `test_model.py` and `test_strategy.py` one final time on the merged codebase
- Update this README with the final Google Drive dataset link
- Optionally record a short demo video of the full pipeline

---

## 10. Team Responsibilities

### 🧠 Abhinaya — Model & Data

**Owns:** `feature_extractor.py`, `player_stats.py`, `simulation_code.py`, `tactical_model_trainer.py`, `tactical_decision_engine.py`, `clean_ipl_dataset.csv`, `models/`

**Claude/AI Prompt:**
```
You are an expert ML engineer improving a cricket tactics prediction system.

Files:
- feature_extractor.py: builds feature vectors from match state
- player_stats.py: returns player performance stats
- tactical_model_trainer.py: trains and saves the model to models/
- tactical_decision_engine.py: converts model output to captain instructions
- clean_ipl_dataset.csv: the training data

Tasks:
1. Add to feature_extractor.py: bowler_type, bowling_style, batsman_role,
   venue, pressure_index
2. Improve player_stats.py: add recent form, head-to-head, venue stats
3. Add XGBoost and Logistic Regression to tactical_model_trainer.py
4. Evaluate models: accuracy, F1-score, inference time
5. (Advanced) Add LSTM in simulation_code.py for over-sequence modelling

RULE: No data leakage. Features must only use info available at ball delivery time.
```

---

### ⚙️ Anjali — Backend API

**Owns:** `api/main.py` (new file to create)

**Claude/AI Prompt:**
```
You are building a FastAPI backend for a cricket AI system.

The trained model is in models/ (scikit-learn or XGBoost .pkl file).
The feature schema comes from feature_extractor.py.
The strategy text comes from tactical_decision_engine.py.

Requirements:
1. Load the model at startup using joblib (not per-request)
2. POST /predict_strategy: input = match state JSON,
   output = {batting_strategy, bowling_strategy, confidence_score}
3. GET /health: returns {"status": "ok"}
4. Add CORSMiddleware for Streamlit frontend
5. Return HTTP 422 with a clear message for invalid/missing input

Write clean, well-commented Python code.
```

---

### 🎨 Ved — Frontend UI

**Owns:** `ui/app.py` (new file to create)

**Claude/AI Prompt:**
```
You are building a Streamlit captain dashboard for a cricket AI system.

Backend: http://127.0.0.1:8000
Endpoint: POST /predict_strategy
Input: batting_team, bowling_team, over (1-20), ball (1-6),
       score, wickets (0-10), venue, target (optional int)
Output: batting_strategy (str), bowling_strategy (str), confidence_score (float)

Build:
1. Dropdowns for all 10 IPL teams (batting + bowling)
2. Number inputs for over, ball, score, wickets, target
3. Venue dropdown (major IPL stadiums)
4. "Get Strategy" button that calls the API
5. Show batting_strategy with bold primary styling (selected team)
6. Show bowling_strategy with secondary/muted styling (opponent)
7. Show confidence_score as st.metric percentage
8. Show clear error if backend is unreachable

Use st.columns() for layout. Keep it clean and functional.
```

---

## 11. Contribution Guide

### Step 1 — Always Pull Before Starting Work

```bash
git pull origin main
```

### Step 2 — Create a Feature Branch

```bash
git checkout -b feature/xgboost-model          # Abhinaya example
git checkout -b feature/fastapi-backend        # Anjali example
git checkout -b feature/streamlit-captain-ui   # Ved example
```

### Step 3 — Run Tests Before Committing

```bash
python test_model.py
python test_strategy.py
```

Both must pass with no errors before you commit anything.

### Step 4 — Commit with a Descriptive Message

```bash
git add .
git commit -m "Add XGBoost model with pressure_index and venue features"
```

Good commit messages describe **what** changed and optionally **why**.

### Step 5 — Push and Open a Pull Request

```bash
git push origin feature/your-branch-name
```

Go to GitHub → Pull Requests → New Pull Request → base: `main`.

> ✅ **Rule:** No one merges their own Pull Request. At least one teammate must review and approve.

---

## 12. Important Rules

| Rule | Detail |
|------|--------|
| ❌ Never commit raw datasets | `ipl_json/` must never be pushed to Git |
| ✅ `clean_ipl_dataset.csv` is the exception | Small enough to stay in the repo |
| ✅ Use Google Drive for raw data | Paste the link in Section 5 of this README |
| ✅ Pin dependency versions | Do not change `scikit-learn==1.3.2` without team consensus |
| ✅ Document model schema | Abhinaya must share feature input/output format before Phase 3 |
| ✅ Document API schema | Anjali must share JSON contract before Phase 4 |
| ✅ Run both test scripts before every PR | `test_model.py` and `test_strategy.py` must pass |
| ✅ One reviewer minimum per PR | No self-merges — always get a teammate to approve |
| ❌ Do not edit `models/` manually | Only `tactical_model_trainer.py` should write to this folder |

---

## 🏁 Final Vision

Synapse Live is a fully functional **AI cricket captain** — a system that reads the live match situation and delivers the same tactical thinking a seasoned captain brings to the field, backed by 16+ years of IPL data.

**Project Classification:**
- Machine Learning System
- Real-Time Simulation Engine
- Decision Intelligence Platform
- Sports Analytics AI

---

*Maintained by [@Raushan1504](https://github.com/Raushan1504) and contributors.*
