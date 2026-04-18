# 🧠🏏 Synapse Live — AI Tactical Cricket Engine

> A real-time AI decision engine that analyzes ball-by-ball IPL data and generates optimal tactical strategies for cricket captains.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Prerequisites](#3-prerequisites)
4. [Clone & Setup](#4-clone--setup)
5. [Dataset Setup](#5-dataset-setup)
6. [Running the Project](#6-running-the-project)
7. [System Architecture](#7-system-architecture)
8. [Development Phases — Execution Order](#8-development-phases--execution-order)
9. [Team Responsibilities](#9-team-responsibilities)
10. [Contribution Guide](#10-contribution-guide)
11. [Important Rules](#11-important-rules)

---

## 1. Project Overview

**Synapse Live** is an AI-powered tactical engine built for real-time cricket decision-making. It processes historical IPL data and simulates live match conditions to generate actionable strategies for batting, bowling, and field placement.

### What it does

- Ingests and processes **ball-by-ball IPL data (2008–2024)**
- Simulates **real-time match scenarios** based on current match state
- Generates three types of output:
  - **Batting Strategy** — shot selection, aggression level, target run-rate pacing
  - **Bowling Strategy** — bowler rotation, line and length, variation recommendations
  - **Captain Instructions** — over-by-over tactical adjustments

### Design Philosophy

- Acts as a **Virtual Intelligent Captain** — not just stats, but actionable decisions
- Built around **real-time tactical adaptation** to changing match conditions
- Modular design so Model, Backend, and Frontend teams can work independently

---

## 2. Project Structure

```
Synapse_Live/
│
├── ipl_json/                   # Ball-by-ball IPL JSON data (NOT in repo — see Section 5)
│
├── data/                       # Processed/cleaned data outputs
│
├── models/                     # Saved trained model files (.pkl, .h5, etc.)
│
├── features/                   # Feature engineering scripts
│   └── feature_extractor.py
│
├── engine/                     # Core tactical decision logic
│   └── strategy_engine.py
│
├── api/                        # FastAPI backend (Team 2)
│   └── main.py
│
├── ui/                         # Streamlit frontend (Team 3)
│   └── app.py
│
├── test_model.py               # Quick model validation script
├── test_strategy.py            # Full pipeline simulation test
├── requirements.txt
└── README.md
```

---

## 3. Prerequisites

Before setting up the project, ensure the following are installed on your machine:

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 or 3.11 | Core runtime |
| pip | Latest | Package management |
| Git | Any recent | Version control |

> ⚠️ **Python 3.12+ is NOT recommended** — some ML libraries (e.g., TensorFlow) may not be compatible yet.

---

## 4. Clone & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Synapse_Live.git
cd Synapse_Live
```

### Step 2 — Create a Virtual Environment (Recommended)

Using a virtual environment avoids conflicts with other Python projects on your machine.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt when it's active.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries (in `requirements.txt`)

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

The IPL dataset is **not included in this repository** due to its size.

### Step 1 — Download the Dataset

Download the ball-by-ball IPL JSON dataset from Google Drive:

```
👉 [PASTE YOUR GOOGLE DRIVE LINK HERE]
```

### Step 2 — Place the Dataset Correctly

After downloading, unzip/extract and place the folder exactly as shown:

```
Synapse_Live/
    └── ipl_json/
            ├── 2008/
            ├── 2009/
            ├── ...
            └── 2024/
```

> ❌ Do **not** rename the folder. The data loading scripts expect the folder to be named `ipl_json`.

---

## 6. Running the Project

Once setup and dataset are complete, verify everything is working with these two scripts:

### Test 1 — Model Validation

```bash
python test_model.py
```

This checks that:
- The trained model loads correctly
- Predictions can be generated from sample input
- Output format is as expected

### Test 2 — Full Strategy Pipeline

```bash
python test_strategy.py
```

This runs a complete simulation:

```
Match Data → Feature Extraction → ML Model → Decision Engine → Strategy Output
```

You should see a printed tactical output in your terminal. If both scripts run without errors, your setup is complete.

---

## 7. System Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  IPL Dataset │────▶│ Feature Extractor│────▶│   ML Models    │
│  (ipl_json/) │     │ (ball-by-ball    │     │ (XGBoost,      │
│              │     │  feature eng.)   │     │  LR, LSTM)     │
└──────────────┘     └──────────────────┘     └───────┬────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Streamlit   │◀────│   FastAPI        │◀────│ Tactical       │
│  Frontend UI │     │   Backend API    │     │ Decision       │
│  (Captain    │     │   (/predict,     │     │ Engine         │
│   Mode)      │     │    /simulate)    │     └────────────────┘
└──────────────┘     └──────────────────┘
```

### Component Roles

| Component | File | Responsibility |
|-----------|------|----------------|
| Feature Extractor | `features/feature_extractor.py` | Converts raw JSON match data into ML-ready features |
| ML Models | `models/` | Learns patterns and predicts strategic outcomes |
| Decision Engine | `engine/strategy_engine.py` | Translates model output into human-readable captain instructions |
| Backend API | `api/main.py` | Serves predictions over HTTP endpoints |
| Frontend UI | `ui/app.py` | Captain-facing dashboard for live match input and strategy display |

---

## 8. Development Phases — Execution Order

> 🚨 **CRITICAL: Follow this phase order strictly. Teams should NOT skip ahead.**  
> Dependencies exist between phases. Backend can't work without a stable model. Frontend can't work without a working backend.

---

### 🔹 Phase 1 — Orientation (ALL TEAMS — Do First)

**Goal:** Everyone understands how the system works end-to-end before writing code.

**Steps:**

1. Clone the repo and complete the setup in Section 4.
2. Download and place the dataset per Section 5.
3. Run both test scripts:
   ```bash
   python test_model.py
   python test_strategy.py
   ```
4. Read through these files in order:
   - `features/feature_extractor.py` — understand what features are generated
   - `engine/strategy_engine.py` — understand how decisions are made
   - `test_strategy.py` — trace the full data flow

**Expected Output:** You should understand the flow:
```
Raw Match JSON → Features → Model Prediction → Strategy Text
```

---

### 🔹 Phase 2 — Model Improvement (Abhinaya — Do Before Phase 3)

**Goal:** Improve the predictive accuracy of the ML model before the backend is built on top of it.

> ⚠️ The backend (Phase 3) will load whatever model is finalized here. Make sure the model interface (input/output format) is stable and documented before handing off.

**Tasks:**

**2a. Feature Engineering** — Add the following new features to `feature_extractor.py`:

| Feature | Description |
|---------|-------------|
| `bowler_type` | Pace vs Spin |
| `bowling_style` | Right-arm, Left-arm, Off-break, etc. |
| `batsman_role` | Opener, Middle-order, Finisher |
| `venue` | Stadium name/location |
| `pressure_index` | Custom metric combining run-rate deficit, wickets remaining, and overs left |

**2b. Add New Models** — Implement and compare:

- **XGBoost** — strong baseline for tabular cricket data
- **Logistic Regression** — for interpretability and fast inference
- *(Advanced)* **LSTM / Transformer** — for sequential, over-by-over pattern learning

**2c. Model Evaluation** — Compare models using:

- Accuracy
- F1-Score (weighted)
- Inference time per prediction

**2d. Avoid Data Leakage** — Do not use future match data (e.g., final score) as an input feature. Features must only reflect what is known at the point of the ball being bowled.

**Deliverable Checklist:**
- [ ] Updated `feature_extractor.py` with new features
- [ ] New model training scripts in `models/`
- [ ] Saved model files (`.pkl` or `.h5`)
- [ ] Comparison table of model metrics documented in a `models/RESULTS.md`
- [ ] Confirmed input/output schema of the final model (share with Team 2)

---

### 🔹 Phase 3 — Backend API (Anjali — Start After Phase 2)

**Goal:** Build a FastAPI server that loads the trained model and serves predictions over HTTP.

> ⚠️ Do **not** start this phase until Team 1 has finalized and saved the model. Coordinate with Team 1 to get the exact input feature format.

**Tasks:**

**3a. Set Up FastAPI App** — Create `api/main.py` with:

```python
from fastapi import FastAPI
app = FastAPI()
```

**3b. Load the Model on Startup** — Load the trained model once at startup (not per-request) for performance.

**3c. Build Endpoints:**

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/predict_strategy` | POST | JSON with current match state | Batting/bowling strategy recommendation |
| `/simulate_match` | POST | JSON with full match config | Over-by-over simulation output |
| `/health` | GET | None | Server status check |

**3d. Input/Output Contract** — Define and document the exact JSON schema for each endpoint. Share this schema with Team 3 so the frontend knows what to send and receive.

Example input schema for `/predict_strategy`:
```json
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
```

**3e. Run the API locally:**

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to see auto-generated API documentation.

**Deliverable Checklist:**
- [ ] Working FastAPI app in `api/main.py`
- [ ] All 3 endpoints implemented and tested
- [ ] Input/output JSON schema documented
- [ ] `requirements.txt` updated if new packages were added
- [ ] Shared endpoint schema with Team 3

---

### 🔹 Phase 4 — Frontend UI (Ved — Start After Phase 3)

**Goal:** Build an interactive Streamlit dashboard for captains to input match state and view tactical recommendations.

> ⚠️ Do **not** start building API calls until Team 2 has the endpoints running and the JSON schema is confirmed.

**Tasks:**

**4a. Match Input Panel:**

- Dropdown to select the current match (or manually input match state)
- Dropdowns for `Batting Team` and `Bowling Team` (all IPL franchises: CSK, MI, RCB, KKR, etc.)
- Input fields for: current over, ball number, score, wickets fallen, target (if chasing)
- Venue selection dropdown

**4b. Captain Mode Strategy Display:**

- Clearly display batting and bowling strategies after receiving API response
- Visual emphasis:
  - **Selected Team (Your Team)** → Primary color / bold styling
  - **Opponent Team** → Secondary/muted styling

**4c. API Integration:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict_strategy",
    json=match_state_dict
)
strategy = response.json()
```

**4d. Error Handling:**

- Show a user-friendly message if the backend is unreachable
- Validate that all required fields are filled before sending the request

**Deliverable Checklist:**
- [ ] Streamlit app runs with `streamlit run ui/app.py`
- [ ] All match input fields working
- [ ] API call integrated and strategy displayed
- [ ] Team styling (primary vs secondary) implemented
- [ ] Error states handled gracefully

---

### 🔹 Phase 5 — Integration Testing (ALL)

**Goal:** Connect all three components and verify the full pipeline works end-to-end.

**Steps:**

1. Team 2 runs the backend: `uvicorn api.main:app --reload`
2. Team 3 runs the frontend: `streamlit run ui/app.py`
3. All teams test the full flow together:
   - Enter a match state in the UI
   - Confirm the request hits the backend
   - Confirm the strategy response is displayed correctly

**Common Issues to Check:**
- Frontend sending wrong JSON field names → Refer to Team 2's schema doc
- Model returning unexpected output format → Team 1 and Team 2 to debug together
- CORS errors when frontend calls backend → Add CORS middleware in FastAPI

**FastAPI CORS fix:**
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

---

### 🔹 Phase 6 — Final Polish (ALL)

**Goal:** Prepare the project for demo and final submission.

- Improve UI aesthetics and layout in Streamlit
- Add plain-language explanations alongside each strategy recommendation (e.g., *"Why this strategy? The run-rate pressure index is high and the opposition has 2 death-over specialists left."*)
- Test with at least 3 different historical match scenarios
- Record a short demo walkthrough video (optional but recommended)
- Update `README.md` with the final Google Drive dataset link

---

## 9. Team Responsibilities

### 🧠 Abhinaya — Model Improvement

**Focus:** Feature Engineering & ML Model Development

**Key Files:** `features/feature_extractor.py`, `models/`, `test_model.py`

**Suggested Prompt for Claude/AI assistance:**

```
You are an expert ML engineer working on a cricket tactics prediction system.

The system uses ball-by-ball IPL data (2008–2024).
Current model: basic classification model.

Your task:
1. Suggest and implement new features: bowler_type, bowling_style, batsman_role, venue, pressure_index
2. Add XGBoost and Logistic Regression models with proper train/test splits
3. Evaluate all models using accuracy, F1-score, and inference time
4. (Advanced) Design an LSTM or Transformer for over-by-over sequence modeling

STRICT RULE: Do not use data leakage. Only use features that are available at the moment a ball is bowled.
```

---

### ⚙️ Anjali — Backend API

**Focus:** FastAPI Server & Model Serving

**Key Files:** `api/main.py`, `requirements.txt`

**Suggested Prompt for Claude/AI assistance:**

```
You are a backend engineer building a FastAPI service for a cricket AI system.

Requirements:
1. Load a trained scikit-learn or XGBoost model from disk at startup
2. Expose a POST endpoint: /predict_strategy
   - Input: JSON with fields: batting_team, bowling_team, over, ball, score, wickets, venue, target
   - Output: JSON with fields: batting_strategy, bowling_strategy, confidence_score
3. Expose a GET endpoint: /health — returns {"status": "ok"}
4. Add CORS middleware to allow frontend requests
5. Handle missing/malformed input with a 422 error response

Return clean, commented Python code.
```

---

### 🎨 Ved — Frontend UI

**Focus:** Streamlit Captain Mode Dashboard

**Key Files:** `ui/app.py`

**Suggested Prompt for Claude/AI assistance:**

```
You are a frontend developer building a Streamlit UI for a cricket AI captain assistant.

Requirements:
1. Match input panel:
   - Dropdowns for batting_team and bowling_team (all IPL teams)
   - Number inputs for: over (1–20), ball (1–6), score, wickets (0–10), target
   - Venue dropdown
2. A "Get Strategy" button that sends a POST request to http://127.0.0.1:8000/predict_strategy
3. Strategy display section:
   - Show batting_strategy and bowling_strategy from the API response
   - Style the selected team's section with a bold primary color
   - Style the opponent's section in a secondary/muted color
4. Show a friendly error message if the API is unreachable

Use st.columns() for layout. Keep the design clean and readable.
```

---

## 10. Contribution Guide

### Step 1 — Always Pull Before Starting Work

```bash
git pull origin main
```

This ensures you have the latest code before making changes.

### Step 2 — Create a Feature Branch

Name your branch clearly based on what you're doing:

```bash
git checkout -b feature/xgboost-model        # Team 1 example
git checkout -b feature/predict-endpoint     # Team 2 example
git checkout -b feature/captain-mode-ui      # Team 3 example
```

### Step 3 — Make Your Changes

Work on your feature. Test it locally before committing.

### Step 4 — Commit with a Clear Message

```bash
git add .
git commit -m "Add XGBoost model with pressure_index feature"
```

Good commit messages explain **what** changed and optionally **why**.

### Step 5 — Push Your Branch

```bash
git push origin feature/your-branch-name
```

### Step 6 — Open a Pull Request

Go to GitHub → Your repository → **Pull Requests** → **New Pull Request**

- Set base branch to `main`
- Write a brief description of what you did
- Tag at least one other team member to review

> ✅ **Rule:** No one merges their own Pull Request. At least one other person must review and approve.

---

## 11. Important Rules

| Rule | Detail |
|------|--------|
| ❌ No large files in repo | Never commit the `ipl_json/` folder or any dataset files to Git |
| ✅ Use Google Drive for data | Share the dataset link in this README and in your team chat |
| ✅ Pin dependency versions | Do not change pinned versions (e.g., `scikit-learn==1.3.2`) without team discussion |
| ✅ Document your model schema | Team 1 must document the final model's input/output format for Team 2 |
| ✅ Document your API schema | Team 2 must share the endpoint JSON schema with Team 3 before Phase 4 begins |
| ✅ Test before pushing | Run `test_model.py` and `test_strategy.py` before opening a PR |
| ✅ Communicate blockers early | If your phase is blocked, tell the team immediately — don't wait |

---

## 🏁 Final Vision

Synapse Live aims to be a fully functional **AI cricket captain** — a system that doesn't just analyze the past, but makes real-time decisions just like a seasoned captain would, combining statistical patterns with situational awareness.

**Project Type:**
- Machine Learning System
- Real-Time Simulation Engine
- Decision Intelligence Platform
- Sports Analytics AI

---

*Last updated: See Git history*
