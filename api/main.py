from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
bowling_model = joblib.load("models/bowling_strategy_model.pkl")
batting_model = joblib.load("models/batting_strategy_model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_strategy")
def predict_strategy(data: dict):
    try:
        # -------- Batting model (23 features) --------
        batting_features = [[
            data["over"],
            data["ball"],
            1,
            6.5,
            120,
            7.5,
            data["score"],
            data["over"] + data["ball"]/6,
            0,
            0,
            0,
            120 - (data["over"] * 6 + data["ball"]),
            0.5,
            1,
            1,
            0.0,
            10 - data["wickets"],
            0,
            0.2,
            0.5,
            20,
            2,
            7.0
        ]]

        # -------- Bowling model (26 features) --------
        bowling_features = [[
            data["over"],
            data["ball"],
            1,
            6.5,
            120,
            7.5,
            data["score"],
            data["over"] + data["ball"]/6,
            0,
            0,
            0,
            120 - (data["over"] * 6 + data["ball"]),
            0.5,
            1,
            1,
            0.0,
            10 - data["wickets"],
            0,
            0.2,
            0.5,
            20,
            2,
            7.0,
            0,
            0,
            0
        ]]

        # Predictions
        batting_pred = batting_model.predict(batting_features)[0]
        bowling_pred = bowling_model.predict(bowling_features)[0]

        return {
            "batting_strategy": str(batting_pred),
            "bowling_strategy": str(bowling_pred),
            "confidence_score": 0.9
        }

    except Exception as e:
        return {"error": str(e)}