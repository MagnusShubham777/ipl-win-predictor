import os
import math
import json
import hashlib
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from jose import jwt
except ImportError:
    jwt = None


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "model" / "ipl_model.joblib"
METRICS_PATH = BASE / "model" / "metrics.json"

SECRET = os.getenv(
    "JWT_SECRET",
    "change-this-secret-in-production"
)

MONGO_URI = os.getenv("MONGO_URI", "")


# ============================================================
# IPL DATA
# ============================================================

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]

CITIES = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Dharamsala",
    "Hyderabad",
    "Jaipur",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Pune"
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="IPL Win Probability API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

origins = [
    x.strip()
    for x in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print("ML model loaded successfully.")
    except Exception as e:
        print("Could not load ML model:", e)


# ============================================================
# LOAD METRICS
# ============================================================

if METRICS_PATH.exists():
    try:
        metrics = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )
    except Exception as e:
        print("Could not load metrics:", e)
        metrics = {
            "models": [],
            "best_model": "fallback"
        }
else:
    metrics = {
        "models": [],
        "best_model": "fallback"
    }


# ============================================================
# MONGODB
# ============================================================

client = None
db = None

if MONGO_URI and MongoClient:

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=2500
        )

        client.admin.command("ping")

        db = client["ipl_predictor"]

        print("MongoDB connected successfully.")

    except Exception as e:

        print("MongoDB unavailable:", e)

        client = None
        db = None


# ============================================================
# FALLBACK STORAGE
# ============================================================

users_fallback = {}
history_fallback = []


# ============================================================
# AUTH HELPERS
# ============================================================

def hash_password(password: str):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()


def create_token(email: str):

    if not jwt:
        return email

    return jwt.encode(
        {
            "sub": email,
            "exp": datetime.utcnow() + timedelta(days=7)
        },
        SECRET,
        algorithm="HS256"
    )


def current_email(
    authorization: Optional[str] = Header(None)
):

    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ", 1)[1]

    if jwt:

        try:

            payload = jwt.decode(
                token,
                SECRET,
                algorithms=["HS256"]
            )

            return payload.get("sub")

        except Exception:

            return None

    return token


# ============================================================
# PYDANTIC MODELS
# ============================================================

class AuthBody(BaseModel):

    email: str

    password: str = Field(
        min_length=6
    )


class PredictionRequest(BaseModel):

    batting_team: str

    bowling_team: str

    city: str

    target: int = Field(
        gt=0,
        le=300
    )

    score: int = Field(
        ge=0,
        le=300
    )

    overs: float = Field(
        ge=0,
        le=20
    )

    wickets: int = Field(
        ge=0,
        le=10
    )


class ScorecardPlayer(BaseModel):

    name: str

    runs: int = 0

    balls: int = 0

    fours: int = 0

    sixes: int = 0

    sr: float = 0


class Scorecard(BaseModel):

    batting_team: str

    bowling_team: str

    score: int

    wickets: int

    overs: float

    target: int

    batters: list[ScorecardPlayer] = Field(
        default_factory=list
    )


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "version": "2.0.0",
        "model": metrics.get(
            "best_model",
            "fallback"
        )
    }


@app.get("/api/teams")
def teams():

    return {
        "teams": TEAMS
    }


@app.get("/api/cities")
def cities():

    return {
        "cities": CITIES
    }


@app.get("/api/metrics")
def get_metrics():

    return metrics


# ============================================================
# AUTH - REGISTER
# ============================================================

@app.post("/api/auth/register")
def register(body: AuthBody):

    email = body.email.lower().strip()

    password = hash_password(
        body.password
    )

    if db:

        if db.users.find_one({
            "email": email
        }):

            raise HTTPException(
                status_code=409,
                detail="Account already exists"
            )

        db.users.insert_one({
            "email": email,
            "password": password,
            "created_at": datetime.utcnow()
        })

    else:

        if email in users_fallback:

            raise HTTPException(
                status_code=409,
                detail="Account already exists"
            )

        users_fallback[email] = password

    return {
        "token": create_token(email),
        "email": email
    }


# ============================================================
# AUTH - LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(body: AuthBody):

    email = body.email.lower().strip()

    password = hash_password(
        body.password
    )

    if db:

        user = db.users.find_one({
            "email": email
        })

        saved = (
            user["password"]
            if user
            else None
        )

    else:

        saved = users_fallback.get(
            email
        )

    if saved != password:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
        "token": create_token(email),
        "email": email
    }


# ============================================================
# AUTH - CURRENT USER
# ============================================================

@app.get("/api/auth/me")
def me(
    email=Depends(current_email)
):

    if not email:

        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    return {
        "email": email
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(req: PredictionRequest):

    balls_left = max(
        0,
        120 - round(req.overs * 6)
    )

    runs_left = max(
        0,
        req.target - req.score
    )

    wickets_left = max(
        0,
        10 - req.wickets
    )

    current_run_rate = (
        req.score / req.overs
        if req.overs > 0
        else 0
    )

    if balls_left > 0:

        required_run_rate = (
            runs_left /
            (balls_left / 6)
        )

    else:

        required_run_rate = (
            999
            if runs_left > 0
            else 0
        )

    return {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "target": req.target,
        "crr": current_run_rate,
        "rrr": required_run_rate,
        "score": req.score,
        "overs": req.overs,

        # These are for API/display/history.
        # They are NOT sent to the ML model.
        "batting_team": req.batting_team,
        "bowling_team": req.bowling_team,
        "city": req.city
    }


# ============================================================
# FALLBACK MODEL
# ============================================================

def fallback_probability(features):

    progress = (
        features["score"] /
        features["target"]
    )

    time_left = (
        features["balls_left"] /
        120
    )

    wicket_factor = (
        features["wickets_left"] /
        10
    )

    pressure = (
        min(
            1.0,
            features["rrr"] / 12
        )
        if features["rrr"] < 999
        else 1
    )

    z = (
        4 * (progress - 0.55)
        + 1.7 * (wicket_factor - 0.5)
        + 1.2 * (0.5 - pressure)
        + 0.7 * (0.5 - time_left)
    )

    return 1 / (
        1 + math.exp(-z)
    )


# ============================================================
# ML PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(
    req: PredictionRequest,
    email=Depends(current_email)
):

    # ----------------------------------------
    # Validation
    # ----------------------------------------

    if req.batting_team == req.bowling_team:

        raise HTTPException(
            status_code=400,
            detail="Teams must be different"
        )

    if req.score > req.target:

        raise HTTPException(
            status_code=400,
            detail="Score cannot exceed target"
        )

    if req.wickets > 10:

        raise HTTPException(
            status_code=400,
            detail="Wickets cannot exceed 10"
        )

    # ----------------------------------------
    # Feature engineering
    # ----------------------------------------

    features = make_features(req)

    # ----------------------------------------
    # IMPORTANT:
    # These MUST match train_model.py
    # ----------------------------------------

    X = pd.DataFrame([{
    "runs_left": features["runs_left"],
    "balls_left": features["balls_left"],
    "wickets_left": features["wickets_left"],
    "target": features["target"],
    "crr": features["crr"],
    "rrr": features["rrr"],
    "score": features["score"],
    "overs": features["overs"],
    "batting_team": features["batting_team"],
    "bowling_team": features["bowling_team"],
    "city": features["city"]
}])

    # ----------------------------------------
    # ML prediction
    # ----------------------------------------

    used_model = "fallback"

    if model is not None:

        try:

            probability = float(
                model.predict_proba(X)[0][1]
            )

            used_model = metrics.get(
                "best_model",
                "trained-model"
            )

            print(
                f"ML prediction successful: "
                f"{probability:.4f}"
            )

        except Exception as e:

            print(
                "ML prediction failed:",
                e
            )

            probability = fallback_probability(
                features
            )

    else:

        probability = fallback_probability(
            features
        )

    # ----------------------------------------
    # Clamp probability
    # ----------------------------------------

    probability = max(
        0.01,
        min(0.99, probability)
    )

    # ----------------------------------------
    # Response
    # ----------------------------------------

    result = {

        "batting_team":
            req.batting_team,

        "bowling_team":
            req.bowling_team,

        "city":
            req.city,

        "batting_probability":
            round(
                probability * 100,
                2
            ),

        "bowling_probability":
            round(
                (1 - probability) * 100,
                2
            ),

        "runs_left":
            features["runs_left"],

        "balls_left":
            features["balls_left"],

        "wickets_left":
            features["wickets_left"],

        "current_run_rate":
            round(
                features["crr"],
                2
            ),

        "required_run_rate":
            (
                round(
                    features["rrr"],
                    2
                )
                if features["rrr"] < 999
                else None
            ),

        "target":
            req.target,

        "score":
            req.score,

        "overs":
            req.overs,

        "wickets":
            req.wickets,

        "model":
            used_model
    }

    # ----------------------------------------
    # Save prediction history
    # ----------------------------------------

    if email:

        doc = {
            **result,
            "email": email,
            "created_at":
                datetime.utcnow()
        }

        if db:

            db.predictions.insert_one(
                doc
            )

        else:

            history_fallback.insert(
                0,
                doc
            )

    return result


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
def history(
    email=Depends(current_email)
):

    if not email:

        return []

    if db:

        docs = list(
            db.predictions
            .find({
                "email": email
            })
            .sort(
                "created_at",
                -1
            )
            .limit(30)
        )

        for doc in docs:

            doc.pop(
                "_id",
                None
            )

            doc.pop(
                "email",
                None
            )

        return docs

    return [
        {
            key: value
            for key, value in doc.items()
            if key not in (
                "email",
                "_id"
            )
        }

        for doc in history_fallback

        if doc.get("email") == email
    ][:30]


# ============================================================
# SCORECARD
# ============================================================

@app.post("/api/scorecard")
def scorecard(
    body: Scorecard
):

    return body.model_dump()


# ============================================================
# LIVE SCORE STATUS
# ============================================================

@app.get("/api/live-status")
def live_status():

    enabled = bool(
        os.getenv(
            "LIVE_SCORE_API_URL"
        )
    )

    return {

        "enabled":
            enabled,

        "message":
            (
                "Configure LIVE_SCORE_API_URL "
                "and provider credentials to "
                "enable live score ingestion."
                if not enabled
                else
                "Live provider configured."
            )
    }