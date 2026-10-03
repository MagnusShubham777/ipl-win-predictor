# IPL Win Probability Predictor — Full Stack AI/ML

A portfolio-grade IPL analytics platform built with **React + FastAPI + scikit-learn/XGBoost + MongoDB**.

## Features

- Interactive React dashboard
- Win probability prediction
- Feature engineering from ball-by-ball match states
- Logistic Regression, Random Forest and optional XGBoost model comparison
- Accuracy, ROC-AUC and log-loss evaluation
- Probability trend chart
- Match scorecard workspace
- User registration/login with JWT
- Prediction history
- MongoDB persistence with an in-memory fallback for local demos
- FastAPI Swagger docs
- Docker + docker-compose
- Production environment examples
- Optional live-score provider hook (requires a licensed/current provider)

## Data

Cricsheet publishes IPL match data and currently lists the Indian Premier League as a club competition. Its JSON format is the primary/current format. Download the IPL JSON archive from:
https://cricsheet.org/downloads/

Extract the JSON files into:

`backend/data/ipl_json/`

Then run:

```bash
cd backend
python train_model.py
```

The trainer builds post-delivery chase states and evaluates multiple classifiers on a holdout set. The best ROC-AUC model is saved to `backend/model/ipl_model.joblib` and metrics to `backend/model/metrics.json`.

### Demo mode

If no Cricsheet JSON files exist, `python train_model.py` generates a synthetic demo dataset so the application still starts. **Do not claim demo-model metrics as real IPL performance.**

## Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
python train_model.py
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open http://localhost:5173

## MongoDB

For local Docker development, `docker-compose.yml` starts MongoDB automatically.
For MongoDB Atlas, put your connection string in `backend/.env` as `MONGO_URI` and never commit it.

## Authentication

The app exposes:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/history`

JWT is used for sessions. Passwords are hashed before persistence. For a production application, use a strong `JWT_SECRET`, HTTPS, secure cookies/token storage, rate limiting, and a production password-hashing algorithm such as Argon2/bcrypt.

## Live scorecard

The project intentionally does not bundle an unlicensed live-score scraper. `GET /api/live-status` exposes the integration state. Set `LIVE_SCORE_API_URL` and implement the provider-specific authentication/normalization for a licensed provider. Keep provider API keys server-side.

## Deployment

### Backend

Build and run the backend Docker image:

```bash
docker build -t ipl-backend ./backend
docker run -p 8000:8000 --env-file backend/.env ipl-backend
```

### Full local stack

```bash
docker compose up --build
```

The frontend will be available at http://localhost:5173 and API at http://localhost:8000.

For cloud deployment, deploy the backend to a service that supports Docker/FastAPI and the frontend to a static hosting platform. Set `VITE_API_URL` to the deployed API origin and set `FRONTEND_ORIGINS` on the backend to the deployed frontend origin.

## Resume bullets

- Built a full-stack IPL analytics platform that predicts real-time match win probability using machine learning and historical ball-by-ball data.
- Engineered chase-state features including runs required, balls remaining, wickets remaining, current run rate and required run rate, and compared Logistic Regression, Random Forest and XGBoost models using ROC-AUC, accuracy and log-loss.
- Developed a FastAPI REST backend and responsive React dashboard with probability trends, scorecard analytics, JWT authentication and MongoDB-backed prediction history.
