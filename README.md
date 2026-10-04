# 🏏 IPL Win Probability Predictor

A full-stack **Machine Learning web application** that predicts the winning probability of two IPL teams during a live match based on the current match situation.

The application uses historical IPL match data and a trained **Logistic Regression** model to estimate the probability of the batting and bowling teams winning the match.

## 🔗 Project Links

**GitHub Repository:**  
https://github.com/MagnusShubham777/ipl-win-predictor

**Live Application:**  
https://ipl-win-predictor-sigma.vercel.app

**Backend API:**  
[Add your Render deployment URL here](https://ipl-win-predictor-8exe.onrender.com)

---

## 🚀 Features

- Predicts IPL match winning probability in real time
- Displays winning probability for both teams
- Takes current match conditions as input
- Machine Learning model trained on historical IPL data
- REST API built using FastAPI
- Interactive frontend
- Backend deployed on Render
- Frontend deployed on Vercel

---

## 📊 Model Inputs

The prediction model uses match information such as:

- Batting Team
- Bowling Team
- Host City
- Target Score
- Current Score
- Overs Completed
- Wickets Lost

From these values, additional features are calculated:

- Runs Left
- Balls Left
- Wickets Remaining
- Current Run Rate (CRR)
- Required Run Rate (RRR)

These features are passed to the trained machine learning model to calculate the winning probability.

---

## 🧠 Machine Learning Model

The project uses **Logistic Regression** for probability prediction.

The ML pipeline includes:

- Data preprocessing
- Feature engineering
- Categorical feature encoding
- Model training
- Probability prediction using `predict_proba()`

### Important Features

```text
runs_left
balls_left
wickets_left
target
current_run_rate
required_run_rate
batting_team
bowling_team
city
```

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

### Backend

- FastAPI
- Uvicorn
- Python

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Vercel — Frontend
- Render — Backend
- GitHub — Version Control

---

## 📂 Project Structure

```text
ipl-win-predictor/
│
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── model/
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/
│
├── README.md
│
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MagnusShubham777/ipl-win-predictor.git
```

Move into the project directory:

```bash
cd ipl-win-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

---

## 🎯 Prediction API

The frontend communicates with the FastAPI backend through the prediction endpoint:

```text
POST /api/predict
```

The API processes the current match situation and returns the predicted winning probabilities of both teams.

Example result:

```json
{
  "batting_team_probability": 72.4,
  "bowling_team_probability": 27.6
}
```

---

## 📈 How It Works

```text
User enters match details
        ↓
Frontend sends request
        ↓
FastAPI Backend
        ↓
Feature Engineering
        ↓
Trained ML Model
        ↓
Win Probability Prediction
        ↓
Frontend displays probabilities
```

---

## 🔮 Future Improvements

- Support newer IPL seasons and teams
- Integrate live IPL match data
- Improve prediction accuracy using advanced ML models
- Add graphical probability visualization
- Add historical match analysis
- Compare Logistic Regression with Random Forest and XGBoost
- Retrain the model automatically with new IPL data

---

## 👨‍💻 Author

**Shubham Negi**

GitHub:  
https://github.com/MagnusShubham777

Project Repository:  
https://github.com/MagnusShubham777/ipl-win-predictor

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
