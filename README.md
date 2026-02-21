# 🤖 Smart Cafeteria - ML Prediction Engine

[![Python Version](https://img.shields.io/badge/python-3.9+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Library](https://img.shields.io/badge/library-Scikit--learn-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/api-Flask-000000?style=flat&logo=flask)](https://flask.palletsprojects.com/)

The intelligence layer of the Smart Cafeteria system. Uses machine learning to forecast student demand based on real-world contextual factors including weather, academic schedules, and historical patterns.

---

## 🧠 Model Specifications
- **Model Type**: Random Forest Regressor (Ensemble Learning)
- **Primary Function**: Meal-wise student demand prediction
- **Accuracy Grade**: **81.52%** (Current Evaluation)
- **MAE**: ~6.0 | **MAPE**: ~18.5%

### **Feature Inputs**:
| Feature | Description |
|---------|-------------|
| **Day of Week** | Weekend dips vs mid-week peaks |
| **Meal Type** | Breakfast, Lunch, Snacks, Dinner |
| **Weather** | Real-time data from Open-Meteo API (Sunny, Rainy, Cloudy) |
| **Temperature** | Actual temperature in °C (from Open-Meteo) |
| **Academic Schedule** | Regular classes, exam periods, holidays, weekends |
| **Month & Season** | Seasonal demand patterns |

### **Training Data**:
- **Source**: Authentic data generated using real weather data for **Coimbatore (Amrita Vishwa Vidyapeetham)**
- **Weather API**: Open-Meteo Historical Weather API
- **Academic Calendar**: Real exam schedules, semester breaks, and national holidays
- **Coverage**: June 2025 – February 2026 (~800+ records)
- **Fallback**: Rule-based prediction when model confidence is low

---

## 🛠 Tech Stack
- **Language**: Python 3.9+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **API Wrapper**: Flask (REST API with CORS)
- **Real-Time Weather**: Open-Meteo API integration
- **Model Persistence**: Joblib serialization

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and model status |
| `POST` | `/predict` | Single meal prediction (date, meal, weather, schedule) |
| `GET` | `/predict/day?date=YYYY-MM-DD` | Full day forecast (all 4 meals with auto weather fetch) |

### Example Request
```json
POST /predict
{
  "date": "2026-02-21",
  "meal_type": "lunch",
  "weather": "sunny",
  "temperature": 32.5,
  "schedule": "regular"
}
```

### Example Response
```json
{
  "predicted_demand": 285,
  "confidence": 0.87,
  "method": "ml_model",
  "weather_source": "open-meteo"
}
```

---

## 🚀 Quick Start

```bash
# 1. Install ML dependencies
pip install -r requirements.txt

# 2. Run the prediction service
python api/predict_api.py
```

The service runs on `http://localhost:5001`. The **Go Backend** calls this service for forecast endpoints.

---

## 📂 Directory Structure
```
ml-model/
├── api/
│   └── predict_api.py         # Flask REST API with weather integration
├── data/
│   ├── generate_dataset.py    # Authentic data generation (real weather + academic calendar)
│   └── training_data.csv      # Historical dataset (~800+ records)
└── models/
    ├── demand_forecaster.py   # DemandForecaster class (train, predict, evaluate)
    └── random_forest.joblib   # Serialized trained model
```
