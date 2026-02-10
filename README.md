# 🤖 Smart Cafeteria - ML Prediction Engine

[![Python Version](https://img.shields.io/badge/python-3.8+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Library](https://img.shields.io/badge/library-Scikit--learn-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/api-Flask-000000?style=flat&logo=flask)](https://flask.palletsprojects.com/)

The "Intelligence" of the system. This module uses machine learning to forecast student demand and provide wait-time predictions based on complex contextual factors.

---

## 🧠 Model Specifications
- **Model Type**: Random Forest Regressor (Ensemble Learning)
- **Primary Function**: Slot-wise student volume prediction.
- **Accuracy Grade**: **81.52%** (Current Evaluation)

### **Key Features (Inputs)**:
1. **Historical Demand**: Past booking volumes for corresponding days/meals.
2. **Day of Week**: Recognizing weekend dips and mid-week peaks.
3. **Weather Conditions**: (Sunny, Rainy, Cloudy) – Affecting student attendance.
4. **Academic Schedule**: Distinguishing between regular classes, exam periods, and holidays.

---

## 🛠 Tech Stack
- **Language**: Python 3.9+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **API Wrapper**: Flask (REST API)
- **Data Source**: CSV and PostgreSQL sync

---

## 🚀 Quick Start (Production)

```bash
# 1. Install ML dependencies
pip install -r requirements.txt

# 2. Run the prediction service
python api/predict_api.py
```

The service runs on `http://localhost:5001`. The **Go Backend** calls this service whenever a 1-week forecast is requested.

---

## 📂 Directory Structure
```
ml-model/
├── api/
│   └── predict_api.py      # Flask REST API for real-time inference
├── data/
│   ├── generate_dataset.py # Synthetic data generation for initial training
│   └── training_data.csv   # Historical dataset
└── models/
    └── random_forest.joblib # Serialized model binary
```
