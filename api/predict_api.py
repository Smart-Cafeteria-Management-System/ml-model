"""
Flask API for demand prediction service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.demand_forecaster import DemandForecaster

app = Flask(__name__)
CORS(app)

# Initialize model
model = DemandForecaster()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'service': 'ML Demand Forecaster'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict demand for given conditions
    
    Expected JSON body:
    {
        "meal_type": "lunch",
        "weather": "sunny",
        "schedule": "regular",
        "day_of_week": "wednesday"
    }
    """
    data = request.get_json()
    
    meal_type = data.get('meal_type', 'lunch')
    weather = data.get('weather', 'sunny')
    schedule = data.get('schedule', 'regular')
    day_of_week = data.get('day_of_week', 'monday')
    
    result = model.predict(meal_type, weather, schedule, day_of_week)
    
    return jsonify({
        'success': True,
        'prediction': result
    })

@app.route('/predict/day', methods=['POST'])
def predict_day():
    """
    Predict demand for all meals on a given day
    
    Expected JSON body:
    {
        "date": "2024-01-24",
        "weather": "sunny",
        "schedule": "regular"
    }
    """
    from datetime import datetime
    
    data = request.get_json()
    
    date_str = data.get('date')
    weather = data.get('weather', 'sunny')
    schedule = data.get('schedule', 'regular')
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        date = datetime.now()
    
    predictions = model.predict_day(date, weather, schedule)
    
    return jsonify({
        'success': True,
        'date': date.strftime('%Y-%m-%d'),
        'predictions': predictions
    })

if __name__ == '__main__':
    print("Starting ML Prediction API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
