"""
Flask API for demand prediction service.
This service exposes the Random Forest model to the Go Backend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Ensure the models directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.demand_forecaster import DemandForecaster

app = Flask(__name__)
CORS(app)

# Initialize the ML Model (Random Forest)
model = DemandForecaster()

@app.route('/health', methods=['GET'])
def health():
    """Verifies that the ML Prediction service is live."""
    return jsonify({'status': 'OK', 'service': 'ML Demand Forecaster'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Real-time Prediction Endpoint.
    Takes Meal Type, Weather, Schedule, and Day of Week as inputs.
    Returns predicted student volume (Demand).
    """
    data = request.get_json()
    
    meal_type = data.get('meal_type', 'lunch')
    weather = data.get('weather', 'sunny')
    schedule = data.get('schedule', 'regular')
    day_of_week = data.get('day_of_week', 'monday')
    
    # Execute inference
    result = model.predict(meal_type, weather, schedule, day_of_week)
    
    return jsonify({
        'success': True,
        'prediction': result
    })

@app.route('/predict/day', methods=['POST'])
def predict_day():
    """
    Predicts demand for all meals (Breakfast, Lunch, Dinner) on a specific date.
    Used for the week-ahead forecasting dashboard.
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
    # Run the Flask app on port 5001 for Go Backend integration
    app.run(host='0.0.0.0', port=5001, debug=True)
