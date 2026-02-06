# Smart Cafeteria ML Model

Demand forecasting model for the Smart Cafeteria system.

## Overview

This module provides ML-powered demand predictions based on:
- Historical demand data
- Weather conditions
- Academic schedules
- Day of week patterns

## Setup

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python api/predict_api.py
```

The API will run on `http://localhost:5001`

## Training Data

Sample training data is provided in `data/sample_training_data.csv`

## Integration

The Express.js backend calls this API for predictions via the `/api/forecasts/predict` endpoint.


