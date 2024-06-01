from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model

with open('forecast_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    future_dates = pd.date_range(start=data['start_date'], periods=data['periods'], freq='D')
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    return jsonify(forecast[['ds','yhat1']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)