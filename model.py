import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle
import os

# Fetch weather data if not already present
if not os.path.exists('weatherAUS.csv'):
    import data_collection
    data_collection.fetch_weather_data()

df = pd.read_csv('weatherAUS.csv')

melb = df[df['Location'] == 'Melbourne']
melb['Date'] = pd.to_datetime(melb['Date'])
melb = melb[melb['Date'] <= '2015-12-31']

data = melb[['Date', 'Temp3pm']].dropna()
data.columns = ['ds', 'y']

m = NeuralProphet()
m.fit(data, freq='D', epochs=1000)

future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)

# Save the trained model
with open('forecast_model.pkl', "wb") as f:
    pickle.dump(m, f)

