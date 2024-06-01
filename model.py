import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle

df = pd.read_csv('weatherAUS.csv')

df.head()

df.Location.unique()

df.dtypes

melb = df[df['Location'] == 'Melbourne']
melb['Date'] = pd.to_datetime(melb['Date'])
melb.head()

melb.dtypes

plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()

melb['Year'] = melb['Date'].apply(lambda x: x.year)
melb = melb[melb['Year'] <= 2015]
plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()

melb = melb[['Date', 'Year', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow']]

melb.head()


data = melb[['Date', 'Temp3pm']]
data.dropna(inplace=True)
data.columns = ['ds', 'y']
data.head()

m = NeuralProphet()
m.fit(data, freq='D', epochs=1000)

future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)
forecast.head()

forecast.tail()

m.plot(forecast)

m.plot_components(forecast)

with open('forecast_model.pkl', "wb") as f:
    pickle.dump(m, f)

