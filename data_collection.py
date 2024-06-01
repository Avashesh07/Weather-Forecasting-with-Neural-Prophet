import requests
import pandas as pd

def fetch_weather_data():
    # Replace with actual API call to get weather data
    response = requests.get('https://api.weather.com/v1/location/Melbourne:4:AU/observations/historical.json?apiKey=YOUR_API_KEY')
    data = response.json()
    df = pd.DataFrame(data['observations'])
    df.to_csv('weatherAUS.csv', index=False)

if __name__ == "__main__":
    fetch_weather_data()
