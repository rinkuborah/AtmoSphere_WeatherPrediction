from flask import Flask, render_template
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Create a Flask web app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def display_predictions():
    # Load the saved model
    model = load_model('regressor-02.h5')

    # Load the dataset and preprocess it as needed
    df = pd.read_csv('sensors_data.csv')

    # Extract the 'Formatted Date' and '_tempm' columns
    data = df[['datetime_utc', '_tempm']].copy()

    # Convert 'Formatted Date' to DateTimeIndex with the appropriate format
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Set 'datetime_utc' as the index
    data.set_index('datetime_utc', inplace=True)

    # Scaling data to get rid of outliers
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    # Get the last 30 days of data for making predictions
    last_30_days = data_scaled[-30:]

    # Prepare input for the model
    X = [last_30_days]
    X = np.array(X)
    X = X.reshape(1, 30, 1)

    # Get the current date
    current_date = datetime.now()

    # Set the current date to the next day
    next_day = current_date + timedelta(days=1)

    # Get the last available date in your dataset
    last_date = data.index[-1]

    # Make predictions for the next seven days
    predictions = []

    # Predict all seven days at once
    for _ in range(7):
        prediction = model.predict(X.reshape(1, 30, 1))
        predictions.append(prediction[0][0])
        X = np.append(X, prediction)
        X = X[1:]

        # Calculate the date for the next day
        next_date = last_date 
        last_date = next_date

    # Inverse transform the predictions to get actual temperature values
    predicted_temperatures = scaler.inverse_transform(np.array(predictions).reshape(7, 1))

    # Display the predictions with dates for the next seven days
    next_dates = [next_day.strftime('%Y-%m-%d')]
    for _ in range(6):
        next_day += timedelta(days=1)
        next_dates.append(next_day.strftime('%Y-%m-%d'))

    return render_template('predictions.html', dates=next_dates, predictions=predicted_temperatures)

if __name__ == '__main__':
    app.run(debug=True)
