import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = load_model('regressor-02.h5')


# Load the dataset and preprocess it as needed
df = pd.read_csv('sensors_data.csv')

df.head()

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

from datetime import datetime, timedelta

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
    predictions.append(prediction)
    X = np.append(X, prediction)
    X = X[1:]

    # Calculate the date for the next day
    next_date = last_date   
    last_date = next_date

# Reshape the predictions to a 2D array
predicted_temperatures = np.array(predictions).reshape(7, 1)

# Inverse transform the predictions to get actual temperature values
predicted_temperatures = scaler.inverse_transform(predicted_temperatures)

# Display the predictions with dates for the next seven days
next_dates = [next_day]  # Set the current date to the next day
for _ in range(6):
    next_day += timedelta(days=1)
    next_dates.append(next_day)

for date, prediction in zip(next_dates, predicted_temperatures):
    print(f"Date: {date.strftime('%Y-%m-%d')}, Min Temp: {prediction[0]:.2f}°C, Max Temp: {prediction[0] + 5:.2f}°C")
