#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime as dt

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import keras_tuner as kt
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler


# In[45]:


data = pd.read_csv('temperature.csv',)

# Print the column names
print("Column names:", data.columns)

data.head(3) 


# In[46]:


def auto_drop_na(data, drop_percent):
    # Calculate the number of rows as a reference for drop calculations
    num_rows = len(data)
    drop_threshold = num_rows * drop_percent / 100  # Convert percentage to actual number of rows
    
    print(f'Drop Percent of the rows is %{drop_percent}')
    print('If the number of NA values in a column is less than the calculated threshold, automatically drop the NA rows.')

    # Get the count of NAs in each column and convert to dictionary
    has_na = data.isna().sum()
    has_na = has_na[has_na > 0].to_dict()
    print(has_na)
    
    # Make decisions about null values
    columns_to_drop = ['datetime', 'Vancouver','Portland', 'San Francisco', 'Seattle', 'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque', 'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City', 'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis', 'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami', 'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal', 'Boston', 'Beersheba', 'Tel Aviv District', 'Eilat', 'Haifa', 'Nahariyya', 'Jerusalem']
    for column, na_count in has_na.items():
        if na_count > drop_threshold:
            print(f"The number of NA values in {column} is {na_count}, which is greater than the threshold {drop_threshold}. Review options before action.")
        else:
            print(f"Automatically dropping rows in {column} where NA values are present.")
            data.dropna(subset=[column], inplace=True)
    return data

# Example usage:
# Assuming 'data' is your DataFrame
data_cleaned = auto_drop_na(data, drop_percent=10)
data_cleaned


# In[47]:


# data_cleaned = data_cleaned[['datetime', 'Vancouver']]
# Convert temperature from Kelvin to Celsius
data_cleaned['Vancouver_Celsius'] = data_cleaned['Vancouver'] - 273.15

# # Convert temperature from Kelvin to Fahrenheit
# data_cleaned['Vancouver_Fahrenheit'] = (data_cleaned['Vancouver'] - 273.15) * 9/5 + 32

data_cleaned


# In[48]:


# import datetime as dt
# # Ensure 'datetime' is in datetime format
# # data_cleaned['datetime'] = pd.to_datetime(data_cleaned['datetime'])

# # # Extract date and hour
# # data_cleaned['date'] = data_cleaned['datetime'].dt.date
# data_cleaned['hour'] = data_cleaned['datetime'].dt.time
# data_cleaned['month'] = data_cleaned['datetime'].dt.month
# data_cleaned['day'] = data_cleaned['datetime'].dt.day


# # # Convert `date` to a numerical format (e.g., days since start of the dataset)
# # data_cleaned['date'] = pd.to_datetime(data_cleaned['date'])
# # data_cleaned['date'] = (data_cleaned['date'] - data_cleaned['date'].min()).dt.days

# # # Convert `hour` to numerical format
# # data_cleaned['hour'] = pd.to_datetime(data_cleaned['hour'], format='%H:%M:%S').dt.hour

# data_cleaned


# In[49]:


data_cleaned = data_cleaned[['datetime','Vancouver_Celsius']]

# Verify the result
data_cleaned.head()
data_cleaned.columns


# In[50]:


# Handle datetime column
datetime_col = 'datetime'
if datetime_col not in data_cleaned.columns:
    raise KeyError(f"The column '{datetime_col}' is not found in the dataset. Available columns: {data_cleaned.columns}")

data_cleaned[datetime_col] = pd.to_datetime(data_cleaned[datetime_col])



# In[51]:


data_cleaned


# In[52]:


# Handle datetime column
data_cleaned['datetime'] = pd.to_datetime(data_cleaned['datetime'])

# Add datetime features as columns
data_cleaned['day_of_year'] = data_cleaned['datetime'].dt.dayofyear
data_cleaned['hour_of_day'] = data_cleaned['datetime'].dt.hour
data_cleaned


# In[64]:


daily = data_cleaned.set_index('datetime')['Vancouver_Celsius'].resample('1D').mean()
daily.plot()


# In[65]:


daily.head()


# In[124]:


def create_sequences(values, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(values) - seq_length):
        X_seq.append(values[i:i + seq_length])
        y_seq.append(values[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


# In[177]:


seq_length = 15


# In[178]:


X, y = create_sequences(daily.values, seq_length)


# In[179]:


X.shape, y.shape


# In[180]:


mu = daily.values.mean()
sd = daily.values.std()


# In[181]:


X_scaled = (X - mu)/sd


# In[182]:


n = int(0.8*X_scaled.shape[0])


# In[183]:


# Split the data
X_train, X_test, y_train, y_test =X_scaled[:n], X_scaled[n:], y[:n], y[n:]


# In[184]:


X_train = X_train[:,:,None]
X_test = X_test[:,:,None]
X_train.shape


# In[185]:


def build_model(input_shape):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(LSTM(units=16,
                   activation='relu',                   
                   return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=8))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
    return model


# In[186]:


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model  = build_model(X_train.shape[1:])


# In[187]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32, callbacks=[early_stopping])


# In[188]:


# Evaluate the model
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# Predict and calculate R^2 score
y_pred = model.predict(X_test).flatten()
test_r2 = r2_score(y_test, y_pred)
print(f"Test R^2: {test_r2}")


# In[189]:


def forecast(vals, seq_length, days):
    vals = vals.copy()
    y_pred = []
    X = np.array(vals[-seq_length:])[None, :]
    
    for i in range(days):
        X_scaled = (X - mu)/sd
        X_scaled = X_scaled[:,:,None]        
        y_cur = model.predict(X_scaled, verbose=0)[0,0]        
        y_pred.append(y_cur)
        vals.append(y_cur)
        X = np.array(vals[-seq_length:])[None, :]
    return y_pred
        


# In[205]:


timestamps = 7
y_pred = forecast(daily.values.tolist(), seq_length, timestamps)


# In[206]:


forecasted = pd.Series(y_pred, index=pd.date_range(daily.index[-1] + pd.Timedelta('1day'), 
                                                   daily.index[-1] + pd.Timedelta(timestamps, unit='D')))


# In[207]:


forecasted


# In[203]:


daily.iloc[-60:].plot()
forecasted.plot()


# In[210]:


def predict_temperature(date):
    return forecasted[date]


# In[211]:


import gradio as gr
# Create the Gradio interface
iface = gr.Interface(
    fn=predict_temperature,
    inputs=[
        gr.Dropdown(choices=forecasted.index.tolist())
    ],
    outputs="number",
    live=True,
    title="Temperature Forecast",
    description="Enter the day of the year and the hour of the day to get the forecasted temperature."
)

# Launch the interface
iface.launch(`share=True`)

