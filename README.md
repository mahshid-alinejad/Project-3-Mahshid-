# Project-3-Mahshid-


Weather Forecasting Project

Overview

This project involves building a weather forecasting model using LSTM (Long Short-Term Memory) networks to predict future temperatures based on historical data. The model has been trained to predict temperatures for various cities, and the results can be accessed through a web interface created with Gradio.

Project Components

Data Preparation: Historical temperature data is used to train the LSTM model. Data preprocessing includes handling missing values, creating sequences, and scaling features.
Model Building: An LSTM model is constructed and tuned using Keras Tuner. Hyperparameters such as the number of units, dropout rates, and learning rates are optimized.
Prediction: The trained model predicts temperatures for specified cities and dates.
Gradio Interface: A user-friendly web interface allows users to input a city and a date to get the forecasted temperature.
Installation

To run this project, ensure you have Python and the following packages installed:

TensorFlow
Keras Tuner
Scikit-learn
Pandas
NumPy
Matplotlib
Gradio
Install the required packages using pip:

bash
Copy code
pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib gradio
Usage

Model Training and Tuning
The model is trained and tuned using a provided script. This script will generate a trained model saved as best_model.h5.

Gradio Interface
The Gradio interface allows you to interact with the model. It uses the trained best_model.h5 file to make predictions based on the city and date inputs.

How to Use the Gradio Interface

Input: Select a city from the dropdown menu and enter a date in the format YYYY-MM-DD.
Output: The forecasted temperature will be displayed in Fahrenheit. The result is constrained to a realistic temperature range.
Example

If you input "Seattle" and "2025-09-01" into the interface, you will receive a forecasted temperature for that city and date.

Error Handling
The interface includes checks to ensure that forecasted temperatures fall within a realistic range. If the forecasted temperature is outside this range, an appropriate error message will be displayed.

