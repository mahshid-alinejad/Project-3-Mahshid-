# Project-3-Mahshid-


# Weather Forecasting Project

  This project aims to predict future temperatures in Vancouver, Canada, using a Long Short-Term Memory (LSTM) model. The model is trained on historical temperature data and utilizes a Gradio interface for easy interaction.

## Table of Contents

  - Project Overview
  - Data Description
  - Data Cleaning and Preparation
  - Feature Selection
  - Model Description
  - Model Comparison
  - Gradio Interface
  - Requirements
  - Usage
  - Future Work
  - Acknowledgements


## Overview

  Weather and temperature is not only an important for individual everyday life, but is also relevant to many other topics such as climate change. Studies from other researchers have shown promising results of using neural networks for predicting and future weather and/or temperature. This project creates a Long Short Term Memory model to predict future temperature, whihc in this case are 7 days out from the last date of dataframe, 2017-10-28.

## Data Description


The original data consists of hourly temperatures for Vancouver spanning 5 years, from 2012 to 2017. To predict the daily temperature for the next 7 days, I resampled the data to daily averages. I then split the dataset into training and testing sets, using an 80/20 split based on time: the first 4 years were used for training, and the last year was used for testing.

![image](https://github.com/user-attachments/assets/f8cd4436-339b-4b85-b234-568c261a2e91)

## Data Cleaning and Preparation

The following steps were performed to clean and prepare the data:

  - Dropped all the cities except Vancouver from the dataset.
  - Dropped all rows with missing values (NANs).
  - Converted temperature values from Kelvin to Celsius using the formula:
  - Celsius = Kelvin − 273.15                      Celsius=Kelvin−273.15

## Model Description

I employed an LSTM model to predict the temperature for the following day using the previous 15 days' temperatures as input features. The model was trained on 4 years of data and achieved an R² score of 93%.

To forecast the temperature for the next 7 days, the model must be run recursively. This involves predicting the temperature for the next day, then appending the predicted value to the feature set and using it for the subsequent day's prediction. As the model progresses through the 7-day forecast, the quality of predictions deteriorates because each prediction relies on previous predictions rather than actual values. Consequently, this type of model is best suited for short-term forecasts, typically up to a few days ahead.

### forecasted
 2017-10-29    14.061722
 2017-10-30    13.360307
 2017-10-31    13.184744
 2017-11-01    13.204847
 2017-11-02    13.258042
 2017-11-03    13.332599
 2017-11-04    13.445753

![Snip20240803_15](https://github.com/user-attachments/assets/62bb5ded-2802-473c-af29-58f7b90e5154)

  Train MAE: 1.1813127994537354
  Test MAE: 1.3242559432983398
  12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
  Test R^2: 0.9395328618648101


### Gradio Interface

A Gradio interface is used to provide an interactive web interface for temperature prediction. Users can input a date, and the interface will display the forecasted temperature in Celsius.

## Usage

To use the Gradio interface for temperature forecasting:
  Ensure the Jupyter notebook is running.
  Locate and run the cell containing the Gradio interface setup and launch code.
  Open the provided local URL or public link to access the interface.


## Future Work

  Potential Features to Add or Remove
Add:Air pressure, Humidity, Wind speed,Precipitation
Remove: Any feature that is irrelevant to temperature prediction

Incorporating additional features like air pressure, humidity, and wind speed could improve the model's accuracy by providing more context to the temperature data.

  7-Day Forecasting Limitation
    Problem
      The model currently forecasts temperatures up to 7 days out. This limitation arises from the nature of the data and the model's design.
    Solution
      To extend the forecasting horizon, additional data and features should be integrated. More advanced modeling techniques and the inclusion of features such as air pressure and humidity could enhance the model's ability to predict temperatures further into the future.
      
## Requirements
  The project requires the following Python libraries:
  - pandas
  - numpy
  - tensorflow
  - gradio
  - matplotlib

## Acknowledgements

This project was inspired by the TempPredict repository. 









