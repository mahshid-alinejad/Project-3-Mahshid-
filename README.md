# Project-3-Mahshid-


Weather Forecasting Project

Overview

Weather and temperature is not only an important for individual everyday life, but is also relevant to many other topics such as climate change. Studies from other researchers have shown promising results of using neural networks for predicting and future weather and/or temperature. This project creates a Long Short Term Memory model to predict future temperature. The model takes in historical weather information such as air pressure and temperature and predicts future temperatures. 


Introduction


The original data consists of hourly temperatures for Vancouver spanning 5 years, from 2012 to 2017. To predict the daily temperature for the next 7 days, I resampled the data to daily averages. I then split the dataset into training and testing sets, using an 80/20 split based on time: the first 4 years were used for training, and the last year was used for testing.

I employed an LSTM model to predict the temperature for the following day using the previous 15 days' temperatures as input features. The model was trained on 4 years of data and achieved an R² score of 93%.

To forecast the temperature for the next 7 days, the model must be run recursively. This involves predicting the temperature for the next day, then appending the predicted value to the feature set and using it for the subsequent day's prediction. As the model progresses through the 7-day forecast, the quality of predictions deteriorates because each prediction relies on previous predictions rather than actual values. Consequently, this type of model is best suited for short-term forecasts, typically up to a few days ahead.






The temperature values in dataset appear to be in Kelvin rather than Celsius or Fahrenheit. Kelvin is a unit of measurement for temperature where 0 K is absolute zero. 

Cleaned the data and drop all cities and only kept one city,Vancouver and dropped all NANs

converted Kelvin to Celsius: 
![image](https://github.com/user-attachments/assets/b05c1328-cc51-4d20-8b24-f4aac9d44616)


Feature Selection:
Ensure that the date and hour features are included if they provide relevant information for forecasting. In your sequence creation, you excluded these features for the LSTM input but included them in the labels. This approach can work if you want the model to predict all features, including date and hour.
Model Output:
Since your Dense layer outputs 5 features, ensure that this matches your forecasting goals. If you're predicting Vancouver, Vancouver_Celsius, Vancouver_Fahrenheit, date, and hour, ensure the output dimension is appropriate.
Evaluation:
After training, evaluate your model on the test set to assess its performance. Consider using metrics like RMSE or MAE in addition to the loss function to get a clearer picture of model accuracy.
Hyperparameter Tuning:
You might need to experiment with the number of LSTM units, epochs, batch size, or other hyperparameters to optimize model performance.
Feature Engineering:
If you find that the date and hour features do not contribute significantly to the predictions, consider excluding them or transforming them differently.







