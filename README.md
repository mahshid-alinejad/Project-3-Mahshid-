# Project-3-Mahshid-


Predicting weather using LSTM neural networks
LSTM Implementation
Predicting weather using LSTM neural networks
Project members: Cassidy Liu and Andrew Yu

Abstract
Weather and temperature is not only an important for individual everyday life, but is also relevant to many other topics such as climate change. Studies from other researchers have shown promising results of using neural networks for predicting and future weather and/or temperature. This project creates a Long Short Term Memory model to predict future temperature. The model takes in historical weather information such as air pressure and temperature and predicts future temperatures. The model we created shows promising results in predicting the sequence when comparing the prediction and the actual temperature at that time.

Introduction
This report explores the use of neural networks in predicting temperature in the future. It will be evaluating the benefits and costs of using neural networks as well as the feasibility and other factors that need to be considered.

Part of what makes this project possible is the availability to detailed, real-time weather data. Modern data science and technology endeavors have allowed us to get accurate data from anywhere in the world. However, there may be varying data availablility for training the neural network, which can impact its accuracy.

Predicting weather patterns are of interest for individuals, weather forecasters, and other weather-related concerns such as energy modeling and daylight analysis. With the growth of machine learning, using neural networks to predict future weather patterns has become more popular. Weather predictions and gathering the predictions’ relevant data is something that has long existed before using, so weather programs already have a variety of ways to predict weather. Our project will aim to mainly use historical weather data to predict future temperature with recurrent neural networks.

One difficulty in predicting temperature in the long run is the addition of global warming into the equation. Global warming has also become a major concern for a lot of stakeholders of weather. Current weather prediction algorithms and equations may also be impacted with this addition as well.

Current meteorologists use numerical weather prediction models, which solve complex sets of mathematical equations based on the physics of air flow, and heat and moisture exchange, etc. This project focuses on implementing a type of recurrent neural network (RNN), Long Short-Term Memory (LSTM), with the use of historical weather data to make weather predictions.

Long Short-Term Memory is a type of recurrent neural network that has been shown to excel at analyzing data sequentially.1 Since weather data is inherently sequential in nature, LSTM has become a popular approach for predicting weather patterns. LSTM models are capable of capturing the temporal dependencies in weather data, which makes them ideal for predicting complex weather phenomena. With the availability of vast amounts of historical weather data and recent advancements in deep learning techniques, LSTM models have become increasingly popular in weather forecasting. This technology has the potential to revolutionize how weather forecasts are made and can provide valuable insights for decision-making in various fields, including agriculture, aviation, and disaster management. In this context, this study will also explore the benefits and challenges associated with using LSTM for weather prediction and discuss the latest developments in this field.

One of challenges we anticipate having to face is having factors that affect the temperature that are not in our dataset. There may be certain geological structures that are impacting how temperature affects a certain (these patterns might be able to be learned in the model). There are also other obscure factors that may affect a place’s temperature this project won’t be considering (out of scope). For example, if a volcano erupts or an urban city has a lot more pollution than normal, these may affect the temperature and will not show up in the dataset in a way that other weather phenomenons would.

Ethical Sweep
Since weather forecasting is already really commonly done (with different methods), there aren’t many more downsides to using neural networks to predict weather. Generally, there aren’t many negative uses of weather data, although a potential concern is that the use of the weather data needed may have varying availability in different areas (so there might limited serviceability). Our model will only help further make different weather prediction services more accessible to people. Weather predictions are important on an individual level for planning and can also be used for resource management and agricultural applications.

Our project team originally had four people, but we cut down the team due to logistical. The current team only has two of us with similar ethnic backgrounds but different academic backgrounds (coming from different colleges and different professions), so that has helped us have more diverse discussions about how.

The addition of using neural networks also usually adds the question of data ethics. A lot of neural networks uses personal information and can cause issues with data privacy. Our web application will take location information that the user provides but does not save this information about the user. From the developers perspective, there will be no way of discerning which user has looked up what location’s weather prediction. Data bias from this data set can be seen from what data was included and excluded. There are more physical factors that could factor into a place’s temperature. For example, there is no data included that is relevant to greenhouse gases and the global warming impact on temperature.

One misinterpretation of results could be that for a lot of users that aren’t familiar with neural nets is that the decision process can seem like a black box and not backed on a real model that was trained by real historical data.

Methods
We utilize numpy and pandas to gather and scrape data from this kaggle dataset. Below is the plot of the temperature data of San Diego from 2012-2017.

image

The data contains hourly weather data for 30 US & Canadian Cities and 6 Israeli Cities from 2012-2017.

imageExample of hourly temperature data for Vancouver, Portland, and San Francisco

imageExample of varying availability in hourly air pressure data for Vancouver, Portland, and San Francisco

We specifically selected 4 files to use from the given kaggle link: humidity, pressure, temperature, and wind speed as we thought those were the main features that would most likely contribute to the temperature. The remaining fields are city attributes, weather description, and wind direction. Weather description data won’t be useful with the model that we have built. The focus of our model is on modeling future temperature based on weather history, so city attributes and wind direction are arbitrary for a location.

We then preprocessed the data by taking the standard deviations of each of the features so the ranges of the value will be minimized to mainly between -3 and 3. And we made 80% of the data to be used to train the model, while the remaining 20% were to be used in the validation process.

The model has a function called multivariate_data() that does a rolling window approach to prepare mulitvariate time series data fro LSTM training. The model has a window size of 48 hours and then rolling window of that size slides along the time series. For each window, we use data within the window as input to the LSTM model and use the output to predict the next value in the series. We then repeat for each subsequent window in the time series.

The result is a set of input/output pairs that can be used for training an LSTM model. Each input/output pair corresponds to a sliding window of the time series data, with the input containing history_size (48 hours) time steps and the output containing target_size (24 hours) time steps. Therefore, the multivariate_data() function can be considered an implementation of the rolling window approach for preparing multivariate time series data for LSTM training.

We then used the frameworks TensorFlow and Keras to create an LSTM (a type of RNN) in python to be trained on the preprocessed data. Specifically, the code defined a multi-step LSTM model using Keras and TensorFlow where it looked back and used data from 48 hours before to predict data 24 hours into the future.

The code was helped with this tutorial. We also partially utilized how data is preprocessed through this blog. However we added upon the data analysis as it only made only a singular prediction rather than multiple predictions that we do in our code. We made it a multi-setp model rather than a single step from the model that we referenced. Our model predicts a sequence of points in the future whereas our reference only predicts a single future point.

Related Works
Kosandal’s article is a good introduction in using recurrent neural networks in predicting weather.2 It runs through an example where it takes in a dataset containing historical temperature, wind speed, wind gust, etc. time series data and uses Python libraries (Pandas, Numpy, Scikit-learn, and Keras) to preprocess, train and run a model that will predict future temperatures.

An article by Tran et. al discusses the uses of neural networks to predict weather data.3 The review shows that LSTMs and RNNs are effective tools to predict weather. Reviewed models include a variety of univariate and multivariate FFNN (feed-forward neural network), FFBF (feed-forward back propagation), GRNN (generalized regression neural network), RBF (radial basis function), CRNN (convlutional recurrent neural network), RNN, and LSTM models.

Another article explores the use of a simple recurrent neural network with convolutional filters, ConvLSTM, and an advanced generative network, the Stochastic Adversarial Video Prediction (SAVP) model.4 The model predicts hourly forecasts for places in Europe using 13 years of historical weather data in Europe. Evaluation show promising results in terms of MSE for SAVP and ConvLSTM models.

A Washington Post article by Deaton explains new research that indicates how global warming will become a bigger factor in making weather predictions.5 While right now, the global warming could just be a small source of error, this error could be compounded and cause bigger issues in weather models as time goes on. Additionally, the impact of global warming will only increase as greenhouse gas emissions continue to increase.

There are also many metereologists that study climatic trends using bottom-up physics-based general circulation and Earth system model approache.6 A study conducted by Ise and Oba detail a top-down approach by training a neural network and a huge dataset about historical global temperature. They were able to obtain 97.0% accuracy by using a LeNet convolutional neural network. The conclusion is that weather forecasting methods should include both a neural network component as well as a numerical method with the traditional physics-based model (using natural physical laws to determine expected temperature).

Discussion
Our code can be found here.
The final project presents a user interface that will give the predicted temperature at a given location (initially will only have a couple options). Currently, our team has a working model, and we are working on having it deployed. Data is evaluated with its accuracy to the actual reading after the predictions. The LSTM model we have is comparable to other atmospheric temperature neural networks; our team drew inspiration from many existing weather models as well as other LSTMs that have similar prediction models. The dataset for this model is a 2012-2017 historical weather dataset that includes field such as historical temperature, air pressure, humidity, city attributes, and wind speed and direction. Below is the interface of our website. image Screenshot of our Gradio interface for predicting temperatures given a cities coordinate.

The reason why we chose an LSTM over other types of neural networks is not only because they can learn long term dependencies in the data, but also aren’t suceptible to the vanishing gradient problem like other RNNs are.

In our case, the LSTM we chose to use an Adam optimizer with a learning rate of 1e-3. The model also utilized the standard MSE function. In addition, we only used data from San Diego from the data compiled in the kaggle dataset. For now, we had a batch size of 256 since our dataset is large and we would like for the model to train efficiently and fast. We also used the tanh activation function as it provided the lowest training loss of all the ones we tried (sigmoid, ReLu, etc.). We also had a lookback of 48 hours, and had the model predict 24 hours into the future (predicted 24 data points).

Model loss
Above is the training loss and validation loss of the model on the San Diego hourly weather data from 2012 to 2017.
Example Prediction
Above is some example of the predictions the model makes. We can see here that the model uses the past 48 hours of data, and predicts 24 hours (shown on the x axis) into the future the temperature in Celcius. We can see the predictions are similiar to the true values of the temperature.
Example Prediction
Another example of the predictions the model makes with temperature in Celcius.
Reflection
The team started this project late because this team decided to split off from the stock market prediction team a little late. If the team were to restart, it would be useful to spread the tasks out over a longer period of time.

We could have also spent more time trying out different models. In our research, we have seen success in the use of RNNs such as ConvLSTM or FFNNs. Continue work on the project would be creating a better user interface as well as getting more recent data, since the model is currently trained on 2012-2017 data.

Another area to expand on in the future is utilizing more complex data to get more accurate information. This would include more obscure data such as elevation, distance from geographical landmarks, weather descriptions (descriptive data), and wind direction.

Also the integration of a physics-based component (that relies on current weather information) in addition to the LSTM that works purely on historical data.

We would also switch to a different library than gradio to deploy our LSTM model. This is because gradio wasn’t very customizeable, and we weren’t able to make the user interface look nice. Thus, in the future, we plan to switch to another framework such as flask.

References
Banoula, Mayank. “Introduction to Long Short-Term Memory(Lstm): Simplilearn.” Simplilearn.com, Simplilearn, 27 Apr. 2023, https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/lstm#:~:text=LSTMs%20are%20able%20to%20process,problem%20that%20plagues%20traditional%20RNNs. ↩

Kosandal, Rohan. “Weather Forecasting with Recurrent Neural Networks.” Medium, Analytics Vidhya, 5 Jan. 2020, https://medium.com/analytics-vidhya/weather-forecasting-with-recurrent-neural-networks-1eaa057d70c3. ↩

Tran, Trang Thi Kieu, et al. “A Review of Neural Networks for Air Temperature Forecasting.” Water, vol. 13, no. 9, May 2021, p. 1294. Crossref, https://doi.org/10.3390/w13091294. ↩

B. Gong, et al. “Temperature Forecasting by Deep Learning Methods.” Vol. 15, 2022, pp. 8931–8956. https://doi.org/10.5194/gmd-15-8931-2022. Accessed 15 Apr. 2023. ↩

Deaton, Jeremy. “Climate Change Could Make Weather Harder to Predict.” The Washington Post, WP Company, 25 Jan. 2022, https://www.washingtonpost.com/weather/2022/01/25/climate-change-weather-unpredictable/. ↩

Ise T and Oba Y (2019) Forecasting Climatic Trends Using Neural Networks: An Experimental Study Using Global Historical Data. Front. Robot. AI 6:32. doi: 10.3389/frobt.2019.00032 ↩
