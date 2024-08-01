# Project-3-Mahshid-


Sales Prediction Project

Project Overview

This project aims to predict sales using a dataset of orders. Various techniques, including feature engineering, data normalization, advanced deep learning architectures, and hyperparameter tuning, have been applied to improve the model's performance. The primary metrics used to evaluate the model are Mean Absolute Error (MAE) and R-squared (R²).

Problem Statement

Predicting sales accurately is crucial for businesses to ensure efficient supply chain management, inventory control, and strategic planning. Given a dataset with various features related to orders, the objective is to build a model that can predict the sales amount for each order.

Key Challenges

Handling diverse features such as dates, categorical variables, and numerical data.

Ensuring data quality and preprocessing for optimal model performance.

Choosing the right model and tuning it to achieve the best possible accuracy.

Data Complexity: Dealing with multiple variables that affect sales

Seasonality and Trends: Accounting for time-based patterns in sales data

Feature Selection: Identifying the most relevant predictors of sales

Model Performance: Achieving high accuracy in sales forecasts


Solution

The solution involves a comprehensive approach that includes data preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation.

Target Variable

Sales: The target variable representing the sales amount.
Features

Date Features: Extracted from 'Order Date' and 'Ship Date'.
Order_Day, Order_Month, Order_Year
Ship_Day, Ship_Month, Ship_Year

Processing_Time: Difference between 'Ship Date' and 'Order Date'

Categorical Features:
Ship Mode
Segment
Country
City
Region
Category
Sub-Category
Models Used


Initial Simple Neural Network:

A basic architecture with a single hidden layer.
Used ReLU activation function and dropout for regularization.
Evaluated and found that it needed improvement for better accuracy.

Improved Deep Learning Model:
Increased the number of hidden layers and units.
Applied L2 regularization to prevent overfitting.
Used Keras Tuner for hyperparameter optimization.
Expanded hyperparameter tuning ranges and added more trials.


Changes and Enhancements
Dropping Unnecessary Columns:
Dropped the 'Postal Code' column as it was not relevant to our prediction.
Handling Missing Values:
Dropped rows with missing values to ensure data quality.


Feature Engineering:
Extracted additional features from 'Order Date' and 'Ship Date'.
Created a Processing_Time feature to represent the time taken to ship an order.

Data Normalization/Standardization:
Applied StandardScaler to normalize numerical features.

Hyperparameter Tuning:
Used keras_tuner to find the best hyperparameters for the model.
Optimized for number of units in each layer, dropout rates, learning rate, and number of layers.

Regularization Techniques:
Added L2 regularization to prevent overfitting.
Training and Validation:
Implemented early stopping to avoid overfitting and reduce training time.
Increased the number of epochs and set a higher patience value for early stopping.


Model Evaluation
The model performance was evaluated using:

Mean Absolute Error (MAE)
R-squared (R²)
Comparison of Model Accuracy
Model	MAE	R²
Initial Simple Neural Network	872.34	0.65
Improved Deep Learning Model	789.21	0.72
Future Improvements
Implement more advanced feature engineering techniques.
Explore ensemble methods to combine multiple models.
Further expand hyperparameter tuning to find more optimal settings.
Implement additional regularization techniques.
Installation


Project Structure

sales_prediction.py: Main script for data preprocessing, model training, and evaluation.
data/: Directory containing the dataset.
requirements.txt: List of required Python packages.
README.md: Project documentation.
Dependencies

pandas
scikit-learn
tensorflow
keras_tuner
matplotlib
Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the Issues if you want to contribute.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgements

pandas for data manipulation and analysis.
scikit-learn for data preprocessing and machine learning tools.
tensorflow and keras_tuner for deep learning and hyperparameter optimization.
matplotlib for visualization.
