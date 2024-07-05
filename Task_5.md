# Sales Prediction Using Python
## Description
In order to improve overall sales performance and profitability, the goal of this sales prediction task is to create a machine learning model that can predict future sales with high accuracy using historical sales data along with other pertinent variables like advertising expenditure, customer demographics, and advertising platforms. By comprehending the complex relationships between these variables and sales outcomes, businesses can optimize their advertising strategies, allocate resources efficiently, and make data-driven decisions. Proper inventory management, budgeting, and strategic planning are all made possible by this predictive capability, which in turn improves customer satisfaction and boosts market competitiveness.
## Key concepts and Challenges
The selection of features, models, assessment metrics, and data pretreatment are important ideas in machine learning-based sales prediction. To guarantee quality and consistency, raw data must be cleaned and transformed as part of the data preprocessing step. Model selection includes selecting the right methods, such as neural networks, decision trees, or linear regression, whereas feature selection concentrates on finding the most pertinent elements impacting sales. The accuracy of the model is evaluated using metrics such as Mean Absolute Error (MAE) and R-squared. Managing the complexity of high-dimensional data, preventing overfitting, handling missing or noisy data, and making sure the model stays accurate over time as customer behavior and market conditions change are some of the challenges this task presents. It can also be difficult to incorporate the model into current business procedures and to keep adding new data to it on a regular basis
## Learning Objectives
Data Preprocessing Skills
Proficiency in Machine Learning Algorithms
Model Evaluation Techniques
# src/main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Creating a DataFrame from the provided data
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8, 66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3, 218.4, 237.4, 13.2, 228.3, 62.3, 262.9, 142.9, 240.1, 248.8, 70.6, 292.9, 112.9, 97.2, 265.6, 95.7, 290.7, 266.9, 74.7, 43.1, 228, 202.5, 177, 293.6, 206.9, 25.1, 175.1, 89.7, 239.9, 227.2, 66.9, 199.8, 100.4, 216.4, 182.6, 262.7, 198.9, 7.3, 136.2, 210.8, 210.7, 5…
Advertising.csv
[12:19, 7/5/2024] Akhila M: import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Step 1: Load the dataset
data = pd.read_csv('advertising.csv')

## Step 2: Prepare the data
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']  # Target variable

## Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Step 4: Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

## Step 5: Predict sales on the testing set
y_pred = model.predict(X_test)

## Step 6: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

## Optional: Plot the results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
![Example Image](images/ft5.jpg)
