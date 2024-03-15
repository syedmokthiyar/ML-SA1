  # ML-SA1
# Register No: 212222230156
# Developed By: Syed Mokthiyar S.M

# Q1. Create a scatter plot between cylinder vs Co2Emission (green color)
# Program:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("FuelConsumption.csv")
df = pd.read_csv("FuelConsumption.csv")
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.title('CYLINDER vs CO2EMISSION')
plt.show()
```
# Output:
![Screenshot 2024-03-15 205759](https://github.com/syedmokthiyar/ML-SA1/assets/118787294/263a738c-99ad-4ec3-a56a-55c67e4a83e7)

# Q2. Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
# Program:
```python
# Scatter plot for cylinder vs Co2Emission (green color)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='CYLINDERS vs CO2EMISSIONS')

# Scatter plot for Enginesize vs Co2Emission (blue color)
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='ENGINESIZE vs CO2EMISSIONS')

plt.xlabel('FEATURE')
plt.ylabel('CO2EMISSIONS')
plt.title('Comparison between Cylinder and Enginesize vs Co2Emission')
plt.legend()
plt.show()
```
# Output:
![Screenshot 2024-03-15 210819](https://github.com/syedmokthiyar/ML-SA1/assets/118787294/f44b64d0-5ad9-4a44-8ad6-50fc9616f41f)

# Q3. Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
# Program:
```python
# Scatter plot for cylinder vs Co2Emission (green color)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder vs Co2Emission')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Enginesize vs Co2Emission')

# Scatter plot for FuelConsumption_comb vs Co2Emission (red color)
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red', label='FuelConsumption_comb vs Co2Emission')
plt.xlabel('Feature')
plt.ylabel('Co2Emission')
plt.title('Comparison between Cylinder, Enginesize, and FuelConsumption_comb vs Co2Emission')
plt.legend()
plt.show()
```
# Output:
![Screenshot 2024-03-15 211324](https://github.com/syedmokthiyar/ML-SA1/assets/118787294/4839f064-b6eb-41db-b0cf-1c7d48554e3d)

# Q4. Train your model with independent variable as cylinder and dependent variable as Co2Emission
```python
# Extracting independent and dependent variables
X = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```
# Output:
![Screenshot 2024-03-15 212118](https://github.com/syedmokthiyar/ML-SA1/assets/118787294/cadcbad5-f559-489d-b3e6-9cef65448174)


# Q5.in your model on different train test ratio and train the models and note down their accuracies
# Program:
```python
# Train models with different train-test ratios and note down accuracies
ratios = [0.2, 0.3, 0.4, 0.5]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Train-Test Ratio: {1-ratio}:{ratio}, MSE: {mse}, R^2 Score: {r2}")
```
# Output:
![Screenshot 2024-03-15 211840](https://github.com/syedmokthiyar/ML-SA1/assets/118787294/efc0c827-288f-4e87-bf3e-85ccb826debe)





