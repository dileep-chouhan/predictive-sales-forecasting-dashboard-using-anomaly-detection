import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2022-01-01', periods=365)
sales = 1000 + 200 * np.sin(2 * np.pi * np.arange(365) / 365) + 50 * np.random.randn(365) #seasonal trend + noise
outliers = [100, 150, 200, 300] #indices of outliers
sales[outliers] += 1000 #introduce outliers
df = pd.DataFrame({'Date': dates, 'Sales': sales})
# --- 2. Anomaly Detection ---
# Simple moving average for smoothing
window_size = 7
df['SMA'] = df['Sales'].rolling(window=window_size,center=True).mean()
df['Anomaly'] = abs(df['Sales'] - df['SMA']) > 200 #Threshold for anomaly detection. Adjust as needed.
# --- 3. Sales Forecasting (Simple Linear Regression for demonstration)---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Prepare data for forecasting
df['Day'] = (df['Date'] - df['Date'].min()).dt.days
X = df[['Day']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
df['Forecast'] = np.nan
df.loc[X_test.index,'Forecast'] = predictions
# --- 4. Visualization ---
#Plot 1: Sales with Anomalies
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Sales')
plt.plot(df['Date'], df['SMA'], label='SMA')
plt.scatter(df.loc[df['Anomaly'],'Date'], df.loc[df['Anomaly'],'Sales'], color='red', label='Anomalies')
plt.title('Sales with Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.savefig('sales_anomalies.png')
print("Plot saved to sales_anomalies.png")
#Plot 2: Sales Forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Actual Sales')
plt.plot(df['Date'], df['Forecast'], label='Forecast', linestyle='--')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.savefig('sales_forecast.png')
print("Plot saved to sales_forecast.png")
#Plot 3: Sales Distribution (with KDE)
plt.figure(figsize=(8,6))
sns.histplot(df['Sales'], kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.savefig('sales_distribution.png')
print("Plot saved to sales_distribution.png")