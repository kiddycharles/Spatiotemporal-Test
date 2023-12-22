import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('resultData.csv', header=[0, 1])

# Extracting the 'Close' prices of 'GOOGL' and 'AAPL'
googl_close = data['Close']['GOOGL']
appl_close = data['Close']['AAPL']

# Creating a DataFrame with both series
df = pd.concat([googl_close, appl_close], axis=1)
df.columns = ['GOOGL', 'AAPL']

# Split the dataset into train and test sets
n_obs = int(len(df) * 0.8)
train, test = df[0:n_obs], df[n_obs:]

# Define and fit the VAR model
lags = 5  # Adjust as per model selection criteria
model = VAR(train)
model_fit = model.fit(lags)

# Make predictions
# Use the last 'lags' observations from the training data to forecast
last_obs = train.values[-lags:]
predictions = model_fit.forecast(last_obs, steps=len(test))


# Evaluate the model
test_values = test.values
mse = mean_squared_error(test_values, predictions)
rmse = np.sqrt(mse)
print('Test RMSE: %.3f' % rmse)

# Plotting the results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))

axes[0].plot(train.index, train['GOOGL'], label='Train (GOOGL)')
axes[0].plot(test.index, test['GOOGL'], label='Test (GOOGL)')
axes[0].plot(test.index, predictions[:, 0], label='Predicted (GOOGL)', color='red')
axes[0].set_title('GOOGL Stock Price Prediction using VAR Model')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend()

axes[1].plot(train.index, train['AAPL'], label='Train (AAPL)')
axes[1].plot(test.index, test['AAPL'], label='Test (AAPL)')
axes[1].plot(test.index, predictions[:, 1], label='Predicted (AAPL)', color='red')
axes[1].set_title('AAPL Stock Price Prediction using VAR Model')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price')
axes[1].legend()

plt.tight_layout()
plt.show()
