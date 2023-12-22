import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('resultData.csv', header=[0, 1])

# Extracting the 'Close' price of 'GOOGL'
googl_close = data['Close']['GOOGL']

# Split the dataset into train and test sets
n_obs = int(len(googl_close) * 0.8)
train, test = googl_close[0:n_obs], googl_close[n_obs:]

# Define and fit the AR model
lags = 5  # You can adjust this based on your model selection criteria
model = AutoReg(train, lags=lags)
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Evaluate the model
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('Test RMSE: %.3f' % rmse)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.title('GOOGL Stock Price Prediction using AR Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
