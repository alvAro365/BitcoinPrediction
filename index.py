
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import seaborn as seabornInstance 
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn import preprocessing

# df = pd.read_csv('./BTC-USD.csv')
start = datetime.datetime(2010, 1, 1)
end = datetime.date.today()

df = web.DataReader("BTC-USD", 'yahoo', start, end)
print(df.tail())

# High Low Percentage and Percentage Change
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# print(dfreg)

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# Separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# Linear regression
# X = df['Low'].values.reshape(-1,1)
# y = df['High'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

# results
print(f"The linear regression confidence is: {confidencereg}")
print(f"The quadratig regression 2 confidence is: {confidencepoly2}")
print(f"The quadratig regression 3 confidence is: {confidencepoly3}")
print(f"The knn regression confidence is: {confidenceknn}")
 
forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

# print(forecast_set)

# last_date = dfreg.iloc[-1].name 
last_date = datetime.date.today() - datetime.timedelta(days = 1)
last_unix = last_date
# print(dfreg)
# print(last_unix)
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
  next_date = next_unix
  next_unix += datetime.timedelta(days=1)
  dfreg.loc[next_date] = [np.nan for _ in
  range(len(dfreg.columns)-1)]+[i]

  dfreg['Adj Close'].tail(500).plot()
  dfreg['Forecast'].tail(500).plot()
  plt.legend(loc=4)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.show()