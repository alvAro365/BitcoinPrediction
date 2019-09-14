import csv # get data from google stock file
import numpy as np # data formatting
from sklearn import linear_model # regression line
import matplotlib.pyplot as plt #visualization

# dates = np.array([], dtype=np.float)
dates = []
prices = []

def get_data(filename):
  with open(filename, 'r') as csvfile:
    csvFileReader = csv.reader(csvfile)
    next(csvFileReader)
    for row in csvFileReader:
      dates.append(int(row[0].split('-')[2]))
      prices.append(float(row[1]))
  return

def show_plot(dates,prices):
  linear_mod = linear_model.LinearRegression()
  dates = np.reshape(dates,(len(dates), 1))
  prices = np.reshape(prices,(len(prices), 1))
  linear_mod.fit(dates,prices)
  plt.scatter(dates,prices,color='yellow')
  plt.plot(dates,linear_mod.predict(dates), color='blue',linewidth=3)
  plt.show()
  return

def predict_price(dates,prices,x):
  # linear_mod = linear_model.LinearRegression()
  # linear_mod = linear_model.Ridge(alpha=.5)
  linear_mod = linear_model.Lasso(alpha=0.1)
  dates = np.reshape(dates,(len(dates), 1))
  prices = np.reshape(prices,(len(prices), 1))
  linear_mod.fit(dates, prices)
  predict_price = linear_mod.predict(x)
  confidence = linear_mod.score(dates, prices)
  return predict_price[0], linear_mod.coef_[0], linear_mod.intercept_[0], confidence
  # return predict_price[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]

get_data('BTC-USD.csv')
print(dates)
print(prices)

# show_plot(dates,prices)
date = 29
predict_price, coefficient, constant, confidence = predict_price(dates, prices, [[date]])
print(f'The stock open price is: {predict_price}')
print(f'The regression coefficent is {coefficient}, and the constant is {constant}')
print(f'The regression confidence is {confidence}')
