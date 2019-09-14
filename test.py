import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics

dataset = pd.read_csv('./BTC-USD.csv')
X = dataset['Low'].values.reshape(-1,1)
y = dataset['High'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
# print(linear.intercept_)

#For retrieving the slope:
# print(linear.coef_)

y_pred = linear.predict(X_test)

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
linearPerformance = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Linear Regression')
print('Root Mean Squared Error:', linearPerformance)

# Ridge Regression
ridge = Ridge(alpha=.5)
ridge.fit(X_train, y_train)

r_pred = ridge.predict(X_test)

ridgePerformance = np.sqrt(metrics.mean_squared_error(y_test, r_pred))
print('Ridge Regression')
print('Root Mean Squared Error:', ridgePerformance)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train) #training the algorithm
lasso.predict(X_test)

l_pred = lasso.predict(X_test)

lassoPerformance = np.sqrt(metrics.mean_squared_error(y_test, l_pred))
print('Lasso Regression')
print('Root Mean Squared Error:', lassoPerformance)

performances = [linearPerformance, ridgePerformance, lassoPerformance]
best_performance = min(performances)
print(f"Best performance: {best_performance}")

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(df)

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle=':', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
