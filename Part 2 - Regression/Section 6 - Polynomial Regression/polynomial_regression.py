import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)

X_poly = poly.fit_transform(X)

linear = LinearRegression()
linear.fit(X_poly, y)

y_pred = linear.predict(poly.fit_transform(6.5))

plt.scatter(X, y, color='red')
plt.plot(X, linear.predict(X_poly), color='blue' ) #linear.predict(poly.fit_transform(X))
plt.title('Polynomial Linear Regression')
plt.xlabel('Posiiton / Level of Employee')
plt.ylabel('Salary')
plt.show()