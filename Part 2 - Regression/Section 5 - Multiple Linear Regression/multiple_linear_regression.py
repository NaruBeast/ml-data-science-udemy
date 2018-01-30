import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_X = LabelEncoder()
X[:,3] = label_X.fit_transform(X[:,3])

encoder = OneHotEncoder(categorical_features=[3])
X = encoder.fit_transform(X).toarray()

X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

"""plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Multiple Linear Regression')"""
