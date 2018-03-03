import pandas as pd
import matplotlib as plt
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X_1 = LabelEncoder()
X[:,1] = le_X_1.fit_transform(X[:,1])

le_X_2 = LabelEncoder()
X[:,2] = le_X_1.fit_transform(X[:,2])

ohe_X_1 = OneHotEncoder(categorical_features=[1])
X = ohe_X_1.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


