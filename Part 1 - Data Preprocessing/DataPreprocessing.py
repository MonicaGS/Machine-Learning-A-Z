# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:58:08 2018

@author: monica g
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the input dataset
data = pd.read_csv("Data.csv")

# Independent and dependent variable vector
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1:].values

# Handle missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_X.fit_transform(Y)

# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=0)

# Feature Scaling













