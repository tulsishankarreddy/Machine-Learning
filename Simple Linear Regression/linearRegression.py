# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:37:03 2019

@author: A TULSI SHANKAR REDD
"""

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 1].values

# splitting the training set into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# fitting simple linear reggression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set result
Y_pred = regressor.predict(X_test)

# visualisin the training set
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experiance (training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# visulaising the test set
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experiance (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()