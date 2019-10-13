# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:35:36 2019

@author: A TULSI SHANKAR REDD
"""

#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4].values

# encoding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[ : , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X= X[ : , 1: ]

#Splitting the data into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#multiple regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set
Y_pred = regressor.predict(X_test)

#building optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
X_opt = X[ : , [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 3, 4, 5]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[ : , [0, 3, 5]]
regressor_OLS = sm.OLS (endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Multiple regression using backward elimination optimization
X_opt_train, X_opt_test, Y_opt_train, Y_opt_test = train_test_split (X_opt, Y, test_size = 0.2, random_state = 0)

regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, Y_opt_train)

Y_pred_opt = regressor_opt.predict(X_opt_test)


#total optimisation
X_opt_final = X[ : , [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt_final).fit()
regressor_OLS.summary()

X_opt_train_final, X_opt_test_final, Y_opt_train_final, Y_opt_test_final = train_test_split (X_opt_final, Y, test_size = 0.2, random_state = 0)

regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train_final, Y_opt_train_final)

Y_pred_opt_final = regressor_opt.predict(X_opt_test_final)