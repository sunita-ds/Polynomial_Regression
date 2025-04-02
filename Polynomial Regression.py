# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset=pd.read_csv("/Users/apple/Desktop/emp_sal.csv")

X=dataset.iloc[:,1:2].values

Y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Simple linear regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

m=lin_reg.coef_
print(m)

c=lin_reg.intercept_
print(c)

lin_model_pred=lin_reg.predict([[6.5]])
print(lin_model_pred)

# Polynomial Regression Model

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=5)

poly_x=poly_reg.fit_transform(X)

poly_reg.fit(poly_x,Y)

lin_reg_2=LinearRegression()

lin_reg_2.fit(poly_x,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title("polynomial regression model")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

