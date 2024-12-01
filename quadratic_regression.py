#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:55:08 2024

@author: nimarzhao
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Step 1: Generate a set of x values
np.random.seed(0)
x = np.linspace(0, 30, 100)

# Step 2: Compute corresponding y values using a quadratic equation with added noise
a, b, c = -0.2, 8, 10
noise = np.random.normal(0, 10, x.shape)
y = a * x**2 + b * x + c + noise

# Step 3: Fit a linear regression model to the data
x_reshaped = x.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(x_reshaped, y)
y_pred = linear_regressor.predict(x_reshaped)

# Step 4: Fit a quadratic regression model to the data
quadratic_featurizer = PolynomialFeatures(degree=2)
quadratic_model = make_pipeline(quadratic_featurizer, LinearRegression())
quadratic_model.fit(x_reshaped, y)
y_quadratic_pred = quadratic_model.predict(x_reshaped)

# Step 2: Plot the original data, the fitted linear regression line, and the fitted quadratic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Quadratic Data with Noise')
plt.plot(x, y_pred, color='red', label=r'$y = \beta_0 + \beta_1 x + e$')
plt.plot(x, y_quadratic_pred, color='green', label=r'$y = \beta_0 + \beta_1 x + \beta_2 x^2 + e$')
plt.title('Quadratic Data with Fitted Linear and Quadratic Regression Lines')
plt.xlabel('District Income', fontsize=16)
plt.ylabel('Test Scores', fontsize=16)
plt.legend()
plt.show()

# Step 5: Plot the original data, the fitted linear regression line, and the fitted quadratic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red', label=r'$y = \beta_0 + \beta_1 x + e$')
plt.plot(x, y_quadratic_pred, color='green', label=r'$y = \beta_0 + \beta_1 x + \beta_2 x^2 + e$')
plt.xlabel('District Income', fontsize=16)
plt.ylabel('Test Scores', fontsize=16)
plt.legend(fontsize=14)
plt.savefig("quadratic_regression.pdf", format='pdf')
plt.show()