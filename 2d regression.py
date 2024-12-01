#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:25:35 2024

@author: nimarzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 3 * x.squeeze() + np.random.randn(100)  # y = 3*x + noise

# Fit the regression model
model = LinearRegression()
model.fit(x, y)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='data points', s=50, alpha=0.6, edgecolors='w')

# Plot the regression line
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label=r'$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$')

# Add labels and title
plt.xlabel(r'$x$ (regressor)', fontsize=12)
plt.ylabel(r'$y$ (outcome)', fontsize=12)
plt.legend()

plt.savefig("2D_regression.pdf", format='pdf')
plt.show()