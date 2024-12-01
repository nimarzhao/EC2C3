#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 01:42:15 2024

@author: nimarzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.random.rand(100, 1)
y_true = 3 * x.squeeze()  # True relationship y = 3*x
y = y_true + np.random.randn(100)  # y = 3*x + noise (observed)

# Introduce classical measurement error in y
measurement_error = np.random.normal(0, 1, size=y.shape)  # Assuming normal error with mean=0 and std=1
y_error = y + measurement_error

# Fit the regression model on the original dataset
model_original = LinearRegression()
model_original.fit(x, y)

# Fit the regression model on the dataset with measurement error
model_error = LinearRegression()
model_error.fit(x, y_error)

# Create the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Plot for the original dataset
axes[0].scatter(x, y, color='blue', label='data points', s=50, alpha=0.6, edgecolors='w')
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = model_original.predict(x_line)
axes[0].plot(x_line, y_line, color='red', linewidth=2, label=r'$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$')
axes[0].set_xlabel(r'$x$ (regressor)', fontsize=12)
axes[0].set_ylabel(r'$y$ (outcome)', fontsize=12)
axes[0].legend()
axes[0].set_title('Original Dataset')

# Plot for the dataset with measurement error
axes[1].scatter(x, y_error, color='green', label='data points with error', s=50, alpha=0.6, edgecolors='w')
y_line_error = model_error.predict(x_line)
axes[1].plot(x_line, y_line_error, color='red', linewidth=2, label=r'$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$')
axes[1].set_xlabel(r'$x$ (regressor)', fontsize=12)
axes[1].set_ylabel(r'$y$ (outcome)', fontsize=12)
axes[1].legend()
axes[1].set_title(r'Dataset with Classical Error in $y$')

# Set the same y-axis range for both plots
y_min = min(y.min(), y_error.min())
y_max = max(y.max(), y_error.max())
axes[0].set_ylim([y_min, y_max])
axes[1].set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("regression_with_classical_y.pdf", format='pdf')
plt.show()