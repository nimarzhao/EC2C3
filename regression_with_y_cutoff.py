#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:24:48 2024

@author: nimarzhao
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 3 * x.squeeze() + np.random.randn(100)  # y = 3*x + noise

# Fit the regression model on the full dataset
model_full = LinearRegression()
model_full.fit(x, y)

# Create a y cutoff
y_cutoff = 2

# Filter the data based on the y cutoff
mask = y < y_cutoff
x_filtered = x[mask]
y_filtered = y[mask]

# Fit the regression model on the filtered dataset
model_filtered = LinearRegression()
model_filtered.fit(x_filtered, y_filtered)

# Create the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4))

# Plot for the full dataset
axes[0].scatter(x, y, color='blue', label='data points', s=50, alpha=0.6, edgecolors='w')
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = model_full.predict(x_line)
axes[0].plot(x_line, y_line, color='red', linewidth=2, label=r'$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$')
axes[0].set_xlabel(r'$x$ (regressor)', fontsize=12)
axes[0].set_ylabel(r'$y$ (outcome)', fontsize=12)
axes[0].legend()
axes[0].set_title('Full Dataset')

# Plot for the filtered dataset
axes[1].scatter(x_filtered, y_filtered, color='green', label='filtered data points', s=50, alpha=0.6, edgecolors='w')
y_line_filtered = model_filtered.predict(x_line)
axes[1].plot(x_line, y_line_filtered, color='red', linewidth=2, label=r'$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$')
axes[1].axhline(y=y_cutoff, color='orange', linestyle='--', linewidth=2, label=f'y cutoff = {y_cutoff}')
axes[1].set_xlabel(r'$x$ (regressor)', fontsize=12)
axes[1].set_ylabel(r'$y$ (outcome)', fontsize=12)
axes[1].legend()
axes[1].set_title(f'Dataset with cutoff')

# Set the same y-axis range for both plots
y_min = min(y.min(), y_filtered.min())
y_max = max(y.max(), y_filtered.max())
axes[0].set_ylim([y_min, y_max])
axes[1].set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("regression_with_y_cutoff.pdf", format='pdf')
plt.show()

