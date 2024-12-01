#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 03:06:10 2024

@author: nimarzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.random.rand(100, 1)

# Homoskedastic errors
y_homo = 3 * x.squeeze() + np.random.randn(100) * 0.5  # y = 3*x + noise with constant variance

# Heteroskedastic errors
y_hetero = 3 * x.squeeze() + np.random.randn(100) *2*( x.squeeze())  # y = 3*x + noise with increasing variance

# Fit the regression models
model_homo = LinearRegression()
model_homo.fit(x, y_homo)
y_line_homo = model_homo.predict(np.linspace(0, 1, 100).reshape(-1, 1))

model_hetero = LinearRegression()
model_hetero.fit(x, y_hetero)
y_line_hetero = model_hetero.predict(np.linspace(0, 1, 100).reshape(-1, 1))

# Create the plots
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# Plot for homoskedastic errors
axes[0].scatter(x, y_homo, color='green', label='Data points', s=50, alpha=0.6, edgecolors='w')
axes[0].plot(np.linspace(0, 1, 100), y_line_homo, color='red', linewidth=2, label='Regression line')
axes[0].set_xlabel(r'$x$ (Regressor)', fontsize=12)
axes[0].set_ylabel(r'$y$ (Outcome)', fontsize=12)
axes[0].set_title('Homoskedastic Errors')
axes[0].legend()

# Plot for heteroskedastic errors
axes[1].scatter(x, y_hetero, color='gray', label='Data points', s=50, alpha=0.6, edgecolors='w')
axes[1].plot(np.linspace(0, 1, 100), y_line_hetero, color='red', linewidth=2, label='Regression line')

# Add lines to show increasing variance
x_range = np.linspace(0, 1, 100)
y_upper = 3 * x_range + 2 * x_range  # y = 3*x + 2*x (upper bound)
y_lower = 3 * x_range - 2 * x_range  # y = 3*x - 2*x (lower bound)
axes[1].plot(x_range, y_upper, color='blue', linestyle='dashed', linewidth=1, label='increasing error variance')
axes[1].plot(x_range, y_lower, color='blue', linestyle='dashed', linewidth=1)

axes[1].set_xlabel(r'$x$ (Regressor)', fontsize=12)
axes[1].set_ylabel(r'$y$ (Outcome)', fontsize=12)
axes[1].set_title('Heteroskedastic Errors')
axes[1].legend()


y_min = min(y_homo.min(), y_hetero.min())
y_max = max(y_homo.max(), y_hetero.max())
axes[0].set_ylim([y_min, y_max])
axes[1].set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("homoskedastic_vs_heteroskedastic.pdf", format='pdf')
plt.show()
