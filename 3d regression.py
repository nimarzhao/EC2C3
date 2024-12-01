#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 05:59:00 2024

@author: nimarzhao
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from matplotlib import cm

# Generate sample data
np.random.seed(0)
x1 = np.random.rand(100)
x2 = np.random.rand(100)
y = 3 * x1 + 2 * x2 + np.random.randn(100)  # y = 3*x1 + 2*x2 + noise

# Prepare the data for regression
X = np.column_stack((x1, x2))

# Fit the regression model
model = LinearRegression()
model.fit(X, y)

# Create the 3D plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
scatter = ax.scatter(x1, x2, y, color='blue', label='Data points', s=50, alpha=0.6, edgecolors='w')

# Create a grid for the plane
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
y_pred = model.intercept_ + model.coef_[0] * x1_grid + model.coef_[1] * x2_grid

# Plot the regression plane with a gradient
surf = ax.plot_surface(x1_grid, x2_grid, y_pred, facecolors=cm.coolwarm((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())), 
                       rstride=10, cstride=10, alpha=0.5, edgecolor='none')

# Add color bar
#mappable = cm.ScalarMappable(cmap=cm.coolwarm)
#mappable.set_array(y_pred)
#fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Predicted y')

# Add labels
ax.set_xlabel(r'$x_1$', fontsize=12, labelpad=10)
ax.set_ylabel(r'$x_2$', fontsize=12, labelpad=10)
ax.set_zlabel(r'$y$', fontsize=12, labelpad=10)


# Add legend
#scatter_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='blue', markerfacecolor='blue', alpha=0.6, markeredgecolor='w')
#ax.legend([scatter_proxy], ['Data points'], numpoints=1, loc='upper left')

# Set axis limits for better visualization
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(y.min() - 1, y.max() + 1)
plt.savefig("3D_regression.pdf", format='pdf')
plt.show()

