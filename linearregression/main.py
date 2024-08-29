"""import matplotlib.pyplot as plt

# Data
temperatures = [15, 20, 25, 30, 35]
rentals = [50, 80, 120, 150, 200]

# Plot
plt.scatter(temperatures, rentals)
plt.xlabel('Temperature (°C)')
plt.ylabel('Number of Rentals')
plt.title('Temperature vs Rentals')
plt.show()
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np



# Data
temperatures = np.array([15, 20, 25, 30, 35]).reshape(-1, 1)
rentals = np.array([50, 80, 120, 150, 200])

# Create model
model = LinearRegression()
model.fit(temperatures, rentals)

# Find slope (m) and intercept (b)
slope = model.coef_[0]
intercept = model.intercept_


# Predictions
predicted_rentals = model.predict(temperatures)  # Predict rentals for the original data

# Calculate MSE
mse = mean_squared_error(rentals, predicted_rentals)  # Compute the MSE
print(f"Mean Squared Error: {mse}")


# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(model, temperatures, rentals, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {abs(cv_scores.mean())}")

# Petrform r2
r_squared = model.score(temperatures, rentals)
print(f"R-squared: {r_squared}")






