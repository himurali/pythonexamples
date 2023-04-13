import numpy as np
import matplotlib.pyplot as plt

# Define the data
ages = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
heights = np.array([40, 44, 46, 48, 50, 52, 55, 58, 61, 63])

# Calculate the slope and y-intercept
m, b = np.polyfit(ages, heights, 1)

print(f"m value is {m}")
print(f"b value is {b}")

# Plot the data and the best-fit line
plt.scatter(ages, heights)
plt.plot(ages, m*ages + b, color='red')
plt.xlabel('Age (years)')
plt.ylabel('Height (inches)')
plt.title('Linear Regression Example')
plt.show()

# Use the best-fit line to make predictions
age = 7
predicted_height = m*age + b
print(f"A {age}-year-old would be about {predicted_height:.2f} inches tall.")
