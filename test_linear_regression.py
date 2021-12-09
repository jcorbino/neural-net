from nn import *
import matplotlib.pyplot as plt

dataset = np.genfromtxt('salaries.csv', delimiter = ',')
x = np.array(dataset[:, 0], ndmin = 2).T
y = np.array(dataset[:, 1], ndmin = 2).T

plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regression(x, y), color = 'blue')
plt.show()

# Or better use NumPy's polyfit + polyval functions
# coeffs = np.polyfit(x[:, 0], y, 1)
# y = np.polyval(coeffs, x)
