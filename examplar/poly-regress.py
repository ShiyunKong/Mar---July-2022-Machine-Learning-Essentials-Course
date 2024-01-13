from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression

def GetPolyData(x,n):    
    return np.sin(x) + 0.1 * rng.randn(n)

rng = np.random.RandomState(1)

n = 250        # elements number
x = list(range(n))
x = [i/100 for i in x]
print(x)

y = GetPolyData(x,n)
train_x = np.array(x)
train_y = np.array(y)

polyModel = PolynomialFeatures(degree = 4)  # play with the degreee
xpol = polyModel.fit_transform(train_x.reshape(-1, 1))
preg = polyModel.fit(xpol,train_y)

linearModel = LinearRegression(fit_intercept = True)
linearModel.fit(xpol, train_y[:, np.newaxis])
polyfit = linearModel.predict(preg.fit_transform(train_x.reshape(-1, 1)))
plt.scatter(train_x, train_y)
plt.plot(train_x, polyfit, color = 'red')
plt.show()
