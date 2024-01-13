import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure()

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, centers=4, random_state=random_state)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title(" Number of Blobs is 4")
plt.show()

