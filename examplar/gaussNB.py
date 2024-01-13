import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

# first need data. fake data or read from csv file
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=2.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');#
plt.show()

# model definition
model = GaussianNB()

# model fitting based on train data
model.fit(X,y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
##print("shape",Xnew.shape)
##print(Xnew[0:5,:])

# checking the model on new data - test data
ynew = model.predict(Xnew)
print(ynew)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()
yprob = model.predict_proba(Xnew ) # predicting probabilities for each label

print(yprob[0:10].round(2))

