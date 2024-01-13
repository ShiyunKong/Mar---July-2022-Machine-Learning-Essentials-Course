import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),cmap='binary', interpolation='nearest', clim=(0, 16))
    plt.show()

# get the data 
digits = load_digits()
#plot it
plot_digits(digits.data)
#add noise
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)
# filter noise wit PCA 
pca = PCA(0.5).fit(noisy)
print("PCA components\n", pca.n_components_)
#reconstruct
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)



