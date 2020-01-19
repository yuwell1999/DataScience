import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM

iris = datasets.load_iris()
cov_data = np.corrcoef(iris.data.T)

print(iris.feature_names)
print(cov_data)

img = plt.matshow(cov_data, cmap=plt.cm.rainbow)
plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)

for x in range(cov_data.shape[0]):
    for y in range(cov_data.shape[1]):
        plt.text(x, y, "%0.2f" % cov_data[x, y],
                 size=12, color='black', ha="center", va="center")

plt.show()

####################### 主成分分析 ######################
pca_2c = PCA(n_components=2)

X_pca_2c = pca_2c.fit_transform(iris.data)
print(X_pca_2c.shape)

plt.scatter(X_pca_2c[:, 0], X_pca_2c[:, 1], c=iris.target,
            alpha=0.8, s=60, marker='o', edgecolors='white')
plt.show()
print(pca_2c.explained_variance_ratio_.sum())

print(pca_2c.components_)

pca_2cw = PCA(n_components=2, whiten=True)
X_pca_1cw = pca_2cw.fit_transform(iris.data)

plt.scatter(X_pca_1cw[:, 0], X_pca_1cw[:, 0], c=iris.target,
            alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()
print(pca_2cw.explained_variance_ratio_.sum())

# PCA变型————RandomizedPCA
rpca_2c = PCA(n_components=2, svd_solver='randomized')
X_rpca_2c = rpca_2c.fit_transform(iris.data)
plt.scatter(X_rpca_2c[:, 0], X_rpca_2c[:, 1], c=iris.target,
            alpha=0.8, s=60, marker='o', edgecolors='white')
plt.show()
print(rpca_2c.explained_variance_ratio_.sum())

# Restricted Boltmann Machine
n_components = 64
olivetti_faces = datasets.fetch_olivetti_faces()
X = preprocessing.binarize(preprocessing.scale(olivetti_faces.data.astype(float)), 0.5)
rbm = BernoulliRBM(n_components=n_components, learning_rate=0.01, n_iter=100)
rbm.fit(X)
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(int(np.sqrt(n_components + 1)),
                int(np.sqrt(n_components)), i + 1)
    plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

plt.subplot(str(n_components),64, fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()
