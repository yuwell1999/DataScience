from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# In[1]:
# mnist = fetch_mldata("MNIST original")
# pickle.dump(mnist, open( "mnist.pickle", "wb"))

# In[2]:
# target_page = ""

# In[3]:
boston = load_boston()
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)

regr = LinearRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)

from sklearn.metrics import mean_absolute_error

print("MAE", mean_absolute_error(Y_test, Y_pred))

# In[4]:
# logistic regression
# 处理二分类问题
import numpy as np

avg_price_house = np.average(boston.target)

high_price_idx = (Y_train >= avg_price_house)

Y_train[high_price_idx] = 1
Y_train[np.logical_not(high_price_idx)] = 0

Y_train = Y_train.astype(np.int8)

high_price_idx = (Y_test >= avg_price_house)

Y_test[high_price_idx] = 1
Y_test[np.logical_not(high_price_idx)] = 0

Y_test = Y_test.astype(np.int8)

# In[5]:
# 输出分类报告
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

# In[6]:
# 朴素贝叶斯
from sklearn import datasets

iris = datasets.load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

# In[7]:
# KNN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

mnist = pickle.load(open("mnist.pickle"), "rb")
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

mnist.data = mnist.data[:1000]
mnist.target = mnist.target[:1000]

X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=0)

# In[8]:
########################### 非线性算法 ####################################
from sklearn.datasets import load_svmlight_file

X_train, Y_train = load_svmlight_file(r'C:\Users\YuYue\PycharmProjects\Learning\ijcnn1.bz2')
first_rows = 2500

# In[9]:
# 无监督学习
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

N_samples = 2000

datasets_1 = np.array(datasets.make_circles(n_samples=N_samples, noise=0.05, factor=0.3)[0])
datasets_2 = np.array(datasets.make_blobs(n_samples=N_samples, centers=4, cluster_std=0.4, random_state=0)[0])

plt.scatter(datasets_1[:, 0], datasets_1[:, 1], alpha=0.8, s=64, edgecolors='white')
plt.show()

plt.scatter(datasets_2[:, 0], datasets_2[:, 1], alpha=0.8, s=64, c='blue', edgecolors='white')
plt.show()

# In[10]:
# 使用K均值算法
from sklearn.cluster import KMeans

K_dataset_1 = 2
km_1 = KMeans(n_clusters=K_dataset_1)
labels_1 = km_1.fit(datasets_1).labels_

plt.scatter(datasets_1[:, 0], datasets_1[:, 1], c=labels_1, alpha=0.8, s=64, edgecolors='white')
plt.scatter(km_1.cluster_centers_[:, 0], km_1.cluster_centers_[:, 1], s=200, c=np.unique(labels_1), edgecolors="black")
plt.show()

K_dataset_2 = 4
km_2 = KMeans(n_clusters=K_dataset_2)
labels_2 = km_2.fit(datasets_2).labels_

plt.scatter(datasets_2[0, :], datasets_2[1, :], c=labels_2, alpha=0.8, s=64, edgecolors='white')
plt.scatter(km_2.cluster_centers_[:, 0], km_2.cluster_centers_[:, 1], marker='s', s=100, c=np.unique(labels_2),
            edgecolors='black')
plt.show()