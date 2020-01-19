#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# In[2]:


iris_filename = r'C:\Users\YuYue\Desktop\iris.csv'
iris = pd.read_csv(iris_filename, header=None,
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
iris.head()

# In[3]:


iris.describe()

# In[4]:


boxes = iris.boxplot(return_type='axes')

# In[5]:


iris.quantile([0.1, 0.9])

# In[6]:


iris.target.unique()

# In[7]:


# 创建共生矩阵，查看特征之间关系
pd.crosstab(iris['petal_length'] >= 3.758667, iris['petal_width'] > 1.1998667)

# In[8]:


scatterplot = iris.plot(kind='scatter', x='petal_width', y='petal_length', s=64, c='blue', edgecolors='white')

# In[9]:


distr = iris.petal_width.plot(kind='hist', alpha=0.5, bins=20)

# In[10]:


cali = datasets.california_housing.fetch_california_housing()
X = cali['data']
Y = cali['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

# In[11]:


regressor = KNeighborsRegressor()
regressor.fit(X_train, Y_train)
Y_est = regressor.predict(X_test)
print("MAE=", mean_squared_error(Y_test, Y_est))

# In[12]:


# 使用Z-标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
regressor = KNeighborsRegressor()
regressor.fit(X_train_scaled, Y_train)
Y_est = regressor.predict(X_test_scaled)
print("MAE=", mean_squared_error(Y_test, Y_est))

# In[13]:


# 鲁棒性缩放
scaler2 = RobustScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)
regressor = KNeighborsRegressor()
regressor.fit(X_train_scaled, Y_train)
Y_est = regressor.predict(X_test_scaled)
print("MAE=", mean_squared_error(Y_test, Y_est))

# In[14]:


# 对特定特征使用非线性修正
non_linear_feat = 5
X_train_new_feat = np.sqrt(X_train[:, non_linear_feat])
X_test_new_feat = np.sqrt(X_test[:, non_linear_feat])

X_train_new_feat.shape = (X_train_new_feat.shape[0], 1)
X_train_extended = np.hstack([X_train, X_train_new_feat])

X_test_new_feat.shape = (X_test_new_feat.shape[0], 1)
X_test_extended = np.hstack([X_test, X_test_new_feat])

scaler = StandardScaler()
X_train_extended_scaled = scaler.fit_transform(X_train_extended)
X_test_extended_scaled = scaler.transform(X_test_extended)

regressor = KNeighborsRegressor()
regressor.fit(X_train_extended_scaled, Y_train)
Y_est = regressor.predict(X_test_extended_scaled)

print("MAE=", mean_squared_error(Y_test, Y_est))
