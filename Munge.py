import csv
import urllib.request

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_boston
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

iris_filename = r'C:\Users\YuYue\Desktop\iris.csv'

iris = pd.read_csv(
    iris_filename, sep=',', decimal='.',
    header=None,
    names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.width', 'Species']
)

print(iris.head())
#
Y = iris['Species']
# print(Y)

X = iris[['Sepal.Length', 'Sepal.Width']]
print(X)
print(Y.shape, X.shape)

iris_chunks = pd.read_csv(
    iris_filename,
    header=None,
    names=['C1', 'C2', 'C3', 'C4', 'C5'],
    chunksize=10
)
# for chunk in iris_chunks:
#     print('Shape:', chunk.shape)
#     print('chunk,','\n')

# Some errors
with open(iris_filename, 'rt') as data_stream:
    # 'rt' mode
    for n, row in enumerate(csv.DictReader(data_stream,
                                           fieldnames=['Sepal.Length', 'Sepal.Width',
                                                       'Petal.Length', 'Petal.Width', 'Species'], dialect='excel')):
        if n == 0:
            print(n, row)
        else:
            break;

'''
my_own_dataset = pd.DataFrame(
    {'Col1': range(5)},
    {'Col2': [1.0] * 5},
    {'Col3': 1.0},
    {'Col4': 'Hello World!'})
'''

######################### 使用分类数据和文本数据 #################################

categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)

count_vect = CountVectorizer()
word_count = count_vect.fit_transform(twenty_sci_news.data)

print(word_count.shape)
# print(word_count[0])

word_list = count_vect.get_feature_names()
for n in word_count[0].indices:
    print('Word "%s" appears %i times' % (word_list[n], word_count[0, n]))

# 计算频率
tf_vect = TfidfVectorizer(use_idf=False, norm='l1')
word_freq = tf_vect.fit_transform(twenty_sci_news.data)
word_list = tf_vect.get_feature_names()
for n in word_freq[0].indices:
    print('Word "%s" has frequency %0.3f' % (word_list[n], word_freq[0, n]))

# using Tfidf
tfidf_vect = TfidfVectorizer()  # default: use_ifdif=True
word_tfidf = tfidf_vect.fit_transform(twenty_sci_news.data)
word_list = tfidf_vect.get_feature_names()
for n in word_tfidf[0].indices:
    print('Word "%s" has tf-idf %0.3f' % (word_list[n], word_tfidf[0, n]))

# 使用Beautiful Soup抓取网页
url = "https://en.wikipedia.org/wiki/William_Shakespeare"  # 不可用
url = "http://qwone.com/~jason/20Newsgroups/"
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)

soup = BeautifulSoup(response, 'html.parser')
print(soup.title)
# print(soup.body)

######################### 使用Numpy进行数据处理 #################################
original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Array_a = original_array.reshape(4, 2)
Array_b = original_array.reshape(4, 2).copy()
Array_c = original_array.reshape(2, 2, 2)
original_array[0] = -1
print(Array_a, '\n', Array_b, '\n', Array_c)

original_array.resize(4, 2)
print("对原数组变形:\n", original_array)

# 产生特定数字序列
ordinal_values = np.arange(9).reshape(3, 3)
# 对数组的值颠倒顺序
print("倒序数组：\n", np.arange(9)[::-1])
# 产生随机数数组
print("随机数组：\n", np.random.randint(low=1, high=10, size=(10, 10)))

# 生成主对角线为1的矩阵
print(np.eye(3, 4))

# 标准正态分布矩阵(均值为0，标准差为1)
print(np.random.normal(size=(3, 3)))

# 指定不同的均值和标准差(loc:均值，分布的中心； scale:概率分步的标准差，对应于分布的宽度)
print(np.random.normal(loc=1.0, scale=3.0, size=(3, 3)))

# 均匀分布(low和high为随机取样的上界和下界)
print(np.random.uniform(low=1.0, high=10.0, size=(3, 3)))

# housing = np.loadtxt('regression_datasets_housing.csv', delimiter=',', dtype=float)
boston = load_boston()
print(boston.data.shape)

################## Numpy快速操作和计算 #####################
a = np.arange(5).reshape(1, 5)
print(a)
a += 1
print(a * a)

a2 = np.array([1, 2, 3, 4, 5] * 5).reshape(5, 5)
print(a2, "\n")
print(a2 * a2.T, "\n")
print(np.multiply(a2, a2.T))

# 矩阵运算
M = np.arange(5 * 5, dtype=float).reshape(5, 5)

coefs = np.array([1., 0.5, 0.5, 0.5, 0.5])
coefs_matrix = np.column_stack((coefs, coefs[::-1]))
print("coefs=\n", coefs, "\n", "coefs_matrix=\n", coefs_matrix, "\n")

print(np.dot(M, coefs))
print(np.dot(coefs, M))
print(np.dot(M, coefs_matrix))
