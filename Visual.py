import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# 画曲线图
x = np.linspace(0, 5, 50)
y_cos = np.cos(x)
y_sin = np.sin(x)

plt.figure()
plt.plot(x, y_cos)  # x—aixs, y-axis
plt.plot(x, y_sin)
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.show()

print(list(mpl.rcParams['axes.prop_cycle']))

plt.subplot(1, 2, 1)
plt.plot(x, y_cos, 'r--')  # r represents red
plt.title('cos')

plt.subplot(1, 2, 2)
plt.plot(x, y_sin, 'g--')
plt.title('sin')

plt.show()

# 画散点图
from sklearn.datasets import make_blobs

D = make_blobs(n_samples=100, n_features=2, centers=3, random_state=7)
groups = D[1]
coordinates = D[0]

# yellow square
plt.plot(coordinates[groups == 0, 0], coordinates[groups == 0, 1], 'ys', label='group 0')
# magenta stars
plt.plot(coordinates[groups == 1, 0], coordinates[groups == 1, 1], 'm*', label='group 1')
# red diamonds
plt.plot(coordinates[groups == 2, 0], coordinates[groups == 2, 1], 'rD', label='group 2')

plt.ylim(-2, 10)
plt.yticks([10, 6, 2, -2])
plt.xticks([-15, -5, 5, -15])
plt.grid()

plt.annotate('Squares', (-12, 2.5))
plt.annotate('Stars', (0, 6))
plt.annotate('Diamonds', (10, 3))
plt.legend(loc='lower left', numpoints=1)
plt.show()

# 画直方图
