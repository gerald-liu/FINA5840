import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde, t, laplace
from matplotlib import pyplot as plt

df = pd.read_excel('HSI.xlsx', sheet_name='data', index_col=0)
ret = df['700 HK'].pct_change().dropna(how='all')
ret.index = pd.to_datetime(ret.index)
mu = ret.mean()
sig = ret.std()

plt.subplot(221)
plt.title('Fig.1.1a. Histogram of empirical returns overlayed with normal density', fontsize=5)
plt.hist(ret, bins=100, density=True, color='grey')
distance = np.linspace(ret.min(), ret.max())
plt.plot(distance, norm.pdf(distance, mu, sig), c='r')
plt.ylabel('density')
# plt.show()

plt.subplot(222)
plt.title('Fig.1.1b. Using Gaussian Kernel to represent empirical returns', fontsize=5)
ret_n = (ret - mu)/sig # normalized returns
distance_n = np.squeeze(np.linspace(ret_n.min(), ret_n.max())) # use ret_n
kernel = gaussian_kde(np.squeeze(ret_n))
plt.plot(distance_n, norm.pdf(distance_n, 0, 1), c='r', label='normal')
plt.plot(distance_n, kernel(distance_n), c='grey', label='empirical')
plt.legend(loc='upper right', fontsize=5)
# plt.show()

plt.subplot(223)
plt.title('Fig.1.1c. Adding the Student-t density', fontsize=5)
plt.plot(distance_n, norm.pdf(distance_n, 0, 1), c='r', label='normal')
plt.plot(distance_n, t.pdf(distance_n, df=3), c='g', label='t-dist, df=3')
plt.plot(distance_n, kernel(distance_n), c='grey', label='empirical')
plt.legend(loc='upper right', fontsize=5)
# plt.show()

plt.subplot(224)
plt.title('Fig.1.1d. Adding Laplace density', fontsize=5)
plt.plot(distance_n, norm.pdf(distance_n, 0, 1), c='r', label='normal')
plt.plot(distance_n, t.pdf(distance_n, df=3), c='g', label='t-dist, df=3')
plt.plot(distance_n, kernel(distance_n), c='grey', label='empirical')
plt.plot(distance_n, laplace.pdf(distance_n), c='y', label='laplace dist')
plt.legend(loc='upper right', fontsize=5)
plt.show()
