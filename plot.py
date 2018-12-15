import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
d0 = np.load('./1/results_0.npz')
d = np.load('./1/results.npz')


y0 = d0['mean']
y = d['mean']


xnew = np.linspace(T.min(),T.max(),300)
y_smooth =

plt.figure()
plt.semilogy(d0['mean'])
plt.semilogy(d['mean'])
plt.xlim([0,1000])
plt.show()
