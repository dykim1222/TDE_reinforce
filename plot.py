import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import pdb


d0 = np.load('/home/kimdy/Desktop/results_0.npz')
d8 = np.load('/home/kimdy/Desktop/results_8.npz')


y0 = d0['mean'][:1000]
y8 = d8['mean'][:1000]
yy0 = d0['median'][:1000]
yy8 = d8['median'][:1000]

xnew = np.linspace(0,len(yy0),100) #300 represents number of points to make between T.min and T.max
xnew8 = np.linspace(0,len(yy8),100)
# pdb.set_trace()

spl0 = make_interp_spline(np.arange(len(yy0)), yy0, k=1) #BSpline object
y0_sm = spl0(xnew)
spl8 = make_interp_spline(np.arange(len(yy8)), yy8, k=1) #BSpline object
y8_sm = spl8(xnew8)

plt.plot(xnew,y0_sm)
plt.plot(xnew8,y8_sm)
# plt.xlim([0,500])
plt.show()

# xnew = np.linspace(T.min(),T.max(),300)
# y_smooth =
#
# plt.figure()
# plt.semilogy(d0['mean'])
# plt.semilogy(d['mean'])
# plt.xlim([0,1000])
# plt.show()
