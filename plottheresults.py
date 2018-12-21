import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import splrep, splev

def errorfill(x, y, yerr, label, color=None, marker = 'o', alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    base_line, = ax.plot(x, y, color=color,linewidth=2,label = label) #,marker = 'o'
    if color is None:
        color = base_line.get_color()
    ax.fill_between(x, ymax, ymin, facecolor=color, alpha=alpha_fill)

def smooth(x,y):
    poly = np.polyfit(x,y,15)
    poly_y = np.poly1d(poly)(x)
    return poly_y

name0 = '/Users/dae/Desktop/bp0.npz'
name  = '/Users/dae/Desktop/bpresults.npz'
x_max = 5000000.0

# name0 = '/Users/dae/Desktop/lunar0.npz'
# name  = '/Users/dae/Desktop/lunarresults.npz'
# x_max = 1000000.0


d0 = np.load(name0)
d = np.load(name)

loss  = d['loss']
loss_length = len(loss)
x_loss = np.linspace(1,x_max,loss_length)

plt.plot(x_loss, loss)
plt.title('TDM Loss for BipedalWokerHardcore')
plt.savefig('lossbp.png',format='png')
plt.show()







mean0 = d0['mean']
mean = d['mean']
mean_length = len(mean0)
x_mean = np.linspace(1,x_max,mean_length)
std0 = d0['std']
std = d['std']

smmean0 = smooth(x_mean, mean0)
smmean  = smooth(x_mean, mean)
smstd0  = smooth(x_mean, std0)
smstd   = smooth(x_mean, std)


plt.plot(x_mean,mean0, linewidth=0.1,color='b')
plt.plot(x_mean,mean,  linewidth=0.1,color='g')
errorfill(x_mean,smmean0,smstd0,'PPO',color='b')
errorfill(x_mean,smmean,smstd,'TDM',color='g')
plt.legend()
plt.title('Episode Reward')
# plt.ylim([0,200])
plt.savefig('rewardbp.png', format='png')
plt.show()

# pdb.set_trace()
