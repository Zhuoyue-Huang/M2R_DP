from DPendulum import Pendulum
import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

def s1(y):
    """Transform onto s1"""
    s1_y = []
    for val in y:
        v = val % (2*np.pi)
        if v > np.pi:
            v = v - 2*np.pi
        s1_y.append(v)
    return s1_y

dt = 0.01
tmax = 100
delta = pi / 180
th_range = np.arange(0, pi+delta/2, delta)
var = 0
varpos = 0
th1 = []
points = np.array([[], [], []])

for j in tqdm(th_range):
    y0 = [j, 0, j, 0]
    pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=tmax, dt=dt, y0=y0)
    y = np.array(pendulum.sol())
    indices = []
    for i in range(len(y[var]) - 1):
        if y[var][i] < varpos and y[var][i+1] > varpos:
            indices.append(i)
    y = y.T
    point = []
    if indices:
        for i in indices:
            x1, x2 = s1(list(y[i])), s1(list(y[i+1]))
            w1, w2 = x1.pop(var), x2.pop(var)
            point.append([(x2[i]*w2 - x1[i]*w1)/(w2-w1) for i in range(3)])
        #print(points)
        len_p = len(point)
        point = np.array(point).T
        points = np.concatenate((points, np.array(point)), axis=1)
        th1 += [j/pi*180]*len_p
plt.scatter(th1, points[0], s=0.3, rasterized=True)
plt.show()