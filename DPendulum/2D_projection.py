import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum

# functions needed for projection
def s1(y):
    """Transform onto s1"""
    s1_y = []
    for val in y:
        v = val % (2*np.pi)
        if v > np.pi:
            v = v - 2*np.pi
        s1_y.append(v)
    return s1_y


def pi_jumps(y):
    """Return list of places where line between i -> i+1 should be removed."""
    no_line = []
    a = (y + np.pi) // (2*np.pi) / np.pi
    for i in range(len(a)-1):
        if a[i] != a[i+1]:
            no_line.append(i)
    return no_line

def plot_minus_jumps(y1, y2):
    index = [i for i in range(len(y1)-1, -1, -1)]
    discont  = set(pi_jumps(y1) + pi_jumps(y2))
    s_y1 = s1(y1)
    s_y2 = s1(y2)
    discont = list(discont)
    discont.sort()
    separated_points = []
    while index:
        run = [[], []]
        while True:
            j = index.pop()
            run[0].append(s_y1[j])
            run[1].append(s_y2[j])
            if j in discont or j == len(y1)-1:
                separated_points.append(run)
                break
    for r in range(len(separated_points)):
        plt.plot(separated_points[r][0], separated_points[r][1], color = 'b')


# projection action
y0 = [np.pi/3, 0, np.pi/3, np.pi/2]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=50, dt=0.05, y0=y0)
y = pendulum.sol()
label = ['theta1', 'theta1dot', 'theta2', 'theta2dot']

for i in range(4):
    for j in range(4):
        if i != j:
            plt.figure(figsize=(15,5))
            a = plt.subplot(1, 3, 1)
            plt.subplots_adjust(wspace=0.5)
            plt.plot(y[i], y[j])
            plt.xlabel(label[i])
            plt.ylabel(label[j])
            b = plt.subplot(1, 3, 2)
            plt.plot(s1(y[i]), s1(y[j]))
            plt.xlabel(label[i])
            plt.ylabel(label[j])
            c = plt.subplot(1, 3, 3)
            plot_minus_jumps(y[i], y[j])
            plt.xlabel(label[i])
            plt.ylabel(label[j])
