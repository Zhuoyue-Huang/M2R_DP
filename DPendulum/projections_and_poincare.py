import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum

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


def twod_projection(*, y0 = np.array([np.pi/3, 0, np.pi/3, np.pi/2]), i=0, j=2):
    pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=50, dt=0.05, y0=y0)
    y = pendulum.sol()
    label = ['theta1', 'theta1dot', 'theta2', 'theta2dot']
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
    plt.show()

def doublependpoincare(pendulum):
    sol = pendulum.sol()
    theta1 = s1(sol[0])
    omega1 = sol[1]
    theta2 = s1(sol[2])
    omega2 = sol[3]
    
    theta_times = [[theta1[i], theta1[i + 1], i, i + 1] for i in range(len(theta1) - 1) \
                   if (theta1[i] * theta1[i + 1] < 0 and abs(theta1[i]) < 1 and omega1[i] > 0)]
    
    interpolated_theta2 = []
    interpolated_omega2 = []
    
    for m in theta_times:
        interpolated_theta2.append((theta2[m[2]] * m[1] - theta2[m[3]] * m[0]) / (m[1] - m[0]))
        interpolated_omega2.append((omega2[m[2]] * m[1] - omega2[m[3]] * m[0]) / (m[1] - m[0]))
    plt.scatter(interpolated_theta2, interpolated_omega2, s=0.1)
    plt.show()


def poincare(pendulum, *, var=0, varpos=0):
    label = ['theta1', 'theta1dot', 'theta2', 'theta2dot']
    label.pop(var)
    y = np.array(pendulum.sol())
    indices = []
    for i in range(len(y[var]) - 1):
        if y[var][i] < varpos and y[var][i+1] > varpos:
                indices.append(i)
    y = y.T
    points = []
    if indices:
        for i in indices:
            x1, x2 = s1(list(y[i])), s1(list(y[i+1]))
            w1, w2 = x1.pop(var), x2.pop(var)
            points.append([(x2[i]*w2 - x1[i]*w1)/(w2-w1) for i in range(3)])
        print(points)
        points = np.array(points).T
        for i in range(3):
            points[i] = s1(points[i])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(points[0], points[1], points[2], s=0.5)
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
        plt.show()

        plt.figure(figsize=(15,5))
        a = plt.subplot(1, 3, 1)
        plt.subplots_adjust(wspace=0.5)
        plt.scatter(points[0], points[1], s=0.5)
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        b = plt.subplot(1, 3, 2)
        plt.scatter(points[0], points[2], s=0.5)
        plt.xlabel(label[0])
        plt.ylabel(label[2])
        c = plt.subplot(1, 3, 3)
        plt.scatter(points[1], points[2], s=0.5)
        plt.xlabel(label[1])
        plt.ylabel(label[2])
        plt.show()
    else:
        print('No crosses')
        return y

y0 = [0.1, 0, 0, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=5000, dt=0.05, y0=y0)
poincare(pendulum)
