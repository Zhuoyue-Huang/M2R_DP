import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
from tqdm import tqdm
from sympy import Symbol




def s1(y):
    """Transform onto s1"""
    s1_y = []
    for val in y:
        v = val % (2*np.pi)
        if v > np.pi:
            v = v - 2*np.pi
        s1_y.append(v)
    return s1_y


def p_poincare(pendulum, *, var=0, varpos=0, s=True, time=0):
    label = [r'$\dot{\theta_1}$', r'$\theta_2$', r'$\dot{\theta_2}$']
    label.pop(var)
    y = np.array(pendulum.sol())
    y[0] = s1(y[0])
    y[2] = s1(y[2])
    indices = []
    for i in range(len(y[var]) - 1):
        if y[var][i] < varpos and y[var][i+1] > varpos:
            indices.append(i)
    y = y.T
    points = []
    if indices:
        for i in indices:
            x1, x2 = list(y[i]), list(y[i+1])
            w1, w2 = x1.pop(var), x2.pop(var)
            points.append([(x2[i]*w2 - x1[i]*w1)/(w2-w1) for i in range(3)])
        print(points)
        points = np.array(points).T
    return points

initial_conds = [[np.pi/20, 0, np.pi/10, 0],
                 [np.pi/10, 0, np.pi/8, 0],
                 [np.pi/12, 0, np.pi/16, 0]]

t = Symbol('t')
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
x = 2
y0 = [0, 0, np.pi/10, 0]
m2 = (1-x**2)**2/(4*x**2)
p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0, m2=m2)
q = p.linearised_deriv()
plt.plot(p.t, p.theta1, linewidth=0.9, color="darkblue")
plt.plot(p.t, [q[0][0].subs(t, r) for r in p.t], linewidth = 0.9, color='orange', ls='--')
plt.legend([r'$\theta_1$', r'$\theta_1$ linearised'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.subplot(3, 1, 2)
x = 5
m2 = (1-x**2)**2/(4*x**2)
p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0, m2=m2)
q = p.linearised_deriv()
plt.plot(p.t, p.theta1, linewidth=0.9, color="darkblue")
plt.plot(p.t, [q[0][0].subs(t, r) for r in p.t], linewidth = 0.9, color='orange', ls='--')
plt.legend([r'$\theta_1$', r'$\theta_1$ linearised'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.subplot(3, 1, 3)
x = 10
m2 = (1-x**2)**2/(4*x**2)
p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0, m2=m2)
q = p.linearised_deriv()
plt.plot(p.t, p.theta1, linewidth=0.9, color="darkblue")
plt.plot(p.t, [q[0][0].subs(t, r) for r in p.t], linewidth = 0.9, color='orange', ls='--')
plt.legend([r'$\theta_1$', r'$\theta_1$ linearised'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.tight_layout()
plt.show()


'''
for y0 in initial_conds:
    p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=100, dt=0.1, y0=y0, m2=m2)
    omega, Theta1, Theta2 = p.fft()
    plt.plot(omega, Theta1, linewidth=0.9, color="#0343DF")
    plt.vlines([np.sqrt(p.g)*w1, np.sqrt(p.g)*w2], 0, 60, colors='yellow')
    plt.show()


for j in tqdm(range(len(initial_conds))):
    label = [r'$\dot{\theta_1}$', r'$\theta_2$', r'$\dot{\theta_2}$']

    y0 = initial_conds[j]
    pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], 
                        tmax=5000, m2=m2, y0=y0)
    points = poincare(pendulum)
    a = plt.subplot(1, 3, j+1, projection='3d')
    plt.subplots_adjust(wspace=0.5)
    a.scatter3D(points[0], points[1], points[2], s=0.3, c="#00008B")
    a.set_xlabel(label[0])
    a.set_ylabel(label[1])
    a.set_zlabel(label[2])
plt.show()
'''
