import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DPendulum import Pendulum


def wrapped(theta):
    """Wrap angular displacement to obtain range of -pi to pi for infinite cylinder plot."""
    theta_wrapped = [None] * len(theta)
    for x in enumerate(theta):
        if (x[1] % (2 * np.pi)) > np.pi:
            theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
        else:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi)
    return theta_wrapped

def heaviside(x):
    if x < 0:
        return 0
    elif x == 0:
        return 1/2
    else:
        return 1

def find_X(sol):
    theta1, z1, theta2, z2 = sol
    theta1 = np.reshape(np.array(wrapped(theta1)), (len(theta1),1))
    z1 = np.reshape(z1, (len(z1), 1))
    theta2 = np.reshape(np.array(wrapped(theta2)), (len(theta2),1))
    z2 = np.reshape(z2, (len(z2), 1))
    return np.concatenate((theta1, z1, theta2, z2), axis=1)

def find_R(X):
    n = len(X)
    distance = []
    for i in X:
        for j in X:
            distance.append(np.linalg.norm(i-j))
    epsilon = np.std(np.array(distance)) * 0.4
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = heaviside(epsilon-np.linalg.norm(X[i]-X[j]))
    return R

# method1: recurrence rate
def RR(R):
    n = len(R)
    rr = sum(sum(R)) / n**2
    return rr

# method2: determinism
def P_hist(R, l):
    s = 0
    n = R.shape[0]
    for i in range(1, n-l):
        for j in range(1, n-l):
            prod = 1
            for k in range(l):
                prod *= R[i+k, j+k]
            s += (1-R[i-1, j-1])*(1-R[i+l, j+l])*prod
    return s

def det(R, l_min):
    n = len(R)
    s1 = 0
    s2 = 0
    for l in range(n):
        p = P_hist(R, l)
        if l >= l_min:
            s1 += l*p
        s2 += l*p
    return s1/s2

tmax = 30
delta = 2
y0 = [np.pi/9, 0, np.pi/9, 0]
p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=tmax, dt=0.01, y0=y0)

X = find_X(p.sol())
R = find_R(X)

# method1
delta = np.pi/180
th1_range = np.arange(0, np.pi, delta)
tmax = 30
rr_val = []
for th1 in tqdm(th1_range):
        y0 = [th1, 0, th1, 0]
        p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=tmax, dt=0.01, y0=y0)
        X = find_X(p.sol())
        R = find_R(X)
        rr_val.append(RR(R))

plt.figure(figsize=(15,5))
plt.plot(th1_range/np.pi*180, rr_val)
plt.xlabel(r"$\theta$", fontsize=17)
plt.ylabel("Recurrence rate", fontsize=17)
plt.xlim(0,180)
plt.ylim(0,0.5)
plt.savefig("RQA.pdf", format="pdf", bbox_inches="tight")
plt.show()


# method2
delta = np.pi/60
th1_range = np.arange(5*np.pi/12, np.pi/2+delta/2, delta)
det_list = []
for th in tqdm(th1_range):
    y0 = [th, 0, th, 0]
    p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=30, dt=0.01, y0=y0)
    X = find_X(p.sol())
    R = find_R(X)
    det_list.append(det(R, 15))
