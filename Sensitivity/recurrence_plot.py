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

def find_epsilon(X):
    distance = []
    for i in tqdm(X):
        for j in X:
            distance.append(np.linalg.norm(i-j))
    return np.std(np.array(distance)) * 0.4

tmax = 30
delta = 0
y0 = [4*np.pi/9-delta, 0, 4*np.pi/9-delta, 0]
p = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=tmax, dt=0.01, y0=y0)
theta1, z1, theta2, z2 = p.sol()
n = len(theta1)

# x, y trajectories
x1 = np.sin(theta1)
y1 = -np.cos(theta1)
x2 = np.reshape((x1 + np.sin(theta2)), (len(x1), 1))
y2 = np.reshape((y1 - np.cos(theta2)), (len(y1), 1))
X = np.concatenate((x2, y2), axis=1)

# phase space
theta1 = np.reshape(np.array(wrapped(theta1)), (len(theta1),1))
z1 = np.reshape(z1, (len(z1), 1))
theta2 = np.reshape(np.array(wrapped(theta2)), (len(theta2),1))
z2 = np.reshape(z2, (len(z2), 1))
X = np.concatenate((theta1, z1, theta2, z2), axis=1)

epsilon = find_epsilon(X)
points1 = []
points2 = []
for i in range(n):
    for j in range(n):
        points1.append(i)
        points2.append(j)

vals = []
for i in tqdm(X):
    for j in X:
        vals.append(heaviside(epsilon-np.linalg.norm(i-j)))

points1, points2 = np.array(points1), np.array(points2)

plt.figure(figsize=(15,15))
plt.scatter(points1, points2, c=vals, cmap='Blues', linewidth=0, marker=".", rasterized=True)
plt.xlim(points1[0], points1[-1])
plt.ylim(points2[0], points2[-1])
plt.axis('off')
plt.tight_layout()
plt.savefig("recurrence3.pdf", format="pdf", bbox_inches="tight")
plt.show()
