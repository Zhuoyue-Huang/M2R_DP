import numpy as np
import matplotlib.pyplot as plt
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

y0 = [0.1, 0, 0.1, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=5000, dt=0.0001, y0=y0)
t = pendulum.t
theta1, z1, theta2, z2 = pendulum.sol()
theta1 = wrapped(theta1)
theta2 = wrapped(theta2)
#t_list = []
z1_list = []
theta2_list = []
z2_list = []
for i in range(len(theta1)-1):
    if theta1[i] <= 0 and theta1[i+1] > 0:
        #t_list.append(t[i])
        z1_list.append(z1[i])
        theta2_list.append(theta2[i])
        z2_list.append(z2[i])

fig = plt.figure(figsize = (15, 10))
ax = plt.axes(projection ="3d")
ax.scatter3D(theta2_list, z2_list, z1_list, s=1, color="#0343DF")
ax.set_xlabel(r"$\theta_2$")
ax.set_ylabel(r"$\dot\theta_2$", fontsize=17)
ax.set_zlabel(r"$\dot\theta_1$", fontsize=17)
plt.tight_layout()
plt.savefig("poin_41.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.scatter(theta2_list, z2_list, s=1, color="#0343DF")
plt.xlabel(r"$\theta_2$", fontsize=17)
plt.ylabel(r"$\dot\theta_2$", fontsize=17)
plt.tight_layout()
plt.savefig("poin_42.pdf", format="pdf", bbox_inches="tight")
plt.show()
