from DPendulum import Pendulum
import numpy as np
from numpy import pi, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

def wrapped(theta):
    """Wrap angular displacement to obtain range of -pi to pi for infinite cylinder plot."""
    theta_wrapped = [None] * len(theta)
    for x in enumerate(theta):
        if (x[1] % (2 * np.pi)) > np.pi:
            theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
        else:
                theta_wrapped[x[0]] = x[1] % (2 * np.pi)
    return theta_wrapped

dt = 0.01
tmax = 100
delta = pi / 60
th_range = np.arange(0, pi+delta/2, delta)
th1 = []
dth1 = []

from numpy import pi
from tqdm import tqdm

def wrapped(theta):
    """Wrap angular displacement to obtain range of -pi to pi for infinite cylinder plot."""
    theta_wrapped = [None] * len(theta)
    for x in enumerate(theta):
        if (x[1] % (2 * np.pi)) > np.pi:
            theta_wrapped[x[0]] = x[1] % (2 * np.pi) -  2 * np.pi
        else:
            theta_wrapped[x[0]] = x[1] % (2 * np.pi)
    return theta_wrapped

dt = 0.01
tmax = 100
delta = pi / 360
th_range = np.arange(0, pi+delta/2, delta)
th1 = []
dth1 = []
t1 = []

for i in tqdm(th_range):
    y0 = [i, 0, i, 0]
    pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=tmax, dt=dt, y0=y0)
    theta1, z1, _, _ = pendulum.sol()
    t = pendulum.t
    theta1 = wrapped(theta1)
    z1_list = []
    #t_list = []
    for j in range(len(theta1)-1):
        if (theta1[j] <= 0 and theta1[j+1] > 0) or (theta1[j] >= 0 and theta1[j+1] < 0):
            z1_list.append(z1[j])
            #t_list.append(t[j])
            #z1_list.append(z1[j]-(z1[j]-z1[j+1])/(theta1[j]-theta1[j+1])*theta1[j])
    th1 += [i/pi*180]*len(z1_list)
    dth1 += z1_list
    #t1 += t_list

plt.figure(figsize=(7,4))
#plt.vlines(60, -10, 10, colors="#7BC8F6", linestyles="dashed", alpha=1, linewidth=0.9)
#plt.plot([0,180], [0, sqrt(9.81)*sqrt(2-sqrt(2))*np.tan(sqrt(9.81)*sqrt(2-sqrt(2))*0.5399102079395084)], color="#00008B")
plt.scatter(th1, dth1, s=0.005, color="#0343DF")
#plt.legend([r"$60^\circ$ line"], prop={'size': 13}, loc="upper left") #, r"Theoretical $\omega_+$, $\omega_-$"
plt.xlim(0, 180)
plt.xlabel(r"$\theta_1$", fontsize=17)
plt.ylabel(r"$\dot\theta_1$", fontsize=17)
plt.title(r"Initial condition $\theta_1=\theta$, $\theta_2=\theta$, $\theta\in [0,\pi]$",  fontsize=20)
plt.tight_layout()
#plt.savefig("bif_4.pdf", format="pdf", bbox_inches="tight")
plt.show()