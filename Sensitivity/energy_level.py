from DPendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m1 = 1
m2 = 1
l1 = 1
l2 = 1
dt = np.pi/180
th1_range = np.arange(-np.pi, np.pi+dt, dt)
th2_range = np.arange(-np.pi, np.pi+dt, dt)
Th1, Th2 = np.meshgrid(th1_range, th2_range)
E = m1*g*(-l1*np.cos(Th1)) + m2*g*(-l1*np.cos(Th1)-l2*np.cos(Th2))

th11 = np.pi/12
th12 = np.pi/15
y0 = [th11, 0, th12, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0)
theta11, _, theta12, _ = pendulum.sol()
x11 = np.sin(theta11)
y11 = -np.cos(theta11)
x12 = x11 + np.sin(theta12)
y12 = y11 - np.cos(theta12)

th21 = np.pi/5
th22 = 0
y0 = [th21, 0, th22, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0)
theta21, _, theta22, _ = pendulum.sol()
x21 = np.sin(theta21)
y21 = -np.cos(theta21)
x22 = x21 + np.sin(theta22)
y22 = y21 - np.cos(theta22)

th31 = np.pi/2
th32 = np.pi/4
y0 = [th31, 0, th32, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0)
theta31, _, theta32, _ = pendulum.sol()
x31 = np.sin(theta31)
y31 = -np.cos(theta31)
x32 = x31 + np.sin(theta32)
y32 = y31 - np.cos(theta32)

th41 = np.pi-0.1
th42 = np.pi-0.1
y0 = [th41, 0, th42, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=20, dt=0.01, y0=y0)
theta41, _, theta42, _ = pendulum.sol()
x41 = np.sin(theta41)
y41 = -np.cos(theta41)
x42 = x41 + np.sin(theta42)
y42 = y41 - np.cos(theta42)

# energy plot
plt.figure(figsize = (8, 6))
plt.contourf(Th1, Th2, E, cmap="YlGnBu_r", levels=20)
plt.scatter([th11, th21, th31, th41], [th21, th22, th32, th42], c=["#7BC8F6", "dodgerblue", "#0343DF", "#00008B"], marker="x")
plt.xlabel(r"$\theta_1$", fontsize=17)
plt.ylabel(r"$\theta_2$", fontsize=17)
plt.colorbar(label="Energy Level")
plt.tight_layout()
plt.savefig("energy.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()


# trjectory plot
plt.figure(figsize = (8, 6))
plt.subplot(221)
plt.plot(x12, y12, linewidth=0.9, color="#7BC8F6")
plt.xlabel("x", fontsize=10)
plt.title(r"Initial condition $\theta1=\pi/12$, $\theta_2=\pi/15$", fontsize=12)

plt.subplot(222)
plt.plot(x22, y22, linewidth=0.9, color="dodgerblue")
plt.xlabel("x", fontsize=10)
plt.title(r"Initial condition $\theta1=\pi/5$, $\theta_2=0$", fontsize=12)

plt.subplot(223)
plt.plot(x32, y32, linewidth=0.9, color="#0343DF")
plt.xlabel("x", fontsize=10)
plt.title(r"Initial condition $\theta1=\pi/2$, $\theta_2=\pi/4$", fontsize=12)

plt.subplot(224)
plt.plot(x42, y42, linewidth=0.9, color="#00008B")
plt.xlabel("x", fontsize=10)
plt.title(r"Initial condition $\theta1=\pi-0.1$, $\theta_2=\pi-0.1$", fontsize=12)

plt.tight_layout()
plt.savefig("trajectory.pdf", format="pdf", bbox_inches="tight")
plt.show()
