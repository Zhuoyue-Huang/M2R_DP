from cProfile import label
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
plt.figure(figsize = (8, 4))
plt.contourf(Th1, Th2, E, cmap="YlGnBu_r", levels=20)
plt.xlabel(r"$\theta_1$", fontsize=17)
plt.ylabel(r"$\theta_2$", fontsize=17)
plt.colorbar(label="Energy Level")
plt.tight_layout()
plt.savefig("energy.pdf", format="pdf", bbox_inches="tight")
plt.show()