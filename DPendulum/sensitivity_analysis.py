from DPendulum.DPendulum import Pendulum
import matplotlib.pyplot as plt
import numpy as np
from scipy import pi

y1 = [pi/2.3, 0, pi/2.275, 0.02]
y2 = [pi/2.3, 0, pi/2.275, 0.02+0.0000001]
pendulum1 = Pendulum(theta1=y1[0], z1=y1[1], theta2=y1[2], z2=y1[3], tmax=50, dt=0.05, y0=y1)
pendulum2 = Pendulum(theta1=y2[0], z1=y2[1], theta2=y2[2], z2=y2[3], tmax=50, dt=0.05, y0=y2)

t1 = pendulum1.t
t2 = pendulum2.t
theta11, z11, theta12, z12 = pendulum1.y.sol(t1)
theta21, z21, theta22, z22 = pendulum2.y.sol(t2)

plt.figure(figsize = (12, 6))
plt.plot(t1, theta12)
plt.plot(t2, theta22)
plt.show()
