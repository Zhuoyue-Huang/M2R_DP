from DPendulum.DPendulum import Pendulum
import matplotlib.pyplot as plt
import numpy as np
from scipy import pi

# FFT plot
y0 = [pi/10, 0, -pi/50, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=25, dt=0.05, y0=y0)
pendulum.fft_plot()

# Reproduce theta plots using two dominant angular velocity
t = pendulum.t
theta1, z1, theta2, z2 = pendulum.y.sol(pendulum.t)
omega1_pval, omega1_pind, omega2_pval, omega2_pind = pendulum.find_peak()

plt.figure(figsize = (15, 6))
plt.subplot(121)
plt.plot(t, theta1)
plt.plot(t, (np.cos(omega1_pval[0]*t)+np.cos(omega1_pval[1]*t))/4)
plt.subplot(122)
plt.plot(t, theta2)
plt.plot(t, (np.sin(omega2_pval[0]*t)+np.sin(omega2_pval[1]*t))/4)
plt.show()