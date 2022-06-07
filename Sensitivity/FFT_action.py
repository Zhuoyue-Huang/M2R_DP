from DPendulum import Pendulum
import matplotlib.pyplot as plt
import numpy as np
from scipy import pi, fft

# FFT plot for small angle osillations
y0 = [pi/12, 0, pi/12, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=200, dt=0.05, y0=y0)
pendulum.fft_plot()

# FFT plot for a periodic case
y0 = [pi/12, 0, np.sqrt(2)*pi/12, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=100, dt=0.05, y0=y0)
pendulum.fft_plot()

# Reproduce theta plots for a periodic case
t = pendulum.t
theta1, z1, theta2, z2 = pendulum.sol()
omega1_pval, omega1_pind, omega2_pval, omega2_pind = pendulum.find_peaks()

plt.figure(figsize = (15, 6))
plt.subplot(121)
plt.plot(t, theta1)
plt.plot(t, np.cos(omega1_pval[0]*t)/4, alpha=0.6)
plt.subplot(122)
plt.plot(t, theta2)
plt.plot(t, np.cos(omega2_pval[0]*t)/3, alpha=0.6)
plt.show()

# Restrict theta to [-pi, pi]
def s1(y):
    """Transform onto s1"""
    s1_y = []
    for val in y:
        v = val % (2*np.pi)
        if v > np.pi:
            v = v - 2*np.pi
        s1_y.append(v)
    return s1_y
y0 = [pi, 10, pi, 10]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=10, dt=0.01, y0=y0)
t = pendulum.t
sr = 1 / pendulum.dt
theta1, z1, theta2, z2 = pendulum.sol()
theta1 = s1(theta1)
theta2 = s1(theta2)
Theta1 = fft(theta1)
Theta2 = fft(theta2)
N = len(Theta1)
n = np.arange(N)
T = N/sr
omega = n/T * (2*pi)
# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
omega = omega[:n_oneside]
Theta1 = Theta1[:n_oneside]
Theta2 = Theta2[:n_oneside]
plt.figure(figsize = (20, 12))
plt.subplot(221)
plt.plot(t, theta1)
plt.xlabel('Time (s)')
plt.ylabel(r'Amplitude of $\theta_1$')
plt.tight_layout()

plt.subplot(222)
plt.plot(omega, Theta1)
plt.legend(["Peak value", "Angular velocity spectrum", "Theoretical angular velocity"])
plt.xlabel(r'Angular velocity of $\theta_1$')
plt.ylabel('Amplitude of the angular velocity')
plt.xlim(0, omega[-1])

plt.subplot(223)
plt.plot(t, theta2)
plt.xlabel('Time (s)')
plt.ylabel(r'Amplitude of $\theta_2$')
plt.tight_layout()

plt.subplot(224)
plt.plot(omega, Theta2)
plt.legend(["Peak value", "Angular velocity spectrum", "Theoretical angular velocity"])
plt.xlabel(r'Angular velocity of $\theta_2$')
plt.ylabel('Amplitude of the angular velocity')
plt.xlim(0, omega[-1])
plt.show()