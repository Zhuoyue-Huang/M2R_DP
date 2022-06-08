from DPendulum import Pendulum
import matplotlib.pyplot as plt
import numpy as np
from scipy import pi, fft

# FFT plot for small angle osillations
y0 = [pi/12, 0, pi/15, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=200, dt=0.1, y0=y0)
pendulum.fft_plot()

# FFT plot for a periodic case
y0 = [pi/10, 0, np.sqrt(2)*pi/10, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=100, dt=0.1, y0=y0)
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

# check sensitive dependence for theta is large
tmax = 30
y1 = [pi/2, 0, pi/4, 0]
y2 = [pi/2, 0, pi/4-0.01, 0]
p1 = Pendulum(theta1=y1[0], z1=y1[1], theta2=y1[2], z2=y1[3], tmax=tmax, dt=0.1, y0=y1)
p2 = Pendulum(theta1=y2[0], z1=y2[1], theta2=y2[2], z2=y2[3], tmax=tmax, dt=0.1, y0=y2)
theta11, z11, theta12, z12 = p1.sol()
theta21, z21, theta22, z22 = p2.sol()
omega1, Theta11, Theta12 = p1.fft()
omega2, Theta21, Theta22 = p2.fft()
plt.figure(figsize = (15, 4))
plt.subplot(121)
plt.plot(p1.t, theta12, linewidth=0.9, color="#7BC8F6")
plt.plot(p1.t, theta22, linewidth=0.9, color="#00008B")
plt.xlabel('Time (s)', fontsize=17)
plt.ylabel(r'Amplitude of $\theta_2$', fontsize=17)
plt.legend([r"$\theta_2$ ($y_1$)", r"$\theta_2$ ($y_2$)"], prop={'size': 13})
plt.subplot(122)
plt.plot(omega1, Theta12, linewidth=0.9, color="#7BC8F6")
plt.plot(omega1, Theta22, linewidth=0.9, color="#00008B")
plt.xlabel(r'$\omega_2$ (m/s)', fontsize=17)
plt.ylabel(r'Amplitude of $\omega_2$', fontsize=17)
plt.legend([r"spectrum of $\omega_2$ ($y_1$)", r"spectrum of $\omega_2$ ($y_2$)"], prop={'size': 13})
plt.tight_layout()
plt.savefig("fft_11.pdf", format="pdf", bbox_inches="tight")
plt.show()