# Animation Reference: https://physicspython.wordpress.com/2019/06/20/double-pendulum-part-3/
# FFT Reference: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
import matplotlib
matplotlib.use('TkAgg') # 'tkAgg' if Qt not present 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import pi
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks, peak_prominences
  
class Pendulum:
    def __init__(self, theta1, z1, theta2, z2, tmax, dt, y0, a1=1, a2=1, m1=1, m2=1):
        self.theta1 = theta1
        self.z1 = z1
        self.theta2 = theta2
        self.z2 = z2

        self.a1 = a1
        self.a2 = a2
        self.m1 = m1
        self.m2 = m2

        self.tmax = tmax
        self.dt = dt
        self.t = np.arange(0, tmax+dt, dt)
        self.ind = 0
        self.g = 9.81
        self.trajectory = [self.polar_to_cartesian()]

        self.y0 = y0
        self.y = solve_ivp(deriv, (0, tmax), y0, method='Radau', dense_output=True,
              args=(self.a1, self.a2, self.m1, self.m2, self.g))
  
    def polar_to_cartesian(self):
        x1 = self.a1 * np.sin(self.theta1)        
        y1 = -self.a1 * np.cos(self.theta1)  
        x2 = x1 + self.a2 * np.sin(self.theta2)
        y2 = y1 - self.a2 * np.cos(self.theta2)
        return np.array([[0.0, 0.0], [x1, y1], [x2, y2]])
      
    def evolve(self):
        "Return the new Cartesian position after time dt."
        theta1, z1, theta2, z2 = self.y.sol(self.t)
        self.theta1 = theta1[self.ind]
        self.z1 = z1[self.ind]
        self.theta2 = theta2[self.ind]
        self.z2 = z2[self.ind]
        self.ind += 1
        new_position = self.polar_to_cartesian()
        self.trajectory.append(new_position)
        return new_position

    def fft(self):
        "Return omega domain and the corresponding amplitude of theta1 and theta2."
        t = self.t
        sr = 1 / self.dt
        theta1, z1, theta2, z2 = self.y.sol(t)
        Theta1 = fft(theta1)
        Theta2 = fft(theta2)
        N = len(Theta1)
        n = np.arange(N)
        T = N/sr
        omega = n/T * (2*pi)
        # Get the one-sided specturm
        n_oneside = N//2
        # get the one side frequency
        omega_oneside = omega[:n_oneside]
        Theta1_oneside = Theta1[:n_oneside]
        Theta2_oneside = Theta2[:n_oneside]
        return (omega_oneside, np.abs(Theta1_oneside), np.abs(Theta2_oneside))

    def fft_plot(self, show_peak=True, peak_num=2):
        "Return 2*2 plots of time domain and omega domain in terms of theta1 and theta2."
        t = self.t
        theta1, z1, theta2, z2 = self.y.sol(t)
        omega, Theta1, Theta2 = self.fft()
        if show_peak:
            omega1_pval, omega1_pind, omega2_pval, omega2_pind = self.find_peaks(peak_num=peak_num)

        plt.figure(figsize = (15, 9))
        plt.subplot(221)
        plt.plot(t, theta1)
        plt.xlabel('Time (s)')
        plt.ylabel(r'Amplitude of $\theta_1$')
        plt.tight_layout()

        plt.subplot(222)
        if show_peak:
            plt.scatter(omega1_pval, Theta1[omega1_pind], color="red", marker="x")
            plt.legend(["Peak value"])
        plt.plot(omega, Theta1)
        plt.xlabel(r'Angular velocity of $\theta_1$')
        plt.ylabel('Amplitude of the angular velocity')
        plt.xlim(0, omega[-1])

        plt.subplot(223)
        plt.plot(t, theta2)
        plt.xlabel('Time (s)')
        plt.ylabel(r'Amplitude of $\theta_2$')
        plt.tight_layout()

        plt.subplot(224)
        if show_peak:
            plt.scatter(omega2_pval, Theta2[omega2_pind], color="red", marker="x")
            plt.legend(["Peak value"])
        plt.plot(omega, Theta2)
        plt.xlabel(r'Angular velocity of $\theta_2$')
        plt.ylabel('Amplitude of the angular velocity')
        plt.xlim(0, omega[-1])
        plt.show()

    def find_peaks(self, peak_num=2):
        "Return a tuple of two np arrays containing peak values of omega and its indices."
        omega, Theta1, Theta2 = self.fft()
        all_peak1 = find_peaks(Theta1)[0]
        all_peak2 = find_peaks(Theta2)[0]
        all_peak1_val = peak_prominences(Theta1, all_peak1)[0]
        all_peak2_val = peak_prominences(Theta2, all_peak2)[0]
        peak1_ind = np.argsort(all_peak1_val)[::-1][0:peak_num]
        peak2_ind = np.argsort(all_peak2_val)[::-1][0:peak_num]
        peak1 = all_peak1[peak1_ind]
        peak2 = all_peak2[peak2_ind]
        return (omega[peak1], peak1, omega[peak2], peak2)


def deriv(t, y, a1, a2, m1, m2, g):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(a1*z1**2*c + a2*z2**2) -
            (m1+m2)*g*np.sin(theta1)) / a1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(a1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
            m2*a2*z2**2*s*c) / a2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

class Animator:
    def __init__(self, pendulum, draw_trace=False):
        self.pendulum = pendulum
        self.draw_trace = draw_trace
        self.time = 0.0
  
        # set up the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_xlim(-2.5, 2.5)
  
        # prepare a text window for the timer
        self.time_text = self.ax.text(0.05, 0.95, '', 
            horizontalalignment='left', 
            verticalalignment='top', 
            transform=self.ax.transAxes)
  
        # initialize by plotting the last position of the trajectory
        self.line, = self.ax.plot(
            self.pendulum.trajectory[-1][:, 0], 
            self.pendulum.trajectory[-1][:, 1], 
            marker='o')
          
        # trace the whole trajectory of the second pendulum mass
        if self.draw_trace:
            self.trace, = self.ax.plot(
                [a[2, 0] for a in self.pendulum.trajectory],
                [a[2, 1] for a in self.pendulum.trajectory])
     
    def advance_time_step(self):
        while self.time <= self.pendulum.tmax:
            self.time += self.pendulum.dt
            yield self.pendulum.evolve()
             
    def update(self, data):
        self.time_text.set_text('Elapsed time: {:6.2f} s'.format(self.time))
         
        self.line.set_ydata(data[:, 1])
        self.line.set_xdata(data[:, 0])
         
        if self.draw_trace:
            self.trace.set_xdata([a[2, 0] for a in self.pendulum.trajectory])
            self.trace.set_ydata([a[2, 1] for a in self.pendulum.trajectory])
        return self.line,
     
    def animate(self):
        self.animation = animation.FuncAnimation(self.fig, self.update,
                         self.advance_time_step, interval=25, blit=False)
