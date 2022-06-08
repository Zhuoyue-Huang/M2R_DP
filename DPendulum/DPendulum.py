# FFT Reference: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
from turtle import color
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from scipy import pi
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks, peak_prominences
from sympy import Matrix, Symbol
from sympy.solvers.ode.systems import matrix_exp


class Pendulum:
    def __init__(self, theta1, z1, theta2, z2, tmax, y0, dt=0.05, L1=1, L2=1, m1=1, m2=1,
                 to_trace=True, trace_delete=True, restart=None, method='Radau'):
        self.theta1 = theta1
        self.z1 = z1
        self.theta2 = theta2
        self.z2 = z2

        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2

        self.tmax = tmax
        self.dt = dt
        self.t = np.arange(0, tmax+dt, dt)
        self.ind = 0
        self.g = 9.81
        self.method = method

        self.to_trace = to_trace
        self.trace_delete = trace_delete
        self.restart = restart
        self.num_frames = int((50/3) * tmax) #250

        self.y0 = y0
        self.trajectory = [self.polar_to_cartesian()]

        self.full_sol = self.sol()
        self.theta1 = self.full_sol[0]
        self.theta2 = self.full_sol[2]

        # if method == "Radau":
        #     self.full_sol = self.sol()
        #     self.theta1 = self.full_sol[0]
        #     self.theta2 = self.full_sol[2]
        
        # elif method == 'RK23':
        #     self.full_sol = self.sol()
        #     self.theta1 = self.full_sol[0]
        #     self.theta2 = self.full_sol[2]

        self.full_sol = self.full_sol
        self.x1 = self.L1 * np.sin(self.theta1)
        self.y1 = -self.L1 * np.cos(self.theta1)
        self.x2 = self.x1 + self.L2 * np.sin(self.theta2)
        self.y2 = self.y1 - self.L2 * np.cos(self.theta2)

    def sol(self):
        "Return theta1, z1, theta2, z2 given the initial condition."
        if self.restart is None:
            if self.method == "Radau":
                y = solve_ivp(deriv, (0, self.tmax), self.y0, method='Radau', dense_output=True,
                            args=(self.L1, self.L2, self.m1, self.m2, self.g))
                theta1, z1, theta2, z2 = y.sol(self.t)
                return [theta1, z1, theta2, z2]
            elif self.method == "RK23":
                y_full = solve_ivp(deriv, np.array((0, self.tmax + self.dt)), self.y0,
                               t_eval=self.t, method='RK23',
                               args=(self.L1, self.L2, self.m1, self.m2, self.g))
                theta1, z1, theta2, z2 = y_full.y
                return [theta1, z1, theta2, z2]
            else:
                raise NotImplementedError
        else:
            return self.iterative_solve()

    def iterative_solve(self):
        T = self.tmax + self.dt
        q, r = np.divmod(T, self.restart)
        if r != 0:
            dt = np.arange(0, r, self.dt)
            sol = solve_ivp(deriv, np.array((0, r)), self.y0, method='RK23', t_eval=dt,
                            args=(self.L1, self.L2, self.m1, self.m2))
            sol = sol.y
        dt_1 = np.arange(0, 1, self.dt)
        for _ in range(int(q)):
            try:
                d_sol = solve_ivp(deriv, np.array((0, 1)), sol[-1], method='RK23',
                                  t_eval=dt_1, args=(self.L1, self.L2, self.m1, self.m2, self.g))
                sol = np.concatenate((sol, d_sol.y))
            except Exception:
                d_sol = solve_ivp(deriv, np.array((0, 1)), self.y0, method='RK23',
                                  t_eval=dt_1, args=(self.L1, self.L2, self.m1, self.m2, self.g))
                sol = deepcopy(d_sol.y)
        theta1, z1, theta2, z2 = sol
        return [theta1, z1, theta2, z2]
    
    
    def polar_to_cartesian(self):
        self.x1 = self.L1 * np.sin(self.theta1)
        self.y1 = -self.L1 * np.cos(self.theta1)
        self.x2 = self.x1 + self.L2 * np.sin(self.theta2)
        self.y2 = self.y1 - self.L2 * np.cos(self.theta2)
        return np.array([[0.0, 0.0], [self.x1, self.y1], [self.x2, self.y2]])
  
    
    def evolve(self):
        "Return the new Cartesian position after time dt."
        theta1, z1, theta2, z2 = self.sol()
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
        sr = 1 / self.dt
        theta1, z1, theta2, z2 = self.sol()
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
        theta1, z1, theta2, z2 = self.sol()
        omega, Theta1, Theta2 = self.fft()
        if show_peak:
            omega1_pval, omega1_pind, omega2_pval, omega2_pind = self.find_peaks(peak_num=peak_num)

        plt.figure(figsize = (15, 4))
        plt.subplot(121)
        plt.plot(t, theta1, linewidth=0.9, color="#0343DF")
        plt.xlabel('Time (s)', fontsize=17)
        plt.ylabel(r'Amplitude of $\theta_1$', fontsize=17)

        plt.subplot(122)
        if show_peak:
            plt.scatter(omega1_pval, Theta1[omega1_pind], color="#00008B", marker="x")
        plt.plot(omega, Theta1, linewidth=0.9, color="#0343DF")
        if (self.m1, self.m2, self.L1, self.L2) == (1, 1, 1, 1):
            plt.vlines((np.sqrt(self.g)*np.sqrt(2-np.sqrt(2)),
                        np.sqrt(self.g)*np.sqrt(2+np.sqrt(2))),
                       0, max(Theta1), colors="#7BC8F6", linestyles="dashed", alpha=0.8)
        plt.legend(["Peak value", "Angular velocity spectrum", "Theoretical angular velocity"], prop={'size': 13})
        plt.xlabel(r'$\omega_1$ (m/s)', fontsize=17)
        plt.ylabel(r'Amplitude of $\omega_1$', fontsize=17)
        plt.xlim(0, omega[-1])
        plt.tight_layout()
        plt.savefig("fft_11.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        plt.figure(figsize = (15, 4))
        plt.subplot(121)
        plt.plot(t, theta2, linewidth=0.9, color="#0343DF")
        plt.xlabel('Time (s)', fontsize=17)
        plt.ylabel(r'Amplitude of $\theta_2$', fontsize=17)

        plt.subplot(122)
        if show_peak:
            plt.scatter(omega2_pval, Theta2[omega2_pind], color="#00008B", marker="x")
        plt.plot(omega, Theta2, linewidth=0.9, color="#0343DF")
        if (self.m1, self.m2, self.L1, self.L2) == (1, 1, 1, 1):
            plt.vlines((np.sqrt(self.g)*np.sqrt(2-np.sqrt(2)),
                        np.sqrt(self.g)*np.sqrt(2+np.sqrt(2))),
                       0, max(Theta2), colors="#7BC8F6", linestyles="dashed", alpha=0.8)
        plt.legend(["Peak value", "Angular velocity spectrum", "Theoretical angular velocity"], prop={'size': 13})
        plt.xlabel(r'$\omega_2$ (m/s)', fontsize=17)
        plt.ylabel(r'Amplitude of $\omega_2$', fontsize=17)
        plt.xlim(0, omega[-1])
        plt.tight_layout()
        plt.savefig("fft_12.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def find_peaks(self, peak_num=2):
        "Return peak values of omega and its indices for theta1 and theta2."
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

    def linearised_deriv(self): 
        '''Return as theta1, theta2, theta1dot, theta2dot.'''
        P = np.array([[(self.m1+self.m2)*self.L1**2, self.m2*self.L1*self.L2], 
                       [self.m2*self.L1*self.L2, self.m2*self.L2**2]])
        Q = - np.array([[(self.m1+self.m2)*self.g*self.L1, 0], 
                        [0, self.m2*self.g*self.L2]])
        A = np.matmul(np.linalg.inv(P), Q)
        a = np.linalg.eig(A)
        C = np.block([[np.zeros([2, 2]), np.identity(2)], [A, np.zeros([2, 2])]])
        t = Symbol('t')
        return (matrix_exp(Matrix(C), t)*
                Matrix([self.y0[0], self.y0[2], self.y0[1], self.y0[3]]), 
                np.imag(np.emath.sqrt(a[0][0])), 
                np.imag(np.emath.sqrt(a[0][1])))
    
    def compare_lin_plot(self):
        t = Symbol('t')
        lin_deriv = self.linearised_deriv()[0]
        plt.plot(self.t, self.theta1, color='red', label='theta1')
        plt.plot(self.t, self.theta2, color='blue', label='theta2')
        plt.plot(self.t, [lin_deriv[0].subs(t, r) for r in self.t], color='magenta', ls=':', label='theta1 linearised')
        plt.plot(self.t, [lin_deriv[1].subs(t, r) for r in self.t], color='cyan', ls=':', label='theta2 linearised')
        plt.xlabel("Time")
        plt.legend()
        plt.show()

def deriv(t, y, L1, L2, m1, m2, g):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
            (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
            m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

