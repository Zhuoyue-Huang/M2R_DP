# Reference: https://physicspython.wordpress.com/2019/06/20/double-pendulum-part-3/
import matplotlib
matplotlib.use('TkAgg') # 'tkAgg' if Qt not present 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.animation as animation
  
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
        theta1, z1, theta2, z2 = self.y.sol(self.t)
        self.theta1 = theta1[self.ind]
        self.z1 = z1[self.ind]
        self.theta2 = theta2[self.ind]
        self.z2 = z2[self.ind]
        self.ind += 1
        new_position = self.polar_to_cartesian()
        self.trajectory.append(new_position)
        return new_position

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