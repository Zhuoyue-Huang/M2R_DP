# Animation Reference: https://physicspython.wordpress.com/2019/06/20/double-pendulum-part-3/
import matplotlib
matplotlib.use('TkAgg') # 'tkAgg' if Qt not present 
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
