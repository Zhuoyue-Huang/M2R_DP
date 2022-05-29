from DPendulum.DPendulum import Pendulum, Animator
import matplotlib
matplotlib.use('TkAgg') # 'tkAgg' if Qt not present 
import matplotlib.pyplot as plt 
import scipy as sp

y0 = [3*sp.pi/7, 0, 3*sp.pi/4, 0]
pendulum = Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], tmax=10, dt=0.01, y0=y0)
animator = Animator(pendulum=pendulum, draw_trace=True)
animator.animate()
plt.show()