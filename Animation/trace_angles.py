from DPendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate(i, p1, trace1, time_text, time_template,
            tr1_x, tr1_y):

    tr1_x.append(p1.theta1[i])
    tr1_y.append(p1.theta2[i])
    trace1.set_data(tr1_x, tr1_y)
    time_text.set_text(time_template % (((p1.tmax + p1.dt) /
                                        p1.num_frames) * i))  # (i*dt))


def show_anim(p1):
    tr1_x, tr1_y = [], []
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_facecolor('w')
    # ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
    # ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlim(-0.45, 0.45)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    time_template = 'time : %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    trace1, = plt.plot([], [], 'r-', alpha=0.75, lw=0.5, markersize=1)

    animate_wrapper = lambda i: animate(i, p1, trace1, # noqa
                                        time_text, time_template, tr1_x,
                                        tr1_y)

    ani = animation.FuncAnimation(fig, animate_wrapper,
                                  frames=int(p1.num_frames),
                                  interval=10, repeat=False)  # 30 -> 500

    # Uncomment function to save animation as a gif
    # ani.save('pendulums.gif', writer='pillow', fps=len(t[t < 1]))

    plt.show()


if __name__ == "__main__":
    del_rate = 50
    eps = np.pi / 20
    y1 = [eps, 0, eps, 0]
    p1 = Pendulum(theta1=y1[0], z1=y1[1], theta2=y1[2], z2=y1[3], y0=y1,
                  method='RK23', to_trace=True, trace_delete=False, tmax=200)
    show_anim(p1)
