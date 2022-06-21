from DPendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

g = 9.81
dt = 0
max_units = 100  # NOTE that 3.5^4 is very close to 150
time_unit = np.sqrt(1 / g)


def step_to_time_unit(s, dt):
    # print(s)
    return (s * dt) / time_unit


t_conv = np.vectorize(step_to_time_unit)


def categorise(x):
    if x < 3.5:
        return 'g'

    elif x <= 12.25:
        return 'r'  # lightsteelblue'

    elif x <= 42.875:
        return 'm'  # 'cornflowerblue'

    elif x <= 150:
        return 'b'

    else:
        return 'w'


cat = np.vectorize(categorise)


def first_flip(theta1, theta2, max_units=100):
    global dt
    p1 = Pendulum(theta1, 0, theta2, 0, tmax=max_units * time_unit,
                  y0=[theta1, 0, theta2, 0], method='RK23')
    dt = p1.dt

    for i, (th1, th2) in enumerate(zip(p1.theta1, p1.theta2)):
        if abs(th1) > np.pi or abs(th2) > np.pi:
            return i

    return float('inf')


delta = np.pi / 100
th1_range = np.arange(-np.pi, np.pi, delta)
th2_range = np.arange(-np.pi, np.pi, delta)


points_th1, points_th2 = [], []
vals = []
for th1 in tqdm(th1_range):
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
        vals.append(first_flip(th1, th2, max_units=max_units))

vals = t_conv(np.array(vals), dt=dt)
col = cat(vals)

points_th1, points_th2 = np.array(points_th1), np.array(points_th2)

# Parametric plot
TH1, TH2 = np.meshgrid(th1_range, th2_range)

F = 2 * np.cos(TH1)
G = np.cos(TH2)


fig, ax = plt.subplots()
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='viridis', s=10,
                     linewidth=0)
cont = plt.contour(TH1, TH2, (F + G - 1), [0], colors='black')
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_title("Time units it takes either pendulum to flip")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=5)
cbar.set_label('number of time units', fontsize=6, rotation=270)
plt.show()
