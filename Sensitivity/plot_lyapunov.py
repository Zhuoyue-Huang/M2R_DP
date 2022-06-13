import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Sensitivity.lyapunov_time_estimate import lyp_exp

# small perturbation
delta = np.pi/20
g = 9.81

th1_range = np.arange(-np.pi, -np.pi, delta)
th2_range = np.arange(np.pi, np.pi, delta)

eps = 0.01  # the initial difference
tmax = 1000


points_th1, points_th2 = [], []
vals = []
for th1 in tqdm(th1_range):
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
        initial_cond = np.array([th1, 0, th2, 0])
        vals.append(lyp_exp(initial_cond, eps, tmax, T=1))


points_th1, points_th2 = np.array(points_th1), np.array(points_th2)

fig, ax = plt.subplots()
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='hot', s=60,
                     linewidth=0)
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_xlim([1.7, 2.3])
ax.set_ylim([1.2, 1.8])
ax.set_title("Maximum Lyapunov exponents")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=5)
cbar.set_label('Lyapunov exponent', fontsize=6, rotation=270)
plt.show()
