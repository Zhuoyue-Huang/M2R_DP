import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Sensitivity.lyapunov_time_estimate import lyp_exp

# small perturbation
g = 9.81
delta = np.pi/180
th1_range = np.arange(0, np.pi+delta/2, delta)
eps = 0.01  # the initial difference
tmax = 500


# line plot
val1 = []
val2 = []
val3 = []
val4 = []
val5 = []

for th1 in tqdm(th1_range):
    # case1
    initial_cond1 = np.array([th1, 0, th1, 0])
    val1.append(lyp_exp(initial_cond1, eps, tmax, T=1))
    # case2
    initial_cond2 = np.array([th1, 0, np.sqrt(2)*th1, 0])
    val2.append(lyp_exp(initial_cond2, eps, tmax, T=1))
    # case3
    initial_cond3 = np.array([th1, 0, -np.sqrt(2)*th1, 0])
    val3.append(lyp_exp(initial_cond3, eps, tmax, T=1))
    # case4
    initial_cond4 = np.array([th1, 0, 3*th1, 0])
    val4.append(lyp_exp(initial_cond4, eps, tmax, T=1))
    # case5
    initial_cond5 = np.array([th1, 0, 3*th1/4, 0])
    val5.append(lyp_exp(initial_cond5, eps, tmax, T=1))

plt.figure(figsize = (10, 8))
plt.plot(np.arange(181), val1, color="#00008B")
plt.plot(np.arange(181), val2, color="#0343DF")
plt.plot(np.arange(181), val3, color="dodgerblue")
plt.plot(np.arange(181), val4, color="#7BC8F6")
plt.plot(np.arange(181), val5, color="purple")
plt.xlabel(r"$\theta_1$", fontsize=17)
plt.ylabel("Lyapunov exponent", fontsize=17)
plt.legend([r"$\theta_2(0)=\theta_1(0)$", r"$\theta_2(0)=\sqrt{2}\theta_1(0)$",
            r"$\theta_2(0)=-\sqrt{2}\theta_1(0)$", r"$\theta_2(0)=3\theta_1(0)$",
            r"$\theta_2(0)=3\theta_1(0)/4$"], prop={'size': 13})
plt.xlim(0,180)
plt.tight_layout()
plt.savefig("lya_bi1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# color plot
delta = np.pi/60
th1_range = np.arange(-np.pi, np.pi, delta)
th2_range = np.arange(-np.pi, np.pi, delta)
points_th1, points_th2 = [], []
for th1 in tqdm(th1_range):
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
vals = np.load("lyapunov_output.npy")
points_th1, points_th2 = np.array(points_th1), np.array(points_th2)

fig, ax = plt.subplots(figsize=(7,9))
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='hot', s=53, linewidth=0, rasterized=True)
ax.plot(np.arange(-np.pi, np.pi, delta), np.arange(-np.pi, np.pi, delta), color="#00008B")
ax.plot(np.arange(-np.pi, np.pi, delta), np.sqrt(2)*np.arange(-np.pi, np.pi, delta), color="#0343DF")
ax.plot(np.arange(-np.pi, np.pi, delta), -np.sqrt(2)*np.arange(-np.pi, np.pi, delta), color="dodgerblue")
ax.plot(np.arange(-np.pi, np.pi, delta), 3*np.arange(-np.pi, np.pi, delta), color="#7BC8F6")
ax.plot(np.arange(-np.pi, np.pi, delta), 3*np.arange(-np.pi, np.pi, delta)/4, color="purple")
ax.set_xlabel(r'$\theta_1$', fontsize=17)
ax.set_ylabel(r'$\theta_2$', fontsize=17)
ax.set_xlim([0, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_title("Maximum Lyapunov exponents", fontsize=20)
cbar = fig.colorbar(scatter, label='Lyapunov exponent')
cbar.ax.tick_params(labelsize=10)
#cbar.set_label('Lyapunov exponent', fontsize=6, rotation=270)
plt.tight_layout()
plt.savefig("lya_bi2.pdf", format="pdf", bbox_inches="tight")
plt.show()

