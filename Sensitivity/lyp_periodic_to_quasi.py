import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Sensitivity.lyapunov_time_estimate import lyp_exp

th1 = np.pi/20
th2_range = np.arange(np.sqrt(2)-0.5, np.sqrt(2)+0.6, 0.1)

eps = 0.01  # the initial difference
tmax = 1000

for th1_fac in [1/40, 1/20, 1/10]:
    lyp_vals = []
    for th2 in tqdm(th2_range):
        initial_cond = np.array([np.pi*th1_fac, 0, np.pi*th1_fac*th2, 0])
        lyp_vals.append(lyp_exp(initial_cond, eps, tmax, T=1))
    plt.plot(th2_range, lyp_vals, label=str(th1_fac))
    print(lyp_vals)
plt.axvline(np.sqrt(2), c='red')
plt.legend()
plt.xlabel(r'$\theta_2 / \theta_1$')
plt.ylabel('Maximum Lyapunov exponent')
plt.show()
