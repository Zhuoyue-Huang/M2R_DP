import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Sensitivity.lyapunov_time_estimate import lyp_exp

points = [[np.pi/20, 0, np.pi/20*np.sqrt(2), 0],
          [4/9*np.pi, 0, 4/9*np.pi, 0],
          [2/3*np.pi, 0, np.pi/2, 0],
          [np.pi/3, 0, np.pi, 0]]
colours = ["#0343DF", "#00008B", "purple", "deepskyblue"]

tmaxes = np.arange(1, 4000, 100)

for i in tqdm(range(len(points))):
    pt = points[i]
    l_vals = []
    for t in tmaxes:
        l_vals.append(lyp_exp(pt, 0.01, t, T=1))
    plt.plot(tmaxes, l_vals, label=str(i), color=colours[i])
    print(l_vals)
plt.xlabel(r'Number of time steps $N$')
plt.ylabel('Maximal Lyapunov exponent')
plt.legend([r"$\theta_1(0)=\frac{1}{20} \pi, \theta_2(0)=\frac{\sqrt{2}}{20} \pi$", 
            r"$\theta_1(0)=\frac{4}{9} \pi, \theta_2(0)=\frac{4}{9} \pi$", 
            r"$\theta_1(0)=\frac{2}{3} \pi, \theta_2(0)=\frac{1}{2} \pi$",
            r"$\theta_1(0)=\frac{1}{3} \pi, \theta_2(0)=\pi$"],
            prop={'size': 10})
plt.show()
