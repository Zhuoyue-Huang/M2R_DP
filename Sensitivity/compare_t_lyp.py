import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Sensitivity.lyapunov_time_estimate import lyp_exp

points = [[np.pi/20, 0, np.pi/20*np.sqrt(2), 0],
          [0, 0, np.pi/3, 0],
          [2/3*np.pi, 0, np.pi/2, 0],
          [np.pi/2, 0, np.pi/4, 0]]

tmaxes = [10, 30, 60, 100, 400, 700, 900, 1500, 3000, 7000, 10000]
lnt = [np.log(t) for t in tmaxes]

for i in tqdm(range(len(points))):
    pt = points[i]
    l_vals = []
    for t in tmaxes:
        l_vals.append(lyp_exp(pt, 0.01, t, T=1))
    plt.plot(lnt, l_vals, label=str(i))
plt.xlabel(r"$log(N)$")
plt.ylabel('Maximum Lyapuonv exponent estimate')
plt.legend()
plt.show()
