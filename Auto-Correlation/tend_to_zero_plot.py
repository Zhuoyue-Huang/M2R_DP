import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
# from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import statistics as stats
from Sensitivity import lyp_exp
from tqdm import tqdm

# i_th1 = np.pi/10
# i_th2 = np.pi/10 * np.sqrt(2)
# i_th1_d = 0
# i_th2_d = 0
tmax = 100
nlags = 200
lags = range(50, nlags + 1)  

def principal(x):
    a = np.mod(x, 2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi
    return a

princ_val = np.vectorize(principal)

# y0 = [i_th1, i_th1_d, i_th2, i_th2_d]
# p1 = Pendulum(i_th1, i_th1_d, i_th2, i_th2_d, 100, y0, method='RK23')
# th1_sol = princ_val(p1.full_sol[2])   #p1.theta1[:-1]
# th2_sol = p1.theta2[:-1]
# n = len(th1_sol)


def principal_th2(th1, th2, th1_d=0, th2_d=0, tmax=100):
    y0 = [th1, th1_d, th2, th2_d]
    p = Pendulum(th1, th1_d, th2, th2_d, tmax, y0, method='RK23')
    return princ_val(p.full_sol[0]), princ_val(p.full_sol[2])


def auto_corr(data, lag):
    n = len(data)
    x_bar = sum(data) / n
    var_x = sum((data - x_bar) ** 2)
    cov_x = 0
    for t in range(lag, n):
        cov_x += (data[t] - x_bar) * (data[t-lag]-x_bar)
    return cov_x / var_x


def auto_corr_l(th1, th2, lags):
    sol_1, sol_2 = principal_th2(th1, th2)

    return np.array([auto_corr((sol_1), l) for l in lags]), np.array([auto_corr((sol_2), l) for l in lags])


def small_frac(th1, th2, lags):
    auto_1, auto_2 = auto_corr_l(th1, th2, lags)
    auto_1, auto_2 = np.abs(auto_1), np.abs(auto_2)

    return (len(auto_1[auto_1 < 0.0576]) + len(auto_2[auto_2 < 0.0576])) / (2 * len(lags))


delta = np.pi/60
th1_range = np.arange(-np.pi, np.pi, delta)
th2_range = np.arange(-np.pi, np.pi, delta)
points_th1, points_th2 = [], []
vals = []

for th1 in tqdm(th1_range):
    for th2 in th2_range:
        points_th1.append(th1)
        points_th2.append(th2)
        vals.append(small_frac(th1, th2, lags))

points_th1, points_th2 = np.array(points_th1), np.array(points_th2)

fig, ax = plt.subplots()
scatter = ax.scatter(points_th1, points_th2, c=vals, cmap='hot', s=30,
                     linewidth=0)
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_title("Fraction of coefficients statistically equal to 0")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=5) 
cbar.set_label('Fraction', fontsize=6, rotation=270)

plt.show()
