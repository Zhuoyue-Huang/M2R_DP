import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
from scipy import stats
from tqdm import tqdm


nlags = 100
lags = range(1, nlags + 1)


def principal(x):
    a = np.mod(x, 2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi
    return a

princ_val = np.vectorize(principal)

def principal_angles(th1, th2, th1_d=0, th2_d=0, tmax=100):
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


def auto_corr_list(data, lags):
    return [auto_corr(data, l) for l in lags]


def ljung_box_stat(data, lags):
    L = 0
    data_n = len(data)
    corr_list = auto_corr_list(data, lags)
    for k in range(nlags):
        L += (corr_list[k] ** 2) / (data_n - k)
    return data_n * (data_n + 2) * L


def ljung_p_value(th1, th2, lags):
    th1_data, th2_data = principal_angles(th1, th2)
    L1 = ljung_box_stat(th1_data, lags)
    L2 = ljung_box_stat(th2_data, lags)
    return (1 - stats.chi2.cdf(L1, nlags)), (1 - stats.chi2.cdf(L2, nlags))


def hyp_test(L1, L2, gamma=0.005):
    return (L1 < gamma) and (L2 < gamma)


def categorise(x):
    if x < 0.05:
        return 'r'

    else:
        return 'g'


cat_vals = np.vectorize(categorise)

if __name__ == "__main__":

    delta = np.pi / 10
    th1_range = np.arange(-np.pi, np.pi, delta)
    th2_range = np.arange(-np.pi, np.pi, delta)
    th1_dense = np.arange(-np.pi, np.pi, 0.01)
    th2_dense = np.arange(-np.pi, np.pi, 0.01)

    points_th1, points_th2 = [], []
    vals = []
    for th1 in tqdm(th1_range):
        for th2 in th2_range:
            points_th1.append(th1)
            points_th2.append(th2)
            th1_sol, th2_sol = principal_angles(th1, th2)
            vals.append(np.log(min(ljung_box_stat(th1_sol, lags),ljung_box_stat(th2_sol, lags))))

    points_th1, points_th2 = np.array(points_th1), np.array(points_th2)
    vals = np.array(vals)
    ax = plt.axes(projection='3d')
    ax.scatter3D(points_th1, points_th2, vals)
    D_1, D_2 = np.meshgrid(th1_dense, th2_dense)

    ax.plot_surface(D_1, D_2, np.ones_like(D_1)*np.log(stats.chi2.ppf(0.995, nlags)),
                    linewidth=0, antialiased=False, color='red')
    ax.set_ylabel("Theta 2")
    ax.set_xlabel("Theta 1")
    ax.set_zlabel("log of the minimum Ljung-Box test statistic")
    ax.set_title("log of the minimum Ljung-Box test statistic for various initial conditions.")
    plt.show()
