import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
from Sensitivity import lyp_exp

tmax = 100
nlags = 100
lags = range(1, nlags + 1) 
eps=0.01

def principal(x):
    a = np.mod(x, 2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi
    return a
    
princ_val = np.vectorize(principal)

def principal_th2(th1, th2, th1_d=0, th2_d=0, tmax=100):
    y0 = [th1, th1_d, th2, th2_d]
    p = Pendulum(th1, th1_d, th2, th2_d, tmax, y0, method='RK23')
    return princ_val(p.full_sol[2])

def auto_corr(data, lag):
    n = len(data)
    x_bar = sum(data) / n
    var_x = sum((data - x_bar) ** 2)
    cov_x = 0
    for t in range(lag, n):
        cov_x += (data[t] - x_bar) * (data[t-lag]-x_bar)
    return cov_x / var_x

def rounded_lyp(th1, th2, th1_d=0, th2_d=0, eps=0.01, tmax=tmax, dp=3):
    return round(lyp_exp([th1, th1_d, th2, th2_d], eps=eps, tmax=1_000), dp)


def auto_corr_l(th1, th2, lags):
    return [auto_corr(principal_th2(th1, th2), l) for l in lags]
