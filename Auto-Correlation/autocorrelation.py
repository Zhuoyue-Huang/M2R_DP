import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
# from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import statistics as stats
from Sensitivity import lyp_exp

# i_th1 = np.pi/10
# i_th2 = np.pi/10 * np.sqrt(2)
# i_th1_d = 0
# i_th2_d = 0
tmax = 100
nlags = 100

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
    return princ_val(p.full_sol[2])


lags = range(1, nlags + 1)  # range(n)
# sa = np.zeros((n))


# def auto_corr_func(data, lag):
#     sum = 0
#     N = 1
#     s_bar = stats.mean(data)
#     sigma = stats.variance(data)
#     for m in range(len(data)):
#         try:
#             sum = sum + ((data[m + lag] - s_bar) * (data[m] - s_bar))
#             N += 1
#         except IndexError:
#             break
#     print(N)
#     return sum / (N * sigma)


def auto_corr(data, lag):
    n = len(data)
    x_bar = sum(data) / n
    var_x = sum((data - x_bar) ** 2)
    cov_x = 0
    for t in range(lag, n):
        cov_x += (data[t] - x_bar) * (data[t-lag]-x_bar)
    return cov_x / var_x



# auto_coeffs = [auto_corr(principal_th2(np.pi/10, -np.sqrt(2)*np.pi/10), l) for l in lags]

# plt.plot(lags, sm.tsa.acf(th1_sol, nlags=lags[-1]))

# plt.plot(lags, auto_coeffs)

# markerline, stemlines, baseline = plt.stem(lags[:len(lags)//2], sm.tsa.acf(th1_sol, nlags=lags[len(lags)//2 - 1]))

# plt.stem(lags, auto_coeffs)

eps=0.01


def rounded_lyp(th1, th2, th1_d=0, th2_d=0, eps=0.01, tmax=tmax, dp=3):
    return round(lyp_exp([th1, th1_d, th2, th2_d], eps=eps, tmax=1_000), dp)


def auto_corr_l(th1, th2, lags):
    return [auto_corr(principal_th2(th1, th2), l) for l in lags]


# print(rounded_lyp(np.pi/10, np.pi/10), rounded_lyp((2/3)*np.pi, 0), rounded_lyp((7/8)*np.pi, (7/8)*np.pi),
#       rounded_lyp(np.pi/36, 0), rounded_lyp(np.pi/10, -np.sqrt(2)*np.pi/10),
#       rounded_lyp(np.pi/5, np.sqrt(2)*np.pi/5))


# 10,000 values: -0.002 0.056 0.104 -0.003 -0.002 -0.002

# 1.000 values: 0.003 0.922 1.389 0.003 0.009 0.005



# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
# fig.subplots_adjust(hspace=1)
# ax1.stem(lags, auto_corr_l(np.pi/10, np.pi/10, lags), linefmt="orange")
# ax1.set_title("theta1=pi/10, theta2=pi/10, max lyapunov exponent=0.003", fontsize=12)
# ax1.set_ylim([-1, 1])
# ax1.set_ylabel('Sample Autocorrelation')
# ax1.set_xlabel('Lag')
# print(1)
# ax2.stem(lags, auto_corr_l((2/3)*np.pi, 0, lags), linefmt="green")
# ax2.set_title("theta1=2*pi/3, theta2=0, max lyapunov exponent=0.922", fontsize=12)
# ax2.set_ylim([-1, 1])
# print(2)
# ax3.stem(lags, auto_corr_l((7/8)*np.pi, (7/8)*np.pi, lags), linefmt="blue")
# ax3.set_title("theta1=7*pi/8, theta2=7*pi/8,, max lyapunov exponent=1.389", fontsize=12)
# ax3.set_ylim([-1, 1])
# print(3)
# ax4.stem(lags, auto_corr_l(np.pi/36, 0, lags), linefmt="magenta")
# ax4.set_title("theta1=pi/36, theta2=0, max lyapunov exponent=0.003", fontsize=12)
# ax4.set_ylim([-1, 1])
# print(4)
# ax5.stem(lags, auto_corr_l(np.pi/10, -np.sqrt(2)*np.pi/10, lags), linefmt="black")
# ax5.set_title("theta1=pi/10, theta2=-sqrt(2)*pi/10, max lyapunov exponent=0.009", fontsize=12)
# ax5.set_ylim([-1, 1])
# print(5)
# ax6.stem(lags, auto_corr_l(np.pi/5, np.sqrt(2)*np.pi/5, lags), linefmt="red")
# ax6.set_title("theta1=pi/5, theta2=sqrt(2)*pi/5, max lyapunov exponent=0.005", fontsize=12)
# ax6.set_ylim([-1, 1])
# plt.show()

fig, ax4 = plt.subplots(1,1)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5)
# ax1.stem(lags, auto_corr_l(np.pi/10, np.pi/10, lags), linefmt="orange")
# # ax1.set_title("theta1=pi/10, theta2=pi/10, max lyapunov exponent=0.003", fontsize=10)
# ax1.set_ylim([-1, 1])
# ax1.set_ylabel('Sample Autocorrelation')
# ax1.set_xlabel('Lag')
# ax1.hlines(y=0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')
# ax1.hlines(y=-0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')
# print(1)

# ax2.stem(lags, auto_corr_l((2/3)*np.pi, 0, lags), linefmt="green")
# # ax2.set_title("theta1=2*pi/3, theta2=0, max lyapunov exponent=0.922", fontsize=10)
# ax2.set_ylim([-1, 1])
# ax2.set_ylabel('Sample Autocorrelation')
# ax2.set_xlabel('Lag')
# ax2.hlines(y=0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')
# ax2.hlines(y=-0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')

# print(2)
# ax3.stem(lags, auto_corr_l(np.pi/10, -np.sqrt(2)*np.pi/10, lags), linefmt="blue")
# # ax3.set_title("theta1=pi/10, theta2=-sqrt(2)*pi/10, max lyapunov exponent=0.009", fontsize=10)
# ax3.set_ylim([-1, 1])
# ax3.set_ylabel('Sample Autocorrelation')
# ax3.set_xlabel('Lag')
# ax3.hlines(y=0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')
# ax3.hlines(y=-0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')

# print(3)
ax4.stem(lags, auto_corr_l(np.pi/5, np.sqrt(2)*np.pi/5, lags), linefmt="magenta")
# ax4.set_title("theta1=pi/5, theta2=sqrt(2)*pi/5, max lyapunov exponent=0.005", fontsize=10)
ax4.set_ylim([-1, 1])
ax4.set_ylabel('Sample Autocorrelation')
ax4.set_xlabel('Lag')
ax4.hlines(y=0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')
ax4.hlines(y=-0.0576, xmin=0, xmax=100, linewidth=2, color='black', linestyle='--')

plt.show()

# ax3.plot(x, y, color="blue")
# ax4.plot(x, y, color="magenta")
# ax5.plot(x, y, color="black")
# ax6.plot(x, y, color="red")

plt.grid()
plt.title('Sample Autocorrelation Function (ACF)')
plt.ylabel('Sample Autocorrelation')
plt.xlabel('Lag')

#plt.plot(lags, auto_coeffs)





