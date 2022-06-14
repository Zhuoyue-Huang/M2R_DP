import numpy as np
import matplotlib.pyplot as plt
from DPendulum import Pendulum
# from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import statistics as stats

i_th1 = np.pi/2
i_th2 = np.pi/2
i_th1_d = 0
i_th2_d = 0 

y0 = [i_th1, i_th2, i_th1_d, i_th2_d]
p1 = Pendulum(i_th1, i_th2, i_th1_d, i_th2_d, 1000, y0)
th1_sol = p1.full_sol[1]  #p1.theta1[:-1]
th2_sol = p1.theta2[:-1]
n = len(th1_sol)

lags = range(n)
sa = np.zeros((n))


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
    return np.dot(data[:-(n-lag)] - x_bar, data[n-lag:] - x_bar) / var_x



# auto_coeffs = [auto_corr(th1_sol, l) for l in lags]

plt.plot(lags[:len(lags)//2], sm.tsa.acf(th1_sol, nlags=lags[len(lags)//2 - 1]))

# markerline, stemlines, baseline = plt.stem(lags[:len(lags)//2], sm.tsa.acf(th1_sol, nlags=lags[len(lags)//2 - 1]))
plt.grid()
plt.title('Sample Autocorrealtion Function (ACF)')
plt.ylabel('Sample Autocorrelation')
plt.xlabel('Lag')

#plt.plot(lags, auto_coeffs)
plt.show()



