import numpy as np
from DPendulum import Pendulum
from tqdm import tqdm


def calc_E(y):
    """Return the total energy of the system."""
    m1 = m2 = L1 = L2 = 1
    g = 9.81

    th1, th1d, th2, th2d = y
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    U = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
                                      2*L1*L2*th1d*th2d*np.cos(th1-th2))

    return U + V


def principal(x):
    a = np.mod(x, 2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi
    return a


def pendulum_vec(v, t):  # v = (th1, th1_dot, th2, th2_dot)
    y0 = [principal(v[0]), v[1], principal(v[2]), v[3]]
    return Pendulum(theta1=y0[0], z1=y0[1], theta2=y0[2], z2=y0[3], y0=y0,
                    tmax=t, to_trace=False, trace_delete=False, method='RK23')


def initial_err_0(eps):
    return np.array([-np.sqrt(eps/4.0) for _ in range(4)])


def lyp_exp(initial_cond, eps, tmax, T=1):
    err_0 = initial_err_0(eps)

    p1 = pendulum_vec(initial_cond, T)
    p2 = pendulum_vec(initial_cond + err_0, T)

    assert p1.dt == p2.dt and p1.tmax == p2.tmax
    lyp_ests = []

    for i in range(tmax):
        p1_state = np.array([i[-1] for i in p1.sol()])
        assert len(p1_state) == 4
        p2_state = np.array([i[-1] for i in p2.sol()])
        err_1 = p1_state - p2_state
        err1_norm = np.linalg.norm(err_1)
        err0_norm = np.linalg.norm(err_0)
        lyp_ests.append((1/T) * (np.log(err1_norm) - np.log(err0_norm)))
        err_0 = (err_1 / err1_norm) * eps  # scaled err_1

        print(calc_E(p1_state), "energy")
        p1 = pendulum_vec(p1_state, T)
        p2 = pendulum_vec(p1_state + err_0, T)
    return sum(lyp_ests) / len(lyp_ests)


if __name__ == "__main__":
    initial_cond = np.array([np.pi/3, 0, np.pi, 0])
    eps = 0.01
    tmax = 10000
    print(lyp_exp(initial_cond, eps, tmax, T=1))
