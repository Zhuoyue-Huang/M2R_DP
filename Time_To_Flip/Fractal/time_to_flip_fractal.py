from DPendulum import Pendulum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

g = 9.81
dt = 0
max_units = 100  # NOTE that 3.5^4 is very close to 150
time_unit = np.sqrt(1 / g)


def step_to_time_unit(s, dt):
    # print(s)
    return (s * dt) / time_unit


t_conv = np.vectorize(step_to_time_unit)


def categorise(x):
    if x < 0:
        return 'w'
    else:
        return 'k'


cat = np.vectorize(categorise)

w_corner_l = [[(1.205, 1.365), (1.470, 0.955)],
            [(1.358, 1.365), (1.570, 1.067)],
            [(1.358, 1.365), (1.635, 1.154)],
            [(1.519, 1.363), (1.670, 1.159)],
            [(1.642, 1.390), (1.731, 1.186)],
            [(1.654, 1.402), (1.812, 1.251)],
            [(1.654, 1.402), (1.884, 1.282)],
            [(1.654, 1.402), (1.979, 1.308)],
            [(1.654, 1.402), (2.047, 1.347)],
            [(1.733, 1.431), (2.051, 1.353)],
            [(1.793, 1.470), (2.082, 1.438)],
            [(1.837, 1.502), (2.110, 1.464)],
            [(1.898, 1.527), (2.110, 1.468)],
            [(1.956, 1.564), (2.139, 1.535)],
            [(2.014, 1.589), (2.133, 1.530)]]

b_corner_l = [[(1.18, 1.758), (2.250, 1.639)],
              [(1.18, 1.758), (1.759, 1.511)],
              [(1.530, 0.763), (2.242, 1.001)],
              [(1.791, 1.001), (2.258, 1.197)],
              [(1.921, 1.120), (2.266, 1.290)],
              [(1.8472, 1.5556), (1.7423, 1.6282)],
              [(1.6663, 1.4623), (1.5081, 1.6748)]]

            #   [(2.1687, 1.2737), (2.2288, 1.4597)],
            #   [(2.1857, 1.4785), (2.2276, 1.5660)]]


def rect_check(p, c):
    max_x, min_x = max(c[0][0], c[1][0]), min(c[0][0], c[1][0])
    max_y, min_y = max(c[0][1], c[1][1]), min(c[0][1], c[1][1])
    if min_x < p[0] and p[0] < max_x and min_y < p[1] and p[1] < max_y:
        return True
    else:
        return False


def all_rect(p, col='w'):
    if col == 'w':
        for l in w_corner_l:
            if rect_check(p, l):
                return True

    elif col == 'k':
        for l in b_corner_l:
            if rect_check(p, l):
                return True

    return False


# def first_flip(a, b, theta1, theta2, max_units=100):
#     global dt, bin_img
#     print(f"{a}/{len(th1_range)}")
#     if all_rect((theta1, theta2), col='w'):
#         bin_img[a][b] = 1
#         return 'w'

#     if all_rect((theta1, theta2), col='k'):
#         return 'k'

#     p1 = Pendulum(theta1, 0, theta2, 0, tmax=max_units * time_unit,
#                   y0=[theta1, 0, theta2, 0], method='RK23')
#     dt = p1.dt

#     for (th1, th2) in zip(p1.theta1, p1.theta2):
#         if abs(th1) > np.pi or abs(th2) > np.pi:
#             return 'k'
#     bin_img[a][b] = 1
#     return 'w'

def first_flip(theta1, theta2, max_units=100):
    global dt, count, start, th1_set
    if (time.time() - start) > 10:
        th1_set.add(theta1)
        print(f"{len(th1_set)} / {len(th1_range)}")
        start = time.time()
    if all_rect((theta1, theta2), col='w'):
        return 1

    if all_rect((theta1, theta2), col='k'):
        return 0

    p1 = Pendulum(theta1, 0, theta2, 0, tmax=max_units * time_unit,
                  y0=[theta1, 0, theta2, 0], method='RK23')
    dt = p1.dt
    th_abs = np.abs(np.concatenate((p1.theta1, p1.theta2)))
    # return int(np.any(th1_abs[th1_abs > np.pi]) or np.any(th2_abs[th2_abs > np.pi]))
    return not((th_abs > np.pi).any())  # int(not(np.any(th_abs[th_abs > np.pi])))

    # if np.abs(th1)[]

    # for (th1, th2) in zip(p1.theta1, p1.theta2):
    #     if abs(th1) > np.pi or abs(th2) > np.pi:
    #         return 0
    return 1


if __name__ == "__main__":
    # delta = np.pi / 100
    power = 9
    # th1_range = np.linspace(1.54, 2.2, 2 ** power)
    # th2_range = np.linspace(1.083, 1.673, 2 ** power)

    th1_range = np.linspace(1.54, 2.4, 2 ** power)
    th2_range = np.linspace(1.083, 1.943, 2 ** power)

    points_th1, points_th2 = [], []
    vals = []

    flip_mat = np.vectorize(first_flip)
    start = time.time()
    th1_set = set()

    X, Y = np.meshgrid(th1_range, th2_range)
    count = 0
    bin_img = flip_mat(X, Y)




    # for a in tqdm(range(len(th1_range))):
    #     th1 = th1_range[a]
    #     for b in range(len(th2_range)):
    #         th2 = th2_range[b]
    #         points_th1.append(th1)
    #         points_th2.append(th2)
    #         vals.append(first_flip(a, b, th1, th2, max_units=max_units))

    # print((sum(vals) / len(vals)) * 100, '%')

   # points_th1, points_th2 = np.array(points_th1), np.array(points_th2)


    # cols = cat(vals)

    # fig, ax = plt.subplots()
    # fig = plt.figure(facecolor='black')
    # plt.scatter(points_th1, points_th2, c=vals, s=10, linewidth=0)
    # plt.axis("off")

    # ax.set_xlabel('Theta 1')
    # ax.set_ylabel('Theta 2')
    # ax.set_xlim([1.4, 2.3])
    # ax.set_ylim([1, 1.75])
    # ax.set_facecolor('black')

    # ax.set_title("Time units it takes either pendulum to flip")

    # cbar = fig.colorbar(scatter)
    # cbar.ax.tick_params(labelsize=5)
    # cbar.set_label('number of time units', fontsize=6, rotation=270)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    np.save('fractal_img_bin_new', bin_img)
    # plt.savefig("fractal_boundary.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
