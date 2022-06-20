from DPendulum import Pendulum
import numpy as np
import time

arr = np.load("fractal_img_bin_new.npy")

power = 9
th1_range = np.linspace(1.54, 2.4, 2 ** power)
step1 = th1_range[1] - th1_range[0]
th2_range = np.linspace(1.083, 1.943, 2 ** power)
step2 = th2_range[1] - th2_range[0]
max_units = 100  # NOTE that 3.5^4 is very close to 150
g = 9.81

val1 = th1_range[-1]
val2 = th2_range[-1]

# for _ in range(50):
#     val1 += step1
#     th1_range = np.append(th1_range, val1)
#     val2 += step2
#     th2_range = np.append(th2_range, val2)


time_unit = np.sqrt(1 / g)


def first_flip(theta1, theta2, max_units=100):
    global start
    if time.time() - start > 30:
        print(f"{theta1}/{th1_range[-1]}")
        start = time.time()
    p1 = Pendulum(theta1, 0, theta2, 0, tmax=max_units * time_unit,
                  y0=[theta1, 0, theta2, 0], method='RK23')

    th_abs = np.abs(np.concatenate((p1.theta1, p1.theta2)))

    return not(int((th_abs > np.pi).any()))


start = time.time()

print(arr.shape)
shift = 300
new_arr = np.zeros((512, 512))
for (i, th1) in enumerate(th1_range[shift:]):
    i += shift
    for (j, th2) in enumerate(th2_range[shift:]):
        j += shift
        arr[i][j] = first_flip(th1, th2)

np.save('fractal_img_bin_new_2', arr)
