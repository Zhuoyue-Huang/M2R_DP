import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
arr_corr = np.load("fractal_img_bin_new_2.npy")
arr_old = np.load("fractal_img_bin_new.npy")

arr_new = deepcopy(arr_old)

arr_new[300:, 300:] = (arr_corr[300:, :(512 - 300)]).T

plt.imshow(np.flipud(arr_new))
plt.show()

np.save("fractal_img_bin_new_3.npy", np.flipud(arr_new))
