import numpy as np
import matplotlib.pyplot as plt


arr = np.load("fractal_img_bin_new_3.npy")

def split_matrix(arr, power):
    m_len = arr.shape[0]
    x = list(range(0, 512, int(512 / (2 ** power))))
    y = list(range(0, 512, int(512 / (2 ** power))))

    x.append(m_len)
    y.append(m_len)
    m_list = []

    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            m_list.append(arr[x[i]:x[i+1], y[j]:y[j+1]])

    return m_list

def fractal_count(m_list):
    count = 0
    for m in m_list:
        if (m == 0).any() and (m == 1).any():
            count = count + 1

    return count

def box_count(arr):
    """2 ** Power is how many intervals to split the rows and columns into. """
    frac_count = []
    r_list = []
    assert int(np.log2(arr.shape[0])) == np.log2(arr.shape[0])
    for power in range(2, int(np.log2(arr.shape[0]))):
        m_list = split_matrix(arr, power)
        r = (2.4 - 1.54) / (2 ** power)
        r_list.append(r)
        frac_count.append(fractal_count(m_list))
    return np.array(r_list), np.array(frac_count)



r_points, f_points = box_count(arr)

r_points, f_points = np.log(1 / r_points), np.log(f_points)

m, c = np.polyfit(r_points, f_points, 1)

fig, ax = plt.subplots()
scatter = ax.scatter(r_points, f_points,
                     linewidth=1)
ax.set_xlabel('log(1/r)')
ax.set_ylabel('log(N(r)')
ax.set_xlim([0, 8])
ax.set_ylim([0, 8])
ax.set_title("Scatter Plot to Determine Fractal Dimension")
ax.text(3, 0.5, f"""Line of best fit:
                       gradient = {round(m, 2)}
                       intercept = {round(c, 2)}""", bbox=dict(facecolor='green', alpha=0.5))
plt.plot(range(0, 8), m*range(0, 8) + c, color='r')
plt.show()
plt.imshow(arr)
plt.show()

