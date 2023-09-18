#%%
import numpy as np
import matplotlib.pyplot as plt

def make_gaussian_kernel(kernel_x, kernel_y, sigma_x, sigma_y):
    x = np.arange(-kernel_x//2, kernel_x//2+1)
    y = np.arange(-kernel_y//2, kernel_y//2+1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)))
    kernel = kernel/np.max(kernel)
    return kernel

# Create matrix of ones with shape 100 x 100
p_mat = np.ones((100, 100))
p_mat = np.pad(p_mat, ((20,20),(20,20)), 'constant', constant_values=np.nan)

# Fake fixation locations
fixlocs = [[50,40],[25,35],[30,30],[70,60],[70,60],[70,60],[70,60],[70,60],[70,60]]

kernel_x = 70
kernel = make_gaussian_kernel(kernel_x, kernel_x, 6, 6)

# Normalization factor
norm_factor = np.sum(kernel)


# Define a relaxation rate (e.g., 0.1)
relax_rate = 0

for fixloc in fixlocs:
    x_idx = np.arange(fixloc[0]-kernel_x//2, fixloc[0]+kernel_x//2 + 1)
    y_idx = np.arange(fixloc[1]-kernel_x//2, fixloc[1]+kernel_x//2 + 1)

    novelty = np.nansum(p_mat[np.ix_(x_idx, y_idx)] * (kernel / norm_factor))
    print(novelty)

    p_mat[np.ix_(x_idx, y_idx)] -= kernel
    p_mat[p_mat <= 0] = 0

    # Relaxation towards 1
    p_mat = p_mat + relax_rate * (1 - p_mat)

    # Visualize the final result
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(-p_mat.T, cmap='viridis')
    plt.show()

# %%
