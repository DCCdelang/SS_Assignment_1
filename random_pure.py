
#%%
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

import scipy.stats as stats
# from scipy.stats import norm

from mandelbrot import mandelbrot
#%%
def plot_figure(result):
    plt.figure(dpi=100)
    plt.imshow(result.T, extent=[-2, 1, -1, 1])
    plt.xlabel('real numbers')
    plt.ylabel('imaginary numbers')
    plt.show()

def pure_random(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2.25, .75
    i_min, i_max = -1.5, 1.5

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )

    # sample real and imaginary numbers from uniform distribution
    real_nrs = np.random.uniform(r_min, r_max, sample_size)
    imag_nrs = np.random.uniform(i_min, i_max, sample_size)
    
    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*total_area

    return area_m



sample_sizes = [10**2, 10**3, 10**4, 10**5, 10**6]
iterations = range(20, 400, 20)

lines = []
for sample_size in sample_sizes:
    line = []
    for iteration in iterations:
        result = pure_random(sample_size, iteration)
        line.append(result)
    print(sample_size)
    plt.plot(iterations, line, label=f"Sample size: {sample_size}")
    lines.append(line)


plt.legend()
plt.show()
#%%