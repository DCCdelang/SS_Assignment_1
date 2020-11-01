import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot


def plot_figure(result):
    plt.figure(dpi=100)
    plt.imshow(result.T, extent=[-2, 1, -1, 1])
    plt.xlabel('real numbers')
    plt.ylabel('imaginary numbers')
    plt.show()

def pure_random():
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2.25, .75
    i_min, i_max = -1.5, 1.5

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )
    sample_size = int(1e6)
    iterations = 100

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
    hits = mb_list.count(100)
    avg = hits/sample_size
    area_m = avg*total_area
    print('Calculated area = ', area_m)

    # plot_figure(result)

pure_random()