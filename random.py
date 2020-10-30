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
    r_min = -2.25
    r_max = .75

    i_min = -1.5
    i_max = 1.5
    


    sample_size = 10000
    iterations = 100

    # rows, cols = 1000, 1000

    mb_list = []
    for i in range(sample_size):
        Re = np.random.uniform(r_min, r_max, sample_size)
        Im = np.random.uniform(i_min, i_max, sample_size)
        print(Re,Im)
        mb = mandelbrot(Re, Im, iterations)
        mb_list.append(mb)
    print(mb_list)
    # plot_figure(result)

pure_random()