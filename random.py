import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot


def plot_figure(result):
    plt.figure(dpi=100)
    plt.imshow(result.T, extent=[-2, 1, -1, 1])
    plt.xlabel('real numbers')
    plt.ylabel('imaginary numbers')
    plt.show()

def random():
    # resolution
    sample_size = 10000

    rows, cols = 1000, 1000


    result = np.zeros([rows, cols])
    iterations = 100
    result = []
    for i in range(sample_size):
        real = np.random(0,rows)
        imag = np.random(0, cols)
        res = mandelbrot(real, imag, iterations)
        result.append(res)

    l = lhsmdu.createRandomStandardUniformMatrix(2, 20) # Monte Carlo sampling
    for real_index, Re in enumerate(np.linspace(-2.25, .75, num=rows )):
        for imag_index, Im in enumerate(np.linspace(-1.5, 1.5, num=cols)):
            result[real_index, imag_index] = mandelbrot(Re, Im, iterations)

    

    # plot_figure(result)

random()