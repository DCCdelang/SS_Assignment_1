import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot

def plot_figure(result,iterations):
    plt.figure(figsize = (6,6), dpi=300) # For everything!
    # plt.axis('scaled')
    plt.imshow(result.T, extent=[-2, 1, -1, 1], aspect = 'auto')
    plt.title("Mandelbrot set with color scheme"+str(iterations))
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    # plt.savefig("Figures/Mandelbrot_visual_"+str(iterations)+"_iter.png")
    plt.show()

def standard():
    # resolution
    rows, cols = 1000, 1000

    result = np.zeros([rows, cols])
    iterations = 20
    
    for real_index, Re in enumerate(np.linspace(-2.25, .75, num=rows )):
        for imag_index, Im in enumerate(np.linspace(-1.5, 1.5, num=cols)):
            result[real_index, imag_index] = mandelbrot(Re, Im, iterations)

    area = np.count_nonzero(result)

    pixelated_proportion = area / (rows*cols)
    print('Proportion of image that contains colored pixels = ', pixelated_proportion)
    avg_pixel_value = np.mean(result)/iterations
    print('Average value per pixel = ', avg_pixel_value)
    total_area = 3*3
    print('Total area =', total_area)

    print('Total area of mandelbrot =', pixelated_proportion * avg_pixel_value * total_area)

    plot_figure(result,iterations)

standard()