import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot

left = -2
right = .75
bottom = -1.25
top = 1.25

def plot_figure(result,iterations):
    # plt.figure(figsize = (5,5), dpi=50) # For everything!
    plt.figure(figsize = (5.5, 5), dpi=300) # For everything!
    # plt.axis('scaled')
    plt.imshow(result.T, extent=[left, right, bottom, top], aspect = 'auto')
    # plt.title(f'Mandelbrot set with color scheme {iterations}')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.savefig("Figures/Mandelbrot_visual_"+str(iterations)+"_iter.png")
    plt.show()

def standard():
    # resolution
    rows, cols = 1000, 1000

    result = np.zeros([rows, cols])
    iterations = 100
    hits = 0
    
    # left = -2
    # right = .75
    # bottom = -1.25
    # top = 1.25


    for real_index, Re in enumerate(np.linspace(left, right, num=rows )):
        for imag_index, Im in enumerate(np.linspace(bottom, top, num=cols)):
            result[real_index, imag_index] = mandelbrot(Re, Im, iterations)
            if result[real_index, imag_index] == iterations:
                hits+=1

    total_area = (abs(left) + right) * (abs(bottom)+top)
    print(total_area)
    avg = hits/(rows*cols)
    area_m = avg*total_area
    print('Iterations=', iterations)
    print('Pixels=', rows*cols)
    print('Estimated area =', area_m)

    plot_figure(result,iterations)

standard()