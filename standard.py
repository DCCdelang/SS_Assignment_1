import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot

def plot_figure(result,iterations):
    plt.figure(figsize = (3,3), dpi=300) # For everything!
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
    iterations = 100
    hits = 0
    
    for real_index, Re in enumerate(np.linspace(-2.25, .75, num=rows )):
        for imag_index, Im in enumerate(np.linspace(-1.5, 1.5, num=cols)):
            result[real_index, imag_index] = mandelbrot(Re, Im, iterations)
            if result[real_index, imag_index] == iterations:
                hits+=1


    avg = hits/(rows*cols)
    area_m = avg*9
    print('Iterations=', iterations)
    print('Pixels=', rows*cols)
    print('Estimated area =', area_m)

    plot_figure(result,iterations)

standard()