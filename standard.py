# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: Dante de Lang (11014083) & Karim Semin (11285990)
"""
import numpy as np
import matplotlib.pyplot as plt

from mandelbrot import mandelbrot

# Setting global variables
left = -2
right = .5
bottom = -1.25
top = 1.25

def plot_figure(result,iterations):
    """
    Plot function to create image of the Mandelbrot set
    """
    # the figsize is based on 2x the total area
    plt.figure(figsize = (5, 5), dpi=300) 
    plt.imshow(result.T, extent=[left, right, bottom, top], aspect = 'auto')
    # plt.title(f'Mandelbrot set with color scheme {iterations}')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.savefig("Figures/Mandelbrot_visual_"+str(iterations)+"_iter.png")
    plt.show()

def standard():
    """
    Function to get all pixels values needed for the plot_figure function
    """
    rows, cols = 1000, 1000
    result = np.zeros([rows, cols])
    iterations = 400
    hits = 0

    for real_index, Re in enumerate(np.linspace(left, right, num=rows )):
        for imag_index, Im in enumerate(np.linspace(bottom, top, num=cols)):
            result[real_index, imag_index] = mandelbrot(Re, Im, iterations)
            if result[real_index, imag_index] == iterations:
                hits+=1

    total_area = (abs(left) + right) * (abs(bottom)+top)
    avg = hits/(rows*cols)
    area_m = avg*total_area
    print('Iterations=', iterations)
    print('Pixels=', rows*cols)
    print('Estimated area =', area_m)

    plot_figure(result,iterations)

standard()