# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: Dante de Lang (11014083) & Karim Semin (11285990)
"""

import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import statistics
import random
import time

from matplotlib import rcParams
from mandelbrot import mandelbrot
rcParams.update({'figure.autolayout': True})

# Global variables: bounds of the undecided square
R_MIN, R_MAX = -2, .5
I_MIN, I_MAX = -1.25, 1.25
TOTAL_AREA = (abs(R_MIN) + abs(R_MAX)) * (abs(I_MIN) + abs(I_MAX))

def orthogonal(sample_size, maxI):
    """
    This function is states the orthogonal sampling method and is 
    predomantly based on this stack overflow discussion:
    https://codereview.stackexchange.com/questions/207610/orthogonal-sampling
    """
    r_tot = abs(R_MIN)+abs(R_MAX)
    i_tot = abs(I_MIN)+abs(I_MAX)

    # assert np.sqrt(ns) % 1 == 0, "Please insert an even number of samples"
    n = int(np.sqrt(sample_size))

    # create datastructure with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
    blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
    points = [] #np.empty((n,2))
    append = points.append # tips of python to fasten up append call

    for block in blocks:
        point = random.choice(blocks[block])
        lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
        lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]

        for col in lst_col:
            blocks[col] = [a for a in blocks[col] if a[1] != point[1]]

        for row in lst_row:
            blocks[row] = [a for a in blocks[row] if a[0] != point[0]]

        #Adjust the points to fit the grid they fall in  
        x =  R_MIN + np.random.uniform((point[0] + n * block[0])/sample_size, 
        (point[0] + n * block[0])/sample_size +1/sample_size ) * r_tot  
        y =  I_MIN + np.random.uniform((point[1] + n * block[1])/sample_size, 
        (point[1] + n * block[1])/sample_size + 1/sample_size ) * i_tot
        point = [x,y]

        append(point)

    mb_list = []
    for point in range(len(points)):
        Re = points[point][0]
        Im = points[point][1]
        mb = mandelbrot(Re, Im, maxI)
        mb_list.append(mb)

    hits = mb_list.count(maxI)
    avg = hits/len(points)
    area_m = avg*TOTAL_AREA

    return area_m, len(points)

def run_orthogonal(simulations, maxI, expS):
    """
    This function iterates over S and I and produces the plots displaying
    the convergence of the area of M using Orthogonal sampling.
    """
    fig, ax = plt.subplots()
    colour = ["b","r", "g", "k"]
    iterations = range(25, maxI+1, 25)
    samplesizes = range(10,expS,10)
    t0 = time.time()
    lines = []
    exp = -1

    for factor in samplesizes:
        exp += 1
        sample_size = factor**2
        line = []
        var_list = []
        ci = []

        for iteration in iterations:
            # print(iteration)
            samples = []
            for _ in range(simulations):
                result = orthogonal(sample_size,iteration)
                area = result[0]
                samples.append(area)
                var_list.append(area)

            ci.append(1.96 * np.std(samples)/np.mean(samples))
            line.append(np.mean(samples))

        # create plot for area
        ax.plot(iterations, line, color=colour[exp], label=f"S: {factor**2}", alpha = .8)
        ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        ax.set(ylabel='Estimated area')
        ax.set_xlim(iterations[0], iterations[-1])
        ax.grid()
        ax.legend(loc='best', bbox_to_anchor=(1, 0.5))

        lines.append(line)

        area_approximation = round(line[-1],3)
        t1 = time.time()

        print("\nSetting = I:", iteration, ",S:", sample_size,
        "\nFinal approx:", area_approximation, 
        "\nUpper bound:", area_approximation + ci[-1], 
        "\nLower bound:", area_approximation - ci[-1],
        "\nVariance:", statistics.variance(var_list), 
        '\nElapsed time:', round(t1-t0, 2))

    plt.xlabel("Iterations")
    t1 = time.time()
    t = t1-t0
    print("Total time:", t)
    plt.show()
    

run_orthogonal(simulations = 30, maxI = 400, expS = 50)