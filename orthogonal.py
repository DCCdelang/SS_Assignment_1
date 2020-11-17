# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: djdcc_000
"""

import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import statistics
import random
import time

from mandelbrot import mandelbrot

# bounds of the undecided square
R_MIN, R_MAX = -2, .5
I_MIN, I_MAX = -1.25, 1.25
TOTAL_AREA = (abs(R_MIN) + abs(R_MAX)) * (abs(I_MIN) + abs(I_MAX))


# Predomantly based on the following stack overflow page:
# https://codereview.stackexchange.com/questions/207610/orthogonal-sampling
def orthogonal(sample_size, maxI):
    r_tot = abs(R_MIN)+abs(R_MAX)
    i_tot = abs(I_MIN)+abs(I_MAX)

    # sample_size = int(sample_size/(sample_size*np.sqrt(sample_size)))
    sample_size_adapt = np.sqrt(sample_size)*sample_size

    # assert np.sqrt(ns) % 1 == 0, "Please insert an even number of samples"
    n = int(np.sqrt(sample_size_adapt))


    # create datastructure with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
    blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
    points = [] #np.empty((n,2))
    append = points.append # tips of python to fasten up append call

    for block in blocks:
        point = random.choice(blocks[block])
        # print(point)
        lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
        lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]

        # print(len(lst_row))
        for col in lst_col:
            blocks[col] = [a for a in blocks[col] if a[1] != point[1]]
        # print(lst_col)

        for row in lst_row:
            blocks[row] = [a for a in blocks[row] if a[0] != point[0]]
            #Adjust the points to fit the grid they fall in  
            point = [(point[0] + n * block[0])/sample_size_adapt, (point[1] + n * block[1])/sample_size_adapt]
            append(point)
        # print(lst_row)

    mb_list = []
    # print(points)
    for point in range(len(points)):
        Re = points[point][0]*r_tot + R_MIN
        Im = points[point][1]*i_tot + I_MIN
        # print(points[point], Re,Im)
        mb = mandelbrot(Re, Im, maxI)
        mb_list.append(mb)
        # plt.scatter(Re,Im,c=mb)
    #     if mb == maxI:
    #         plt.scatter(Re, Im, c="r")
    #     else:
    #         plt.scatter(Re, Im, c="b")
    # plt.show()

    print("mb:",len(mb_list))
    print("Points:", len(points))
    hits = mb_list.count(maxI)
    avg = hits/len(points)
    print(avg)
    print(TOTAL_AREA)
    area_m = avg*TOTAL_AREA
    print((hits/sample_size)*TOTAL_AREA)
    print("Area:", area_m)

    return area_m, len(points)

# orthogonal(100, 400)

def run_orthogonal(simulations, maxI, expS):
    _, ax = plt.subplots()
    colour = [0,0,"k","b", "g", "r"]
    iterations = range(20, maxI, 20)

    t0 = time.time()
    lines = []

    for exp in range(2,expS):
        sample_size = 10**exp
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

        plt.title("Area vs iterations (I) for different S")
        plt.xlabel("Max iterations")
        plt.ylabel("Area mandelbrot")
        # ax.fill_between(iterations, (np.array(area_list2)-np.array(ci)), (np.array(area_list2)+np.array(ci)), color=colour[exp], alpha=.1)
        # plt.plot(I_list,area_list2,color=colour[exp], label  = "S=10^"+str(exp))

        # print(line)
        ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        plt.plot(iterations, line,color=colour[exp], label=f"Sample size: {sample_size}")
        lines.append(line)
        # print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

    plt.legend()
    # plt.savefig("Figures/latincube_S_and_I.png",dpi = 300)
    plt.show()
    t1 = time.time()
    t = t1-t0
    print("Total time:", t)

run_orthogonal(simulations = 10, maxI = 420, expS = 5)