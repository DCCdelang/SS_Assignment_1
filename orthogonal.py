# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: djdcc_000
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs

import random
import time

from mandelbrot import mandelbrot


#%%
# Predomantly based on the following stack overflow page 
# CHECK IF THIS CAN BE DONE?????
# See also post on canvas
# https://codereview.stackexchange.com/questions/207610/orthogonal-sampling
def orthogonal(maxI, ns):
    # assert np.sqrt(ns) % 1 == 0, "Please insert an even number of samples"
    n = int(np.sqrt(ns))
    # Making a datastructure of a dict with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
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
            point = [(point[0] + n * block[0])/ns, (point[1] + n * block[1])/ns]
            append(point)

    mb_list = []
    r_min = -2.25
    r_max = 0.75
    r_tot = abs(r_min)+abs(r_max)
    i_min = -1.5 
    i_max = 1.5
    i_tot = abs(i_min)+abs(i_max)

    for point in range(len(points)):
        Re = points[point][0]*r_tot + r_min
        Im = points[point][1]*i_tot + i_min
        # mb_fast = numba.jit("void(f4[:])")(mandelbrot(Re, Im, 20))
        mb = mandelbrot(Re, Im, maxI)
        mb_list.append(mb)

    return mb_list

# N_sample = 100
# t0 = time.time()
# sample = orthogonal(maxI,N_sample)
# t1 = time.time()

#%%
area_list1 = []
S_list = []

for exp in range(1,5):
    I = 200
    S = 10**exp

    S_list.append(S)

    t0 = time.time()
    sample = orthogonal(I,S)
    t1 = time.time()
    t = t1-t0

    hit = sample.count(I)
    area_list1.append((hit/S)*9)
    print("\nTotal time:", t)
    print("Percentage hits:",hit/S)
    print("Area mandelbrot:",(hit/S)*9)

# print(S_list,area_list1)
plt.title("Area vs samplesize (S) - I=200")
plt.xlabel("Samplesize (log)")
plt.ylabel("Area mandelbrot")
plt.plot(S_list,area_list1)
plt.xscale("log")
plt.show()

#%%
area_list2 = []
I_list = []
for mult in range(1,16):
    I = 20*mult
    S = 10**3

    I_list.append(I)

    t0 = time.time()
    sample = orthogonal(I,S)
    t1 = time.time()
    t = t1-t0

    hit = sample.count(I)
    # print(hit,len(sample))
    area_list2.append((hit/len(sample))*9)
    print("Total time:", t)
    print("Percentage hits:",hit/len(sample))
    print("Area mandelbrot:",(hit/len(sample))*9)

# print(I_list,area_list2)

plt.title("Area vs iterations (I) - S=10^5")
plt.xlabel("Max iterations")
plt.ylabel("Area mandelbrot")
plt.plot(I_list,area_list2)
plt.show()

#%%

for exp in range(2,4):
    area_list2 = []
    I_list = []
    for mult in range(1,16):
        I = 20*mult
        S = 10**exp

        I_list.append(I)

        t0 = time.time()
        sample = orthogonal(I,S)
        t1 = time.time()
        t = t1-t0

        hit = sample.count(I)
        # print(hit,len(sample))
        area_list2.append((hit/len(sample))*9)
        print("Total time:", t)
        # print("Percentage hits:",hit/len(sample))
        # print("Area mandelbrot:",(hit/len(sample))*9)

    # print(I_list,area_list2)

    plt.title("Area vs iterations (I) - S=10^5")
    plt.xlabel("Max iterations")
    plt.ylabel("Area mandelbrot")
    plt.plot(I_list,area_list2)
    plt.show()