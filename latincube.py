# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: djdcc_000
"""
# import numpy as np
# import lhsmdu
# from numba import jit
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
from mandelbrot import mandelbrot

# @jit("void(i1[:])")
# @jit(nopython=True)
def latincube(maxI,N_sample):
    r_min = -2.25
    r_max = 0.75
    r_tot = abs(r_min)+abs(r_max)
    i_min = -1.5 
    i_max = 1.5
    i_tot = abs(i_min)+abs(i_max)
    t0 = time.time()
    k = lhs(2, samples = N_sample)
    # k = np.array(lhsmdu.sample(2, N_sample)) # Latin Hypercube Sampling with multi-dimensional uniformity
    t1 = time.time()
    t = t1-t0
    # print("Sampletime:", t)

    mb_list = []
    
    for sample in range(N_sample):
        Re = k[sample][0]*r_tot + r_min
        Im = k[sample][1]*i_tot + i_min
        # mb_fast = numba.jit("void(f4[:])")(mandelbrot(Re, Im, 20))
        mb = mandelbrot(Re, Im, maxI)
        mb_list.append(mb)
    return mb_list

area_list1 = []
S_list = []

for exp in range(2,7):
    I = 200
    S = 10**exp

    S_list.append(S)

    t0 = time.time()
    sample = latincube(I,S)
    t1 = time.time()
    t = t1-t0

    hit = sample.count(I)
    area_list1.append((hit/S)*9)
    # print("Total time:", t)
    # print("Percentage hits:",hit/S)
    # print("Area mandelbrot:",(hit/S)*9)

# print(S_list,area_list1)
plt.title("Area vs samplesize (S) - I=200")
plt.xlabel("Samplesize (log)")
plt.ylabel("Area mandelbrot")
plt.plot(S_list,area_list1)
plt.xscale("log")
plt.show()

area_list2 = []
I_list = []
for mult in range(1,16):
    I = 20*mult
    S = 10**5

    I_list.append(I)

    t0 = time.time()
    sample = latincube(I,S)
    t1 = time.time()
    t = t1-t0

    hit = sample.count(I)
    area_list2.append((hit/S)*9)
    # print("Total time:", t)
    # print("Percentage hits:",hit/S)
    # print("Area mandelbrot:",(hit/S)*9)

# print(I_list,area_list2)

plt.title("Area vs iterations (I) - S=10^5")
plt.xlabel("Max iterations")
plt.ylabel("Area mandelbrot")
plt.plot(I_list,area_list2)
plt.show()