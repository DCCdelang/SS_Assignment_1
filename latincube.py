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
    print("Sampletime:", t)

    mb_list = []
    
    for sample in range(N_sample):
        Re = k[sample][0]*r_tot + r_min
        Im = k[sample][1]*i_tot + i_min
        # mb_fast = numba.jit("void(f4[:])")(mandelbrot(Re, Im, 20))
        mb = mandelbrot(Re, Im, maxI)
        mb_list.append(mb)
    return mb_list

I = 200
S = 1000000
t0 = time.time()
sample = latincube(I,S)
t1 = time.time()
t = t1-t0

hit = sample.count(I)
print("Total time:", t)
print("Percentage hits:",hit/S)
print("Area mandelbrot:",(hit/S)*9)