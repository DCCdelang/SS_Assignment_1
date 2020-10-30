# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: djdcc_000
"""
import numpy as np
import lhsmdu
from numba import jit

# mandelbrot formula
def mandelbrot(real_nr, imag_nr, max_iter):
    c = complex(real_nr, imag_nr)
    z = 0.0j # define z as complex nr
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iter

r_min = -2
r_max = 1
r_tot = abs(r_min)+abs(r_max)
i_min = -1 
i_max = 1
i_tot = abs(i_min)+abs(i_max)

N_sample = 20

# l = lhsmdu.createRandomStandardUniformMatrix(2, 20) # Monte Carlo sampling
k = np.array(lhsmdu.sample(2, N_sample)) # Latin Hypercube Sampling with multi-dimensional uniformity


mb_list = []

for sample in range(N_sample):
    Re = k[0][sample]*r_tot + r_min
    Im = k[1][sample]*i_tot + i_min
    print(Re,Im)
    # mb_fast = numba.jit("void(f4[:])")(mandelbrot(Re, Im, 20))
    mb = mandelbrot(Re, Im, I)
    mb_list.append(mb)
print(mb_list)
