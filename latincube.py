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
import numpy as np

#%%
# Latin Cube Function
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

#%%
# Plotting I for variable S
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

#%%
# Plotting S for variable I
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

plt.title("Area vs iterations (I) - S^5")
plt.xlabel("Max iterations")
plt.ylabel("Area mandelbrot")
plt.plot(I_list,area_list2)
plt.show()
# %%
# Plot for different Samplesize with respect to Iterations and Area

fig, ax = plt.subplots()
colour = [0,0,"k","b", "g", "r"]
for exp in range(2,6):
    area_list2 = []
    I_list = []
    N_samples = 3
    ci = []
    t0 = time.time()

    for mult in range(1,16):
        I = 20*mult
        S = 10**exp
        I_list.append(I)
        samples = []

        for _ in range(N_samples):
            sample = latincube(I,S)
            hit = sample.count(I)
            samples.append((hit/S)*9)
        ci.append(1.96 * np.std(samples)/np.mean(samples))
        area_list2.append(np.mean(samples))

    t1 = time.time()
    t = t1-t0
    print("Total time:", t)

    plt.title("Area vs iterations (I) for different S")
    plt.xlabel("Max iterations")
    plt.ylabel("Area mandelbrot")
    ax.fill_between(I_list, (np.array(area_list2)-np.array(ci)), (np.array(area_list2)+np.array(ci)), color=colour[exp], alpha=.1)
    plt.plot(I_list,area_list2,color=colour[exp], label  = "S=10^"+str(exp))

plt.legend()
plt.savefig("Figures/latincube_S_and_I.png",dpi = 300)
plt.show()

#%%
# Plot Delta for different Samplesize with respect to Iterations and Area

colour = [0,0,"k","b", "g", "r"]
fig, ax = plt.subplots()
for exp in range(2,6):
    area_list2 = [0]
    I_list = []
    delta_area = []
    N_samples = 3
    ci = []
    t0 = time.time()

    for mult in range(1,21):
        I = 20*mult
        S = 10**exp
        I_list.append(I)
        samples = []

        for _ in range(N_samples):
            sample = latincube(I,S)
            hit = sample.count(I)
            samples.append((hit/S)*9)
        Area = np.mean(samples)

        Delta = Area - area_list2[mult-1]
        area_list2.append(Area)
        if mult > 1:
            ci.append(1.96 * np.std(samples)/np.mean(samples))
            delta_area.append(Delta)

    t1 = time.time()
    t = t1-t0
    print("Total time:", t)

    plt.title("Delta area vs iterations (I) for different S")
    plt.xlabel("Max iterations")
    plt.ylabel("Area mandelbrot")
    plt.plot(I_list[1:],delta_area,color=colour[exp], label  = "S=10^"+str(exp))
    ax.fill_between(I_list[1:], (np.array(delta_area)-np.array(ci)), (np.array(delta_area)+np.array(ci)), color=colour[exp], alpha=.1)

plt.legend()
plt.savefig("Figures/latincube_Delta_S_and_I.png", dpi = 300)
plt.show()
