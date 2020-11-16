# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:54:28 2020

@author: djdcc_000
"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import statistics
import scipy.stats as stats
import time
from mandelbrot import mandelbrot
from pyDOE import lhs

# Setting random seed
from numpy.random import RandomState
rs = RandomState(420)

# Latin Cube Function

def latincube(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    r_tot = abs(r_min) + abs(r_max)
    i_tot = abs(i_min) + abs(i_max)
    total_area = r_tot * i_tot

    k = lhs(2, samples = sample_size)

    # Latin Hypercube Sampling with multi-dimensional uniformity
    mb_list = []
    
    for sample in range(sample_size):
        Re = k[sample][0]*r_tot + r_min
        Im = k[sample][1]*i_tot + i_min
        mb = mandelbrot(Re, Im, iterations)
        mb_list.append(mb)

    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*total_area

    return area_m

# Plot for different Samplesize with respect to Iterations and Area

def run_latin_cube(delta, simulations, maxI, expS):
    fig, ax = plt.subplots()
    colour = [0,0,"k","b", "g", "r"]
    iterations = range(20, maxI, 20)

    t0 = time.time()
    lines = []

    for exp in range(2,expS):
        sample_size = 10**exp
        line = []
        # iterations = []
        var_list = []
        ci = []
        area_list= []
        
        for iteration in iterations:
            samples = []

            if delta == False:
                for sim in range(simulations):
                    result = latincube(sample_size,iteration)
                    samples.append(result)
                    var_list.append(result)

                ci.append(1.96 * np.std(samples)/np.mean(samples))
                line.append(np.mean(samples))

            elif delta == True:
                for sim in range(simulations):
                    sample = latincube(iterations,sample_size)
                    hit = sample.count(iterations)
                    samples.append((hit/sample_size)*9)
                Area = np.mean(samples)

                Delta = Area - area_list[-1]
                area_list.append(Area)
                if len(area_list) > 1:
                    ci.append(1.96 * np.std(samples)/np.mean(samples))
                    line.append(Delta)

        plt.title("Area vs iterations (I) for different S")
        plt.xlabel("Max iterations")
        plt.ylabel("Area mandelbrot")
        # ax.fill_between(iterations, (np.array(area_list2)-np.array(ci)), (np.array(area_list2)+np.array(ci)), color=colour[exp], alpha=.1)
        # plt.plot(I_list,area_list2,color=colour[exp], label  = "S=10^"+str(exp))

        print(line)
        ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        plt.plot(iterations, line,color=colour[exp], label=f"Sample size: {sample_size}")
        lines.append(line)
        print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

    plt.legend()
    # plt.savefig("Figures/latincube_S_and_I.png",dpi = 300)
    plt.show()
    t1 = time.time()
    t = t1-t0
    print("Total time:", t)

run_latin_cube(delta = True, simulations = 10, maxI = 420, expS = 3)