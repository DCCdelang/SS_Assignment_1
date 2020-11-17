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

from pyDOE import lhs
from matplotlib import rcParams

from matplotlib import rcParams
from mandelbrot import mandelbrot

# Setting random seed
from numpy.random import RandomState
rs = RandomState(420)

# define min and max real and imaginary numbers for mandelbrot function
R_MIN, R_MAX = -2, .5
I_MIN, I_MAX = -1.25, 1.25

# create extra space in plots for labels
rcParams.update({'figure.autolayout': True})

def latincube(sample_size, iterations):

    # define total area of real and imaginary numbers
    r_tot = abs(R_MIN) + abs(R_MAX)
    i_tot = abs(I_MIN) + abs(I_MAX)
    total_area = r_tot * i_tot

    # print(sample_size)
    k = lhs(2, samples = sample_size)

    # Latin Hypercube Sampling with multi-dimensional uniformity
    mb_list = []
    
    for sample in range(sample_size):
        Re = k[sample][0]*r_tot + R_MIN
        Im = k[sample][1]*i_tot + I_MIN
        mb = mandelbrot(Re, Im, iterations)
        mb_list.append(mb)

    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*total_area

    return area_m

# Plot for different Samplesize with respect to Iterations and Area

def run_latin_cube(delta, simulations, maxI, expS):
    
    fig, ax = plt.subplots()

    colour = [0,0,"b", "g", "r", "k"]
    iterations = range(25, maxI, 25)

    t0 = time.time()
    lines = []

    for exp in range(2,expS):
        sample_size = 10**exp
        line = []
        var_list = []
        ci = []
        area_list= [0]
        
        for iteration in iterations:
            samples = []

            if delta == False:
                for _ in range(simulations):
                    result = latincube(sample_size,iteration)
                    samples.append(result)
                    var_list.append(result)

                ci.append(1.96 * np.std(samples)/np.mean(samples))
                line.append(np.mean(samples))

            elif delta == True:
                for sim in range(simulations):
                    result = latincube(sample_size, iteration)
                    samples.append(result)
                    var_list.append(result)
                Area = np.mean(samples)

                Delta = Area - area_list[-1]
                area_list.append(Area)
                if len(area_list) > 1:
                    ci.append(1.96 * np.std(samples)/np.mean(samples))
                    line.append(Delta)


        # create plot for area
        ax.plot(iterations, line, color=colour[exp], label=f"S: 10^{exp}", alpha = .8)
        ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        ax.set(ylabel='Estimated area')
        ax.set_xlim(iterations[0], iterations[-1])
        ax.grid()
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

        lines.append(line)

        t1 = time.time()
        t = t1-t0
        
        area_approximation = round(line[-1],3)

        print("\nDelta:", delta,", Setting = I:", iteration, ",S:", sample_size,
        "\nFinal approx:", area_approximation, 
        "\nUpper bound:", area_approximation + ci[-1], 
        "\nLower bound:", area_approximation - ci[-1],
        "\nVariance:", statistics.variance(var_list), 
        '\nElapsed time:', round(t1-t0, 2))


    t1 = time.time()
    t = t1-t0
    print("\nThe simulation took:", round(t,3), 'seconds')

    
    plt.tight_layout()
    # plt.savefig("Figures/latincube_S_and_I.png",dpi = 300)
    plt.show()

run_latin_cube(delta=False, simulations = 30, maxI = 420, expS = 6)