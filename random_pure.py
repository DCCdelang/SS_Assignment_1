import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

import math
import time
import statistics

from mandelbrot import mandelbrot

# Setting random seed
from numpy.random import RandomState
rs = RandomState(420)

""" Defining functions for three different approaches to pure random sampling """

def pure_random(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max))

    # sample real and imaginary numbers from uniform distribution
    real_nrs = rs.uniform(r_min, r_max, sample_size)
    imag_nrs = rs.uniform(i_min, i_max, sample_size)

    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*total_area

    return area_m

def pure_random_circle(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max))

    Sub_area = total_area - 0.347 # Approach of deleted area

    # sample real and imaginary numbers from uniform distribution
    real_nrs = rs.uniform(r_min, r_max, sample_size)
    imag_nrs = rs.uniform(i_min, i_max, sample_size)

    # Importance sampling circle:
    Imp_real = []
    Imp_imag = []
    for i in range(sample_size):
        if real_nrs[i]**2 + imag_nrs[i]**2 <= 4:
            Imp_real.append(real_nrs[i])
            Imp_imag.append(imag_nrs[i])

    imp_sample_size = len(Imp_real)

    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(imp_sample_size):
        mb = mandelbrot(Imp_real[i], Imp_imag[i], iterations)
        mb_list.append(mb)
    
    hits = mb_list.count(iterations)
    avg = hits/imp_sample_size
    area_m = avg*Sub_area

    return area_m

def pure_random_antithetic(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )

    # sample real and imaginary numbers from uniform distribution
    real_pos1 = rs.uniform(-2, 2, int(0.5*sample_size))
    real_neg1 = [-x for x in real_pos1]
    # real_pos2 = rs.uniform(0, -2, int(0.25*sample_size))
    # real_neg2 = [-x for x in real_pos2]
    real_nrs = [*real_pos1, *real_neg1,]

    imag_pos1 = rs.uniform(i_min, i_max, int(0.5*sample_size))
    imag_neg1 = [-x for x in imag_pos1]
    # imag_pos2 = rs.uniform(0, i_min, int(0.25*sample_size))
    # imag_neg2 = [-x for x in imag_pos2]
    imag_nrs = [*imag_pos1, *imag_neg1]

    # Fitting
    Imp_real = []
    Imp_imag = []
    for i in range(sample_size):
        # if real_nrs[i]**2 + imag_nrs[i]**2 <= 4:
        if real_nrs[i] > -2 and real_nrs[i] < .5:
            Imp_real.append(real_nrs[i])
            Imp_imag.append(imag_nrs[i])
    sub_size = len(Imp_real)
    
    print(Imp_real)
    print(Imp_imag)

    plt.scatter(real_nrs,imag_nrs)
    plt.show()
    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sub_size
    area_m = avg*total_area

    return area_m

""" Possible algorithms to choose are Classic (CL), Circle (CI) or Antithetic (AN). Plotting all
"""
def run_random_pure(algorithm, simulations, maxI, expS):
    iterations = range(25, maxI+1, 25)
    fig, ax = plt.subplots()
    colour = [0,0,"k","b", "r", "g"]
    t0 = time.time()
    lines = []

    # First loop with exponents for 
    for exp in range(2,expS):
        sample_size = 10**exp
        line = []
        var_list = []
        ci = []

        for iteration in iterations:
            subresult = []
            for subsamp in range(simulations):
                if algorithm == "CI":
                    result = pure_random(sample_size, iteration)
                elif algorithm == "CL":
                    result = pure_random_circle(sample_size, iteration)
                elif algorithm == "AN":
                    result = pure_random_antithetic(sample_size, iteration)

                subresult.append(result)
                var_list.append(result)

            ci.append(1.96 * np.std(subresult)/np.mean(subresult))
            line.append(np.mean(subresult))

        ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        plt.plot(iterations, line,color=colour[exp+1], label=f"Sample size: {sample_size}")
        lines.append(line)
        print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

    plt.legend()
    plt.show()
    t1 = time.time()
    t = t1-t0
    print("Time is:", t)

run_random_pure(algorithm = "CI", simulations = 10, maxI = 400, expS = 5)
# run_random_pure(algorithm = "CL", simulations = 4, maxI = 420, expS = 5)
# run_random_pure(algorithm = "AN", simulations = 1, maxI = 60, expS = 3)



