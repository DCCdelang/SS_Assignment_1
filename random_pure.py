import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

import math
import statistics
import time
import random


from mandelbrot import mandelbrot

# Setting random seed
from numpy.random import RandomState
rs = RandomState(420)

""" Defining functions for three different approaches to pure random sampling
pure_random serves as a bases for all three.  
"""

def pure_random(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    rs = RandomState(np.random.randint(0,10**7))
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

    # Approach of area - deleted area 
    # Math based on https://byjus.com/maths/area-segment-circle/
    Sub_area = total_area - 0.347 

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
    
    # print(len(Imp_real))
    # print(Imp_imag)

    # plt.scatter(Imp_real,Imp_imag)
    # plt.show()
    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(sub_size):
        mb = mandelbrot(Imp_real[i], Imp_imag[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sub_size
    area_m = avg*total_area

    return area_m

def pure_random_antithetic_karim(sample_size, iterations):


    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )

    # real_uni = rs.uniform(-1.25, 1.25, int(0.5*sample_size))
    # imag_uni = rs.uniform(-1.25, 1.25, int(0.5*sample_size))

    # real_dup = [-x for x in real_uni]
    # imag_dup = [-x for x in imag_uni]

    # real_nrs = [*real_uni, *real_dup]
    # imag_nrs = [*imag_uni, *imag_dup]

    # # shift nr .75 to the left, so we get the "undecided square"
    # real_nrs = [x-.75 for x in real_nrs]

    real_nrs = np.linspace(r_min, r_max, int(sample_size)).tolist()
    imag_nrs = rs.uniform(-1.25, 1.25, int(sample_size))
    # random.shuffle(real_nrs)
    
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

""" 
Possible algorithms to choose are Classic (CL), Circle (CI) or Antithetic (AN). 
Plotting all combinations of S and I
"""
def run_random_pure(algorithm, simulations, maxI, expS):
    iterations = range(25, maxI+1, 25)
    fig, (ax1, ax2) = plt.subplots(2)
    
    colour = [0 , 0,"b", "r", "g", "k"]
    t0 = time.time()
    lines = []
    delta_lines = []

    # First loop with exponents for 
    for exp in range(2,expS):
        sample_size = 10**exp
        line = []
        var_list = []
        ci = []

        area_list = []
        delta_line = []

        for iteration in iterations:
            
            subresult = []
            
            for _ in range(simulations):
                if algorithm == "Normal":
                    result = pure_random(sample_size, iteration)
                elif algorithm == "Circle":
                    result = pure_random_circle(sample_size, iteration)
                elif algorithm == "Antithetic":
                    result = pure_random_antithetic(sample_size, iteration)
                elif algorithm == "Antithetic2":
                    result = pure_random_antithetic_karim(sample_size, iteration)

                subresult.append(result)
                var_list.append(result)
            
            # calculate area and delta 
            area = np.mean(subresult)
            delta = area - area_list[-1] if len(area_list) != 0 else 0

            # save results for lines + confidence interval
            area_list.append(area)
            ci.append(1.96 * np.std(subresult)/area)
            line.append(np.mean(subresult))
            delta_line.append(delta)

        # add line area
        ax1.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        ax1.plot(iterations, line,color=colour[exp], label=f"Sample size: 10^{exp}")
        ax1.set(xlabel='Iterations', ylabel='Estimated area')

        # add line for delta
        ax2.plot(iterations, delta_line,color=colour[exp], label=f"Sample size: 10^{exp}")
        ax2.set(xlabel = 'Iterations', ylabel = 'delta')

        lines.append(line)
        delta_lines.append(delta_line)

        t1 = time.time()
        t = t1-t0
    
        print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list), '\nElapsed time:', round(t1-t0,2))


    t1 = time.time()
    t = t1-t0
    print("The simulation took:", round(t,3), 'seconds')

    plt.legend()
    plt.show()
    


run_random_pure(algorithm = "Normal", simulations = 30, maxI = 300, expS = 5)
# run_random_pure(algorithm = "Antithetic2", simulations = 100, maxI = 300, expS = 3)

# pure_random_antithetic(sample_size = 10, iterations = 300)

