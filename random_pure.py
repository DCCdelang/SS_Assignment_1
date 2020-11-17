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
R_MIN, R_MAX = -2, .5
I_MIN, I_MAX = -1.25, 1.25
TOTAL_AREA = (abs(R_MIN) + abs(R_MAX)) * (abs(I_MIN) + abs(I_MAX))


def pure_random(sample_size, iterations):
    '''
    Samples values from a uniform distribution 
    and calculates A_M based on the values.
    '''
    # define min and max real and imaginary numbers for mandelbrot function
    rs = RandomState(np.random.randint(0,10**7))
    

    # sample real and imaginary numbers from uniform distribution
    real_nrs = rs.uniform(R_MIN, R_MAX, sample_size)
    imag_nrs = rs.uniform(I_MIN, I_MAX, sample_size)

    # create list to save mandelbrot iterations
    mb_list = []
    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*TOTAL_AREA

    return area_m

def pure_random_circle(sample_size, iterations):
    '''
    
    '''
    # Approach of area - deleted area 
    # Math based on https://byjus.com/maths/area-segment-circle/
    Sub_area = TOTAL_AREA - 0.347 

    # sample real and imaginary numbers from uniform distribution
    real_nrs = rs.uniform(R_MIN, R_MAX, sample_size)
    imag_nrs = rs.uniform(I_MIN, I_MAX, sample_size)

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
    '''
    Simulates antihetic values for .5 times the sample size
    '''
    # define min and max for evenly spaced out square, to simulate anithetic samples
    MIN, MAX = -1.25, 1.25

    real_orig = rs.uniform(MIN, MAX, int(0.5*sample_size))
    imag_orig = rs.uniform(MIN, MAX, int(0.5*sample_size))

    real_copy = [-x for x in real_orig]
    imag_copy = [-x for x in imag_orig]

    real_nrs = [*real_orig, *real_copy]
    imag_nrs = [*imag_orig, *imag_copy]

    # shift nr .75 to the left, so we get the "undecided square"
    real_nrs = [x-.75 for x in real_nrs]

    mb_list = []
    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*TOTAL_AREA

    return area_m

def pure_random_stratified(sample_size, iterations):
    # get evenly spread out real nrs
    real_nrs = np.linspace(R_MIN, R_MAX, int(sample_size)).tolist()
    # randomly simulate imaginary nrs
    imag_nrs = rs.uniform(-1.25, 1.25, int(sample_size))
    
    # create list to save mandelbrot iterations
    mb_list = []

    # iterate over entire sample
    for i in range(sample_size):
        mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
        mb_list.append(mb)
    
    # calculate area of the mandelbrot
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*TOTAL_AREA

    return area_m


def run_random_pure(algorithm, simulations, maxI, expS):
    """ 
    Plots all combinations of s and i

    Possible algorithms to choose from are "Normal", "Circle", "Antithetic" and "Stratified"
    """
    iterations = range(25, maxI+1, 25)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

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
                elif algorithm == "Stratified":
                    result = pure_random_stratified(sample_size, iteration)

                subresult.append(result)
                var_list.append(result)

            
            # calculate area and delta 
            area = np.mean(subresult)
            if len(area_list) != 0:
                delta = area - area_list[-1] 
                delta_line.append(delta)

            # save results for lines + confidence interval
            area_list.append(area)
            ci.append(1.96 * np.std(subresult)/area)
            line.append(np.mean(subresult))
        
        # calculate confidence interval for delta
        ci_delta = 1.96 * (np.std(delta_line))
        print('ci_delta', ci_delta)

        # add line area
        ax1.plot(iterations, line, color=colour[exp], label=f"Sample size: 10^{exp}")
        ax1.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
        ax1.set(xlabel='Iterations', ylabel='Estimated area')
        ax1.set_xlim(iterations[0], iterations[-1])

        # add line for delta
        ax2.plot(iterations[1:], delta_line, color=colour[exp], label=f"Sample size: 10^{exp}")
        ax2.fill_between(iterations[1:], delta_line-ci_delta, delta_line+ci_delta, color=colour[exp], alpha=.1)
        # add line on x-axis
        ax2.axhline(0, color='black', linewidth=.5)
        ax2.set(xlabel = 'Iterations', ylabel = 'delta')
        ax2.set_xlim(iterations[1], iterations[-1])
        
        lines.append(line)
        delta_lines.append(delta_line)

        t1 = time.time()
        t = t1-t0
        area_approximation = round(line[-1],3)
        print("\nAlgorithm:", algorithm,", Setting = I:", iteration, ",S:", sample_size)
        print("\nFinal approx:", area_approximation, "\nUpper bound:", area_approximation + ci[-1], "\nLower bound:", area_approximation - ci[-1],"\nVariance:", round(statistics.variance(var_list),4), '\nElapsed time:', round(t1-t0,2))


    t1 = time.time()
    t = t1-t0
    print("The simulation took:", round(t,3), 'seconds')

    plt.legend()
    plt.show()



# run_random_pure(algorithm = "Normal", simulations = 30, maxI = 400, expS = 6)
run_random_pure(algorithm = "Stratified", simulations = 300, maxI = 400, expS = 6)


