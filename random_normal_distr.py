import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

import scipy.stats as stats

from mandelbrot import mandelbrot
#%%
def normal_distribution():
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2.25, .75
    i_min, i_max = -1.5, 1.5

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )
    sample_size = int(1e4)
    iterations = 100

    samples = 10000
    results = []
    append = results.append # define append prior to the loop should make the code faster
    for sample in range(samples):
        # sample real and imaginary numbers from uniform distribution
        real_nrs = np.random.uniform(r_min, r_max, sample_size)
        imag_nrs = np.random.uniform(i_min, i_max, sample_size)
        
        # create list to save mandelbrot iterations
        mb_list = []
        # iterate over entire sample
        for i in range(sample_size):
            mb = mandelbrot(real_nrs[i], imag_nrs[i], iterations)
            mb_list.append(mb)
        
        # calculate area of the mandelbrot
        hits = mb_list.count(100)
        avg = hits/sample_size
        area_m = avg*total_area
        
        print(sample)
        append(area_m) # this should make the algorithm faste

    return results


results =  normal_distribution()
#%%

def plot_results(results, bin_amount):
    # plot distribution of runs of MC random sampling
    plt.hist(results, bins=bin_amount, edgecolor='black')
    plt.grid()
    plt.show()

def plot_normalized_results(results, bin_amount):
    # create bins
    n, bin_edges = np.histogram(results, bin_amount)
    # calculate p per bin
    bin_probability = n/float(n.sum())
    # get mid points for bins
    bin_avg = (bin_edges[1:]+bin_edges[:-1])/2.
    # get width of bins
    bin_width = bin_edges[1]-bin_edges[0]


    # get norm distribution
    (mu, sigma) = stats.norm.fit(results)
    # calculate normalized curve for probilities
    norm_curve = stats.norm.pdf(bin_avg, mu, sigma)*bin_width

    ci = 1.96 * sigma

    plt.plot(bin_avg, norm_curve, 'r', linewidth=2)
    plt.fill_between(bin_avg, norm_curve,  color='red', alpha=.5, where = (bin_avg > mu-ci) & (bin_avg < mu + ci), hatch="/", edgecolor='black')
    plt.bar(bin_avg, bin_probability, width=bin_width, edgecolor='black', alpha=.5)
    plt.grid(True)
    plt.show()

bin_amount = 50
plot_results(results, bin_amount)
plot_normalized_results(results, bin_amount)



# %%
