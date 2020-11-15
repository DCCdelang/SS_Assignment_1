
#%%
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import statistics
import scipy.stats as stats
# from scipy.stats import norm
import time
from mandelbrot import mandelbrot

# Setting random seed
from numpy.random import RandomState
rs = RandomState(420)

#%%
# Function for plotting and random sampling 

def plot_figure(result):
    plt.figure(dpi=100)
    plt.imshow(result.T, extent=[-2, 1, -1, 1])
    plt.xlabel('real numbers')
    plt.ylabel('imaginary numbers')
    plt.show()

def pure_random(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2, .5
    i_min, i_max = -1.25, 1.25

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max) )

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
    real_uni = rs.uniform(0, 1.25, int(0.5*sample_size))
    real_pos = [x-0.75 for x in real_uni]
    real_neg = [-x-0.75 for x in real_uni]
    real_nrs = [*real_pos, *real_neg]

    imag_pos = rs.uniform(0, i_max, int(0.5*sample_size))
    imag_neg = [-x for x in imag_pos]
    imag_nrs = [*imag_pos, *imag_neg]

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


#%%
# Function tests

sample_sizes = [10**2,10**3,10**4,10**5]
iterations = range(20, 420, 20)

t0 = time.time()
lines = []
for sample_size in sample_sizes:
    line = []
    for iteration in iterations:
        result = pure_random(sample_size, iteration)
        line.append(result)
    plt.plot(iterations, line, label=f"Sample size: {sample_size}")
    lines.append(line)
    print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(line))

plt.legend()
plt.show()
t1 = time.time()
t = t1-t0
print("Time is:", t)

t0 = time.time()
lines = []
for sample_size in sample_sizes:
    line = []
    for iteration in iterations:
        result = pure_random_circle(sample_size, iteration)
        line.append(result)
    plt.plot(iterations, line, label=f"Sample size: {sample_size}")
    lines.append(line)
    print("Setting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(line))

plt.legend()
plt.show()
t1 = time.time()
t = t1-t0
print("Time is:", t)


# %%
# With confidence intervals

print("CLASSIC")
# sample_sizes = [10**2,10**3,10**4,10**5]
iterations = range(20, 420, 20)
conf_samples = 100
fig, ax = plt.subplots()
colour = [0,0,"k","b", "g", "r"]

t0 = time.time()
lines = []
for exp in range(3,4):
    sample_size = 10**exp
    line = []
    var_list = []
    ci = []
    for iteration in iterations:
        subresult = []
        for subsamp in range(conf_samples):
            result = pure_random(sample_size, iteration)
            subresult.append(result)
            var_list.append(result)

        ci.append(1.96 * np.std(subresult)/np.mean(subresult))
        line.append(np.mean(subresult))

    ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
    plt.plot(iterations, line,color=colour[exp], label=f"Sample size: {sample_size}")
    lines.append(line)
    print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

plt.legend()
plt.show()
t1 = time.time()
t = t1-t0
print("Time is:", t)

#%%
# With confidence intervals with circle limitation
print("CIRCLE")
# sample_sizes = [10**2,10**3,10**4,10**5]
iterations = range(20, 420, 20)
conf_samples = 100
fig, ax = plt.subplots()
colour = [0,0,"k","b", "g", "r"]

t0 = time.time()
lines = []
for exp in range(3,4):
    sample_size = 10**exp
    line = []
    var_list = []
    ci = []
    for iteration in iterations:
        subresult = []
        for subsamp in range(conf_samples):
            result = pure_random_circle(sample_size, iteration)
            subresult.append(result)
            var_list.append(result)

        ci.append(1.96 * np.std(subresult)/np.mean(subresult))
        line.append(np.mean(subresult))

    ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
    plt.plot(iterations, line,color=colour[exp], label=f"Sample size: {sample_size}")
    lines.append(line)
    print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

plt.legend()
plt.show()
t1 = time.time()
t = t1-t0
print("Time is:", t)

#%%
# With confidence intervals and Antithetic variables

print("ANTITHETIC")

iterations = range(20, 420, 20)
conf_samples = 100
fig, ax = plt.subplots()
colour = [0,0,"k","b", "g", "r"]

t0 = time.time()
lines = []
for exp in range(3,4):
    sample_size = 10**exp
    line = []
    var_list = []
    ci = []
    for iteration in iterations:
        subresult = []
        for subsamp in range(conf_samples):
            result = pure_random_antithetic(sample_size, iteration)
            subresult.append(result)
            var_list.append(result)

        ci.append(1.96 * np.std(subresult)/np.mean(subresult))
        line.append(np.mean(subresult))

    ax.fill_between(iterations, (np.array(line)-np.array(ci)), (np.array(line)+np.array(ci)), color=colour[exp], alpha=.1)
    plt.plot(iterations, line,color=colour[exp], label=f"Sample size: {sample_size}")
    lines.append(line)
    print("\nSetting = I:", iteration, ",S:", sample_size, "\nFinal approx:", line[-1],"\nVariance:", statistics.variance(var_list))

plt.legend()
plt.show()
t1 = time.time()
t = t1-t0
print("Time is:", t)