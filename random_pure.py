
#%%
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import time
import scipy.stats as stats
# from scipy.stats import norm

from mandelbrot import mandelbrot
#%%
def plot_figure(result):
    plt.figure(dpi=100)
    plt.imshow(result.T, extent=[-2, 1, -1, 1])
    plt.xlabel('real numbers')
    plt.ylabel('imaginary numbers')
    plt.show()

def pure_random(sample_size, iterations):
    # define min and max real and imaginary numbers for mandelbrot function
    r_min, r_max = -2.25, .75
    i_min, i_max = -1.5, 1.5

    # define total area of real and imaginary numbers
    total_area = (abs(r_min) + abs(r_max)) * (abs(i_min) + abs(i_max))

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
    hits = mb_list.count(iterations)
    avg = hits/sample_size
    area_m = avg*total_area

    return area_m


def single_plot():
    '''
    Shows calculated mandelbrot area for different sample sizes and amount of iterations
    (in a single plot)
    '''
    t0 = time.time()
    sample_sizes = [10**2, 10**3, 10**4]
    iterations = range(20, 400, 20)
    experiments = 50
    results, lines = [], []
    
    for sample_size in sample_sizes:
        line = []
        for iteration in iterations:
            # get mean for number of experiments
            for x in range(experiments):
                result = pure_random(sample_size, iteration)
                results.append(result)
            print(np.mean(results))
            line.append(np.mean(result))

        ci = 1.96 * np.std(line)/np.mean(line)
        error = round(np.std(line),3)
        plt.plot(iterations, line, label=f"n = {sample_size} (error = {error})")
        plt.fill_between(iterations, (line-ci), (line+ci), alpha=.1)
        lines.append(line)

        # calculate elapsed time
        t1= time.time()
        print('sample_size:', sample_size, f't: {round(t1-t0, 2)}')


    plt.legend()
    plt.show()
#%%

def separate_plots():
    '''
    Shows calculated mandelbrot area for different sample sizes and amount of iterations
    (in separate plots)
    doesn't work yet!
    '''
    sample_sizes = [10**2, 10**3, 10**4, 10**5]
    iterations = range(20, 400, 20)

    lines = []
    fig = plt.figure()

    for i in range(len(sample_sizes)):
        line = []
        for iteration in iterations:
            result = pure_random(sample_sizes[i], iteration)
            line.append(result)

        ci = 1.96 * np.std(line)/np.mean(line)
        error = round(np.std(line),3)
        ax[i] = fig.add_subplot()
        ax[i].plot(iterations, line)
        ax[i].set_title('f"n = {sample_size} (error = {error})"')
        ax[i].fill_between(iterations, (line-ci), (line+ci), alpha=.1)
        lines.append(line)


    plt.legend()
    plt.show()

single_plot()