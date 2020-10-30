import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from numba import jit

# from numba import 

# mandelbrot formula
def mandelbrot(real_nr, imag_nr, max_iter):
    c = complex(real_nr, imag_nr)
    z = 0.0j # define z as complex nr
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iter

# resolution
rows, cols = 1000, 1000


result = np.zeros([rows, cols])

r_min = -2
r_max = 1
i_min = -1 
i_max = 1

for real_index, Re in enumerate(np.linspace(r_min, r_max, num=rows )):
    for imag_index, Im in enumerate(np.linspace(i_min, i_max, num=cols)):
        result[real_index, imag_index] = mandelbrot(Re, Im, 20)

# result[500][500]

plt.figure(dpi=200)
plt.imshow(result.T, extent=[-2, 1, -1, 1])
plt.xlabel('real numbers')
plt.ylabel('imaginary numbers')
plt.show()


l = lhsmdu.createRandomStandardUniformMatrix(2, 20) # Monte Carlo sampling
k = lhsmdu.sample(2, 20) # Latin Hypercube Sampling with multi-dimensional uniformity

r_min = -2
r_max = 1
i_min = -1 
i_max = 1

Real = np.linspace(r_min, r_max, num=rows )
Imag = np.linspace(i_min, i_max, num=cols)

for sample in :
    sample[]
mandelbrot(Re, Im, 20)