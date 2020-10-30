import numpy as np
import matplotlib.pyplot as plt

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

for real_index, Re in enumerate(np.linspace(-2, 1, num=rows )):
    for imag_index, Im in enumerate(np.linspace(-1, 1, num=cols)):
        result[real_index, imag_index] = mandelbrot(Re, Im, 100)



plt.figure(dpi=100)
plt.imshow(result.T, extent=[-2, 1, -1, 1])
plt.xlabel('real numbers')
plt.ylabel('imaginary numbers')
plt.show()


