import numpy as np

# mandelbrot formula
def mandelbrot(real_nr, imag_nr, max_iter):
    c = complex(real_nr, imag_nr)
    z = 0.0j # define z as complex nr
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iter
