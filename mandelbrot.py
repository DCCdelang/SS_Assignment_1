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
rows, cols = 500, 500


result = np.zeros([rows, cols])

for real_index, Re in enumerate(np.linspace(-2.25, .75, num=rows )):
    for imag_index, Im in enumerate(np.linspace(-1.5, 1.5, num=cols)):
        result[real_index, imag_index] = mandelbrot(Re, Im, 100)

area = np.count_nonzero(result)

pixelated_proportion = area / (rows*cols)
print('Proportion of image that contains colored pixels = ', pixelated_proportion)
avg_pixel_value = np.mean(result)/100
print('Average value per pixel = ', np.mean(result)/100)
total_area = 3*3
print('Total area = 3 * 2 =', total_area)

print('Total area of mandelbrot =', pixelated_proportion * avg_pixel_value * total_area)


plt.figure(dpi=100)
plt.imshow(result.T, extent=[-2, 1, -1, 1])
plt.xlabel('real numbers')
plt.ylabel('imaginary numbers')
plt.show()


