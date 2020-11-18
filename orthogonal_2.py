import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import statistics
import random
import time

from mandelbrot import mandelbrot

# bounds of the undecided square
R_MIN, R_MAX = -2, .5
I_MIN, I_MAX = -1.25, 1.25
TOTAL_AREA = (abs(R_MIN) + abs(R_MAX)) * (abs(I_MIN) + abs(I_MAX))


def ortho(ns):
    assert(np.sqrt(ns) % 1 == 0), f"Please insert an even number of samples {ns}"
    n = int(np.sqrt(ns))
    # Making a datastructure of a dict with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
    blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
    points = []#np.empty((n,2))
    append = points.append # tips of python to fasten up append call

    for block in blocks:
        point = random.choice(blocks[block])
        lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
        lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]

        for col in lst_col:
            blocks[col] = [a for a in blocks[col] if a[1] != point[1]]

        for row in lst_row:
            blocks[row] = [a for a in blocks[row] if a[0] != point[0]]

        #Adjust the points to fit the grid they fall in  
        point = (point[0] + n * block[0], point[1] + n * block[1])
        append(point)

    return points

def scale_points(sample_size):
    r_tot = abs(R_MIN)+abs(R_MAX)
    i_tot = abs(I_MIN)+abs(I_MAX)
    x_l = []
    y_l = []
    p = ortho(sample_size)
    # print(p)
    points = len(p)
    maximum = len(p) 
    scaling =[ 1/maximum * i for i in range(len(p))]
    result = np.zeros((points,2))

    for idx, scale in enumerate(scaling):
        x =  R_MIN + np.random.uniform(p[idx][0]/maximum, p[idx][0]/maximum +1/maximum ) * r_tot  # 4 is just max - min which is in my case 4
        y =  I_MIN + np.random.uniform(p[idx][1]/maximum, p[idx][1]/maximum + 1/maximum ) * i_tot
        result[idx, :] = [x,y]

    for i in result:
        x_l.append(i[0])
        y_l.append(i[1])
    
    x_l = np.array(x_l)
    y_l = np.array(y_l)
    # print(x_l,y_l)
    return x_l, y_l


def orthogonal(sample_size, maxI):
    # points = ortho(sample_size)
    x_l,y_l = scale_points(sample_size)
    mb_list = []
    print(len(x_l))
    for i in range(len(x_l)):
        mb = mandelbrot(x_l[i], y_l[i], maxI)
        mb_list.append(mb)

    hits = mb_list.count(maxI)
    avg = hits/len(x_l)
    area_m = avg*TOTAL_AREA
    print("Area:", area_m)

    return area_m, len(x_l)

t0 = time.time()
orthogonal(2500, 100)
print(time.time()-t0)


# assert(np.sqrt(2500) % 1 == 0)
