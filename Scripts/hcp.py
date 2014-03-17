import numpy as np
import matplotlib.pyplot as pp
import vtk
from vtk.util import numpy_support
import butils

def hex_lattice(n):
    rs = []
    for i in range(n):
        y = np.sqrt(3) * i / 2.0
        for j in range(2 * n - 1 - i):
            x = (-2 * n + i + 2) / 2.0 + j
            rs.append([x, y])
            if y != 0.0:
                rs.append([x, -y])
    rs = np.array(rs).reshape([-1, 2])
    f = 2.0 * rs.max() + 1.0
    rs /= f
    R = 0.5 / f
    return rs, R

if __name__ == '__main__':
    n = 5

    d = 1.0
    dim = 3

    rs, R = hex_lattice(n)
    rs = butils.pad_to_3d(rs)

    L = 2 * 1.1 * rs.max()

    # for r in rs:
    #     pp.gca().add_artist(pp.Circle(r, radius=0.5))
    # l = [2 * rs.min(), 2 * rs.max()]
    # pp.xlim(l)
    # pp.ylim(l)
    # pp.gca().set_aspect('equal')
    # pp.show()
