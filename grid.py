import numpy as np
from OceanPy.projections import rotatexy
from scipy.interpolate import griddata
from matplotlib.path import Path


def xyz_in_rectangle(x, y, z, ll, lr, ur, ul):
    domain = Path([ll, lr, ur, ul, ll], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    condition = domain.contains_points(list(zip(x, y)))
    xrect = np.extract(condition, x)
    yrect = np.extract(condition, y)
    zrect = np.extract(condition, z)

    return xrect, yrect, zrect

def ascii(x, y, var, dx, dy):

    nx = int((max(x) - min(x)) / dx)+1
    ny = int((max(y) - min(y)) / dy)+1
    xi = np.linspace(min(x), max(x), nx)
    yi = np.linspace(min(y), max(y), ny)

    xgrid, ygrid = np.meshgrid(xi, yi)
    vargrid = griddata((x, y), var, (xgrid, ygrid), method='linear')

    return xgrid, ygrid, vargrid

def xbeach(x, y, z, xori, yori, alfa, dx, dy, xdist, ydist):
    ''' alfa is the coastal orientation angle '''

    rot_coords = rotatexy(xori, yori, x, y, alfa)
    xprime, yprime = list(zip(*rot_coords))

    nx = int((max(xprime) - min(xprime)) / dx) + 1
    ny = int((max(yprime) - min(yprime)) / dy) + 1
    xi = np.linspace(min(xprime), max(xprime), nx)
    yi = np.linspace(min(yprime), max(yprime), ny)

    xgrid, ygrid = np.meshgrid(xi, yi)
    zb = griddata((xprime, yprime), z, (xgrid, ygrid), method='linear')

    xgrid_flat = np.reshape(xgrid, np.prod([xgrid.shape]))
    ygrid_flat = np.reshape(ygrid, np.prod([ygrid.shape]))

    grid_coords = rotatexy(xori, yori, xgrid_flat, ygrid_flat, -alfa)

    x, y = list(zip(*grid_coords))

    xgrid = np.reshape(x, xgrid.shape)
    ygrid = np.reshape(y, ygrid.shape)

    return xgrid, ygrid, zb


