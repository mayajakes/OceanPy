import numpy as np
from scipy.interpolate import griddata

def grid_ascii(x, y, var, res_x, res_y):

    nx = int((max(x) - min(x)) / res_x)+1
    ny = int((max(y) - min(y)) / res_y)+1
    xi = np.linspace(min(x), max(x), nx)
    yi = np.linspace(min(y), max(y), ny)

    xgrid, ygrid = np.meshgrid(xi, yi)
    vargrid = griddata((x, y), var, (xgrid, ygrid), method='linear')

    return xgrid, ygrid, vargrid

def write_ascii(filename, array, xll, yll, cellsize):
    header = "ncols     %s\n" % array.shape[1]
    header += "nrows    %s\n" % array.shape[0]
    header += "xllcorner %f\n" % xll
    header += "yllcorner %f\n" % yll
    header += "cellsize %f\n" % cellsize
    header += "NODATA_value -9999\n"

    np.savetxt(filename, array, header=header, fmt="%1.4f")
