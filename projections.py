__author__ = 'jaap.meijer'

# https://pypi.python.org/pypi/utm

import pyproj

# SWEREF99 13 30
def swe2utm(xswe, yswe, zswe=None):

    SWE99 = pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=13.5 +k=1 +x_0=150000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    UTM33 = pyproj.Proj("+proj=utm +zone=33U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    if zswe == None:
        xutm, yutm = pyproj.transform(SWE99, UTM33, xswe, yswe)
        return xutm, yutm
    else:
        xutm, yutm, zutm = pyproj.transform(SWE99, UTM33, xswe, yswe, zswe)
        return xutm, yutm, zutm

WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
UTM33 = pyproj.Proj("+proj=utm +zone=33U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
ETRS89 = pyproj.Proj('+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs')
SWE99 = pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=13.5 +k=1 +x_0=150000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
OSM = pyproj.Proj('+ellps=WGS84 +proj=tmerc +lat_0=0 +lon_0=015d48.377m +k=1.0000056 +x_0=1500064.1 +y_0=-668.0')

def rotate(xpnt, ypnt, zpnt, lstx, lsty, lstz, theta, dn=None):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,0,xpnt],
                       [0,1,0,ypnt],
                       [0,0,1,zpnt],
                       [0,0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,0,-xpnt],
                           [0,1,0,-ypnt],
                           [0,0,1,-zpnt],
                           [0,0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0,0,1,0],
                     [0,0,0,1]])

    # combined (inverse)translation and rotation matrix (does not work)
    #  rot_mat = np.matrix([[np.cos(alpha[1]), -np.sin(alpha[1]), 0, (p*np.cos(alpha[1]) - q*np.sin(alpha[1]) - p)],
    #                      [np.sin(alpha[1]), np.cos(alpha[1]), 0, (p*np.sin(alpha[1]) + q*np.cos(alpha[1]) - q)],
    #                      [0,0,1,0],
    #                      [0,0,0,1]])

    if dn is not None:
        xyzvector = [np.matrix([[lstx[i]],[lsty[i]],[lstz[i]],[1]]) for i in range(0,len(lstx),dn)]
    else:
        xyzvector = [np.matrix([[lstx[i]],[lsty[i]],[lstz[i]],[1]]) for i in range(0,len(lstx))]

    xprime = []
    yprime = []
    zprime = []

    for a in xyzvector:
        xyzprimeinv = np.dot(inv_trans,a)
        xyzprimerot = np.dot(rot, xyzprimeinv)
        xyzprimetrans = np.dot(trans, xyzprimerot)
        xprime.append(float(xyzprimetrans[0]))
        yprime.append(float(xyzprimetrans[1]))
        zprime.append(float(xyzprimetrans[2]))

    return (xprime, yprime, zprime)

def rotate1pnt(xpnt, ypnt, zpnt, x, y, z, theta):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,0,xpnt],
                       [0,1,0,ypnt],
                       [0,0,1,zpnt],
                       [0,0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,0,-xpnt],
                           [0,1,0,-ypnt],
                           [0,0,1,-zpnt],
                           [0,0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0,0,1,0],
                     [0,0,0,1]])

    # combined (inverse)translation and rotation matrix (does not work)
    #  rot_mat = np.matrix([[np.cos(alpha[1]), -np.sin(alpha[1]), 0, (p*np.cos(alpha[1]) - q*np.sin(alpha[1]) - p)],
    #                      [np.sin(alpha[1]), np.cos(alpha[1]), 0, (p*np.sin(alpha[1]) + q*np.cos(alpha[1]) - q)],
    #                      [0,0,1,0],
    #                      [0,0,0,1]])

    xyzvector = np.matrix([[x],[y],[z],[1]])

    xprime = []
    yprime = []
    zprime = []

    xyzprimeinv = np.dot(inv_trans,xyzvector)
    xyzprimerot = np.dot(rot, xyzprimeinv)
    xyzprimetrans = np.dot(trans, xyzprimerot)
    xprime = float(xyzprimetrans[0])
    yprime = float(xyzprimetrans[1])
    zprime = float(xyzprimetrans[2])

    return (xprime, yprime, zprime)

def rotate2D(xpnt, ypnt, lstx, lsty, theta, dn=None):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,xpnt],
                       [0,1,ypnt],
                       [0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,-xpnt],
                           [0,1,-ypnt],
                           [0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

    if dn is not None:
        xyvector = [np.matrix([[lstx[i]],[lsty[i]],[1]]) for i in range(0,len(lstx),dn)]
    else:
        xyvector = [np.matrix([[lstx[i]],[lsty[i]],[1]]) for i in range(0,len(lstx))]

    xprime = []
    yprime = []
    zprime = []

    for a in xyvector:
        xyprimeinv = np.dot(inv_trans,a)
        xyprimerot = np.dot(rot, xyprimeinv)
        xyprimetrans = np.dot(trans, xyprimerot)
        xprime.append(float(xyprimetrans[0]))
        yprime.append(float(xyprimetrans[1]))

    return (xprime, yprime)