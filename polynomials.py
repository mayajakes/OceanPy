#!/usr/bin/evn oceanv3

import numpy as np

def polyfit_1d(x, y, order=2, grid=True):
    '''
    Uses x, y data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    y = ax + b                              (order = 1 or 'linear')
    y = ax^2 + bx + c                       (order = 2 or 'quadratic')
    y = ax^3 + bx^2 + cx + d                (order = 3 or 'cubic'
    '''

    xm = np.linspace(x.min(), x.max()) if grid else x.copy()

    if order == 1:
        A = np.vstack([np.ones(len(x)), x]).T
        coef = np.linalg.lstsq(A, y)[0]
        # ym = coef[1] * xm + coef[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm], coef)

    elif order == 2:
        A = np.vstack([np.ones(len(x)), x, x**2]).T
        coef = np.linalg.lstsq(A, y)[0]
        # ym = coef[2] * xm**2 + coef[1] * xm + coef[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm**2], coef)

    elif order == 3:
        A = np.vstack([np.ones(len(x)), x, x**2, x**3]).T
        coef = np.linalg.lstsq(A, y)[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm ** 2, xm**3], coef)

    elif order == 4:
        A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4]).T
        coef = np.linalg.lstsq(A, y)[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm ** 2, xm**3, xm**4], coef)

    else:
        raise ValueError('This function does not solve higher than 4th order relations')

    return xm, ym


def polyfit_2d(x, y, z, order=2, gridsize=(50, 100)):
    '''
    Uses x, y and z data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    z = ax + by + c                                                     (order = 1 or 'linear')
    y = ax^2 + by^2 + cxy + dx + ey + f                                 (order = 2 or 'quadratic')
    y = ax^3 + by^3 + cxy^2 + dx^2y + ex^2 + fy^2 + gxy + hx + iy + j   (order = 3 or 'cubic'
    '''


    xm, ym = np.meshgrid(np.linspace(x.min(), x.max(), gridsize[0]),
                       np.linspace(y.min(), y.max(), gridsize[1]))
    xf, yf = xm.flatten(), ym.flatten()

    if order == 1:
        A = np.vstack([np.ones(len(x)), x, y]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf], coef).reshape(xm.shape)

    elif order == 2:
        A = np.vstack([np.ones(len(x)), x, y, x * y, x**2, y**2]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf * yf, xf**2, yf**2], coef).reshape(xm.shape)

    elif order == 3:
        A = np.vstack([np.ones(len(x)), x, y, x * y, x**2, y**2, x**2 * y, x * y**2, x**3, y**3]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf*yf, xf**2, yf**2, xf**2 * yf, xf * yf**2, xf**3, yf**3],
                    coef).reshape(xm.shape)

    elif order == 4:
        A = np.vstack([np.ones(len(x)), x, y, x*y, x**2, y**2, x**2 * y, x * y**2, x**3, y**3,
                       x ** 2 * y ** 2, x**3 * y, x * y**3, x**4, y**4]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf*yf, xf**2, yf**2, xf**2 * yf, xf * yf**2, xf**3, yf**3,
                          xf ** 2 * yf ** 2, xf**3 * yf, xf * yf**3, xf**4, yf**4],
                    coef).reshape(xm.shape)

    else:
        raise ValueError('This function does not solve higher than 4th order relations')


    return xm, ym, zm
