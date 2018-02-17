#!/usr/bin/evn oceanv3

import numpy as np

def polyfit_1d(x, y, order=1, grid=True):
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


def polyfit_2d(x, y, z, order=1, gridsize=(50, 100)):
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

def polyfit1d(x, y, order=1, grid=True):
    '''
    Uses x, y data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    y = ax + b                              (order = 1 or 'linear')
    y = ax^2 + bx + c                       (order = 2 or 'quadratic')
    y = ax^3 + bx^2 + cx + d                (order = 3 or 'cubic'
    '''

    if grid:
        xm, ym = np.linspace(x.min(), x.max(), len(y)), np.zeros(len(y))
    else:
        xm, ym = x.copy(), np.zeros(len(y))

    nterms = int(order + 1)

    P = np.zeros((x.size, nterms))
    pascal_triangle = [i for i in range(order + 1) if i <= order]
    for k, i in enumerate(pascal_triangle):
        P[:, k] = x ** i

    A = np.linalg.lstsq(P, y)[0]
    for alpha, i in zip(A, pascal_triangle):
        ym += alpha * xm ** i

    return xm, ym


def polyfit2d(x, y, z, order=1, gridsize=False, point=False):
    '''
    Uses x, y and z data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    z = ax + by + c                                                     (order = 1 or 'linear')
    y = ax^2 + by^2 + cxy + dx + ey + f                                 (order = 2 or 'quadratic')
    y = ax^3 + by^3 + cxy^2 + dx^2y + ex^2 + fy^2 + gxy + hx + iy + j   (order = 3 or 'cubic')
    :param x:
    :param y:
    :param z:
    :param order:
    :param gridsize:
    :return:
    '''

    # TODO: built in residuals and RMSE

    if point:
        xm, ym, zm = point[0], point[1], 0
    elif gridsize:
        xg, yg = np.meshgrid(np.linspace(x.min(), x.max(), gridsize[0]),
                             np.linspace(y.min(), y.max(), gridsize[1]))
        xm, ym = xg.flatten(), yg.flatten()
        zm = np.zeros(xm.shape)
    else:
        xm, ym, zm = x.copy(), y.copy(), np.zeros(z.shape)

    nterms = int((order ** 2 + 3 * order + 2) / 2)

    P = np.zeros((x.size, nterms))
    pascal_triangle = [(i, j) for i in range(order + 1) for j in range(order + 1) if i + j <= order]
    for k, (i, j) in enumerate(pascal_triangle):
        P[:, k] = x ** i * y ** j

    A = np.linalg.lstsq(P, z)[0]
    for alpha, (i, j) in zip(A, pascal_triangle):
        zm += alpha * xm ** i * ym ** j

    # residuals = zm - z
    # rmse = np.sqrt(((zm - z) ** 2).mean())

    if gridsize is not False:
        xm, ym, zm = xg, yg, zm.reshape(xg.shape)

    return xm, ym, zm#, rmse


# http://www.ce.udel.edu/faculty/kaliakin/appendix_poly.pdf

# def test(order=3):
#     "Stupid test function"
#     # ij = []
#     # for i in range(0, order + 1):
#     #     for j in range(0, order + 1):
#     #         if i + j <= order:
#     #             ij.append((i, j))
#     ij = [(i, j) for i in range(order+1) for j in range(order+1) if i + j <= order]
#
#
# if __name__=='__main__':
#     from timeit import Timer
#     t = Timer("test()", "from __main__ import test")
#     print(t.timeit())