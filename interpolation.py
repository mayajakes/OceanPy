__author__ = 'jaap.meijer'

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interpolate_1h(df, n):

    # df = pd.DataFrame.from_csv('./CRAF/waves/wind.txt')
    # if strings have to be maintained
    # df = df[50:65].resample('H', how='first', limit=5)

    df = df.resample('H')
    #df = df.interpolate(limit=2)

    headers = list(df.columns.values)

    for i in range(0,len(df)-1,n):

        if ~np.isnan(df[headers[0]][i + n]):
            a1 = (df[headers[0]][i + n] - df[headers[0]][i]) / n
            a2 = (df[headers[1]][i + n] - df[headers[1]][i]) / n
            df[headers[0]][i + 1] = a1 * (df.index.hour[i + 1] - df.index.hour[i]) + df[headers[0]][i]
            df[headers[0]][i + 2] = a1 * (df.index.hour[i + 2] - df.index.hour[i]) + df[headers[0]][i]
            df[headers[1]][i + 1] = a2 * (df.index.hour[i + 1] - df.index.hour[i]) + df[headers[1]][i]
            df[headers[1]][i + 2] = a2 * (df.index.hour[i + 2] - df.index.hour[i]) + df[headers[1]][i]

    return df

def interp1d_nan(arr, kind='linear'):
    '''
    interpolate over nan values
    '''
    indices = np.arange(arr.shape[0])
    finite = np.where(np.isfinite(arr))
    f = interp1d(indices[finite], arr[finite], kind=kind, bounds_error=False)
    arrout = np.where(np.isfinite(arr), arr, f(indices))

    return arrout


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
    if any(type(l) is list for l in (x, y, z)):
        x, y, z = np.array(x), np.array(y), np.array(z)

    # TODO: built in residuals and RMSE

    if point:
        xm, ym, zm = point[0], point[1], 0
    elif gridsize:
        xg, yg = np.meshgrid(np.linspace(min(x), max(x), gridsize[0]),
                             np.linspace(min(y), max(y), gridsize[1]))
        xm, ym = xg.flatten(), yg.flatten()
        zm = np.zeros(xm.shape)
    else:
        xm, ym, zm = x.copy(), y.copy(), np.zeros(z.shape)

    nterms = int((order ** 2 + 3 * order + 2) / 2)

    P = np.zeros((len(x), nterms))
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
