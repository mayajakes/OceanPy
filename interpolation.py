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


def polyfit1d(x, y, order=1):
    '''
    Uses x, y data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    y = ax + b                              (order = 1 or 'linear')
    y = ax^2 + bx + c                       (order = 2 or 'quadratic')
    y = ax^3 + bx^2 + cx + d                (order = 3 or 'cubic'
    '''

    if any(type(l) is list for l in (x, y)):
        x, y = np.array(x), np.array(y)

    nterms = int(order + 1)

    P = np.zeros((x.size, nterms))
    pascal_triangle = [i for i in range(order + 1) if i <= order]
    for k, i in enumerate(pascal_triangle):
        P[:, k] = x ** i

    A = np.linalg.lstsq(P, y)[0]
    def get_yy(xx):
        yy = np.zeros(xx.shape)
        for alpha, i in zip(A, pascal_triangle):
            yy += alpha * xx ** i
        return yy

    # for alpha, i in zip(A, pascal_triangle):
    #     ym += alpha * xm ** i

    return get_yy


def polyfit2d(x, y, z, order=1):
    '''
    Uses x, y and z data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    z = ax + by + c                                                     (order = 1 or 'linear')
    z = ax^2 + by^2 + cxy + dx + ey + f                                 (order = 2 or 'quadratic')
    z = ax^3 + by^3 + cxy^2 + dx^2y + ex^2 + fy^2 + gxy + hx + iy + j   (order = 3 or 'cubic')
    :param x:
    :param y:
    :param z:
    :param order:
    :param gridsize:
    :return: function f(x, y) to calculate interpolated z values based on original x and y values or grid arrays
    '''

    if any(type(l) is list for l in (x, y, z)):
        x, y, z = np.array(x), np.array(y), np.array(z)

    # TODO: built in residuals and RMSE

    nterms = int((order ** 2 + 3 * order + 2) / 2)

    P = np.zeros((len(x), nterms))
    pascal_triangle = [(i, j) for i in range(order + 1) for j in range(order + 1) if i + j <= order]
    for k, (i, j) in enumerate(pascal_triangle):
        P[:, k] = x ** i * y ** j

    A = np.linalg.lstsq(P, z)[0]
    def get_zz(xx, yy):
        shp = xx.shape
        if len(shp) > 1:
            xx, yy = xx.flatten(), yy.flatten()
        zz = np.zeros(xx.shape)
        for alpha, (i, j) in zip(A, pascal_triangle):
            zz += alpha * xx ** i * yy ** j
        if len(shp) > 1:
            zz = np.reshape(zz, shp)
        return zz

    # residuals = zm - z
    # rmse = np.sqrt(((zm - z) ** 2).mean())

    return get_zz#, rmse

def OI(x, y, obs_fld, Lx, Ly=False, xx=None, yy=None, bg_fld=None, gridsize=None):
    '''
    Optimal Interpolation scheme based on Kalnay, 2003
    Multivariate analysis: http://www.atmosp.physics.utoronto.ca/PHY2509/ch3.pdf
    http://modb.oce.ulg.ac.be/wiki/upload/diva_intro.pdf
    :param x:
    :param y:
    :param obs_fld:
    :param L:
    :param xx:
    :param yy:
    :param bg_fld:
    :param gridsize:
    :return:
    '''

    # checks if background field is provided
    lst = [True if x != None else False for x in [xx, yy, bg_fld]]
    if all(lst):
        ny, nx = bg_fld.shape
        if xx.ndim == 1 and yy.ndim == 1:
            xi, yi = xx.copy(), yy.copy()
            xx, yy = np.meshgrid(xi, yi)
        elif xx.ndim == 2 and yy.ndim == 2:
            xi, yi = xx[0,:], yy[:,0]
        elif xx.ndim != yy.ndim:
            raise InputError('xx and yy do not have the same dimensions, received x, y: %s, %s' % xx.ndim, yy.ndim)
        else:
            raise InputError('Optimal interpolation works only for 1 or 2-dimensional arrays')

        # Lx, Ly = abs(xi[-1] - xi[0]), abs(yi[-1] - yi[0])
        dx, dy = abs(xi[-1] - xi[0]) / (nx - 1), abs(yi[-1] - yi[0]) / (ny - 1)
        xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2
    # creates background field by interpolating a linear plane through the observations
    elif not all(lst):

        if gridsize is None:
            gridsize = (20, 20)
        ny, nx = gridsize
        xi, dx = np.linspace(min(x), max(x), nx, retstep=True)
        yi, dy = np.linspace(min(y), max(y), ny, retstep=True)
        # Lx, Ly = abs(max(x) - min(x)), abs(max(y) - min(y))
        xx, yy = np.meshgrid(xi, yi)
        xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2

        f = polyfit2d(x, y, obs_fld, order=1)
        bg_fld = f(xx, yy)

    else:
        raise InputError('If background field is provided, grid (xx, yy) should be provided too.')

    N = nx * ny
    P = len(obs_fld)

    # BACKGROUND ERROR COVARIANCE MATRIX
    # Gaussian function to model the correlation between analysis point i and analysis point j
    # gamma_ij = np.exp(-(r_ij/L)**2)
    # r_ij is the distance between i and j
    # L length scale, in the ocean mesoscale processes have a length scale on the order of the radius of deformation

    def Bmatrix(varian_b, Lx, Ly=Ly):

        B = np.matlib.ones((N, N))
        for m in range(1, N):
            mj = int(m / nx)
            mi = m - mj * nx

            xm = xc + (mi - int(nx / 2)) * dx
            ym = yc + (mj - int(ny / 2)) * dy

            for l in range(0, m):
                lj = int(l / nx)
                li = l - lj * nx

                xl = xc + (li - int(nx / 2)) * dx
                yl = yc + (lj - int(ny / 2)) * dy

                dist2 = (xm - xl) ** 2 + (ym - yl) ** 2
                cov = np.exp(-dist2 / (2 * Lx ** 2)) if Ly is False else np.exp(-dist2 / (Lx**2 + Ly**2))
                B[m, l] = cov
                B[l, m] = cov

        # variance background field
        for m in range(0, N):
            B[m, m] = varian_b

        return B

    varian_b = np.var(bg_fld)
    B = Bmatrix(varian_b, Lx, Ly)

    # OBSERVATION ERROR COVARIANCE MATRIX
    varian_r = np.var(obs_fld)
    R = np.identity(P)
    R = varian_r * R

    # FORWARD OPERATOR OR OBSERVATION OPERATOR MATRIX
    def Hmatrix():

        H = np.matlib.zeros((P, N))
        for k in range(P):

            # llcrnr of grid cell
            xo = int(nx / 2) - np.ceil(xc / dx) + x[k] / dx
            yo = int(ny / 2) - np.ceil(yc / dy) + y[k] / dy

            # index of llcrnr of grid cell
            i, j = int(xo), int(yo)

            if 0 <= i <= nx - 1 and 0 <= j <= ny - 1:

                # normalized weighting factor in x, y direction
                wx = xo - i
                wy = yo - j

                i = i - 1 if i == nx - 1 else i
                j = j - 1 if j == ny - 1 else j

                # fill matrix with weighting factors
                H[k, j * nx + i] = (1 - wx) * (1 - wy)
                H[k, j * nx + i + 1] = wx * (1 - wy)
                H[k, j * nx + nx + i] = (1 - wx) * wy
                H[k, j * nx + nx + i + 1] = wx * wy

                # print('Check sum: %s' % (wx * (1 - wy) + (1 - wx) * (1 - wy) + wx * wy + (1 - wx) * wy))


            else:
                raise ValueError('Observation point (%s, %s) is not within grid domain.' % (x[k], y[k]))
        return H

    H = Hmatrix()


    # BACKGROUND FIELD VECTOR AT GRID POINTS
    x_b = np.reshape(bg_fld, (N, 1))

    # BACKGROUND FIELD VECTOR AT OBSERVATION POINTS
    y_b = H * x_b

    # if convert:
        

    # OBSERVATION FIELD VECTOR AT OBSERVATION POINTS
    y_o = np.matrix(obs_fld).T

    # INNOVATION OR OBSERVATIONAL INCREMENTS VECTOR
    d = y_o - y_b

    # WEIGHT OR GAIN MATRIX
    W = B * H.T * (R + H * B * H.T).I

    # ANALYSIS FIELD VECTOR
    x_a = x_b + W * d

    # reshape analysis filed vector into grid array and make matrix an array
    ana_fld = np.asarray(np.reshape(x_a, (ny, nx)))

    # ANALYSIS ERROR COVARIANCE MATRIX
    I = np.identity(N)
    A = (I - W * H) * B

    return xx, yy, bg_fld, ana_fld

