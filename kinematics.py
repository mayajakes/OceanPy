import pyproj
import numpy as np
from scipy.interpolate import griddata, UnivariateSpline
from gsw import f, grav

from OceanPy.projections import haversine, rotatexy
from OceanPy.utilities import contour_length
# TODO: mask if any of the variables is nan
import xarray as xr

def gradient_balance_from_ssh(xr_ds, coord, variables=('adt', 'ugos', 'vgos'),
dimensions=('longitude', 'latitude'), fcor=1e-4, gravity=9.81, transform=None, time=None):

    # select which timestep
    if time is not None:
        xr_ds = xr_ds.sel(time=time)

    # take Absolute Dynamic Topography and geostrophic velocities from SSH xarray
    adt = xr_ds[variables[0]] if hasattr(xr_ds, variables[0]) else xr_ds.copy()
    ugos = xr_ds[variables[1]] if hasattr(xr_ds, variables[1]) else None
    vgos = xr_ds[variables[2]] if hasattr(xr_ds, variables[2]) else None

    # check if field dimensions are 2-D
    if adt.ndim != 2:
        raise ValueError('Field can have a maximum number of 2 dimension but got %s', adt.ndim)

    # transform polar in cartesian coordinate system
    if transform is not None:
        WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        lnln, ltlt = np.meshgrid(xr_ds[dimensions[0]].data, xr_ds[dimensions[1]].data)
        xx, yy = pyproj.transform(WGS84, transform, lnln, ltlt)
        x, y = pyproj.transform(WGS84, transform, *coord)
    else:
        xx, yy = np.meshgrid(xr_ds[dimensions[0]].data, xr_ds[dimensions[1]].data)
        x, y = coord
    dx, dy = np.unique(np.diff(xr_ds[dimensions[0]]))[0], np.unique(np.diff(xr_ds[dimensions[1]]))[0]

    # calculate geostrophy parameters
    if transform is not None:
        fcor = f(coord[1])
        gravity = grav(coord[1], p=0)

    # interpolate adt to coordinate location
    points = np.array((xx.flatten(), yy.flatten())).T
    adt_flat = adt.data.flatten()
    adt_coord = griddata(points, adt_flat, (x, y))

    # calculate geostrophic velocities at coordinate location
    if ugos is None or vgos is None:
        adtx = griddata(points, adt_flat, ([x - (dx / 2), x + (dx / 2)], [y, y]))
        adty = griddata(points, adt_flat, ([x, x], [y - (dy / 2), y + (dy / 2)]))

        dzetadx = np.diff(adtx) / dx
        dzetady = np.diff(adty) / dy
        ug = -(gravity / fcor) * dzetady
        vg = (gravity / fcor) * dzetadx
    else:
        ugos_flat = ugos.data.flatten()
        vgos_flat = vgos.data.flatten()
        ug = griddata(points, ugos_flat, (x, y))
        vg = griddata(points, vgos_flat, (x, y))
    # if ug is positive, t is positive in x-direction and n in positive y direction
    xpos = True if ug > 0 else False
    ypos = True if vg > 0 else False

    # find contour points close to interested data point
    def strictly_increasing(L):
        return all(i0 < i1 for i0, i1 in zip(L, L[1:]))
    def strictly_decreasing(L):
        return all(i0 > i1 for i0, i1 in zip(L, L[1:]))

    coords_ct = contour_length(xr_ds=adt, contour=adt_coord, time_sel=time, timemean=False,
                               lon_sel=slice(coord[0] - 2*dx, coord[0] + 2*dx),
                               lat_sel=slice(coord[1] - 2*dy, coord[1] + 2*dy))[1:3]
    if not (strictly_increasing(coords_ct[0]) | strictly_decreasing(coords_ct[0])):
        # TODO: if transform: distance
        idx = np.argsort(np.sqrt((coords_ct[0] - coord[0])**2 + (coords_ct[1] - coord[1])**2))[0:2]
        coords_ct = (np.append(coords_ct[0][idx], coord[0]), np.append(coords_ct[1][idx], coord[1]))

    # determine normal/ tangential resolution
    x0 = (xr_ds[dimensions[0]].min() + xr_ds[dimensions[0]].max()) / 2
    y0 = (xr_ds[dimensions[1]].min() + xr_ds[dimensions[1]].max()) / 2
    if transform is not None:
        x_ct, y_ct = pyproj.transform(WGS84, transform, *coords_ct)
        dn = haversine([x0 - dx / 2, x0 + dx / 2], [y0 - dy / 2, y0 + dy / 2])[0][0]
    else:
        x_ct, y_ct = coords_ct
        dn = np.sqrt(dx**2 + dy**2)

    # calculate radius of curvature and orientation angle velocity vector
    try:
        fx = UnivariateSpline(x_ct, y_ct)
    except ValueError:
        # print('x is not increasing')
        x_ct, y_ct = [lst for lst in zip(*sorted(zip(x_ct, y_ct), key=lambda pair: pair[0]))]
        try:
            fx = UnivariateSpline(x_ct, y_ct)
            dydx = fx.derivative(1)(x)
            d2ydx2 = fx.derivative(2)(x)
        except:
            dydx = np.gradient(y_ct)[1] / np.gradient(x_ct)[1]
            d2ydx2 = dydx / np.gradient(x_ct)[1]
    else:
        dydx = fx.derivative(1)(x)
        d2ydx2 = fx.derivative(2)(x)

    Rcurv = (1 + dydx**2)**(3 / 2) / d2ydx2
    orientation = np.arctan(dydx) if xpos else np.arctan(dydx) + np.pi

    # determine locations of points normal to interested data point
    xi = np.array([x - dn / 2, x + dn / 2])
    yi = y * np.ones(len(xi))
    ti, ni = zip(*rotatexy(x, y, xi, yi, orientation + (np.pi / 2)))

    # interpolate ssh.adt to normal/ tangential points
    adti = griddata(points, adt_flat, (ti, ni))

    # geostrophic speed
    dDdn = np.diff(adti) / dn
    Vg = -(gravity / fcor) * dDdn

    # gradient speed from Holten, 2004
    hemisphere = 'SH' if fcor < 0 else 'NH'
    root = np.sqrt(((fcor**2 * Rcurv**2) / 4) + (fcor * Rcurv * Vg))
#     print(hemisphere, 'Vg', Vg, 'first term', -(fcor * Rcurv / 2), 'root', root)
#     print('plus root', -fcor * Rcurv / 2 + root, 'min root', -fcor * Rcurv / 2 - root)
    if hemisphere == 'NH':
        if Rcurv > 0 and Vg > 0:
            # regular low
            V = -(fcor * Rcurv / 2) + root
        elif Rcurv < 0 and Vg > 0:
            # regular high
            V = -(fcor * Rcurv / 2) - root
        else:
            V = np.nan

    elif hemisphere == 'SH':
        if Rcurv < 0 and Vg > 0:
            # regular low
            V = -(fcor * Rcurv / 2) + root
        elif Rcurv > 0 and Vg > 0:
            # regular high
            V = -(fcor * Rcurv / 2) - root
        # elif Rcurv < 0 and Vg < 0:
        #     # TODO: not sure, anomalous high
        #     V = -(fcor * Rcurv / 2) + root
        # elif np.isnan(root) and np.isfinite(Vg):
        #     V = Vg.copy()
        #     print('geostrophic')
        else:
            V = np.nan

    return V, Vg, orientation, ug, vg

def gradient_wind_from_ssh(xr_ds, variables=('adt', 'ugos', 'vgos'),
                           dimensions=('time', 'latitude', 'longitude'), transform=None):

    # take Absolute Dynamic Topography from SSH xarray
    adt = xr_ds[variables[0]] if hasattr(xr_ds, variables[0]) else xr_ds.copy()
    ugeos = xr_ds[variables[1]] if hasattr(xr_ds, variables[1]) else None
    vgeos = xr_ds[variables[2]] if hasattr(xr_ds, variables[2]) else None

    orientation = np.arctan(vgeos / ugeos)
    Vgeos = np.sqrt(ugeos**2 + vgeos**2)

    # transform polar to cartesian coordinate system
    if transform is not None:
        WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        lnln, ltlt = np.meshgrid(xr_ds[dimensions[2]].data, xr_ds[dimensions[1]].data)
        xx, yy = pyproj.transform(WGS84, transform, lnln, ltlt)
    else:
        xx, yy = np.meshgrid(xr_ds[dimensions[2]].data, xr_ds[dimensions[1]].data)

    shp = adt.shape

    gravity = grav(ltlt, p=0)
    fcor = f(ltlt)
    if adt.ndim != 2:
        gravity = np.broadcast_to(gravity, shp)
        fcor = np.broadcast_to(fcor, shp)

    detadx = np.ma.masked_all(shp)
    detady = detadx.copy()
    d2etadx2, d2etady2, d2etadxdy = detadx.copy(), detadx.copy(), detadx.copy()
    for it in range(len(xr_ds[dimensions[0]])):
        detadx[it,] = np.gradient(adt[it,])[1] / np.gradient(xx)[1]
        detady[it,] = np.gradient(adt[it,])[0] / np.gradient(yy)[0]

        d2etadxdy[it,] = np.gradient(detadx[it,])[0] / np.gradient(yy)[0]

        d2etadx2[it,] = np.gradient(detadx[it,])[1] / np.gradient(xx)[1]
        d2etady2[it,] = np.gradient(detady[it,])[0] / np.gradient(yy)[0]

    kappa = (-(d2etadx2*detady**2) -(d2etady2*detadx**2) + (2*d2etadxdy*detadx*detady)) / (detadx**2 + detady**2)**(3/2)
    Rcurv = 1 / kappa

    root = np.sqrt(((fcor**2 * Rcurv**2) / 4) + (fcor * Rcurv * Vgeos))

    Vgrad = np.ma.masked_all(shp).flatten()
    fcor, Vgeos, Rcurv, root = fcor.flatten(), Vgeos.data.flatten(), Rcurv.flatten(), root.flatten()
    for i in range(len(Vgrad)):
        # Northern Hemisphere
        if fcor[i] >= 0:
            if (Rcurv[i] < 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) - root[i]
            elif (Rcurv[i] > 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) + root[i]
            else:
                Vgrad[i] = np.nan
        # Southern Hemisphere
        elif fcor[i] < 0:
            if (Rcurv[i] < 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) + root[i]
            elif (Rcurv[i] > 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) - root[i]
            else:
                Vgrad[i] = np.nan

    Vgrad, Vgeos = Vgrad.reshape(shp), Vgeos.reshape(shp)
    ugrad, vgrad = Vgrad * np.cos(orientation), Vgrad * np.sin(orientation)
    xr_ds_new = xr.Dataset(data_vars={'Vgrad': (dimensions, Vgrad),
                                      'Vgeos': (dimensions, Vgeos),
                                      'orientation': (dimensions, orientation),
                                      'ugrad': (dimensions, ugrad),
                                      'vgrad': (dimensions, vgrad)},
                           coords={dimensions[0]: xr_ds[dimensions[0]],
                                   dimensions[1]: xr_ds[dimensions[1]],
                                   dimensions[2]: xr_ds[dimensions[2]]})

    return xr_ds_new
