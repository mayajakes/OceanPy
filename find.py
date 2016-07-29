__author__ = 'jaap.meijer'

def find(lstlon, lstlat, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat):
    result = []
    for index, (i,j) in enumerate(zip(lstlon,lstlat)):
        if i >= llcrnrlon and i <= urcrnrlon and j >=llcrnrlat and j <= urcrnrlat:
            result.append(index)
    return result

""" FIND FLOATS/INTEGERS IN LIST/SERIES"""
def find_finite(lst):
    import numpy as np
    result = []
    for index, i in enumerate(lst):
        if np.isfinite(i):
            result.append(index)
    return result

""" FIND NAN IN LIST/SERIES OF FLOATS/INTEGERS"""
def find_nan(lst):
    import numpy as np
    result = []
    for index, i in enumerate(lst):
        if np.isnan(i):
            result.append(index)
    return result

""" FIND NAN IN LIST/SERIES OF STRINGS"""
def find_nanstr(lst_str):
    import pandas as pd
    result = []
    for index, i in enumerate(lst):
        if pd.isnull(i):
            result.append(index)
    return result

def find_str(lst_str,string):
    """ FIND STRING IN LIST/SERIES OF STRINGS"""

    import pandas as pd
    if type(string) == str:
        result = []
        for index, i in enumerate(lst):
            if i == string:
                result.append(index)
                return result
    else:
        print('input value is not a string')

def find_closest(array,value):
    """ FIND INDEX IN LIST CLOSEST TO PREDEFINED VALUE """

    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx#, array[idx]

def closestgridpnt(coords, xgrid, ygrid):
    """ FIND COORDINATE IN GRID CLOSEST TO PREDEFINED COORDINATES """

    from scipy import spatial

    # gridID = []
    griddata = list(zip(xgrid.ravel(), ygrid.ravel()))
    gridcoords = []
    for coord in coords:
        _, index = spatial.KDTree(griddata).query(coord)
        # gridpt = grid[spatial.KDTree(grid).query(pt)[1]] # or grid[index]
        # gridID.append(grid_inds[index])
        gridcoords.append(griddata[index])
    return gridcoords

# def findy(lst, lat1, lat2):
#     result = []
#     for index, i in enumerate(lst):
#         if i >= lat1 and i < lat2:
#             result.append(index)
#     return result