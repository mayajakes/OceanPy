from OceanPy.projections import *
import pyproj

import shapefile
from shapely.geometry import Polygon, Point


def coords_from_line(shpfile_line, projection=None):
    coords = []
    sf_line = shapefile.Reader(shpfile_line)
    for shape in sf_line.shapes():
        for x, y in shape.points:
            if projection is not None:
                coords.append(pyproj.transform(projection, WGS84, x, y))
            else:
                coords.append(x, y)
    return coords


def coords_in_polygon(shpfile_points, shpfile_polygons, projection=None):

    coords = []

    try:
        sf_points = shapefile.Reader(shpfile_points)
        sf_polygons = shapefile.Reader(shpfile_polygons)
    except shapefile.ShapefileException:
        sf_points = shpfile_points
        sf_polygons = shpfile_polygons

    for shape in sf_points.shapes():
        if projection is not None:
            x, y = shape.points[0]
            point = Point(pyproj.transform(projection, WGS84, x, y))
        else:
            point = Point(shape.points[0])
        for poly in sf_polygons.shapes():
            if Polygon(poly.points).contains(point):
                coords.append(point.coords[0])
    return coords