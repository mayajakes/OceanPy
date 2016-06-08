import OceanPy as op
import pyproj

import shapefile
from shapely.geometry import Polygon, Point


def coords_from_line(shpfile_line, projection=None):
    coords = []
    sf_line = shapefile.Reader(shpfile_line)
    for shape in sf_line.shapes():
        for x, y in shape.points:
            if projection is not None:
                coords.append(pyproj.transform(projection, op.WGS84, x, y))
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
            point = Point(pyproj.transform(projection, op.WGS84, x, y))
        else:
            point = Point(shape.points[0])
        for poly in sf_polygons.shapes():
            if all([len(x) > 2 for x in poly.points]):
                poly = [[x, y] for x, y, _, _ in poly.shapes()[0].points]
            if Polygon(poly.points).contains(point):
                coords.append(point.coords[0])
    return coords

def transform_polygons(shpfile_polygons, projin, projout):

    try:
        sf_polygons = shapefile.Reader(shpfile_polygons)
    except shapefile.ShapefileException:
        sf_polygons = shpfile_polygons

    polygons = shapefile.Writer(shapefile.POLYGON)
    for shape in sf_polygons.shapes():
        points = shape.points
        poly = []
        for point in points:
            poly.append(list(pyproj.transform(projin, projout, point[0], point[1])))
        polygons.poly(parts=[poly])

    return polygons