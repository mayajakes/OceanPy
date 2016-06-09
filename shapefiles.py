import OceanPy as op
import pyproj

import shapefile
import shapely.geometry as slgeo
import matplotlib.patches as mplpatch

class Shapefile(object):

    shpfile = ''

    def __init__(self, shpfile=None, projin=None, projout=None):
        self.shpfile = shpfile
        self.projin = projin
        self.projout = projout
        self.get_coords()
        self.get_patch()

    def read_shpfile(self):
        try:
            self.reader = shapefile.Reader(self.shpfile)
        except shapefile.ShapefileException:
            self.reader = self.shpfile

    def get_coords(self):
        self.read_shpfile()
        self.coordinates = []
        for shape in self.reader.shapes():
            for x, y in shape.points:
                if self.projin is not None and self.projout is not None:
                    self.coordinates.append(pyproj.transform(self.projin, self.projout, x, y))
                else:
                    self.coordinates.append((x, y))

    def get_patch(self):
        if self.reader.shapeType == 5:
            self.read_shpfile()
            self.patches = []
            if self.projin is not None and self.projout is not None:
                self.transform_shapes()
                for shape in self.writer.shapes():
                    self.patches.append(mplpatch.Polygon(shape.points))
            else:
                for shape in self.reader.shapes():
                    self.patches.append(mplpatch.Polygon(shape.points))
        else:
            pass


    def transform_shapes(self):
        if self.reader.shapeType == 1 or self.reader.shapeType == 3:
            pass
        elif self.reader.shapeType == 5:
            self.writer = shapefile.Writer(shapefile.POLYGON)
            for shape in self.reader.shapes():
                points = shape.points
                poly = []
                for point in points:
                    poly.append(list(pyproj.transform(self.projin, self.projout, point[0], point[1])))
                self.writer.poly(parts=[poly])
        return self.writer


def shapes_in_polygons(shpfile_shapes, shpfile_polygons):

    coords = []
    if shpfile_shapes.shapeType == 1:
        for shape in shpfile_shapes.shapes():
            point = slgeo.Point(shape.points[0])
            for poly in shpfile_polygons.shapes():
                if all([len(x) > 2 for x in poly.points]):
                    poly = [[x, y] for x, y, _, _ in poly.points]
                    if slgeo.Polygon(poly).contains(point):
                        coords.append(point.coords[0])
                elif slgeo.Polygon(poly.points[0]).contains(point):
                    coords.append(point.coords[0])
        return coords
    else:
        pass

# def coords_in_polygon(shpfile_points, shpfile_polygons, projection=None):
#
#     coords = []
#
#     try:
#         sf_points = shapefile.Reader(shpfile_points)
#         sf_polygons = shapefile.Reader(shpfile_polygons)
#     except shapefile.ShapefileException:
#         sf_points = shpfile_points
#         sf_polygons = shpfile_polygons
#
#     for shape in sf_points.shapes():
#         if projection is not None:
#             x, y = shape.points[0]
#             point = Point(pyproj.transform(projection, op.WGS84, x, y))
#         else:
#             point = Point(shape.points[0])
#         for poly in sf_polygons.shapes():
#             if all([len(x) > 2 for x in poly.points]):
#                 poly = [[x, y] for x, y, _, _ in poly.points]
#                 if Polygon(poly).contains(point):
#                     coords.append(point.coords[0])
#             elif Polygon(poly.points[0]).contains(point):
#                 coords.append(point.coords[0])
#     return coords