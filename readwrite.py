__author__ = 'jaap.meijer'

import numpy as np

def readxyz(filename,step=None,xy=False):

    xyz = open(filename)
    if xy is False:
        x = []
        y = []
        z = []

        for line in xyz:
            x1,y1,z1 = line.split()
            x.append(float(x1))
            y.append(float(y1))
            if z1 == '-9999':
                z.append(float('nan'))
            else:
                z.append(float(z1))
        xyz.close()

        if step is not None:
            xstep  = [x[i] for i in range(0,len(x),step)]
            ystep  = [y[i] for i in range(0,len(y),step)]
            zstep  = [z[i] for i in range(0,len(z),step)]

            return xstep, ystep, zstep

        else:
            return x, y, z

    else:
        x = []
        y = []

        for line in xyz:
            x1,y1 = line.split()
            x.append(float(x1))
            y.append(float(y1))
        xyz.close()

        return (x, y)

def writexyz(filename, x, y, z=None):

    if z is None:
        with open(filename, "w") as xy:

            for i in range(0,len(x)):
                xy.write(str(x[i]) + " " + str(y[i]) + '\n')

    else:
        with open(filename, "w") as xyz:
            if isinstance(x, list):
                for i in range(0,len(x)):
                    xyz.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + '\n')
            else:
                xyz.write(str(x) + " " + str(y) + " " + str(z))

def addxyz(filenamelst,step=None):

    x = []
    y = []
    z = []

    for i in range(0,len(filenamelst)):

        xtemp, ytemp, ztemp = readxyz(filenamelst[i],step)

        x = x + xtemp
        y = y + ytemp
        z = z + ztemp

        del xtemp, ytemp, ztemp

    return x, y, z

def readxytxt(filename):
    x = np.genfromtxt(filename, usecols=(0), delimiter=' ', dtype=None) # comma: delimiter=','
    y = np.genfromtxt(filename, usecols=(1), delimiter=' ', dtype=None)

    return x, y

def readxls(filename, colnos, rowstart = 0, sheetno = 0):
    import xlrd

    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(sheetno)

    x = []
    y = []

    for row in range(rowstart,sheet.nrows):
        x.append(sheet.cell_value(row,colnos[0]))
        y.append(sheet.cell_value(row,colnos[1]))
    return x, y


# def readdata(filename, ncols, nhead):
#     with open(filename, 'r') as file:
#         data = file.read()
#
#         if ncols > 6 or ncols < 2:
#             print('Function not applicable for the number of colums: ' + str(ncols))
#         elif ncols == 2:
#             var1=[]; var2=[]
#         elif ncols == 3:
#             var1=[]; var2=[]; var3=[]
#         elif ncols == 4:
#             var1=[]; var2=[]; var3=[]; var4=[]
#         elif ncols == 5:
#             var1=[]; var2=[]; var3=[]; var4=[]; var5=[]
#         elif ncols == 6:
#             var1=[]; var2=[]; var3=[]; var4=[]; var5=[]; var6=[]
#
#             for line in data:
#                 if ncols == 2:
#                     v1, v2 = line.split()
#                 elif ncols == 3:
#
#                 elif ncols == 4:
#
#                 elif ncols == 5:
#
#                 elif ncols == 6:
#
