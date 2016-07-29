#!/usr/bin/env python

'''
nc_animation makes an animation of a NetCDF file
'''

__author__ = 'Jan Jaap Meijer'
__email__ = 'janjaapmeijer@gmail.com'

import os

from matplotlib import animation
# from pylab import *
from netCDF4 import Dataset
from OceanPy.colormaps import *
from OpenEarthTools.plot.colormap_vaklodingen import *

# class ncAnimation:
#     filename = ''
#     workdir = ''
#     anim = 0 #
#
def __init__(self, filename=None):#
    self.filename = input('Provide filename for animation: ')#
    self.workdir = os.path.join(input('Provide root of the project: '), 'Animations')
#     self.anim = 0
#     self.x = 0
#     self.y = 0
#     self.z = 0)

def play1D(t, x, y):
    fig, ax = plt.subplots()
    graph, = ax.plot(x, y[0])

    def init():
        graph.set_data([], [])
        return graph

    def animate(i):
        graph.set_data(x, y[i])
        return graph

    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=100, blit=False)

    plt.show()

    return anim

def play1D_vars(vararray, t, x, y=None, colors=None):
    if y is not None:
        dist = [0]
        for i in range(0, len(x)-1):
            dist.append(dist[i] + np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2))
        x=dist

    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(vararray.min(), vararray.max()))

    graphs = [ax.plot([], [], color=colors[j])[0] for j in range(vararray.shape[0])]

    def init():
        for graph in graphs:
            graph.set_data([], [])
        return graphs

    def animate(i):
        for gnum, graph in enumerate(graphs):
            graph.set_data(x, vararray[gnum, i])
        return graphs

    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=100, blit=False)

    plt.show()

    return anim


def play2D(x, y, z, cmin=None, cmax=None, save=False):
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect('equal')
    pcol = plt.pcolor(x,y,z[0])

    # pcol.set_clim([0,1])
    fig.colorbar(pcol,ax=ax)

    plt.tight_layout()

    def init():
        pcol.set_array([])
        pcol.set_clim([])
        return pcol

    # animation function.  This is called sequentially
    def animate(i):
        pcol.set_array(z[i,:-1,:-1].ravel())
        if cmin is None and cmax is None:
            pcol.set_clim([z[i,:-1,:-1].min(), z[i,:-1,:-1].max()])
        else:
            pcol.set_clim([cmin, cmax])
        return pcol

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=z.shape[0], init_func=init, interval=1, blit=False)

    def nc_animation_save(path, filename, dpi):
        writer = animation.writers['ffmpeg'](fps=30)
        anim.save(os.path.join(path, filename + '.mp4'),writer=writer,dpi=dpi)

    if save:
        path = input('Provide path to write the animation to: ')
        filename = input('Provide filename for animation: ')
        if not os.path.exists(path):
            os.makedirs(path)
        nc_animation_save(path, filename, dpi=300)

    plt.show()

    return anim

def nc_animation_play2(x, y, z1, z2, h, points, cmin=None, cmax=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    bathy = plt.pcolor(x,y,z1[0], cmap=vaklodingen_colormap(), linewidth=0,)
    plt.scatter(list(zip(*points))[0], list(zip(*points))[1], facecolors='k', edgecolors='k', marker='o', alpha=0.6)
    pcol = plt.pcolor(x,y,z2[0], cmap=cmap_sea_surface(), alpha=0.6, edgecolors=(1, 1, 1, 0.6), linewidth=0)
    ax.axis('off')

    # pcol.set_edgecolor('none')

    fig.colorbar(pcol,ax=ax)

    tight_layout()

    def init():
        bathy.set_array([])
        bathy.set_clim([])
        pcol.set_array([])
        pcol.set_clim([])
        pcol.set_edgecolor([])

        return bathy, pcol

    # animation function.  This is called sequentially
    def animate(i):
        bathy.set_array(z1[i,:-1,:-1].ravel())
        pcol.set_array(ma.masked_where(h[i] == 0.005, z2[i])[:-1,:-1].ravel())
        pcol.set_edgecolor('face')
        if cmin is None and cmax is None:
            bathy.set_clim([-20, 10])
            pcol.set_clim([ma.masked_where(h[i] == 0.005, z2[i])[:-1,:-1].min(),
                           ma.masked_where(h[i] == 0.005, z2[i])[:-1,:-1].max()])
        else:
            bathy.set_clim([-20, 10])
            pcol.set_clim([cmin, cmax])
        return bathy, pcol

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=shape(z2)[0], init_func=init, interval=1, blit=False)

    def nc_animation_save(path, filename, dpi):
        writer = animation.writers['ffmpeg'](fps=20, codec='libx264', bitrate=-1)
        anim.save(os.path.join(path, filename + '.mp4'),writer=writer,dpi=dpi)

    if save:
        path = os.path.abspath(os.path.join(os.sep, 'Users', 'jaap.meijer', 'Dropbox', 'RISCKIT_Jaap', 'Animations'))# input('Provide path to write the animation to: ')
        filename = input('Provide filename for animation: ')
        if not os.path.exists(path):
            os.makedirs(path)
        nc_animation_save(path, filename, dpi=200)

    plt.show()

    return anim
