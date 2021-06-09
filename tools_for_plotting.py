#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:55:10 2021

Definition of functions used for saving
of plots, .dat files, .npy solutions
creating paths etc.
for the purposes of the Liquid film solver

@author: tsveti
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import matplotlib.ticker as tick
from matplotlib.ticker import LinearLocator
from pathlib import Path
import os
import gc

# spectrogram
from scipy.signal import spectrogram
import pandas as pd
import datetime
from scipy.fft import fftshift


# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 10}
rc('font', **font)


#################################################
def plot_surfaces(h, X, Z, n,
                  directory,
                  filename,
                  conf_key):
    '''
    Plot the height of the liquid film
    as a surface in (x,z)
    '''
    plt.close()

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    h_max = 0.3
    h_min = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')
    ax.plot_surface(X, Z, h,
                    cmap = 'BuPu',
                    vmin = h_min,
                    vmax = h_max,
                    linewidth = 0,
                    antialiased = 'True',
                    rstride = 3,
                    cstride = 3)
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_major_formatter('{x:.1f}')
    ax.zaxis.set_major_formatter('{x:.2f}')
    ax.set_title('Dimensionless liquid film height')
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('z')
    ax.set_zlabel('h')
    ax.set_zlim(h_min, h_max)
    ax.set_ylim(Z.min(), Z.max())
    ax.set_box_aspect((8, 3, 1.5))
    ax.view_init(elev = 18, azim = 48) # 3D

    plt.contourf(X, Z, h,
                 cmap = 'BuPu',
                 levels = np.linspace(h_min, h_max, 100))
    plt.colorbar(shrink = 0.4,
                 ticks = np.linspace(h_min, h_max, 5),
                 format = '%.2f',
                 anchor = (0.5, 0.5))
    plt.savefig(directory \
                + os.sep \
                + filename \
                + '_frame{:.0f}.png'
                .format(n/100),
                format      = 'png',
                dpi         = 300,
                pad_inches  = 0.1,
                bbox_inches = 'tight')
    fig.clear()
    plt.clf()
    plt.close(fig)
    gc.collect()



#################################################
def plot_contourfs(h, X, Z,
                   n,
                   directory,
                   filename):
    h_min = 0.17
    h_max = 0.23
    fig = plt.figure(figsize = (12,4))
    plt.plot()
    contourplot = plt.contourf(-X, Z, h,
                               alpha = 0.9,
                               cmap = 'Purples',
                               levels = np.linspace(h_min,
                                                    h_max,
                                                    200),
                               extend = 'both')
    plt.colorbar(contourplot,
                 ticks = np.linspace(h_min, h_max, 5),
                 format = '%.2f',
                 anchor = (0.5, 0.5))
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.title('Contour plot of the dimensionless film height')
    plt.xlabel('domain length x, [-]')
    plt.ylabel('domain width z, [-]')
    plt.tight_layout()

    plt.savefig(directory \
                + os.sep \
                + 'contourf_' + filename + '.png'
                .format(n),
                format      = 'png',
                dpi         = 300,
                pad_inches  = 0.1,
                bbox_inches = 'tight')
    # plt.show()
    fig.clear()
    plt.clf()
    plt.close(fig)
    gc.collect()



#################################################
def plot_limiters(nx, dx, nz, dz,
                  Phi_x, Phi_z,
                  directory_lim,
                  scheme_choice,
                  n):
    '''
    Plot the flux limiters
    '''
    x1 = [np.mgrid[0:nx-2]*dx, np.mgrid[0:nx]*dx]
    z1 = [np.mgrid[0:nz]*dz, np.mgrid[0:nz-2]*dz]
    Phis = [Phi_x, Phi_z]
    indicator = ['x', 'z']

    for i in range(len(Phis)):
        # Create matrices of the coordinate variables
        [Z1, X1] = np.meshgrid(z1[i], x1[i])
        fig  = plt.figure()
        ax   = fig.add_subplot(111, projection= '3d')
        ax.plot_surface(X1, Z1, Phis[i],
                        cmap = 'BuPu',
                        linewidth = 0,
                        antialiased = 'True',
                        rstride = 3,
                        cstride = 3)
        ax.set_title('slope limiter ' + indicator[i])
        plt.savefig(directory_lim + os.sep \
                    + 'Phi_' + indicator[i] \
                    + '_' + scheme_choice \
                    + '_limiter_dx{:.3f}_frame{:.0f}.png'
                    .format(dx, n/1000),
                    format      = 'png',
                    dpi         = 200,
                    pad_inches  = 0.1,
                    bbox_inches = 'tight')
        fig.clear()
        plt.clf()
        plt.close(fig)
        gc.collect()
