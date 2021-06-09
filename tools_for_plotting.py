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

# filtered plots
from findiff import FinDiff, coefficients, Coefficient
from Functions_Miguel import filt_X


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
def filtered(h, X, Z,
             dx, dz,
             nx, nz,
             n,
             directory,
             filename):

    # First derivatives operators:
    d_dx = FinDiff((0, dx, 1))
    d_dz = FinDiff((1, dz, 1))

    # Filter the height along x and along z:
    H_Xf = filt_X(h[1:-1,1:-1],31,boundaries="extrap",s=0.2)
    H_Zf = filt_X(h[1:-1,1:-1].T,31,boundaries="extrap",s=0.2)
    H_Zf = H_Zf.T

    # Computations of the third derivatives
    # that are included in sources S2 and S3:

    # d3_dx3 h:
    # take the first derivative:
    hx = d_dx(H_Xf)
    # filter it along x:
    hx_Xf = filt_X(hx,31,boundaries="extrap",s=0.2)
    # differentiate again:
    hxx = d_dx(hx_Xf)
    # filter again:
    hxx_Xf = filt_X(hxx,31,boundaries="extrap",s=0.2)
    # take the third derivative:
    hxxx0 = d_dx(hxx_Xf)
    # and filter it:
    hxxx = filt_X(hxxx0,31,boundaries="extrap",s=0.2)

    # d3_dz3 h:
    # hzzz = d_dz(hzz)
    #new attemp:
    hz = d_dz(H_Zf)
    hz_Zf = filt_X(hz.T,31,boundaries="extrap",s=0.2)
    hzz = d_dz(hz_Zf.T)
    hzz_Zf = filt_X(hzz.T,31,boundaries="extrap",s=0.2)
    hzzz0 = d_dz(hzz_Zf.T)
    hzzz = filt_X(hzzz0,31,boundaries="extrap",s=0.2)

    # d3_dxdz2 h:
    hxzz0 = d_dx(hzz_Zf.T)
    hxzz = filt_X(hxzz0,31,boundaries="extrap",s=0.2)

    # d3_dzdx2 h:
    # (some computations repeaded as for d3_dx3)
    hxx_Zf = filt_X(hxx.T,31,boundaries="extrap",s=0.2)
    hzxx0 = d_dz(hxx_Zf.T)
    hzxx = filt_X(hzxx0,31,boundaries="extrap",s=0.2)


    # Cell indices at which to take slices
    # and compare derivatives profiles:
    slice_at_x = int(1.3*nx/2)
    slice_at_z = int(1.3*nz/2)

    # Unfiltered derivatives along x:
    unfiltered_x = [
                  d_dx(d_dx(d_dx(h)))[:,slice_at_z],
                  d_dx(d_dz(d_dz(h)))[:,slice_at_z],
                  d_dz(d_dz(d_dz(h)))[:,slice_at_z],
                  d_dz(d_dx(d_dx(h)))[:,slice_at_z]
                  ]
    # Filtered derivatives slices along x:
    third_derivatives_slices_x = [
                                hxxx[:,slice_at_z],
                                hxzz[:,slice_at_z],
                                hzzz[:,slice_at_z],
                                hzxx[:,slice_at_z]
                                ]

    # Unfiltered derivatives along z:
    unfiltered_z = [
                  d_dx(d_dx(d_dx(h)))[slice_at_x,:],
                  d_dx(d_dz(d_dz(h)))[slice_at_x,:],
                  d_dz(d_dz(d_dz(h)))[slice_at_x,:],
                  d_dz(d_dx(d_dx(h)))[slice_at_x,:]
                  ]
    # Filtered derivatives slices along z:
    third_derivatives_slices_z = [
                                hxxx[slice_at_x,:],
                                hxzz[slice_at_x,:],
                                hzzz[slice_at_x,:],
                                hzxx[slice_at_x,:]
                                ]

    # Collect all third derivatives in a list to loop over them
    # for plotting:
    third_derivatives = [hxxx, hxzz, hzzz, hzxx]
    labels = ['hxxx', 'hxzz', 'hzzz', 'hzxx']

    # axes for the slices:
    X1 = -X[:,slice_at_z]
    Z1 = Z[slice_at_x,:]

    # domain for the contourplots:
    X = -X[1:-1,1:-1]
    Z = Z[1:-1,1:-1]

    # plot all derivatives:
    for i in range(len(third_derivatives)):
        # Controurplots:
        fig = plt.figure(figsize=(12,4))
        plt.plot()
        contourplot = plt.contourf(X, Z, third_derivatives[i],
                                   alpha = 0.9,
                                   cmap = 'Blues')
        plt.colorbar(contourplot,
                     format = '%.2f',
                     anchor = (0.5, 0.5))
        plt.grid(color='k', linestyle='--', linewidth=0.5)
        plt.title('Contour plot of the filtered ' + labels[i])
        plt.xlabel('domain length x, [-]')
        plt.ylabel('domain width z, [-]')
        plt.tight_layout()
        plt.savefig(directory \
                    + os.sep \
                    + labels[i] + '_' + filename + '.png',
                    format      = 'png',
                    dpi         = 300,
                    pad_inches  = 0.1,
                    bbox_inches = 'tight')
        # plt.show()
        plt.close()


        # Third derivatives profiles along x
        # comparing with the filtered versions:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(X1, unfiltered_x[i], '.-',
                 label = 'third derivative')
        plt.plot(X1[1:-1], third_derivatives_slices_x[i],
                 c = 'deeppink',
                 label = 'filtered derivative')
        plt.title('Comparison for the third derivative ' + labels[i])
        plt.xlabel('domain length x, [-]')
        plt.ylabel(labels[i])
        plt.tight_layout()
        plt.legend(loc='lower right')
        plt.savefig(directory \
                    + os.sep \
                    + labels[i] + '_xcomparison_' + filename + '.png',
                    format      = 'png',
                    dpi         = 300,
                    pad_inches  = 0.1,
                    bbox_inches = 'tight')
        # plt.show()
        plt.close()


        # Third derivatives profiles along z
        # comparing with the filtered versions:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(Z1, unfiltered_z[i], '.-',
                 label = 'third derivative')
        plt.plot(Z1[1:-1], third_derivatives_slices_z[i],
                 c = 'deeppink',
                 label = 'filtered derivative')
        plt.title('Comparison for the third derivative ' + labels[i])
        plt.xlabel('domain width z, [-]')
        plt.ylabel(labels[i])
        plt.tight_layout()
        plt.legend(loc='lower right')
        plt.savefig(directory \
                    + os.sep \
                    + labels[i] + '_zcomparison_' + filename + '.png',
                    format      = 'png',
                    dpi         = 300,
                    pad_inches  = 0.1,
                    bbox_inches = 'tight')
        # plt.show()
        plt.close()

        fig.clear()
        plt.clf()
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
