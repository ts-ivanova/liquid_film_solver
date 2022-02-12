#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:55:10 2021

Definition of functions used for saving
of plots, .dat files, .npy solutions
creating paths etc.
for the purposes of the Liquid film solver

@author: tsvetelina ivanova
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

#################################################
def create_directories(surface_tension,
                       liquids_key, conf_key,
                       scheme_choice,
                       dim,
                       h0, A,
                       dx, nx, dz, nz,
                       CFL, dt, final_time,
                       Epsilon, Re, freq):
    # DIRECTORIES
    # To store the numerical data, check if the
    # directories exists. If not, create them.

    # The reduced Reynolds number is defined as:
    delta = Epsilon*Re
    # A string containing the initial film height value
    # and the perturbation amplitude:
    h0_amplitude_and_delta = "h{:.2f}".format(h0) \
                             + "_A{:.2f}".format(A) \
                             + "_delta{}".format('%07.3F' % delta)

    if surface_tension:
        all_st = "_with_surf_ten_" + 'Re{:.0f}'.format(Re)
        results_dir = "RESULTS_January_fixed" \
                        + os.sep + dim + liquids_key + all_st

        Path(results_dir)\
            .mkdir(parents=True, exist_ok=True)
        os.chdir(results_dir)

    else:
        no_st = "_without_surf_ten_" + 'Re{:.0f}'.format(Re)
        results_dir = "RESULTS_January_fixed" \
                        + os.sep + dim + liquids_key + no_st

        Path(results_dir)\
            .mkdir(parents=True, exist_ok=True)
        os.chdir(results_dir)

    Path("PLOTS").mkdir(parents=True, exist_ok=True)
    Path("PLOTS/FRAMES").mkdir(parents=True, exist_ok=True)
    # To store solution every few steps for the configuration:
    saving = conf_key \
             + '_' + scheme_choice \
             + '_CFL{:.2f}'.format(CFL) \
             + '_dx{:.4f}'.format(dx) \
             + '_nx{0:04d}'.format(nx) \
             + '_dz{:.3f}'.format(dz) \
             + '_nz{0:04d}'.format(nz) \
             + '_Re{:.0f}'.format(Re) \
             + '_f{:.3f}'.format(freq)

    filename = conf_key \
             + '_' + scheme_choice \
             + '_dx{:.4f}'.format(dx) \
             + '_T{0:03d}'.format(final_time) \
             + '_dt{:.3f}'.format(dt)

    # The directory in which the solutions are saved:
    # -> separated by folders dx
    # directory_n = "SOLUTIONS_n" \
    #                 + os.sep + 'dx{:.4f}'.format(dx) \
    #                 + os.sep + saving
    # -> separated by folders Re::
    Re_folder = 'Re{:.0f}'.format(Re) + '_' + h0_amplitude_and_delta

    directory_n = "SOLUTIONS_n" \
                    + os.sep + Re_folder \
                    + os.sep + saving
    Path(directory_n).mkdir(parents=True, exist_ok=True)

    # To store plots every few steps for the configuration:
    directory_plots = "PLOTS/FRAMES" \
                    + os.sep + Re_folder \
                    + os.sep + saving
    Path(directory_plots).mkdir(parents=True, exist_ok=True)

    # To store plots every few steps for the configuration:
    directory_lim = "PLOTS/Limites"
    Path(directory_lim).mkdir(parents=True, exist_ok=True)

    return  results_dir, directory_n, directory_plots, \
            filename, directory_lim


#################################################
def save_to_dat(h, qx, qz,
                nx, nz,
                directory_n,
                filename,
                n):
    '''
    Save to .dat files a slice of the wave along x;
    (choose an index that is not proportional
    to the wavelength)
    '''
    # To store the height profile along x:
    directory_n_xslice = directory_n \
                        + os.sep + 'h_xslice'
    Path(directory_n_xslice).mkdir(parents=True, exist_ok=True)
    
    # choose a slice location
    xslice = int(1.2*nz/2)

    h_val  = [float(i) for i in h[:,xslice]]
    qx_val = [float(i) for i in qx[:,xslice]]
    qz_val = [float(i) for i in qz[:,xslice]]
    data   = np.array([h_val, qx_val, qz_val])
    data   = data.T

    with open(directory_n_xslice + os.sep \
               + "h_" + filename \
               + "_nz_" + str(xslice) \
               + "_n{0:05d}.dat".format(n),
              'wb') \
    as f:
        np.savetxt(f,
                   data,
                   fmt       = ['%.4f']*3,
                   delimiter = '  ',
                   newline   = '\n',
                   header    = 'h | qx | qz',
                   footer    = '',
                   comments  = '# ',
                   encoding  = None)


#################################################
def save_matrix(h, directory_n,
                filename,
                n):
    '''
    Save the height as a matrix in (x,z)-dimensions.
    '''
    # To store the height matrices:
    directory_n_matrix = directory_n \
                        + os.sep + 'h_matrix'
    Path(directory_n_matrix).mkdir(parents=True, exist_ok=True)

    # height:
    h_matrix = np.asmatrix(h)
    with open(directory_n_matrix + os.sep \
              + "hm_" \
              + filename \
              + "_n{0:05d}.dat".format(n),
              'wb') \
    as f:
        for line in h_matrix:
            np.savetxt(f, line, fmt='%.3f')


#################################################
def save_np(h, directory_n,
            filename,
            n):
    '''
    Save the height as an .npy-file in (x,z)-dimensions.
    '''
    # To store the height .npy's:
    directory_n_np = directory_n \
                        + os.sep + 'q_np'
    Path(directory_n_np).mkdir(parents=True, exist_ok=True)

    # height:
    file = directory_n_np + os.sep \
            + "h_np_" \
            + filename \
            + "_n{0:05d}".format(n)
    np.save(file, h)
