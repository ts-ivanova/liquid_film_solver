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
                       liquids_key, 
                       scheme_choice,
                       h0,
                       dx, nx, dz, nz,
                       CFL, dt, final_time,
                       Epsilon, Re, freq):
    # DIRECTORIES
    # To store the numerical data, check if the
    # directories exists. If not, create them.

    # The reduced Reynolds number is defined here as:
    delta = 3*Epsilon*Re
    # A string containing the initial film height value
    # and the perturbation amplitude:
    h0_amplitude_and_delta = "h{:.2f}".format(h0) \
                             + "_delta{}".format('%07.3F' % delta)

    main_res = 'RESULTS_fixed_substrate'
    
    if surface_tension:
        all_st = "_with_surf_ten_" + 'Re{0:03d}'.format(Re)
        results_dir = main_res \
                        + os.sep + liquids_key + all_st
        Path(results_dir)\
            .mkdir(parents=True, exist_ok=True)
        os.chdir(results_dir)

    else:
        no_st = "_without_surf_ten_" + 'Re{0:03d}'.format(Re)
        results_dir = main_res \
                        + os.sep + liquids_key + no_st
        Path(results_dir)\
            .mkdir(parents=True, exist_ok=True)
        os.chdir(results_dir)


    # To store solution every few steps for the configuration:
    saving = scheme_choice \
             + '_CFL{:.2f}'.format(CFL) \
             + '_dx{:.4f}'.format(dx) \
             + '_nx{0:04d}'.format(nx) \
             + '_dz{:.3f}'.format(dz) \
             + '_nz{0:04d}'.format(nz) \
             + '_Re{0:03d}'.format(Re) \
             + '_f{:.3f}'.format(freq)

    filename = scheme_choice \
             + '_dx{:.4f}'.format(dx) \
             + '_T{0:03d}'.format(final_time) \
             + '_dt{:.3f}'.format(dt)

    # The directory in which the solutions are saved:
    # -> separated by folders dx
    # directory_n = "SOLUTIONS_n" \
    #                 + os.sep + 'dx{:.4f}'.format(dx) \
    #                 + os.sep + saving
    # -> separated by folders Re::
    Re_folder = 'Re{0:03d}'.format(Re) + '_' + h0_amplitude_and_delta

    directory_n = "SOLUTIONS_n" \
                    + os.sep + Re_folder \
                    + os.sep + saving
    Path(directory_n).mkdir(parents=True, exist_ok=True)


    return  results_dir, directory_n, filename



#################################################
def save_np(h, #qx, 
            directory_n,
            filename,
            n):
    '''
    Save the height and qx as an .npy-file in (x,z)-dimensions.
    '''
    
    # To store the height .npy's:
    directory_h_np = directory_n \
                        + os.sep + 'h_np'
    Path(directory_h_np).mkdir(parents=True, exist_ok=True)
    # height:
    file = directory_h_np + os.sep \
            + "h_np_" \
            + filename \
            + "_n{0:05d}".format(n)
    np.save(file, h)
    
    # To store the qx .npy's:
    #directory_q_np = directory_n \
    #                    + os.sep + 'q_np'
    #Path(directory_q_np).mkdir(parents=True, exist_ok=True)
    ## qx:
    #file = directory_q_np + os.sep \
    #        + "q_np_" \
    #        + filename \
    #        + "_n{0:05d}".format(n)
    #np.save(file, qx)

