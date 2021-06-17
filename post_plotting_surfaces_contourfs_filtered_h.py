#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 02 18:00:03 2021

@author: tsveti

Plot the height surfaces from the liquid film solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path

import gc

# Import home-made script for saving things in directories etc.
import tools_for_plotting as save_plots


####################
os.chdir('RESULTS/')

# Gather all computed configurations:
# LIQUIDS = natsorted(glob.glob('2D*')) + natsorted(glob.glob('3D*'))
LIQUIDS = natsorted(glob.glob('*'))

# Choose which plots to produce:
Surfaces  = False     # liquid film height surfaces h
Contourfs = True      # contour plot of the last computed time step of h
Filtered  = False     # filtered third derivatives contourplots
                      # and comparisons

if Filtered:
    LIQUIDS = ['3D_WATER_with_surface_tension_coarse_XZ1/',
               '3D_WATER_with_surface_tension_coarse_XZ2/',
               '3D_WATER_with_surface_tension_whole_zdomain_coarseXZ1/']


print('Going to process ', LIQUIDS)

for LIQUID in LIQUIDS:
    print('Case ', LIQUID)
    os.chdir(LIQUID + os.sep + 'SOLUTIONS_n')
    FOLDERS = natsorted(glob.glob('dx*'))

    for FOLDER in FOLDERS:
        # Change working directory:
        os.chdir(FOLDER)
        subfolder = natsorted(glob.glob("P*"))

        for i in range(len(subfolder)):
            print('Processing ', subfolder[i])

            conf_key = subfolder[i][:4]

            # extract information from naming convention:
            dx = float(subfolder[i][20:26])
            nx = int(subfolder[i][29:33])
            dz = float(subfolder[i][36:40])
            nz = int(subfolder[i][44:48])

            # Spatial dimensions:
            x = np.mgrid[0:nx]*dx
            z = np.mgrid[0:nz]*dz
            # Create matrices of the coordinate variables
            [Z,X] = np.meshgrid(z,x)

            os.chdir(subfolder[i])
            filenames = natsorted(glob.glob('h_np' + os.sep \
                                           + '*.npy'))


            if Surfaces:
                print('Plotting height surfaces from ', subfolder[i])

                # To save the plot surfaces:
                directory = "../../../POSTPROCESSED/height_surfaces"
                Path(directory).mkdir(parents=True, exist_ok=True)

                directory_plots  =  directory + os.sep \
                                    + subfolder[i]
                Path(directory_plots).mkdir(parents=True, exist_ok=True)

                # if the files are too many, plot once every 5 files:
                if len(filenames) > 50:
                    skipfiles = int(len(filenames)/25)
                else:
                    skipfiles = 1

                for j in range(0, len(filenames), skipfiles):
                    h = np.load(filenames[j])
                    save_plots.plot_surfaces(h, X, Z,
                                             j*100,
                                             h0,
                                             directory_plots,
                                             filenames[j][5:-11],
                                             conf_key)


            if Contourfs:
                print('Plotting contour plots from ', subfolder[i])

                # To save the contourf's:
                directory = "../../../POSTPROCESSED/contourfs"
                Path(directory).mkdir(parents=True, exist_ok=True)

                directory_plots  =  directory + os.sep \
                                    + subfolder[i]
                Path(directory_plots).mkdir(parents=True, exist_ok=True)

                # Plot contourf's:
                h = np.load(filenames[-1])
                save_plots.plot_contourfs(h, X, Z,
                                          len(filenames),
                                          directory_plots,
                                          filenames[-1][5:-11])


            if Filtered:
                print('Plotting filtered derivatives from ',
                      subfolder[i])

                # To save the plot surfaces:
                directory = "../../../POSTPROCESSED/filtered"
                Path(directory).mkdir(parents=True, exist_ok=True)

                directory_plots  =  directory + os.sep \
                                    + subfolder[i]
                Path(directory_plots).mkdir(parents=True, exist_ok=True)

                skipfiles = 10

                for j in range(0, len(filenames), skipfiles):
                    h = np.load(filenames[j])
                    save_plots.filtered(h, X, Z,
                                        dx, dz,
                                        nx, nz,
                                        j*100,
                                        directory_plots,
                                        filenames[j][5:-4])


            print('Completed plotting for ', subfolder[i])
            os.chdir('../')

            gc.collect()
            del filenames
            del h
            del x
            del X
            del z
            del Z

        print('Completed all from ', FOLDER)
        os.chdir('../')
    print('Completed all for Case ', LIQUID)
    os.chdir('../../')
os.chdir('../')