#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:48:30 2021

@author: tsveti

Post-processing the liquid film waves:
Plotting the maximum values of each position over time
in order to see if the perturbations decrease or increase.
"""

import numpy as np
from statistics import mean

# plots:
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path
# Produce gifs
import imageio

import gc


# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}
rc('font', **font)

# transparency level
alpha1 = 0.9

####################
os.chdir('RESULTS/')
# Gather all computed 2D configurations:
LIQUIDS = natsorted(glob.glob('2D*'))


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
            print('Plotting envelopes from ', subfolder[i])

            conf_key = subfolder[i][:4]

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

            # To save the plots:
            directory = "../../../POSTPROCESSED/envelopes"
            Path(directory).mkdir(parents=True, exist_ok=True)

            directory_plots  =  directory + os.sep \
                                + subfolder[i]
            Path(directory_plots).mkdir(parents=True, exist_ok=True)

            # Initialize a dictionary in which the heights
            # will be stored for the different time steps.
            h_n = dict()

            # Initialize a dictionary that will store a maximum
            # in time of the height, for each position x.
            maxs       = dict()
            mins       = dict()
            means_dict = dict()

            # Create the subdirectory to save the .png's
            subdirectory = directory + os.sep + subfolder[i]
            Path(subdirectory).mkdir(parents=True, exist_ok=True)

            # The list of the .dat files in each subfolder:
            # filelist = natsorted(glob.glob(subfolder[i] + os.sep + '*.dat'))
            filenames = natsorted(glob.glob('h_np' + os.sep \
                                           + '*.npy'))
            # if the files are too many, plot once every 5 files:
            if len(filenames) > 50:
                skipfiles = int(len(filenames)/25)
            else:
                skipfiles = 1

            # Extract values from the filename:
            filename = filenames[0]
            # extract timestep dt:
            dt = float(filename[-16:-11])

            # Loop over all data files to extract
            # the heights and put them in the dictionary:
            for j in range(0, len(filenames), skipfiles):
                print('Extracting data from file ', filenames[j])
                h = np.load(filenames[j])
                h_n[j] = h[:,int(1.2*nz/2)]

            # Loop over all data files to extract
            # the height at each location:
            for ii in range(0, len(filenames), skipfiles):
                print('Processing file ', filenames[ii])
                # A list to store the height values in time
                # for a fixed point in space:
                fixed_x = []
                # Loop over each line of the files
                for jj in range(nx):
                    fixed_x.append(h_n[ii][jj])
                    maxs[jj] = (max(fixed_x))
                    mins[jj] = (min(fixed_x))
                    means_dict[jj] = h_n[ii][jj]

                # THE QUANTITIES WHICH WE WILL PLOT:
                # maximums of h-axis:
                h_maxs = list(maxs.values())
                h_mins = list(mins.values())
                h_means = list(means_dict.values())

                # PLOT THE RESULTS
                plt.close()
                plt.figure(figsize = (12,4))
                title1 = 'Height envelopes in time ' \
                            + 'for each position x, ' \
                            + 'time = {0:03d}'.format(int(dt*ii*100))
                xlabel1 = 'length x [-], left:inlet, right:outlet'
                ylabel1 = 'film thickness h [-]'
                plt.title(title1)
                # plt.title(subfolder[i])
                plt.xlabel(xlabel1)
                plt.ylabel(ylabel1)
                plt.plot(-x, h_maxs,
                         linestyle = 'dashed',
                         color = '#9e2ec9',
                         linewidth = 2,
                         alpha = alpha1,
                         label = 'height max')
                plt.plot(-x, h_mins,
                         linestyle = 'dashdot',
                         color = '#2878b5',
                         linewidth = 2,
                         alpha = alpha1,
                         label = 'height min')
                plt.plot(-x, h_means,
                         linestyle = 'solid',
                         color = '#574d57',
                         linewidth = 2,
                         alpha = alpha1,
                         label = 'height')
                plt.legend(loc = 'upper right')
                plt.ylim(0.1, 0.4)
                plt.grid()

                plt.savefig(subdirectory + os.sep \
                            + 'envelopes_' \
                            + subfolder[i] \
                            + '_n' + str(ii*10).zfill(5) \
                            + '.png',
                            dpi = 200,
                            bbox_inches = 'tight')

                plt.clf()
                plt.close()
                gc.collect()


            # Produce .gifs
            # Change working directory:
            os.chdir(subdirectory)
            # To save the gifs:
            directory_gifs = "../../envelopes_gifs"
            # create if not existing:
            Path(directory_gifs).mkdir(parents=True, exist_ok=True)

            # Extract frame names:
            files = list(natsorted(os.listdir('./')))

            # Specify frames per second
            fpss = 10

            # Loop to produce the animated .gif:
            # for j in range(0, len(files)):
            images = [imageio.imread(file) \
                      for file in files]
            imageio.mimwrite(directory_gifs \
                             + os.sep + subfolder[i] + '.gif',
                             images, fps=fpss)

            gc.collect()
            os.chdir('../../../SOLUTIONS_n/' + FOLDER)

    #     os.chdir('../')
    # # return to directory to loop configurations
    # os.chdir('../')
        print('Completed all from ', FOLDER)
        os.chdir('../')
    print('Completed all for Case ', LIQUID)
    os.chdir('../../')

print('Completed.')
# return to origin
os.chdir('../../../')
