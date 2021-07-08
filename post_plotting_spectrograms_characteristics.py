#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:50:30 2021

@author: tsveti

Post-processing the liquid film waves:
Plotting the spectrograms in space,
as well as characteristics.

The spectrograms give information on whether
the wave evolves towards other frequencies.
The characteristic lines indicate whether
the nonlinear effects are weak or not.
"""

import numpy as np

# plots:
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

# spectrogram
from scipy.signal import spectrogram
import pandas as pd
import datetime
from scipy.fft import fftshift

# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path


# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)


####################
os.chdir('RESULTS-processed/')
# Gather all computed configurations:
# LIQUIDS = natsorted(glob.glob('2D*')) + natsorted(glob.glob('3D*'))
LIQUIDS = natsorted(glob.glob('*'))

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

            # Extract information from the naming conventions:
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
            filelist = natsorted(glob.glob('h_np' + os.sep \
                                           + '*.npy'))

            # To save the contourf's:
            directory = "../../../POSTPROCESSED/spectrogr_charact"
            Path(directory).mkdir(parents=True, exist_ok=True)

            H_XT_list = []
            # Loop over all data files to extract the height
            # and attach them to the dictionary:
            for ii in range(len(filelist)):
                h_np = np.load(filelist[ii])
                h = h_np[:,int(1.2*nz/2)]
                # 1.2*nz/2 - an arbitrary chosen slice of the domain

                # store in matrix the height along x for each time
                H_XT_list.append([])
                H_XT_list[ii].append(h)

            print('Computing spectrograms ... ')

            # extract timestep dt:
            dt = float(filelist[-1][-16:-11])
            # final_time:
            T  = int(filelist[-1][-22:-19])


            H_XT = np.asarray(H_XT_list).reshape(len(filelist), nx)
            print('H_XT.shape = ', H_XT.shape)

            # Construct space and time axes:
            # x-axis:
            x_axis = -x
            # (flip because the inlet is at the last index -1.)

            # t-axis:
            n_files_in_time = len(filelist)
            t_axis = 100*dt*np.linspace(0, len(filelist), len(filelist))
            # ^ multiplication by 100 because the saved solutions
            # are chosen to be every 100 steps (can be modified).

            H_XF = np.zeros(H_XT.shape)

            # This is the numpy tool for freq axis:
            Freqs = fftshift(np.fft.fftfreq(len(filelist)))*1/dt

            for j in range(1,nx):
                # Profile at location j
                Prof_time = H_XT[:,j]
                Prof_Freq = np.abs(fftshift(np.fft.fft(Prof_time
                                                       - Prof_time.mean())))
                H_XF[:,j] = Prof_Freq
            plt.close()
            fig, ax = plt.subplots()
            contourplot_sp = plt.contourf(Freqs/100,
                                          x_axis,#-x_axis.min(),
                                          H_XF.T,
                                          cmap = 'PuBu')
            # plt.colorbar(contourplot_sp,
            #              format = '%.2f',
            #              anchor = (0.5, 0.5))
            # print('H_XF.T', H_XF.T)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('frequency, [-]')
            plt.ylabel('length x, [-]')
            plt.xlim(0,)
            plt.title('Spectrogram')
            # plt.title(subfolder[i])
            plt.savefig(directory + os.sep \
                        + 'spect_' + subfolder[i] \
                        + '.png',
                        format      = 'png',
                        dpi         = 200,
                        bbox_inches = 'tight')
            print('Plotting characteristics ... ')
            plt.close()
            contourplot_ch = plt.contourf(t_axis,
                                          x_axis,#-x_axis.min(),
                                          H_XT.T,
                                          cmap = 'PuBu')
            plt.colorbar(contourplot_ch,
                         format = '%.2f',
                         anchor = (0.5, 0.5))
            plt.xlabel('time, [-]')
            plt.ylabel('length x, [-]')
            plt.title('Characteristics')
            # plt.title(subfolder[i])
            plt.savefig(directory + os.sep \
                        + 'char_' + subfolder[i] \
                        + '.png',
                        format      = 'png',
                        dpi         = 200,
                        bbox_inches = 'tight')

            os.chdir('../')

        os.chdir('../')
    os.chdir('../../')
os.chdir('../')
