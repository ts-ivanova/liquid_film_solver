#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:50:30 2021

@author: tsvetelina ivanova

Post-processing the liquid film waves:
inlet in time
"""

import numpy as np

# plots:
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

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
        'size'   : 12}
rc('font', **font)

# transparency level
alpha1 = 0.9

# time step for the fixed plate case:
dt = 0.01
H_S = 0.000334339580339541# [m], scale for case 2


# Validation data:
A = np.loadtxt('./doro_fig5')
x_doro = A[:,0]
y_doro = A[:,1]

# shift to developed state and crop out unnecessary profile:
shift  = 30
total  = 140 # this is Lx
factor = ((total-shift)/0.12)
offset = shift/factor # this is to substract from x-axis 
# in order to have the same dimensional axes in m.
# 0.12m is the validation case domain,
# and it fits in 110 dimensionless length units of mine
# (this is how factor is defined).


####################
# os.chdir('RESULTS-processed/')
os.chdir('RESULTS_fixed_substrate/')

# Gather all computed configurations:
LIQUIDS = natsorted(glob.glob('water_without_surf_ten_*'))
# LIQUIDS = natsorted(glob.glob('3*'))

print('Going to process ', LIQUIDS)



for LIQUID in LIQUIDS:
    print('Case ', LIQUID)

    os.chdir(LIQUID + os.sep + 'SOLUTIONS_n')
    # FOLDERS = natsorted(glob.glob('dx*'))
    FOLDERS = natsorted(glob.glob('R*'))

    for FOLDER in FOLDERS:
        # Change working directory:
        os.chdir(FOLDER)
        subfolder = natsorted(glob.glob("*"))


        for i in range(len(subfolder)):
            print('Processing ', subfolder[i])
            
            # Extract information 
            # from the naming conventions:
            
            h0 = float(FOLDER[7:11])
            print('h0 = ', h0)
            # delta = Epsilon*Re:
            delta = FOLDER[-7:]
            # Freq:
            f = float(subfolder[i][-4:])
            print('f = ', f)
            # Re:
            Re = float(subfolder[i][-10:-7])
            print('Re = {:.0f}'.format(Re))
            
            conf_key = subfolder[i][:4]

            dx = float(subfolder[i][15:21])
            nx = int(subfolder[i][24:28])

            dz = float(subfolder[i][31:35])
            nz = int(subfolder[i][39:43])
            

            # Spatial dimensions:
            x = np.mgrid[0:nx]*dx
            z = np.mgrid[0:nz]*dz
            # Create matrices of the coordinate variables
            [Z,X] = np.meshgrid(z,x)

            os.chdir(subfolder[i])
            filelist = natsorted(glob.glob('h_np' + os.sep \
                                            + '*.npy'))

            # To save the contourf's:
            directory = "../../../POSTPROCESSED" \
                        + os.sep + "fixed_substrate" \
                        + "_Re{:.0f}".format(Re) \
                        + "_h{:.1f}".format(h0) \
                        + "_delta" + delta
             
            Path(directory).mkdir(parents=True,
                                  exist_ok=True)

            # initialize the height along x in time list:
            H_XT_list = []

            # an arbitrary chosen xslice of the domain:
            # xslice = int(0.5*nz/8)
            xslice = int(0.46*nz)

            # Loop over all data files to extract the height
            # and attach them to the dictionary:
            for ii in range(len(filelist)):
                h_np = np.load(filelist[ii])
                h = h_np[:,xslice]

                # store in matrix the height along x 
                # for each time
                H_XT_list.append([])
                H_XT_list[ii].append(h)

            H_XT = np.asarray(H_XT_list).reshape(len(filelist), nx)
            print('H_XT.shape = ', H_XT.shape)

            # Construct space and time axes:
            # x-axis:
            x_axis = x
            # For falling film on a fixed plate, keep positive sign.


            # %%
            print('Plotting ... ')

            for j in range(1,len(filelist)):
                plt.close()
                plt.figure(figsize = (10,5))

                plt.plot((x_axis)/factor - offset, #/1000 to turn to m
                         H_XT.T[:,j]*H_S*1000,
                         linestyle = 'solid',
                         color = '#574d57',
                         linewidth = 2,
                         alpha = alpha1,
                         label = 'IBL simulation')


                plt.xlabel('length x, [m]')
                plt.ylabel('film thickness h, [mm]')
                plt.title('A snapshot of the wave profile. ' \
                             + 'Frequency = 16 Hz, Reynolds = 15', \
                             pad=10, fontsize = 12)
                plt.legend(loc = 'upper right')
                plt.ylim(0.1, 0.6)
                plt.xlim(0.065,0.12)
                plt.grid()
                plt.plot(x_doro,#*((total-shift)/0.12)+shift)/1000-offset,
                         y_doro,
                         '.-',
                         label = 'Doro et al., 2013')
                plt.legend()
                plt.savefig(directory + os.sep \
                            + 'env_' + 'f_' + str(f) \
                            + '_Re' + str(Re) \
                            + '_frame_' + str(j)
                            + '.png', \
                            format      = 'png', \
                            dpi         = 200, \
                            bbox_inches = 'tight')

            os.chdir('../')
        os.chdir('../')
    os.chdir('../../')
    
os.chdir('../')
print("completed")
