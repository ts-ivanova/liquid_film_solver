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

####################
# os.chdir('RESULTS-processed/')
os.chdir('RESULTS_fixed_plate/')
# Gather all computed configurations:
LIQUIDS = natsorted(glob.glob('w*'))
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
            xslice = int(0.45*nz)

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
            # Flip for moving substrate only because 
            # the inlet is at the last index -1.
            # For falling film on a fixed plate, keep positive sign.

            plt.close()

            # %%
            print('Plotting ... ')
            for j in range(1,len(filelist),100):
                plt.close()
                plt.figure(figsize = (12,4))
                #plt.ylim(0.1,0.4)

                plt.plot(x_axis,
                         H_XT.T[:,j],
                         linestyle = 'solid',
                         color = '#574d57',
                         linewidth = 2,
                         alpha = alpha1,
                         label = 'height')


                plt.xlabel('length x, [-]')
                plt.ylabel('film thickness h, [-]')
                plt.suptitle('Profile in x, time ' \
                            + '{:.2f}'.format(dt*j), y=1.04)
                plt.title('Frequency ' + str(f) \
                            + ' Reynolds ' + str(Re))
                plt.legend(loc = 'upper right')
                plt.ylim(h0 - 0.6, h0 + 1.5)
                plt.grid()
                # plt.title(subfolder[i])
                plt.savefig(directory + os.sep \
                            + 'env_' + 'f_' + str(f) \
                            + '_Re' + str(Re) \
                            + '_frame_' + str(j)
                            + '.png', \
                            format      = 'png', \
                            dpi         = 200, \
                            bbox_inches = 'tight')
            #plt.show()


            os.chdir('../')
        os.chdir('../')
    os.chdir('../../')
    
os.chdir('../')
print("completed")
