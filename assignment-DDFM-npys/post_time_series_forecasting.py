#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:16:30 2021

@author: tsveti

Post-processing the liquid film waves:
Time series forecasting
"""

# https://scikit-learn.org/stable/modules/gaussian_process.html

# Tutorial slightly adapted from
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py

import numpy as np

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

# time series:
from sklearn.gaussian_process \
    import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, \
        RationalQuadratic, ExpSineSquared

# Show code documentation:
# print(__doc__)

####################
# PLOT CUSTOMIZATIONS:
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
rc('font', **font)
# transparency levels
alpha1 = 0.25
alpha2 = 0.9
alpha3 = 0.5



####################
os.chdir('../RESULTS/')

LIQUID = '2D_WATER_without_surface_tension'

FOLDER = LIQUID + os.sep + 'SOLUTIONS_n/dx0.0275'
# Change working directory:
os.chdir(FOLDER)
subfolder = natsorted(glob.glob("P*"))
####################

# Save the whole plot and/or a zoomed area:
zoom_list = [False, True]
# To save the plots:
directory = "../../POSTPROCESSED/predictions-time/"
Path(directory).mkdir(parents=True, exist_ok=True)


# Kernels:
kernels_list = [
                0.514**2 * RBF(length_scale=1e-05) \
                + 0.514**2 * RBF(length_scale=1e-05) \
                * ExpSineSquared(length_scale=1e-05, periodicity=1e+05)\
                + 0.388**2 * \
                RationalQuadratic(alpha=1e+05, length_scale=1e-05) \
                + 0.236**2 * RBF(length_scale=1e-05) \
                + WhiteKernel(noise_level=0.265)
                ]*4

#%%%%%
# Generate all plots for all subfolders/configurations:
for i in range(len(subfolder)):

    dx = float(subfolder[i][20:26])
    nx = float(subfolder[i][29:33])
    dz = float(subfolder[i][36:40])
    nz = float(subfolder[i][44:48])

    # cell number/location along x
    # a little after the middle
    # at which to collect time data and predict:
    pos1 = int(nx - nx*7/12)


    print('Now running in ', subfolder[i])
    print('at cell index ', pos1)
    print('out of total cells along x: ', nx)
    print('(inlet is last cell, outlet is 0th cell)')

    # Initialize a dictionary in which the heights will
    # be stored for the different time steps.
    h_n = dict()

    # The list of the .dat files in each subfolder:
    filelist =  natsorted(glob.glob(subfolder[i] \
                                    + os.sep + 'h_np' \
                                    + os.sep + '*.npy'))

    # Extract values from the filename:
    filename = filelist[0]
    # extract timestep dt:
    dt = float(filename[-16:-11])


    # Loop over all data files to exctract the height
    # and attach them to the dictionary:
    for ii in range(len(filelist)):
        # For the exponential growth/decay:
        # h = np.genfromtxt(filelist[ii], skip_header=1).T[1]
        # h_n[ii] = h
        h_np = np.load(filelist[ii])
        h_n[ii] = h_np[:,int(1.3*nz/2)].reshape(-1,1)
    # Loop over each line of the files,
    # note that len(h_n[0]) is the same number for each h_n[i]:
    for jj in range(len(h_n[0])):
        # A list to store the heigh values in time
        # for a fixed point in space:
        fixed_x = []
        # Go over all times:
        for j in range(len(filelist)):
            # For the exponential growth/decay:
            fixed_x.append(h_n[j][pos1])

    start        = int(0.3*len(fixed_x))
    predict_from = int(0.7*len(fixed_x))

    DATA = fixed_x[start:]

    full_data    = len(fixed_x[0:])
    available    = len(DATA)


    #%%
    # SHAPE THE DATA:

    # t is the time unit index:
    t = dt*np.linspace(0+start, available-1+start, available) \
                    .reshape(-1,1)

    # h is the liquid film height, dimensionless:
    h = np.asarray(DATA).reshape(-1,1)
    # Full solution (take all elements in DATA):
    t_full = dt*np.linspace(0, full_data-1, full_data) \
                        .reshape(-1,1)
    h_full = np.asarray(fixed_x[0:]) \
                        .reshape(-1,1)



    # LEARNED kernel
    kernel = kernels_list[0]

    # Improve the kernel:
    gp = GaussianProcessRegressor(kernel = kernel,
                                  alpha = 0,
                                  normalize_y = True)
    gp.fit(t, h)

    # Print the learned kernel:
    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp.log_marginal_likelihood(gp.kernel_.theta))


    #%%
    # This function prepares the data:
    t_ = np.linspace(t.min(), t.max(), 1000)\
                    [:, np.newaxis]
    y_pred, y_std = gp.predict(t_, return_std = True)

    #%% PLOTS:
    plt.close()

    # Create the figure
    fig, ax = plt.subplots(figsize=(18, 6))

    # Plot full data (numerical solution)
    # to compare with the prediction:
    plt.plot(t_full, h_full
             , '.k-'
             , markersize = 8
             , alpha = alpha1
             , label = 'height - numerical solution')

    # Convert the y_pred(iction) nested list
    # into a flat list in order to be able
    # to plot the prediction ranges:
    y_pred = [item for elem in y_pred for item in elem]

    # Plot the prediction curve
    # with its ranges of uncertainty:
    plt.plot(t_, y_pred
             , 'm--'
             , linewidth = 2
             , label = 'prediction curve'
             , alpha = alpha2)
    # uncertainties of 95% confidence interval:
    plt.fill_between(t_[:,0]
                     , y_pred + 1.96*y_std
                     , y_pred - 1.96*y_std
                     , alpha = alpha3
                     , label = 'uncertainty for C.I. = 95%')

    suptitle1 = 'Prediction of the liquid film height'

    title1 = 'Start of prediction at time unit = {:.2f}'\
            .format(predict_from*dt)

    plt.suptitle(suptitle1, fontsize = 18)
    plt.title(title1, pad = 20)

    plt.xlabel('t, time units')
    plt.ylabel('h, dimensionless liquid film height')
    plt.tight_layout()
    plt.legend(loc = 'upper left', fontsize=10)
    plt.grid(linewidth = 0.8)
    # Mark with a line where the data is taken into account:
    plt.axvline(x=dt*start, c='g', ls=':', lw=0.5)
    # Mark with a line where the prediction starts:
    plt.axvline(x=dt*predict_from)
    # Save the figure:
    plt.savefig(directory + os.sep \
                + 'predict_from_time_' \
                + str(int(predict_from)) \
                + '_' + subfolder[i] + '.png' \
                , format = 'png'
                , dpi = 200
                , bbox_inches = 'tight')

    # zoom determines whether the whole plot is saved
    # or a specified range of it.
    for j in range(len(zoom_list)):
        zoom = zoom_list[j]
        if zoom == True:
            plt.xlim(start, t_full.max())
            plt.savefig(directory \
                        + os.sep + 'time_predict_from_' \
                        + str(int(predict_from))
                        + 'zoomed_' \
                        + subfolder[i] + '.png' \
                        , format = 'png'
                        , dpi = 400
                        , bbox_inches = 'tight')
    # plt.show()

os.chdir('../../../../')
