# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:02:30 2021
Modified in April 2021

@author: mendez
Course: Tools for Scientific Computing

Adaptations by ivanova for liquid film height predictions.

GAUSSIAN PROCESS REGRESSION

The imported data represents numerical solutions
of an integral boundary layer model for h, qx, qz
(height, flow rate along x and along z)
and it is obtained from the 3D liquid film solver
developed during RM2021.
"""

# Tutorial adapted from
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
# "The tutorial from the above link
# on “Gaussian Processes for Machine Learning”
# illustrates an example of complex kernel engineering
# and hyperparameter optimization using gradient ascent
# on the log-marginal-likelihood.
# The objective is to model predictions
# as a function of the time t."

# Import libraries:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path

# garbage collector
import gc

# For the predictions model:
from sklearn.gaussian_process \
    import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, \
        RationalQuadratic, ExpSineSquared

# Show code documentation:
# print(__doc__)


# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)

# transparency levels
alpha1 = 0.5
alpha2 = 0.9
alpha3 = 0.5

# %%

# Save the whole plot and/or a zoomed area:
zoom_list = [False, True]

# Gather all cases for prediction:
LIQUIDS = sorted(glob.glob('SOLUTIONS' + os.sep + 'W*'))
print('Going to process ', LIQUIDS)

# Create the path to save the plots, if it does not exist:
directory = "TS_plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# Loop over all cases:
for LIQUID in LIQUIDS:
    print('\n Now solving for Case ', LIQUID)
    # Go into case directory:
    os.chdir(LIQUID)
    # Gather solution folders
    # (they start with P (from Perturbations))
    subfolder = natsorted(glob.glob("P*"))

    # Go over all solution folders:
    for i in range(len(subfolder)):
        print('Processing ', subfolder[i])

        # Extract name of configuration:
        conf_key = subfolder[i][:4]

        # Extract cell sizes and numbers of cells along x and z:
        dx = float(subfolder[i][20:26])
        nx = float(subfolder[i][29:33])
        dz = float(subfolder[i][36:40])
        nz = float(subfolder[i][44:48])

        # Full spatial dimension:
        x_full = -np.mgrid[0:nx]*dx
        x_full = x_full.reshape(-1,1)

        # Ignore first few cells to avoid issues:
        cut = int(0.92*nx)
        # Specify until when to take data into account:
        last_cell = int(0.3*nx)

        # Construct axes:
        x = -np.mgrid[last_cell:cut]*dx
        x = x.reshape(-1,1)
        # z = np.mgrid[0:nz]*dz

        # Go into the subfolder to get the .npy
        os.chdir(subfolder[i])
        directory_plots  =  "../../../" + os.sep \
                            + directory + os.sep \
                            + subfolder[i]
        Path(directory_plots).mkdir(parents=True, exist_ok=True)

        filenames = natsorted(glob.glob('*.npy'))

        # Dimensionless liquid film height in npy format:
        h_np = np.load(filenames[0])
        # Check object lenghts
        print('len x ', len(h_np[:,int(1.3*nz/2)]))
        print(nx)
        print('len z ', len(h_np[int(nx/2),:]))
        print(nz)

        # Take a slice along the x-axis (streamwise):
        # full numerical solution
        h_full = h_np[:,int(1.3*nz/2)].reshape(-1,1)
        # Take part of the solution used in the prediction model
        h = h_full[last_cell:cut].reshape(-1,1)



        # Start on the predictions,
        # Learned kernels:
        if   LIQUID == 'SOLUTIONS/WATER_without_surface_tension_2D':
            k = 0.0244**2 * RBF(length_scale=1e+05) \
                + 1.88**2 * RBF(length_scale=4.51e+04) \
                * ExpSineSquared(length_scale=1.11, periodicity=38)\
                + 0.00316**2 \
                * RationalQuadratic(alpha=1e+05, length_scale=1e+05) \
                + 0.0902**2 * RBF(length_scale=2.58) \
                + WhiteKernel(noise_level=1e-05)

        else:
            k = 0.00316**2 * RBF(length_scale=1e+05) \
                + 0.988**2 * RBF(length_scale=1e-05) \
                * ExpSineSquared(length_scale=1e-05, periodicity=1e-05)\
                + 0.00316**2 * \
                RationalQuadratic(alpha=1e+05, length_scale=1e+05) \
                + 0.00316**2 * RBF(length_scale=1e+05) \
                + WhiteKernel(noise_level=0.0236)


        # The kernel is the sum of all contributions:
        # kernel = k1 + k2 + k3 + k4
        kernel = k

        # Improve the kernel:
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0,
                                      normalize_y=True)
        gp.fit(x, h)

        # Print the learned kernel:
        print("\nLearned kernel: %s" % gp.kernel_)
        print("Log-marginal-likelihood: %.3f"
              % gp.log_marginal_likelihood(gp.kernel_.theta))


        #%%
        # How many cells ahead to predict:
        predict_cells = cut - len(h)


        # Predictions:
        x_ = np.linspace(x.min(), x.max() + (predict_cells)*dx,
                         100)[:, np.newaxis]
        y_pred, y_std = gp.predict(x_, return_std = True)


        #%% PLOTS:
        plt.close()

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot full data (numerical solution)
        # to compare with the prediction:
        plt.plot(x_full, h_full
                 , '.k-'
                 , markersize = 4
                 , alpha = alpha1
                 , label = 'height - numerical solution')

        # Convert the y_pred(iction) nested list
        # into a flat list in order to be able
        # to plot the prediction ranges:
        y_pred = [item for elem in y_pred for item in elem]

        # Plot the prediction curve
        # with its ranges of uncertainty:
        plt.plot(x_, y_pred
                 , 'm--'
                 , linewidth = 2
                 , label = 'prediction curve'
                 , alpha = alpha2)
        # uncertainties of 95% confidence interval:
        plt.fill_between(x_[:,0]
                         , y_pred + 1.96*y_std
                         , y_pred - 1.96*y_std
                         , alpha = alpha3
                         , label = 'uncertainty for C.I. = 95%')

        suptitle1 = 'Prediction of the liquid film height ' + \
                    'along the domain length. '

        title1 = 'Start of prediction at x = {:.2f}'\
                 .format(-(last_cell)*dx)

        plt.suptitle(suptitle1)
        plt.title(title1, pad = 10)

        plt.xlabel('domain length x, [-]')
        plt.ylabel('film height h, [-]')
        plt.tight_layout()
        plt.legend(loc = 'upper left')
        plt.grid(linewidth = 0.8)
        # Mark with a line the cutted data in the beginning:
        plt.axvline(x=-(cut)*dx, c='grey', ls = ':')
        # Mark with a line where the prediction starts:
        plt.axvline(x=-(last_cell)*dx)

        # Save the figure:
        plt.savefig(directory_plots + os.sep \
                    + 'predict_x' \
                    + str(int(-(last_cell)*dx)) \
                    + '_' + subfolder[i] \
                    + '.png' \
                    , format = 'png'
                    , dpi = 400
                    , bbox_inches = 'tight')

        # zoom determines whether the whole plot is saved
        # or a specified range of it.
        for zoom in zoom_list:
            if zoom == True:
                plt.xlim(-(0.5*nx)*dx, x_full.max())
                plt.savefig(directory_plots + os.sep \
                            + 'predict_x' \
                            + str(int(-(last_cell)*dx)) \
                            + '_' + subfolder[i] \
                            + '_zoomed' \
                            + '.png'
                            , format = 'png'
                            , dpi = 400
                            , bbox_inches = 'tight')
        # plt.show()

        # Cleanup:
        plt.clf()
        plt.close()
        gc.collect()

        os.chdir('../')
        print('Completed predictions for ', subfolder[i])
    print('of Case ', LIQUID)
    os.chdir('../../')
