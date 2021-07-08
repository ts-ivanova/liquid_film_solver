# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:02:30 2021
Modified in April 2021

@author: mendez
Adaptations by Tsveti for liquid film height predictions.

GAUSSIAN PROCESS REGRESSION

The imported data represents numerical solutions
of an integral boundary layer model for h, qx, qz
(height, flow rate along x and along z)
and it is obtained from the 3D liquid film solver
developed during RM2021.
"""


# Tutorial adapted from
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import os

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
        'size'   : 18}
rc('font', **font)

# transparency levels
alpha1 = 0.5
alpha2 = 0.9
alpha3 = 0.5


#%%
# Tsveti
# Data folder from Numerical solver for liquid films
FOLDER = 'SOLUTIONS'

# Options to choose among for wave height prediction:
# (see descriptions below)
Opt1 = 0 # the 2D OpenFOAM validation test case
Opt2 = 1 # a generated 3D wave
Opt3 = 2 # another generated 3D wave

# The list of options to predict for:
option_list = [Opt1, Opt2, Opt3]

# Save the whole plot and/or a zoomed area:
zoom_list = [False, True]


# Generate all plots with a single run:
for i in range(len(option_list)):
    option = option_list[i]
    print('Now running option ', option)

    # Discard (or not) first few cells
    # as they may cause issues
    # because of the shape of the wave
    # (especially for the 2D OF wave)
    cut = 1680
    # (no discarding, 1680 is the total
    # number of cells in the domain)
    last_cell = 680
    # (last_cell is actually cell 1000 = 1680 - 680)

    if option == Opt1:
        # Option 1: The OpenFOAM validation test case,
        # a 2D wave data file:
        name = ['BlendedScheme_mesh_0.046_T80_C1']
        cut = 1650
        last_cell = 650
    elif option == Opt2:
        # Option 2: a 3D wave generated with
        # perturbations at the inlet data file:
        name = ['LW_mesh_0.046_T100_C3']
    elif option == Opt3:
        # Option 3: a 3D wave generated with
        # perturbations at the inlet data file:
        name = ['LW_mesh_0.046_T100_C4']

    # Assign and generate data:
    Name = FOLDER + os.sep + name[0] + '.dat'
    DATA = np.genfromtxt(Name, skip_header=1)

    #%%
    # SHAPE THE DATA:

    # x is the cell index:
    realign = len(DATA[:cut,0])
    x = np.asarray(-DATA[last_cell:cut,0] + realign)\
        .reshape(-1,1)
    # minus sign and added term 'realign'
    # to readjust domain orientation
    # (left - inlet; right - outlet).

    # h is the liquid film height, dimensionless:
    h = np.asarray(DATA[last_cell:cut,1])\
        .reshape(-1,1)

    # Full solution (take all elements in DATA):
    x_full = np.asarray(-DATA[:cut,0] + realign)\
            .reshape(-1,1)
    h_full = np.asarray(DATA[:cut,1])\
            .reshape(-1,1)

    #%% ADJUST KERNEL WEIGHTS
    # Kernel with optimized parameters
    # (found/learned via HPO!)

    # Overall the kernel is k = k1 + k2 + k3 + k4,
    # with k1 - long term smooth rising trend,
    #      k2 - seasonal component,
    #      k3 - medium term irregularities,
    #      k4 - noise terms.

    # What has been printed in the console
    # from the learning:
    if option == Opt1:
        # learned kernel for 2D OpenFOAM wave:
        k1 = 0.0243**2 * RBF(length_scale=4.71e+03)
        k2 = 1.37**2 * RBF(length_scale=2.19e+03) \
            * ExpSineSquared(length_scale=1.92,
                             periodicity=407)
        k3 = 0.072**2 * \
            RationalQuadratic(alpha=0.00325,
                              length_scale=151)
        k4 = 0.00316**2 * RBF(length_scale=0.1) \
            + WhiteKernel(noise_level=1e-05)

    elif option in (Opt2, Opt3):
        # learned kernel for the 3D waves:
        k1 = 0.0246**2 * RBF(length_scale=1e+05)
        k2 = 1.14**2 * RBF(length_scale=334) \
            * ExpSineSquared(length_scale=1.13,
                             periodicity=41.1)
        k3 = 0.00316**2 * RationalQuadratic(alpha=1e+05,
                                            length_scale=1e+05)
        k4 = 0.00316**2 * RBF(length_scale=0.1) \
            + WhiteKernel(noise_level=1e-05)


    #%%
    # THE KERNEL IS THE SUM OF ALL THESE CONTIBUTIONS:
    kernel = k1 + k2 + k3 + k4

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
    predict_cells = len(DATA[:cut,0]) - len(x)

    # This function prepares the data:
    x_ = np.linspace(x.min(), x.max() + predict_cells, 1000)\
                    [:, np.newaxis]
    y_pred, y_std = gp.predict(x_, return_std = True)

    #%% PLOTS:
    plt.close()

    # Create the figure
    if   option == Opt1:
        fig, ax = plt.subplots(figsize=(10, 5))
    elif option in (Opt2, Opt3):
        fig, ax = plt.subplots(figsize=(18, 6))

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
            'along the length x of the domain. '

    title1 = 'Start of prediction at cell index = {:.0f}'\
             .format(len(h))

    plt.suptitle(suptitle1)
    plt.title(title1, pad = 10)

    plt.xlabel('cell index')
    plt.ylabel('film thickness [-]')
    plt.tight_layout()
    plt.legend(loc = 'upper left')
    plt.grid(linewidth = 0.8)
    # Mark with a line where the prediction starts:
    plt.axvline(x=len(h))

    # Save the figure:
    plt.savefig('TS_plots_dats/predict-' \
                + str(int(predict_cells)) \
                + '-cells-' + name[0] \
                + '.png' \
                , format = 'png'
                , dpi = 400
                , bbox_inches = 'tight')

    # zoom determines whether the whole plot is saved
    # or a specified range of it.
    for j in range(len(zoom_list)):
        zoom = zoom_list[j]
        if zoom == True:
            plt.xlim(950, 1450)
            plt.savefig('TS_plots_dats/predict-' \
                        + str(int(predict_cells)) \
                        + '-cells-' + name[0] \
                        + '-zoomed' \
                        + '.png'
                        , format = 'png'
                        , dpi = 400
                        , bbox_inches = 'tight')
    # plt.show()
