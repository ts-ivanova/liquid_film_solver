#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:40:46 2021

@author: tsveti

Compare numerical solutions from .dat files
for different mesh sizes and different schemes
and validate with OpenFOAM 2D case from JFM paper.
"""

import numpy as np
import io
import matplotlib
import matplotlib.pyplot as plt
# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path

# Check if directory exists. If not, create it.
Path("PLOTS").mkdir(parents=True, exist_ok=True)

# PLOT CUSTOMIZATIONS:
plt.close()
font = {'family'    : 'DejaVu Sans',
        'weight'    : 'normal',
        'size'      : 14}
matplotlib.rc('font', **font)

# DATA FOLDERS:
FOLDERS = ['different_schemes',
           'mesh_sensitivity'
           ]

# Plot titles and labels:
titles = ['Different schemes',
          'Mesh sensitivity'
          ]

xlabel1 = 'domain length x [-]'#, left:inlet, right:outlet'
ylabel1 = 'film thickness h [-]'

colors = ['#9e2ec9','#574d57','#2878b5']
types  = ['-.','--',':','']

for j in range(len(titles)):
    # plots properties
    plt.figure(figsize=(12,4))
    plt.suptitle(titles[j])
    plt.xlabel(xlabel1)
    plt.ylabel(ylabel1)
    alpha1 = 0.7

    # collect the .dat files
    os.chdir(FOLDERS[j])
    names = natsorted(glob.glob('*.dat'))
    os.chdir('../')

    # dictionaries to store the data
    var_holder  = {}
    data_holder = {}
    x_holder    = {}
    h_holder    = {}

    for i in range(len(names)):
        var_holder['H' + str(i)] = FOLDERS[j] + os.sep + names[i]
        locals().update(var_holder)

        data_holder['DATA_h'+str(i)] = \
            np.genfromtxt(var_holder['H' + str(i)],
                          skip_header=1)

        # extract dx value from the filenames:
        dx = float(names[i][14:20])

        factor = dx/0.0275
        if factor > 1:
            factor = round(factor, 0)
        else:
            factor = round(factor, 1)

        # number of cells along x
        nx = int(2810/factor)
        x_axis = -dx*np.linspace(0, nx-1, nx)

        h_holder['h'+str(i)] = \
            data_holder['DATA_h'+str(i)][:,0]

        plt.plot(x_axis,
                 h_holder['h'+str(i)]
                 .reshape(-1,1),
                 types[i],
                 linewidth = 2.2,
                 color = colors[i],
                 label = names[i][7:11] \
                 + ' dx = {:.4f}'.format(dx),
                 alpha = alpha1)

    plt.grid()
    plt.legend(loc='upper left',
               fontsize=12,
               numpoints=4,
               handlelength=4,
               markerscale=0.6)
    plt.tight_layout()
    plt.savefig('PLOTS' + os.sep + FOLDERS[j] + '.png',
                format='png', dpi=400)


    DATFILE = 'Validation_data_2D_OF_JFM' \
                + os.sep + 'validation_JFM_OF'+'.dat'
    VALIDATA = np.genfromtxt(DATFILE, delimiter=',', skip_header=1)

    h_OF  = VALIDATA[:,0].reshape(-1,1)
    x_OF  = VALIDATA[:,1].reshape(-1,1)

    x_OF = x_OF.reshape(-1,1)
    # Domain length 77.28 = nx*dx = 1680*0.046 from JFM paper
    plt.plot(x_OF - 77.28,
             h_OF,
             '.',
             markersize = '7',
             color = '#088bd1',
             label = 'OpenFOAM case',
             alpha = alpha1)

    plt.legend(loc='upper left',
               fontsize=12,
               numpoints=4,
               handlelength=4,
               markerscale=0.6)
    plt.savefig('PLOTS' + os.sep + 'OF_' + FOLDERS[j] + '.png',
                format='png', dpi=400)
