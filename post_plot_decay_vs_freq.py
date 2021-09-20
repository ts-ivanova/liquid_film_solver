#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:14:21 2021

@author: ivanova

Plots of the decay rates VS frequencies

"""


import numpy as np
import math

# plots:
from matplotlib import pyplot as plt
from matplotlib import rc

# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob



# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)

# transparency level
alpha1 = 0.9
markers = ['o', 'X', 'v']


####################
os.chdir('RESULTS/decay/')

Re_paths = natsorted(glob.glob('R*'))

for Re_path in Re_paths:
    os.chdir(Re_path)
    Paths = natsorted(glob.glob('3*'))
    for Path in Paths:

        os.chdir(Path)
        
        freqs_list = natsorted(glob.glob('f*'))
        decays_list = natsorted(glob.glob('d*'))
        
        for i in range(len(freqs_list)):
                
            h0 = float(decays_list[i][-7:-4])
            print('h0 = ', h0)
            Re = float(Re_path[-3:])
            print('Re = {:.0f}'.format(Re))
            
            freqs = np.load(freqs_list[i])
            k = 2*math.pi/(1/freqs)
            decay_rates = np.load(decays_list[i])
            
            plt.plot(freqs, decay_rates, 
                      marker=markers[i], 
                      linestyle='--',
                      alpha = alpha1,
                      label = '$h_0$ = ' + str(h0)
                      )    
            plt.xlim(0.04,0.15)
            # plt.plot(k, decay_rates, 
            #          marker=markers[i], 
            #          linestyle='--',
            #          alpha = alpha1,
            #          label = '$h_0$ = ' + str(h0)
            #          )   
        
        plt.xlabel('frequency, [-]')
        # plt.xlabel('wave number k, [-]')
        plt.ylabel('decay rate, [-]')
        plt.legend(loc = 'upper left')
        plt.grid()
        plt.title('Decay rates of amplitudes, Re = {:.0f}'
                  .format(Re))
        
        plt.savefig('plot_' + 'Re{:.0f}'.format(Re) \
                    + '.png',
                    format      = 'png',
                    dpi         = 200,
                    bbox_inches = 'tight')
        plt.show()
            
        os.chdir('../')
    os.chdir('../')