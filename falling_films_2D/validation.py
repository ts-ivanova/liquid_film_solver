#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivanova
date: 2022-02-14
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['legend.frameon'] = 1
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['figure.frameon'] = False
plt.rcParams["axes.formatter.min_exponent"] = True

plt.close()
A = np.loadtxt('./doro_fig5')
x_doro = A[:,0]
y_doro = A[:,1]

plt.plot(x_doro,y_doro, '.')
plt.show()

