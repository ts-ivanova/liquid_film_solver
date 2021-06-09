# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:14:53 2021

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt


# Here is some test to implemente the filtering

# Name = 'Blend_LWend_dx0.500_dt0.050_n08000.dat'
# H = np.genfromtxt(Name, skip_header=1).T[0]
# H = H.T
# I import the fake thickness map
# data = np.load('Fake_H.npz')
# H=data['H'].T # I do not know in which shape you have stored H in your code
data = np.load('Blend_minmo_h_dx0.200_dt0.020_n02500.npy')
H=data
# I assume the x directory is along the columns

# # Check extension functions for you to see the options
# from Functions_Miguel import Bound_EXT
# #### Test the extensions first
# h=H[:,40] # This is just one curve

# # Check the different options to learn what this function is doing
# h_EXT=Bound_EXT(h,15,'extrap')
# # Here is the result
# plt.plot(h_EXT)
# plt.show()

# This function will be implemented inside the filter.
# Go to see what is inside filt_X to know more

# Now we implement the filtering.
# Test an horizontal (along x?) filter:
from Functions_Miguel import filt_X
# Filter along x:
H_Xf=filt_X(H,31,boundaries="extrap",s=0.2)


# Compare results
fig = plt.figure(figsize=(8, 4))
# plt.plot(H[:,4])
# plt.plot(H_Xf[:,4])
plt.plot(H)
plt.plot(H_Xf)
plt.title('Signal')
plt.show()

# Make a stupid derivative comparison (Zoom it and see the difference!)
fig = plt.figure(figsize=(8, 4))
# plt.plot((np.diff(H[:,4])))
# plt.plot((np.diff(H_Xf[:,4])))
plt.plot((np.diff(H)))
plt.plot((np.diff(H_Xf)))
plt.title('First Derivative')
plt.show()


# Make a stupid derivative comparison (Zoom it and see the difference!)
fig = plt.figure(figsize=(8, 4))
# plt.plot(np.diff(np.diff(H[:,4])))
# plt.plot(np.diff(np.diff(H_Xf[:,4])))
plt.plot(np.diff(np.diff(H)))
plt.plot(np.diff(np.diff(H_Xf)))
plt.title('Second Derivative')
plt.show()
