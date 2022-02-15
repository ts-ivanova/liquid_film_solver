#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:27:43 2021

SOLVER FOR A FALLING LIQUID FILM ON A FIXED SUBSTRATE

@author: tsvetelina ivanova

"""

#######################################################################

# Import libraries:
import numpy as np
import math
import sys

# Import numerical scheme procedures:
import lax_wendroff_richtmyer_fixed_plate as lw
#^not performing well
import lax_friedrichs_fixed_plate as lf
#^performing well
import lax_wendroff_friedrichs_blended_fixed_plate as bl
#^not performing well

# To monitor the execution time:
import os
import logging
from datetime import datetime

# Import home-made script for saving things in directories etc.
import tools_for_boundary_conditions_fixed_plate as comp
import tools_for_saving_fixed_plate as save_data

# For freeing up some space while computing:
import gc



#######################################################################

# Possible numerical schemes:
schemes  = {
            'LWendr' : 0, # Lax-Wendroff (2-step Richtmyer)
            'LFried' : 1, # Lax-Friedrichs
            'LWFble' : 2  # Blended Lax-Wendroff -- Lax-Friedrichs
            }

# Possible flux limiters for the Blended scheme:
limiters = [
            'LxFr',  # retrieves the Lax-Friedrichs scheme
            'LxWe',  # retrieves the Lax-Wendroff scheme
            'BLmi',  # minmode      (also used in 2D BLEW)
            'BLal',  # van Albada   (also used in 2D BLEW)
            'BLa2',  # van Albada 2
            'BLhc',  # HCUS
            'BLhq',  # HQUICK
            'BLko',  # Koren
            'BLvl'   # van Leer
            ]

# Choice of the numerical scheme:
scheme = schemes['LWFble']
# scheme = schemes['LFried']
# scheme = schemes['LWendr']

# If the selected scheme is the Blended scheme, 
# then specify more precisely which flux limiters
# are used (by un-commenting a line):
if scheme == schemes['LWFble']:
    # for naming purposes, 
    # to indicate which limiters are used:
    # scheme_choice = 'LxFr'
    # scheme_choice = 'LxWe'
    scheme_choice = 'BLmi'
    # ... more options can be added 
    # for the other limiter types if they are used.
else:
    # extract the string for the naming conventions:
    scheme_choice = \
        list(schemes.keys())[scheme][:4]



#######################################################################

# Specify whether to include surface tension terms
# in the sources (the third derivatives of the height h):
surface_tension = True
# (False = do not include)

# for naming purposes:
liquids_key = 'water'



#######################################################################

# PHYSICAL PROPERTIES AND SCALING
# LIQUID TYPE: DMSO
Re          = 15 # Reynolds number
F           = 16 # [Hz], perturbation frequency
rho         = 1098.3 # [kg/m^3], density of DMSO
g           = 9.78 # [m/s^2]
sigma       = 0.0484 #0.073 # [N/m]
nu          = 2.85*10**(-6) #1*10**(-6) # [m^2/s], kinematic viscosity
mu          = nu*rho # dynamic viscosity
t_nu        = (nu/g**2)**(1/3) # [s], time scale
l_nu        = (nu**2/g)**(1/3) # [m], length scale
l_sigma     = (sigma/(rho*g))**(1/2) # capillary length
Ka          = sigma/(g**(1/3)*nu**(4/3)*rho) # Kapitza
H_S         = l_nu*(3*Re)**(1/3) # [m], height scale
WE          = sigma/(rho*g*H_S**2) # Weber
Epsilon     = WE**(-1/3) # [-], long-wave parameter
X_S         = H_S/Epsilon # [m], reference
U_S         = g/nu*H_S**2 # [m/s], reference
T_S         = X_S/U_S # [s], ref time
Epsilon     = (3*Re)**(2/9)/(Ka**(1/3)) # [-]
delta       = 3*Epsilon*Re # [-]

f_hP = F*T_S # dimensionless perturbation freq
points_per_period = 32
# Perturbation period:
PERIOD_PROB = 1/f_hP
# Miguel's saving times:
dtS = PERIOD_PROB/points_per_period
freq = f_hP
# perturbation amplitude
q_a = 0.05
#freq = f_hP/(10*dtS)
#freq = 1/53

print('Epsilon = ', Epsilon)
print('Re = ', Re)
print('Ka = ', Ka)
print('H_S = ', H_S)
print('F = ', F)
print('f_hP = ', f_hP)
print('PERIOD_PROB = ', PERIOD_PROB)
print('dtS = ', dtS)
print('freq = ', freq)

# Initial dimensionless film height:
h0 = 1.047 #1 # [-]
U_REF = 5 # [m/s], reference velocity

#breakpoint()



#######################################################################

# SPACE INFORMATION
# dx - cell size along x-axis, [-]
# nx - number of cells along x-axis
# dz - cell size along z-axis, [-]
# nz - number of cells along z-axis
# L - domain length along x, [-]

# x
dx    = 0.1
L     = 140 
nx    = int(L/dx)
# z
dz    = 0.1
Lz    = 10
nz    = int(Lz/dz)
# Mesh:
x = np.mgrid[0:nx]*dx
z = np.mgrid[0:nz]*dz
# Create matrices of the coordinate variables
[Z,X] = np.meshgrid(z,x)



#######################################################################

# TIME INFORMATION
# Time is dimensionless.
# final_time - total simulation length, [-]
#final_time = int(PERIOD_PROB*100) #1200
final_time = int(PERIOD_PROB*36) #1200
# CFL number:
CFL = 0.3
# Timestep (unit) from the CFL formula:
dt = CFL*dx/U_REF
print('dt = ', dt)

# Number of timesteps:
nt = int(final_time/dt) 
time_steps = np.arange(1,nt+1,1)*dt

# Time between outputs:
output_interval = 1.0
# timesteps between outputs:
tsteps_btwn_out = np.fix(output_interval/dt)
noutput = int(np.ceil(nt/tsteps_btwn_out))
# noutput is the number of output frames.



#######################################################################

# Initialize h, qx, qz:
h = h0*np.ones((nx,nz), dtype='float32')
# Initial values for the flow rates
qx = (1/3)*h**3
qz = np.zeros((nx,nz), dtype='float32')



#######################################################################

# Create directories for storing data:
results_dir, directory_n, filename = \
    save_data.create_directories(surface_tension,
                                 liquids_key,
                                 scheme_choice,
                                 h0, dx, nx, dz, nz,
                                 CFL, dt, final_time,
                                 Epsilon, Re, freq)



#######################################################################
# INITIALIZE THE 3D ARRAYS
# where the output data will be stored
qx_save = np.zeros((nx, nz, noutput),
                   dtype='float32')
qz_save = np.zeros((nx, nz, noutput),
                   dtype='float32')
h_save  = np.zeros((nx, nz, noutput),
                   dtype='float32')
t_save  = np.zeros((noutput,1),
                   dtype='float32')
# Index of stored data
i_save = 0



#######################################################################
# Information about the running case:
info =  '\n Results directory: ' + results_dir \
        + '\n Liquid type: ' + liquids_key \
        + '\n surface_tension: ' + str(surface_tension) \
        + '\n Computing configuration: ' \
        + '\n ' + directory_n \
        + '\n ' + filename \
        + '\n domain length L: {:.2f}'.format(L)
print(info)



#######################################################################
# mark the beginning of the computations:
startTime = datetime.now()
# MAIN LOOP
for n in range(0, nt): #0,1,2,3,...,nt-1
    # Save fields for every fixed
    # number of timesteps
    if np.mod(n, tsteps_btwn_out) == 0:
        print("T = {:.2f} (max {:.0f}), [-]".format(n*dt, final_time))
        qx_save[:,:,i_save] = qx
        qz_save[:,:,i_save] = qz
        h_save[:,:,i_save]  = h
        t_save[i_save]      = n*dt
        i_save              = i_save + 1

    ###################################################
    # Call the scheme to move one step forward:
    if scheme == schemes['LWendr']:
        h_new, qxnew, qznew, \
        hzzz, hxxx, hzxx, hxzz = lw.lax_wendroff(surface_tension, 
                                                 scheme_choice,
                                                 dx, dz, dt,
                                                 h, qx, qz,
                                                 Epsilon, Re, nx, nz)
    elif scheme == schemes['LWFble']:
        h_new, qxnew, qznew, \
        hzzz, hxxx, hzxx, hxzz = bl.blended_lw_lf(surface_tension,
                                                  scheme_choice,
                                                  dx, dz, dt,
                                                  h, qx, qz,
                                                  Epsilon, Re, nx, nz)
    elif scheme == schemes['LFried']:
        h_new, qxnew, qznew, \
        hzzz, hxxx, hzxx, hxzz = lf.lax_friedrichs(surface_tension,
                                                   scheme_choice,
                                                   dx, dz, dt,
                                                   h, qx, qz,
                                                   Epsilon, Re, nx, nz)

    ###################################################
    # UPDATE THE FIELDS:
    h[1:-1,1:-1]  = h_new
    qx[1:-1,1:-1] = qxnew
    qz[1:-1,1:-1] = qznew
    if np.isnan(h).any():
        print("Reached NaNs. Stopping computation.")
        print("Logging summary...")
        info = '\n COMPUTATION REACHED NaNs.' + info
        break

    ###################################################
    # ENFORCE BOUNDARY CONDITIONS:
    # Note: indices for the four
    # boundaries are as follows:
    # Inlet (bottom): [0,:]
    # Outlet (top):   [-1,:]
    # Left:           [:,0]
    # Right:          [:,-1]

    # INTRODUCING PERTURBATIONS
    # AT THE INLET [0,:]

    # Miguel style:
    qx[0,:] = q_a*(1/3)*np.sin(2*np.pi*freq*n*dt) + (1/3)
    h[0,:] = (3*qx[0,:])**(1/3)
    #qx[0,:] = 1/3
    #h[0,:] = (3*qx[0,:])**(1/3)
    
    # Fabien Style
    #qx[0,:] = (1+q_a\
    #          *np.sin(2*np.pi\
    #          *freq*time_steps[n]))\
    #          *1/3

    # BOUNDARY CONDITIONS (LEFT, RIGHT, and TOP):                    
    # Linear extrapolation along z-azis
    # (LEFT and RIGHT BOUNDARIES)
    # and along the
    # TOP BOUNDARY (OUTLET):
    comp.linear_extrap_lrt(h, qx, qz)


    ###################################################
    # SAVE .dat AND .npy FILES every 100 steps:
    if n%100 < 0.0001:
        # Save the np solutions:
        save_data.save_np(h, #qx, 
                          directory_n, filename, n)
        # (most efficient format for post-processing)
        # pick up trash:
        gc.collect()

    # PRINT REMINDERS EVERY 10000 STEPS:
    if n%10000 < 0.0001:
        # Remind me what I've been doing
        # in the terminal:
        print("\n Reminder:" + "\n" + info)

        if surface_tension:
            # Check derivatives values:
            print('hxxx', hxxx)
            print('hzzz', hzzz)
            print('hxzz', hxzz)
            print('hzxx', hzxx)

        print('h.min() = ', h.min())
        print('h.max() = ', h.max())
        gc.collect()



#######################################################
# Go back to the solver directory:
os.chdir('../../')

# Summary of the simulation:
summary = info + '\n Execution time: ' \
        + str(datetime.now()-startTime) \
        + '\n h.max() = {:.4f}'.format(h.max()) \
        + '\n h.min() = {:.4f}'.format(h.min())

# Save the summary to a logfile:
logging.basicConfig(level    = logging.INFO,
                    filename = "zlogfile",
                    filemode = "a+",
                    format   = "%(asctime)-15s \
                                %(levelname)-8s \
                                %(message)s")
logging.info(summary)
print(summary)
print('Summary logged.')

# Clean up a bit:
gc.collect()
del h_new
del qxnew
del qznew
del h
del qx
del qz

print('Auf Wiedersehen.')
