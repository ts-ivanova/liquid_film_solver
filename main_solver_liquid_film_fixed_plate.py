#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:27:43 2021

SOLVER FOR A LIQUID FILM ON A MOVING SUBSTRATE

@author: ivanova

This script solves a simplified version of a 3D integral model
for a liquid film on a moving substrate.
The unknowns of this integral model are
the liquid film height h,
the flow rate along the streamwise direction qx,
and the flow rate along the spanwise direction qz
(check RM2021 report).

The derived 3D integral model is dimensionless.

    \partial_t h  + \partial_x F11 + \partial_z F21 = S1
    \partial_t qx + \partial_x F12 + \partial_z F22 = S2
    \partial_t qz + \partial_x F13 + \partial_z F23 = S3

In a simplified version of the model, 
the following terms are neglected:
pressure gradients, interface shear stresses,
and the most challenging to implement numerically -
surface tension terms (the third derivatives of the height h).
"""

#######################################################################

# Import libraries:
import numpy as np
import math
import sys

# Import numerical scheme procedures:
import lax_wendroff_richtmyer as lw
import lax_friedrichs_fixed_plate as lf
import lax_wendroff_friedrichs_blended_fixed_plate as bl

# To monitor the execution time:
import os
import logging
from datetime import datetime

# Import home-made script for saving things in directories etc.
import tools_for_boundary_conditions as comp
import tools_for_saving as save_data
import tools_for_plotting as save_plots

# For freeing up some space while computing:
import gc

#######################################################################

# Possible liquid types:
liquids = {
            'WATER' : 0,
            'ZINC'  : 1
          }

# Possible configurations:
conf = {
        # Perturbations only on x (PX) of qx (OpenFOAM case):
        'PX01' : 0,
        # Perturbations on x and on z (PXZ):
        # - 3D perturbations of the flow rate qx:
        'PXZ1' : 1,
        # - 3D perturbations of the height h:
        'PXZ2' : 2
        }

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



#######################################################################
######                   MAKE THE CHOICES HERE                   ###### 
######               (by un-commenting lines etc.)               ######
#######################################################################

# Selection of the fluid type:
liquid_list = [
               liquids['WATER']#,
               #liquids['ZINC']
               ]

# Selection of the configuration:
configurations = [
                  conf['PX01']#, # 2D cases + OpenFOAM JFM case
                  # conf['PXZ1'] # 3D cases
                  # conf['PXZ2'] # 3D cases
                  ]

# Choice of the numerical scheme:
scheme = schemes['LWFble']
# scheme = schemes['LFried']
# scheme = schemes['LWendr']

# Specify whether to include surface tension terms
# in the sources (the third derivatives of the height h):
surface_tension = False
# (False = do not include)

# Dimensionless substrate velocity:
U_substr = 1
# Wave travel speed for CFL:
u = 2*U_substr
# Time between outputs:
output_interval = 1.0

# Initial film height:
h0 = 0.2 # [-]
# h0 = 0.1 # [-] can be suitable for higher Re numbers



#######################################################################
######              RUN A LOOP FOR ALL LIQUID TYPES,             ###### 
######      CONFIGURATIONS, REYNOLDS NUMBERS, FREQUENCIES...     ######
#######################################################################

for liquid in liquid_list:
    # LIQUID PARAMETERS:
    # if water has been selected:
    if liquid == liquids['WATER']:
        # Set WATER parameters from JFM 2020 paper:
        Epsilon = 0.23918 # Long-wave parameter, [-]
        #
        # To run for a lower value of delta:
        # Epsilon = 0.0023918
        #
        Re_list = [319] # Re number in OpenFOAM JFM
        # Re_list = [319, 2*319]
    # or if zinc has been selected:
    elif liquid == liquids['ZINC']:
        # Set liquid ZINC parameters from JFM 2020 paper:
        Epsilon = 0.15460 # Long-wave parameter, [-]
        # Re = 478 # Reynolds number
        Re_list = [478, 2*478]

    # The reduced Reynolds number is defined as:
    # delta = Epsilon*Re_list
    # (It is useful to keep track on its value
    # when investigating stability)

    # RUN FOR ALL REYNOLDS NUMBER VALUES IN Re_list:
    for Re in Re_list:
        # Run for all specified configurations
        # in the list:
        for configuration in configurations:
            # Configuration:
            conf_key = list(conf.keys())[configuration]
            # Liquid type:
            liquids_key = list(liquids.keys())[liquid]


            # freq_list = list(np.arange(0.005, 0.205, 0.015))
            # frequencies = [round(elem, 3) for elem in freq_list]
            # [-] low, medium and high freqs
            # frequencies = [0.05]
            frequencies = [0.02]


            # if the configuration is 2D:
            if configuration == conf['PX01']:
                dim = '2D_' # to use for namings in tools_for_saving.py

                surface_tension = False

                # Select the Lax-Friedrichs scheme 
                # since it is robust and stable for the 2D waves:
                scheme = schemes['LFried']

                # The CFL number is defined as u*dt/dx
                # from which dt is evaluated below:
                CFL = 0.3

                # OpenFOAM case in JFM:
                # frequencies = [0.05] # [-] as in 2D JFM

                # Set the amplitude for the flow rate perturbations
                # as in 2D JFM:
                A = 0.2 # [-]

            # else if the configuration is 3D:
            else: # parameters for the 3D waves:
                dim = '3D_' # to use for namings in tools_for_saving.py

                # CFL number is defined as u*dt/dx
                # from which dt is evaluated below:
                if surface_tension:
                    CFL = 0.1
                else:
                    CFL = 0.3

                # frequencies = [0.05]

                if configuration == conf['PXZ1']:
                    A = 0.2 # amplitude for the flow rate perturbations
                elif configuration == conf['PXZ2']:
                    A = 0.07 # amplitude for the height perturbations

            # If the selected scheme is the Blended scheme, 
            # then specify more precisely which flux limiters
            # are used (by un-commenting a line):
            if scheme == schemes['LWFble']:
                # scheme_choice = 'LxFr'
                # scheme_choice = 'LxWe'
                scheme_choice = 'BLmi'
                # ... more options can be added 
                # for the other limiter types, if they are used.
            else:
                # extract the string for the naming conventions:
                scheme_choice = \
                    list(schemes.keys())[scheme][:4]

            # RUN FOR ALL FREQUENCIES:
            for freq in frequencies:

                # mark the beginning of the computations:
                startTime = datetime.now()

                # SPACE INFORMATION
                # dx - cell size along x-axis, [-]
                # nx - number of cells along x-axis
                # dz - cell size along z-axis, [-]
                # nz - number of cells along z-axis
                # L - domain length along x, [-]
                # lambd - dimensionless wavelength [-] along x
                # npoin - number of points per lambd
                # final_time - total simulation length, [-]
                lambd    = U_substr/freq
                factor   = 1
                freq_JFM = 0.05

                # Depending on whether the surface tension is
                # taken into account or not, the number of points
                # per wavelength are specified below:

                # Currently (end of RM2021),
                # the presence of surface tension terms
                # causes issues when the cell size is too small,
                # therefore for now a working option is of the order of
                # 30-40 points per wavelength:
                if surface_tension:
                    npoin = int((U_substr/freq_JFM)/(0.0275*2*10))

                # Otherwise without surface tension,
                # the minimum npoin is around 363 for which
                # the numerical dissipation is negligible,
                # as confirmed with the validation test case:
                else:
                    npoin = int((U_substr/freq_JFM)/(0.0275*2))

                dx    = (lambd/(npoin))*factor
                L     = 8*lambd
                nx    = int(L/dx)
                final_time = int((L+lambd)/U_substr)

                #######################################################
                # For the OpenFOAM case in JFM:
                # (for the validation)
                if configuration == conf['PX01']:
                    # nx = 2810 is extracted from JFM data
                    # by using Lx = nx*dx, where Lx = 77.28
                    # and dx = 0.0275 from the JFM paper 2020.
                    npoin = int(lambd/0.0275) # is approx. 727
                    dx = (lambd/(npoin))*factor
                    L  = 4*lambd
                    nx = int(2810/factor)
                    # z-dimension:
                    dz = 1e-2/10
                    nz = 100
                #######################################################

                # wavelength lambd_z [-] along z
                lambd_z = 1
                dz      = lambd_z/100
                nz      = int(lambd_z/dz)*3

                #######################################################
                # TIME INFORMATION
                # Time is dimensionless;
                # Timestep (unit) from the CFL formula:
                dt = CFL*dx/U_substr

                # Number of timesteps:
                nt = int(np.fix(final_time/dt)+1)
                time_steps = np.linspace(0,final_time,nt)
                tsteps_btwn_out = np.fix(output_interval/dt)
                noutput = int(np.ceil(nt/tsteps_btwn_out))
                # noutput is the number of output frames.

                # Initialize h, qx, qz:
                h = h0*np.ones((nx,nz), dtype='float32')
                # Initial values for the flow rates
                # Quasi-steady state:
                qx = (1/3)*h**3-h
                qz = np.zeros((nx,nz), dtype='float32')

                #######################################################
                # Spatial dimensions:
                x = np.mgrid[0:nx]*dx
                z = np.mgrid[0:nz]*dz
                # Create matrices of the coordinate variables
                [Z,X] = np.meshgrid(z,x)

                #######################################################
                # Create directories for storing data:
                results_dir, \
                directory_n, \
                directory_plots, \
                filename, \
                directory_lim = \
                    save_data.create_directories(surface_tension,
                                                 liquids_key,
                                                 conf_key,
                                                 scheme_choice,
                                                 dim,
                                                 h0, A,
                                                 dx, nx, dz, nz,
                                                 CFL, dt, final_time,
                                                 Epsilon, Re, freq)

                #%%
                #######################################################
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

                #######################################################
                # Information about the running case:
                info =  '\n Results directory: ' \
                        + results_dir \
                        + '\n Liquid type: ' \
                        + liquids_key \
                        + '\n surface_tension: ' \
                        + str(surface_tension) \
                        + '\n Computing configuration: ' \
                        + '\n ' + directory_n \
                        + '\n ' + filename \
                        + '\n lambd: {:.2f}'.format(lambd) \
                        + '\n number of points per lambd: ' \
                        + str(npoin) \
                        + '\n domain length L: {:.2f}'.format(L)
                print(info)

                #######################################################
                # MAIN LOOP
                for n in range(0, nt):
                    # Save fields for every fixed
                    # number of timesteps
                    if np.mod(n, tsteps_btwn_out) == 0:
                        print("T = {:.2f} (max {:.0f}), [-]"
                              .format((n)*dt, final_time))
                        qx_save[:,:,i_save] = qx
                        qz_save[:,:,i_save] = qz
                        h_save[:,:,i_save]  = h
                        t_save[i_save]      = n*dt
                        i_save              = i_save + 1

                    ###################################################
                    # Call the scheme to move one step forward:
                    if scheme == schemes['LWendr']:
                        h_new, qxnew, qznew = \
                            lw.lax_wendroff(surface_tension,
                                            scheme_choice,
                                            dx, dz, dt,
                                            h, qx, qz,
                                            Epsilon, Re,
                                            nx, nz)
                    elif scheme == schemes['LWFble']:
                        h_new, qxnew, qznew, \
                        Phi_x, Phi_z, \
                        hzzz, hxxx, hzxx, hxzz = \
                            bl.blended_lw_lf(surface_tension,
                                             scheme_choice,
                                             dx, dz, dt,
                                             h, qx, qz,
                                             Epsilon, Re,
                                             nx, nz)
                    elif scheme == schemes['LFried']:
                        h_new, qxnew, qznew = \
                            lf.lax_friedrichs(surface_tension,
                                              scheme_choice,
                                              dx, dz, dt,
                                              h, qx, qz,
                                              Epsilon, Re,
                                              nx, nz)

                    ###################################################
                    # UPDATE THE FIELDS:
                    h[1:-1,1:-1]  = h_new
                    qx[1:-1,1:-1] = qxnew
                    qz[1:-1,1:-1] = qznew

                    ###################################################
                    # ENFORCE BOUNDARY CONDITIONS:
                    # Note: indices for the four
                    # boundaries are as follows:
                    # Inlet (bottom): [-1,:]
                    # Outlet (top):   [0,:]
                    # Left:           [:,0]
                    # Right:          [:,-1]

                    # INTRODUCING PERTURBATIONS
                    # AT THE INLET [-1,:]

                    # 2D perturbation along x of qx:
                    if configuration == conf['PX01']:
                        # BOTTOM BOUNDARY (INLET):
                        # BCs at inlet: Dirichlet conditions
                        h[-1,:] = h0*np.ones(nz)
                        # From quasi-steady formula,
                        # compute qx with introduced
                        # perturbations as in JFM:
                        qx[-1,:] = (1/3*h[-1,:]**3 \
                                    - h[-1,:])\
                                    *(1 + \
                                     # A*np.sin(2*np.pi\
                                     #           *freq\
                                     #           *time_steps[n])\
                                    A*np.sin(2*np.pi\
                                              *time_steps[n]/\
                                              lambd)\
                                    *np.ones((nz,)))/100
                        # Set qz to zeros at the inlet:
                        qz[-1,:] = np.zeros(nz)

                    # 3D perturbations of qx along x and z:
                    elif configuration == conf['PXZ1']:
                        # BOTTOM BOUNDARY (INLET):
                        # BCs at inlet: Dirichlet conditions
                        h[-1,:] = h0*np.ones(nz)
                        # From the quasi-steady formula,
                        # compute qx with introduced perturbations:
                        qx[-1,:] = (1/3*h[-1,:]**3 \
                                    - h[-1,:])\
                                    *(1 + \
                                      A*np.sin(2*np.pi\
                                                *freq\
                                                *time_steps[n])\
                                      *np.sin((2*np.pi\
                                                   /lambd_z)\
                                                  *z)\
                                      *np.ones((nz,))\
                                      *np.exp(-(z-z.mean())**2\
                                              /(2*(0.4)**2))\
                                      /(0.4*np.sqrt(2*math.pi))
                                      )
                        # Note: the perturbations are sinusoidal
                        # along x and along z, 
                        # and the Gaussian function regulates
                        # the width of the domain which is affected
                        # by the perturbations.

                        # Set qz to zeros at the inlet:
                        qz[-1,:] = np.zeros(nz)

                    # 3D perturbations of h along x and z:
                    elif configuration == conf['PXZ2']:
                        # BCs at inlet: Dirichlet conditions
                        # BOTTOM BOUNDARY (INLET):
                        h[-1,:] = h0*np.ones(nz) + \
                                A*np.sin(2*np.pi*freq\
                                            *time_steps[n])\
                                *np.sin((2*np.pi/lambd_z)*z)\
                                # *np.exp(-(z-z.mean())**2\
                                #         /(2*(0.4)**2))\
                                # /(0.4*np.sqrt(2*math.pi))
                        # From the quasi-steady formula,
                        # compute qx:
                        qx[-1,:] = 1/3*h[-1,:]**3 - h[-1,:]
                        # Set qz to zeros at the inlet:
                        qz[-1,:] = np.zeros(nz)


                    else:
                        print('Unknown configuration %f'
                              % configuration)
                        sys.exit()

                    # BOUNDARY CONDITIONS (LEFT, RIGHT, and TOP):                    
                    # Linear extrapolation along z-azis
                    # (LEFT and RIGHT BOUNDARIES)
                    # and along the
                    # TOP BOUNDARY (OUTLET):
                    comp.linear_extrap_lrt(h, qx, qz)


                    ###################################################
                    # SAVE .dat AND .npy FILES every 100 steps:
                    if n%100 < 0.0001:
                        # # Save to .dat files a slice
                        # # of the wave along x:
                        # save_data.save_to_dat(h, qx, qz,
                        #                       nx, nz,
                        #                       directory_n,
                        #                       filename,
                        #                       n)
                        # # Save the whole height
                        # # as a matrix:
                        # save_data.save_matrix(h, directory_n,
                        #                       filename,
                        #                       n)

                        # Save the np solutions:
                        save_data.save_np(h, directory_n,
                                          filename,
                                          n)
                        # (most efficient format for post-processing)

                        # pick up trash:
                        gc.collect()

                    # SAVE PLOTS AND PRINT REMINDERS EVERY 2000 STEPS:
                    if n%2000 < 0.0001:
                        # Save .png's:
                        save_plots.plot_surfaces(h, X, Z, n,
                                                 h0,
                                                 directory_plots,
                                                 filename,
                                                 conf_key)

                        # Remind me what I've been doing
                        # in the terminal:
                        print("\n Reminder:" + "\n" + info)

                        # Monitor the behaviour of the limiters
                        # for the blended scheme (only for 3D):
                        if scheme == schemes['LWFble'] \
                        and configuration != conf['PX01']:
                            # Option to plot the limiters:
                            # save_plots.plot_limiters(nx, dx, nz, dz,
                            #                    Phi_x, Phi_z,
                            #                    directory_lim,
                            #                    scheme_choice,
                            #                    n)
                            # Check limiters values:
                            print('Phi_x' , Phi_x)
                            print('Phi_z' , Phi_z)
                            # Check derivatives values:
                            # print('hxxx', hxxx)
                            # print('hzzz', hzzz)
                            # print('hxzz', hxzz)
                            # print('hzxx', hzxx)
                        gc.collect()

                #%%
                #######################################################
                # Go back to the solver directory:
                os.chdir('../../')

                # Summary of the simulation:
                summary = info \
                        + '\n Execution time: ' \
                        + str(datetime.now()-startTime) \
                        + '\n h.max() = {:.4f}'.format(h.max()) \
                        + '\n h.min() = {:.4f}'.format(h.min())

                # Save the summary to a logfile:
                logging.basicConfig(level    = logging.INFO,
                                    filename = "zlogfile",
                                    filemode = "a+",
                                    format   = \
                                    "%(asctime)-15s \
                                    %(levelname)-8s \
                                    %(message)s")
                logging.info(summary)
                print(summary)
                print('Summary logged.')

                # Clean up a bit before the next case in the loop:
                gc.collect()
                del h_new
                del qxnew
                del qznew
                del h
                del qx
                del qz
                if configuration != conf['PX01'] \
                and scheme == schemes['LWFble']:
                    del Phi_x
                    del Phi_z

print('Auf Wiedersehen.')
