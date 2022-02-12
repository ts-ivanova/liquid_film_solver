#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:56:58 2021

@author: tsvetelina ivanova

A script that provides different limiter types for
the blended scheme (Lax-Wendroff -- Lax-Friedrichs).
"""

import numpy as np

def flux_limiters(scheme_choice, h, qx, qz):
    '''
    Flux limiters for the blending of the schemes
    for the Liquid film solver
    '''
    # FLUX LIMITERS OPTIONS:
    if scheme_choice == 'LxFr':
        # If all Phi's are zeros,
        # the Lax-Friedrichs scheme is restored:
        Phi_x = np.zeros(np.array(h[1:-1,:]).shape)
        Phi_z = np.zeros(np.array(h[:,1:-1]).shape)

    elif scheme_choice == 'LxWe':
        # If all Phi's are ones,
        # the Lax-Wendroff scheme is restored:
        Phi_x = np.ones(np.array(h[1:-1,:]).shape)
        Phi_z = np.ones(np.array(h[:,1:-1]).shape)

    else:
        # For the other limiter types,
        # compute first the consecutive gradients:

        rx = (h[1:-1,:] - h[:-2,:])/(h[2:,:] - h[1:-1,:])
        rz = (h[:,1:-1] - h[:,:-2])/(h[:,2:] - h[:,1:-1])

        rx = np.nan_to_num(rx)
        rz = np.nan_to_num(rz)

        rx[rx==np.inf] = 0
        rz[rz==np.inf] = 0

        rx[rx<0] = 0
        rz[rz<0] = 0

        rx = np.nan_to_num(rx)
        rz = np.nan_to_num(rz)

        if scheme_choice == 'BLmi':
            # MINMODE, also used in 2D BLEW
            Phi_x = np.array(np.maximum(0,
                                        np.minimum(1,np.array(rx))))
            Phi_z = np.array(np.maximum(0,
                                        np.minimum(1,np.array(rz))))

        if scheme_choice == 'BLal':
            # van Albada 1, also used in 2D BLEW
            Phi_x = (np.array(rx)**2 + np.array(rx))/\
                    (np.array(rx)**2 + 1)
            Phi_z = (np.array(rz)**2 + np.array(rz))/\
                    (np.array(rz)**2 + 1)

        if scheme_choice == 'BLa2':
            # van Albada 2
            Phi_x = (2*np.array(rx))/(np.array(rx)**2 + 1)
            Phi_z = (2*np.array(rz))/(np.array(rz)**2 + 1)

        if scheme_choice == 'BLhc':
            # HCUS
            Phi_x = 1.5*(np.array(rx) \
                         + abs(np.array(rx)))/(np.array(rx) + 2)
            Phi_z = 1.5*(np.array(rz) \
                         + abs(np.array(rz)))/(np.array(rz) + 2)

        if scheme_choice == 'BLhq':
            # HQUICK
            Phi_x = 2*(np.array(rx) + abs(np.array(rx)))/\
                    (np.array(rx) + 3)
            Phi_z = 2*(np.array(rz) + abs(np.array(rz)))/\
                    (np.array(rz) + 3)

        if scheme_choice == 'BLko':
            # Koren
            Phi_x = max(0, min(2*np.array(rx),
                               min((1+2*np.array(rx))/3,2)))
            Phi_z = max(0, min(2*np.array(rz),
                               min((1+2*np.array(rz))/3,2)))

        if scheme_choice == 'BLvl':
            # van Leer
            Phi_x = (np.array(rx) + abs(np.array(rx)))/\
                    (1 + abs(np.array(rx)))
            Phi_z = (np.array(rz) + abs(np.array(rz)))/\
                    (1 + abs(np.array(rz)))

        Phi_x = np.nan_to_num(Phi_x)
        Phi_x[Phi_x==np.inf] = 0
        Phi_z = np.nan_to_num(Phi_z)
        Phi_z[Phi_z==np.inf] = 0

        Phi_x[Phi_x<0] = 0
        Phi_x[Phi_x>1] = 1

        Phi_z[Phi_z<0] = 0
        Phi_z[Phi_z>1] = 1


    return Phi_x, Phi_z
