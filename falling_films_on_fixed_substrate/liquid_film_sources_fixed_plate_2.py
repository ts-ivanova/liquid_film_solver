#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:56:58 2021

@author: tsvetelina ivanova
"""

import numpy as np

def liquid_film_sources(surface_tension,
                        dx, dz, nx, nz,
                        h, qx, qz,
                        Epsilon, Re):
    '''
    Compute the sources for the falling liquid film model.
    '''
    
    # when surface_tension = True, compute the source terms
    # including the third derivatives:
    if surface_tension:
        # WITH FILTERING: 
        # from findiff import FinDiff
        # from Functions_Miguel import filt_X
        # from Functions_Miguel import smoothstep
        from Functions_Miguel import Surf_T

        # Recreate Xg Zg
        # Create the Analytic function for validation purposes
        # n_x=1000; n_z=500 # Number of points along x and y.
        # Define the 1D grids
        z=np.linspace(0,(nx)*dx,nx); x=np.linspace(0,(nz)*dz,nz)
        # Define the mesh grid
        Xg,Zg=np.meshgrid(x,z)        

        h_xxx,h_xzz, h_zxx,h_zzz=Surf_T(h,Xg,Zg,P=500,Frac=2)



        # Here we are 2D:
        # hzzz = 0
        # hzxx = 0
        # hxzz = 0

    
        # sources S1 for the h-eqn:
        delta1 = 3*Epsilon*Re

        S1 = np.zeros((nx-2,nz-2))
    
        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(3*Epsilon*Re) \
                - (3*qx[1:-1,1:-1])/ \
                (delta1*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(delta1)) \
                *(h_xxx[1:-1,1:-1])#+h_xzz[1:-1,1:-1])
        # sources S3 for the qz-eqn:
        S3 = np.zeros((nx-2,nz-2))
               #  -3*qz[1:-1,1:-1]/ \
               #  (delta1*h[1:-1,1:-1]**2) \
               #  + (h[1:-1,1:-1]/(delta1)) \
               #  *(hzzz + hzxx)*M[1:-1,1:-1]


    # when surface_tension = False, compute the source terms
    # without the third derivatives:
    else:
        # sources S1 for the h-eqn:
        delta1 = 3*Epsilon*Re

        S1 = np.zeros((nx-2,nz-2))

        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(3*Epsilon*Re) \
                - (3*qx[1:-1,1:-1])/ \
                (delta1*h[1:-1,1:-1]**2)

        # sources S3 for the qz-eqn:
        S3 = -3*qz[1:-1,1:-1]/ \
                (delta1*h[1:-1,1:-1]**2)

        # set the third derivatives to zero as they have to be returned
        h_zzz = 0
        h_xxx = 0
        h_zxx = 0
        h_xzz = 0

    return S1, S2, S3, h_zzz, h_xxx, h_zxx, h_xzz
