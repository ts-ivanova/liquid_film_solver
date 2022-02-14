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
        from findiff import FinDiff
        #from Functions_Miguel import filt_X

        # First derivatives operators:
        d_dx = FinDiff((0, dx, 1))
        d_dz = FinDiff((1, dz, 1))

        H_X = h[1:-1,1:-1]
        H_Z = h[1:-1,1:-1]
        # Computations of the third derivatives
        # that are included in sources S2 and S3:

        # d3_dx3 h:
        hx = d_dx(H_X)
        hxx = d_dx(hx)
        hxxx = d_dx(hxx)

        hz = d_dz(H_Z)
        hzz = d_dz(hz)
        hzzz = d_dz(hzz)

        # d3_dxdz2 h:
        hxzz = d_dx(hzz)

        # d3_dzdx2 h:
        hzxx = d_dz(hxx)

    
        # sources S1 for the h-eqn:
        delta1 = 3*Epsilon*Re

        S1 = np.zeros((nx-2,nz-2))
    
        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(3*Epsilon*Re) \
                - (3*qx[1:-1,1:-1])/ \
                (delta1*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(delta1)) \
                *(hxzz + hxxx)
        # sources S3 for the qz-eqn:
        S3 = -3*qz[1:-1,1:-1]/ \
                (delta1*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(delta1)) \
                *(hzzz + hzxx)

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
                #(3*Epsilon*Re*h[1:-1,1:-1]**2)

        # sources S3 for the qz-eqn:
        S3 = -3*qz[1:-1,1:-1]/ \
                (delta1*h[1:-1,1:-1]**2)
                #(3*Epsilon*Re*h[1:-1,1:-1]**2)

        # set the third derivatives to zero as they have to be returned
        hzzz = 0
        hxxx = 0
        hzxx = 0
        hxzz = 0

    return S1, S2, S3, hzzz, hxxx, hzxx, hxzz