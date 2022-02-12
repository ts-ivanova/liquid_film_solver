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
    Compute the sources for the 3D liquid film integral model.

    Currently solving simplified equations:
     - neglected pressure gradient because it affects
    a narrow region of the domain;
     - neglected interface shear stress;
     - problems with implementing the third derivatives.

    If the variable surface_tension is set to True in the main loop
    script, then the terms in S2 are taken into account.
    If it is set to False, then
    the surface tension terms are neglected.

    Important:
    The filtering of the third derivatives performs well
    (checked with the postprocessing scripts for plotting).
    Therefore the problems with the third derivatives are possibly
    related to the the blended scheme which has to be checked
    as there are issues with the sources in S3 (the solution explodes).
    Most of the implemented types of limiters have been tested.
    Nevertheless, when running with included sources in S2,
    the solver performs well with about 30-40 points per wavelength
    which yields ~10 times larger cell sizes
    than the ones without surface tension.
    This results in significant numerical dissipation.
    '''
    
    # when surface_tension = True, compute the source terms
    # including the third derivatives:
    if surface_tension:
        from findiff import FinDiff
        from Functions_Miguel import filt_X

        # First derivatives operators:
        d_dx = FinDiff((0, dx, 1))
        d_dz = FinDiff((1, dz, 1))

        # Filter the height along x and along z:
        H_Xf = filt_X(h[1:-1,1:-1],31,boundaries="extrap",s=0.2)
        H_Zf = filt_X(h[1:-1,1:-1].T,31,boundaries="extrap",s=0.2)
        H_Zf = H_Zf.T

        # Computations of the third derivatives
        # that are included in sources S2 and S3:

        # d3_dx3 h:
        # take the first derivative:
        hx = d_dx(H_Xf)
        # filter it along x:
        hx_Xf = filt_X(hx,31,boundaries="extrap",s=0.2)
        # take the second derivative:
        hxx = d_dx(hx_Xf)
        # filter again:
        hxx_Xf = filt_X(hxx,31,boundaries="extrap",s=0.2)
        # take the third derivative:
        hxxx0 = d_dx(hxx_Xf)
        # and filter it:
        hxxx = filt_X(hxxx0,31,boundaries="extrap",s=0.2)

        # d3_dz3 h:
        hz = d_dz(H_Zf)
        hz_Zf = filt_X(hz.T,31,boundaries="extrap",s=0.2)
        hzz = d_dz(hz_Zf.T)
        hzz_Zf = filt_X(hzz.T,31,boundaries="extrap",s=0.2)
        # hzzz0 = d_dz(hzz_Zf.T)
        # hzzz = filt_X(hzzz0,31,boundaries="extrap",s=0.2)

        # d3_dxdz2 h:
        hxzz0 = d_dx(hzz_Zf.T)
        hxzz = filt_X(hxzz0,31,boundaries="extrap",s=0.2)

        # d3_dzdx2 h:
        # hxx_Zf = filt_X(hxx.T,31,boundaries="extrap",s=0.2)
        # hzxx0 = d_dz(hxx_Zf.T)
        # hzxx = filt_X(hzxx0,31,boundaries="extrap",s=0.2)

        ########################
        # sources S1 for the h-eqn:
        S1 = np.zeros((nx-2,nz-2))

        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(Epsilon*Re) \
                - (6*h[1:-1,1:-1] + 6*qx[1:-1,1:-1])/ \
                (2*Epsilon*Re*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(Epsilon*Re)) \
                *(hxzz + hxxx)

        # sources S3 for the qz-eqn:
        S3 = -6*qz[1:-1,1:-1]/ \
                (2*Epsilon*Re*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(Epsilon*Re)) \
                *(hzzz + hzxx)

        hzzz = 0
        hzxx = 0

    # when surface_tension = False, compute the source terms
    # without the third derivatives:
    else:
        # sources S1 for the h-eqn:
        S1 = np.zeros((nx-2,nz-2))

        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(Epsilon*Re) \
                - (6*h[1:-1,1:-1] + 6*qx[1:-1,1:-1])/ \
                (2*Epsilon*Re*h[1:-1,1:-1]**2)

        # sources S3 for the qz-eqn:
        S3 = -6*qz[1:-1,1:-1]/ \
                (2*Epsilon*Re*h[1:-1,1:-1]**2)

        # set the third derivatives to zero as they have to be returned
        hzzz = 0
        hxxx = 0
        hzxx = 0
        hxzz = 0

    return S1, S2, S3, hzzz, hxxx, hzxx, hxzz
