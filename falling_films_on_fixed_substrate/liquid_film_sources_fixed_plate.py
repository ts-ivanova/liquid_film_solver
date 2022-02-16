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
        from findiff import FinDiff
        from Functions_Miguel import filt_X
        from Functions_Miguel import smoothstep

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
        #hzzz0 = d_dz(hzz_Zf.T)
        #hzzz = filt_X(hzzz0,31,boundaries="extrap",s=0.2)

        # d3_dxdz2 h:
        hxzz0 = d_dx(hzz_Zf.T)
        hxzz = filt_X(hxzz0,31,boundaries="extrap",s=0.2)

        # # d3_dzdx2 h:
        #hxx_Zf = filt_X(hxx.T,31,boundaries="extrap",s=0.2)
        #hzxx0 = d_dz(hxx_Zf.T)
        #hzxx = filt_X(hzxx0,31,boundaries="extrap",s=0.2)

        # Mask:
        xn=np.arange(1,nx-1,1)
        zn=np.arange(1,nz-1,1)

        ## Here I construct the smoothed step in x
        X_L = smoothstep(xn, 200, 600,N=10)
        X_H = smoothstep(xn, 3000, 3400,N=10)
        STEP_X=X_L-X_H
        
        ## Here I construct the smoothed step in z
        Z_L = smoothstep(zn, 10, 50,N=10)
        Z_H = smoothstep(zn, 350, 390,N=10)
        STEP_Z=Z_L-Z_H
        
        MASK_2D=np.outer(STEP_X,STEP_Z)
        

        # # WITHOUT FILTERING:
        # # First derivatives operators:
        # d_dx = FinDiff((0, dx, 1))
        # d_dz = FinDiff((1, dz, 1))

        # H_X = h[1:-1,1:-1]
        # H_Z = h[1:-1,1:-1]
        # # Computations of the third derivatives
        # # that are included in sources S2 and S3:

        # # d3_dx3 h:
        # hx = d_dx(H_X)
        # hxx = d_dx(hx)
        # hxxx = d_dx(hxx)

        # # d3_dz3 h:
        # hz = d_dz(H_Z)
        # hzz = d_dz(hz)
        # hzzz = d_dz(hzz)

        # # d3_dxdz2 h:
        # hxzz = d_dx(hzz)

        # # d3_dzdx2 h:
        # hzxx = d_dz(hxx)

        # Here we are 2D:
        hzzz = 0
        hzxx = 0
        hxzz = 0

    
        # sources S1 for the h-eqn:
        delta1 = 3*Epsilon*Re

        S1 = np.zeros((nx-2,nz-2))
    
        # sources S2 for the qx-eqn:
        S2 = h[1:-1,1:-1]/(3*Epsilon*Re) \
                - (3*qx[1:-1,1:-1])/ \
                (delta1*h[1:-1,1:-1]**2) \
                + (h[1:-1,1:-1]/(delta1)) \
                *(hxxx)*MASK_2D #(hxxx+hxzz)
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
        hzzz = 0
        hxxx = 0
        hzxx = 0
        hxzz = 0

    return S1, S2, S3, hzzz, hxxx, hzxx, hxzz
