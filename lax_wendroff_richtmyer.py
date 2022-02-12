#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tsvetelina ivanova

Two-step two dimensional Richtmyer-Lax-Wendroff scheme.
"""

import liquid_film_fluxes as fluxes
import liquid_film_sources as sources

def lax_wendroff(surface_tension,
                 scheme_choice,
                 dx, dz, dt,
                 h, qx, qz,
                 Epsilon, Re,
                 nx, nz):
    '''
    This function performs the Richtmyer-Lax-Wendroff scheme
    on a dimensionless integral model for a liquid film
    on a moving substrate.
    Unknowns: height: h,
              flow rate along x: qx,
              flow rate along z: qz.
    '''

    # Compute the fluxes:
    F11, F21, F12, F22, F13, F23 = \
        fluxes.liquid_film_fluxes(h, qx, qz)
    # Compute the sources:
    S1, S2, S3, \
    hzzz, hxxx, hzxx, hxzz \
        = sources.liquid_film_sources(surface_tension,
                                      dx, dz,
                                      nx, nz,
                                      h, qx, qz,
                                      Epsilon, Re)


    # Compute mid-point values in time and space for all quantities:
    h_mid_xt = 0.5*(h[1:,:] + h[0:-1,:]) \
        -(0.5*dt/dx)*(qx[1:,:] - qx[0:-1,:])
    h_mid_zt = 0.5*(h[:,1:] + h[:,0:-1]) \
        -(0.5*dt/dz)*(qz[:,1:] - qz[:,0:-1])

    qx_mid_xt = 0.5*(qx[1:,:] + qx[0:-1,:]) \
        -(0.5*dt/dx)*(F12[1:,:] - F12[0:-1,:])
    qx_mid_zt = 0.5*(qx[:,1:] + qx[:,0:-1]) \
        -(0.5*dt/dz)*(F22[:,1:] - F22[:,0:-1])

    qz_mid_xt = 0.5*(qz[1:,:] + qz[0:-1,:]) \
        -(0.5*dt/dx)*(F13[1:,:] - F13[0:-1,:])
    qz_mid_zt = 0.5*(qz[:,1:] + qz[:,0:-1]) \
        -(0.5*dt/dz)*(F23[:,1:] - F23[:,0:-1])


    # Compute mid-point fluxes along x and along z:
    F11_mid_xt, F21_mid_xt, F12_mid_xt, \
    F22_mid_xt, F13_mid_xt, F23_mid_xt = \
    fluxes.liquid_film_fluxes(h_mid_xt, qx_mid_xt, qz_mid_xt)

    F11_mid_zt, F21_mid_zt, F12_mid_zt, \
    F22_mid_zt, F13_mid_zt, F23_mid_zt = \
    fluxes.liquid_film_fluxes(h_mid_zt, qx_mid_zt, qz_mid_zt)


    # Now use the mid-point values to predict the values
    # at the next timestep:
    h_new = h[1:-1,1:-1] \
            - (dt/dx)*(F11_mid_xt[1:,1:-1] -
                       F11_mid_xt[0:-1,1:-1]) \
            - (dt/dz)*(F21_mid_zt[1:-1,1:] -
                       F21_mid_zt[1:-1,0:-1]) \
            + dt*S1

    qx_new = qx[1:-1,1:-1] \
            - (dt/dx)*(F12_mid_xt[1:,1:-1] -
                       F12_mid_xt[0:-1,1:-1]) \
            - (dt/dz)*(F22_mid_zt[1:-1,1:] -
                       F22_mid_zt[1:-1,0:-1]) \
            + dt*S2

    qz_new = qz[1:-1,1:-1] \
            - (dt/dx)*(F13_mid_xt[1:,1:-1] -
                       F13_mid_xt[0:-1,1:-1]) \
            - (dt/dz)*(F23_mid_zt[1:-1,1:] -
                       F23_mid_zt[1:-1,0:-1]) \
            + dt*S3


    return h_new, qx_new, qz_new
