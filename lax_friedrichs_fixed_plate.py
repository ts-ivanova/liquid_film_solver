#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:21:59 2021

"""

import liquid_film_fluxes_fixed_plate as fluxes
import liquid_film_sources_fixed_plate as sources

def lax_friedrichs(surface_tension,
                  scheme_choice,
                  dx, dz, dt,
                  h, qx, qz,
                  Epsilon, Re,
                  nx, nz):
    '''
    This function performs the Lax-Friedrichs scheme
    on the dimensionless integral model for a liquid film
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

    # Predict the values h_new, qx_new, qz_new
    # at the next timestep:
    h_new = (1/4)*(h[:-2,1:-1] + h[2:,1:-1] \
                   + h[1:-1,:-2] + h[1:-1,2:]) \
            - 0.5*(dt/dx)*(F11[2:,1:-1] - F11[:-2,1:-1]) \
            - 0.5*(dt/dz)*(F21[1:-1,2:] - F21[1:-1,:-2]) \
            + dt*S1

    qx_new = (1/4)*(qx[:-2,1:-1] + qx[2:,1:-1] \
                    + qx[1:-1,:-2] + qx[1:-1,2:]) \
            - 0.5*(dt/dx)*(F12[2:,1:-1] - F12[:-2,1:-1]) \
            - 0.5*(dt/dz)*(F22[1:-1,2:] - F22[1:-1,:-2]) \
            + dt*S2

    qz_new = (1/4)*(qz[:-2,1:-1] + qz[2:,1:-1] \
                    + qz[1:-1,:-2] + qz[1:-1,2:]) \
            - 0.5*(dt/dx)*(F13[2:,1:-1] - F13[:-2,1:-1]) \
            - 0.5*(dt/dz)*(F23[1:-1,2:] - F23[1:-1,:-2]) \
            + dt*S3

    return h_new, qx_new, qz_new
