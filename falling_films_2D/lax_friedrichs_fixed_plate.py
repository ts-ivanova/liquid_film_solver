#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:21:59 2021

"""

import liquid_film_fluxes_fixed_plate as fluxes
import liquid_film_sources_fixed_plate_2D as sources

def lax_friedrichs(surface_tension,
                  scheme_choice,
                  dx, dt,
                  h, qx,
                  Epsilon, Re,
                  nx):
    '''
    This function performs the Lax-Friedrichs scheme
    on the dimensionless integral model 
    for a falling liquid film on a fixed substrate.
    
    Unknowns: height: h,
              flow rate along x: qx,
              flow rate along z: qz.
    '''

    # Compute the fluxes:
    F11, F12 = fluxes.liquid_film_fluxes(h, qx)

    # Compute the sources:
    S1, S2, hxxx = sources.liquid_film_sources(surface_tension,
                                               dx, nx,
                                               h, qx,
                                               Epsilon, Re)

    # Predict the values h_new, qx_new, qz_new
    # at the next timestep:
    h_new = (1/2)*(h[:-2] + h[2:]) \
            - 0.5*(dt/dx)*(F11[2:] - F11[:-2]) \
            + dt*S1

    qx_new = (1/2)*(qx[:-2] + qx[2:]) \
            - 0.5*(dt/dx)*(F12[2:] - F12[:-2]) \
            + dt*S2


    # return the computed quantities
    # as well as the limiters to monitor/plot their behaviour
    return h_new, qx_new, hxxx
