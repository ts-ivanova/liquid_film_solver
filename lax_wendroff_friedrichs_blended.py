#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivanova

Blended Lax-Wendroff & Lax-Friedrichs scheme
for solving the integral model for liquid films.

RM2021
"""

import liquid_film_fluxes as fluxes
import liquid_film_sources as sources
import liquid_film_flux_limiters as limiters

def blended_lw_lf(surface_tension,
                  scheme_choice,
                  dx, dz, dt,
                  h, qx, qz,
                  Epsilon, Re,
                  nx, nz):
    '''
    This function performs the blended
    Lax-Wendroff (high-order) + Lax-Friedrichs (low-order) scheme
    on a dimensionless integral model
    for a liquid film on a moving substrate.
    Unknown quantities: height: h,
                        flow rate along x: qx,
                        flow rate along z: qz.
    Equations:
        \partial_t h  + \partial_x F11 + \partial_z F21 = S1
        \partial_t qx + \partial_x F12 + \partial_z F22 = S2
        \partial_t qz + \partial_x F13 + \partial_z F23 = S3
    '''

    # Compute the fluxes for the three eqns:
    F11, F21, F12, \
    F22, F13, F23 \
    = fluxes.liquid_film_fluxes(h, qx, qz)

    # # Compute the sources for the three eqns:
    S1, S2, S3, \
    hzzz, hxxx, hzxx, hxzz \
        = sources.liquid_film_sources(surface_tension,
                                      dx, dz,
                                      nx, nz,
                                      h, qx, qz,
                                      Epsilon, Re)

    # Compute mid-point values in time and space
    # for all quantities:
    h_mid_xt = 0.5*(h[1:,:] + h[:-1,:]) \
                - (0.5*dt/dx)*(F11[1:,:] - F11[:-1,:])
    h_mid_zt = 0.5*(h[:,1:] + h[:,:-1]) \
                - (0.5*dt/dz)*(F21[:,1:] - F21[:,:-1])

    qx_mid_xt = 0.5*(qx[1:,:] + qx[:-1,:]) \
                - (0.5*dt/dx)*(F12[1:,:] - F12[:-1,:])
    qx_mid_zt = 0.5*(qx[:,1:] + qx[:,:-1]) \
                - (0.5*dt/dz)*(F22[:,1:] - F22[:,:-1])

    qz_mid_xt = 0.5*(qz[1:,:] + qz[:-1,:]) \
                - (0.5*dt/dx)*(F13[1:,:] - F13[:-1,:])
    qz_mid_zt = 0.5*(qz[:,1:] + qz[:,:-1]) \
                - (0.5*dt/dz)*(F23[:,1:] - F23[:,:-1])


    # Compute mid-point fluxes along x and along z:

    # (Lax-Wendroff high-order scheme, half-steps along x)
    F11_high_mid_xt, F21_high_mid_xt, F12_high_mid_xt, \
    F22_high_mid_xt, F13_high_mid_xt, F23_high_mid_xt = \
    fluxes.liquid_film_fluxes(h_mid_xt, qx_mid_xt, qz_mid_xt)

    # (Lax-Wendroff high-order scheme, half-steps along z)
    F11_high_mid_zt, F21_high_mid_zt, F12_high_mid_zt, \
    F22_high_mid_zt, F13_high_mid_zt, F23_high_mid_zt = \
    fluxes.liquid_film_fluxes(h_mid_zt, qx_mid_zt, qz_mid_zt)

    # (Lax-Friedrichs low-order scheme, half-steps along x)
    F11_low_mid_xt = F11[1:,:] + (dt/dx)*(h_mid_xt - h[1:,:])
    # F21_low_mid_xt = F21[1:,:] + (dt/dx)*(h_mid_xt - h[1:,:])

    F12_low_mid_xt = F12[1:,:] + (dt/dx)*(qx_mid_xt - qx[1:,:])
    # F22_low_mid_xt = F22[1:,:] + (dt/dx)*(qx_mid_xt - qx[1:,:])

    F13_low_mid_xt = F13[1:,:] + (dt/dx)*(qz_mid_xt - qz[1:,:])
    # F23_low_mid_xt = F23[1:,:] + (dt/dx)*(qz_mid_xt - qz[1:,:])

    # (Lax-Friedrichs low-order scheme, half-steps along z)
    # F11_low_mid_zt = F11[:,1:] + (dt/dx)*(h_mid_zt - h[:,1:])
    F21_low_mid_zt = F21[:,1:] + (dt/dx)*(h_mid_zt - h[:,1:])

    # F12_low_mid_zt = F12[:,1:] + (dt/dx)*(qx_mid_zt - qx[:,1:])
    F22_low_mid_zt = F22[:,1:] + (dt/dx)*(qx_mid_zt - qx[:,1:])

    # F13_low_mid_zt = F13[:,1:] + (dt/dx)*(qx_mid_zt - qx[:,1:])
    F23_low_mid_zt = F23[:,1:] + (dt/dx)*(qz_mid_zt - qz[:,1:])


    # Flux limiter functions:
    Phi_x, Phi_z = limiters.flux_limiters(scheme_choice,
                                          h, qx, qz)


    # Fluxes used for the computations of the new steps:

    # along x:
    F11_mid_xt_1 = F11_low_mid_xt[1:,1:-1] \
                    - Phi_x[:,1:-1]*(F11_low_mid_xt[1:,1:-1] \
                                     - F11_high_mid_xt[1:,1:-1])
    F11_mid_xt_2 = F11_low_mid_xt[:-1,1:-1] \
                    - Phi_x[:,:-2]*(F11_low_mid_xt[:-1,1:-1] \
                                     - F11_high_mid_xt[:-1,1:-1])

    F12_mid_xt_1 = F12_low_mid_xt[1:,1:-1] \
                    - Phi_x[:,1:-1]*(F12_low_mid_xt[1:,1:-1] \
                                     - F12_high_mid_xt[1:,1:-1])
    F12_mid_xt_2 = F12_low_mid_xt[:-1,1:-1] \
                    - Phi_x[:,:-2]*(F12_low_mid_xt[:-1,1:-1] \
                                     - F12_high_mid_xt[:-1,1:-1])

    F13_mid_xt_1 = F13_low_mid_xt[1:,1:-1] \
                    - Phi_x[:,1:-1]*(F13_low_mid_xt[1:,1:-1] \
                                     - F13_high_mid_xt[1:,1:-1])
    F13_mid_xt_2 = F13_low_mid_xt[:-1,1:-1] \
                    - Phi_x[:,:-2]*(F13_low_mid_xt[:-1,1:-1] \
                                     - F13_high_mid_xt[:-1,1:-1])

    # along z:
    F21_mid_zt_1 = F21_low_mid_zt[1:-1,1:] \
                    - Phi_z[1:-1,:]*(F21_low_mid_zt[1:-1,1:] \
                                     - F21_high_mid_zt[1:-1,1:])
    F21_mid_zt_2 = F21_low_mid_zt[1:-1,:-1] \
                    - Phi_z[:-2,:]*(F21_low_mid_zt[1:-1,:-1] \
                                     - F21_high_mid_zt[1:-1,:-1])

    F22_mid_zt_1 = F22_low_mid_zt[1:-1,1:] \
                    - Phi_z[1:-1,:]*(F22_low_mid_zt[1:-1,1:] \
                                     - F22_high_mid_zt[1:-1,1:])
    F22_mid_zt_2 = F22_low_mid_zt[1:-1,:-1] \
                    - Phi_z[:-2,:]*(F22_low_mid_zt[1:-1,:-1] \
                                     - F22_high_mid_zt[1:-1,:-1])

    F23_mid_zt_1 = F23_low_mid_zt[1:-1,1:] \
                    - Phi_z[1:-1,:]*(F23_low_mid_zt[1:-1,1:] \
                                     - F23_high_mid_zt[1:-1,1:])
    F23_mid_zt_2 = F23_low_mid_zt[1:-1,:-1] \
                    - Phi_z[:-2,:]*(F23_low_mid_zt[1:-1,:-1] \
                                     - F23_high_mid_zt[1:-1,:-1])


    # Now use these mid-point values
    # to predict the values
    # at the next timestep:
    h_new = h[1:-1,1:-1] \
            - (dt/dx)*(F11_mid_xt_1 - F11_mid_xt_2) \
            - (dt/dz)*(F21_mid_zt_1 - F21_mid_zt_2) \
            + dt*S1

    qx_new = qx[1:-1,1:-1] \
            - (dt/dx)*(F12_mid_xt_1 - F12_mid_xt_2) \
            - (dt/dz)*(F22_mid_zt_1 - F22_mid_zt_2) \
            + dt*S2

    qz_new = qz[1:-1,1:-1] \
            - (dt/dx)*(F13_mid_xt_1 - F13_mid_xt_2) \
            - (dt/dz)*(F23_mid_zt_1 - F23_mid_zt_2) \
            + dt*S3

    # return the computed quantities
    # as well as the limiters to monitor/plot their behaviour
    return h_new, qx_new, qz_new, \
           Phi_x, Phi_z, \
           hzzz, hxxx, hzxx, hxzz
