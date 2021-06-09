#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:56:58 2021

@author: tsveti
"""

def liquid_film_fluxes(h, qx, qz):
    '''
    Compute the fluxes for the 3D liquid film integral model.
    '''
    # Fluxes in the equation for h:
    F11 = qx
    F21 = qz
    # Fluxes in the equation for qx:
    F12 = (144*qx**2 + 48*h*qx + 24*h**2)/(120*h)
    F22 = (144*qx*qz + 24*h*qz)/(120*h)
    # Fluxes in the equation for qz:
    F13 = F22
    F23 = (144*qz**2)/(120*h)

    return F11, F21, F12, F22, F13, F23
