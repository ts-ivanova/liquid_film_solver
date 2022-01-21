#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:56:58 2021

@author: ivanova

"""

def liquid_film_fluxes(h, qx, qz):
    '''
    Compute the fluxes for the 3D liquid film integral model.
    Simplified version: interface shear stress is neglected.
    '''
    # Fluxes in the equation for h:
    F11 = qx
    F21 = qz
    # Fluxes in the equation for qx:
    F12 = (6*qx**2)/(5*h)
    F22 = (6*qx*qz)/(5*h)
    # Fluxes in the equation for qz:
    F13 = F22
    F23 = (6*qz**2)/(5*h)

    return F11, F21, F12, F22, F13, F23
