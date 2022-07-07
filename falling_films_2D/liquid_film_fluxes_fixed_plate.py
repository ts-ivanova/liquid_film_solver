#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:56:58 2021

@author: tsvetelina ivanova

"""

def liquid_film_fluxes(h, qx):
    '''
    Compute the fluxes for the falling liquid film model.
    '''
    # Fluxes in the equation for h:
    F11 = qx
    # Fluxes in the equation for qx:
    F12 = (6*qx**2)/(5*h)

    return F11, F12