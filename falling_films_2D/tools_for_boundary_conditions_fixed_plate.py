#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:33:33 2021

@author: tsvetelina ivanova


Definition of a function for imposing linear extrapolation
as boundary conditions for the Liquid film solver.
"""

#################################################
def linear_extrap_lrt(h, qx):
    '''
    Boundary conditions: linear extrapolation.
    This function enforces linear extrapolation
    on the left, right, and top boundaries (lrt)
    (sides and outlet respectively)
    of the moving substrate.
    '''

	# BOTTOM BOUNDART (INLET):

	# Linear extrapolation along x
    qx[-1]  = 2*qx[-2] - qx[-3]
    h[-1]   = 2*h[-2] - h[-3]
