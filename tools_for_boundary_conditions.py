#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:33:33 2021

@author: tsveti


Definition of a function for imposing linear extrapolation
as boundary conditions for the Liquid film solver.
"""

#################################################
def linear_extrap_lrt(h, qx, qz):
    '''
    Boundary conditions: linear extrapolation.
    This function enforces linear extrapolation
    on the left, right, and top boundaries (lrt)
    (sides and outlet respectively)
    of the moving substrate.
    '''
    # LEFT AND RIGHT BOUNDARIES (SLICES):

    # Linear extrapolation along z
    qz[:,0]  = 2*qz[:,1] - qz[:,2]
    qx[:,0]  = 2*qx[:,1] - qx[:,2]
    h[:,0]   = 2*h[:,1]  - h[:,2]

    qz[:,-1] = 2*qz[:,-2] - qz[:,-3]
    qx[:,-1] = 2*qx[:,-2] - qx[:,-3]
    h[:,-1]  = 2*h[:,-2]  - h[:,-3]

    # TOP BOUNDARY (OUTLET):

    # Linear extrapolation along x
    qz[0,:]  = 2*qz[1,:] - qz[2,:]
    qx[0,:]  = 2*qx[1,:] - qx[2,:]
    h[0,:]   = 2*h[1,:] - h[2,:]
