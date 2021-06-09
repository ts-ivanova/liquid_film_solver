#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:33:33 2021

Definition of functions used for computing
for the purposes of the Liquid film solver

@author: tsveti
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
    h[:,0]   = 2*h[:,1] - h[:,2]
    # 3rd deriv cond:
    # h[:,0] = 0.5*(5*h[:,1] - 4*h[:,2] + h[:,3])

    qz[:,-1] = 2*qz[:,-2] - qz[:,-3]
    qx[:,-1] = 2*qx[:,-2] - qx[:,-3]
    h[:,-1]  = 2*h[:,-2] - h[:,-3]
    # 3rd deriv cond:
    # h[:,-1] = 0.5*(5*h[:,-2] - 4*h[:,-3] + h[:,-4])

    # TOP BOUNDARY (OUTLET):

    # Linear extrapolation along x
    qz[0,:]  = 2*qz[1,:] - qz[2,:]
    qx[0,:]  = 2*qx[1,:] - qx[2,:]
    h[0,:]   = 2*h[1,:] - h[2,:]
    # 3rd deriv cond:
    # h[0,:] = 0.5*(5*h[1,:] - 4*h[2,:] + h[3,:])
