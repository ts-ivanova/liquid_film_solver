#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:17:02 2021
@author: mendez
"""

import numpy as np
from scipy.signal import firwin # To create FIR kernels
from scipy import signal
from scipy.special import comb

def smoothstep(x, x_min=0, x_max=1, N=100):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result

def Bound_EXT(S,Ex,boundaries):
    """
    This function computes the extension of a 1D signal for
    filtering purposes
    :param S: The Input signal
    :param Ex: Ex The extension to be introduced
    (must be an odd number!)
    :param boundaries: The type of extension:
        ‘reflect’ (d c b a | a b c d | d c b a)
        The input is extended by reflecting about
        the edge of the last pixel.
        ‘nearest’ (a a a a | a b c d | d d d d)
        The input is extended by replicating the last pixel.
        ‘wrap’ (a b c d | a b c d | a b c d)
        The input is extended by wrapping around to
        the opposite edge.
        ‘extrap’ Extrapolation
        The input is extended via linear extrapolation.
        
    """
    # We first perform a zero padding
    #Ex = int((Nf-1)/2) # Extension on each size

    # Compute the size of the extended signal:
    size_Ext = 2*Ex+len(S)
    # Initialize extended signal:
    S_extend = np.zeros((size_Ext))
    # Assign the Signal on the zeroes:
    S_extend[Ex:(int((size_Ext)-Ex))] = S

    if boundaries == "reflect":
        # Prepare the reflection on the left:
        LEFT = np.flip(S[0:Ex])
        # Prepare the reflection on the right:
        RIGHT = np.flip(S[len(S)-Ex:len(S)])
        S_extend[0:Ex] = LEFT;
        S_extend[len(S_extend)-Ex:len(S_extend)] = RIGHT
    elif boundaries == "nearest":
        # Prepare the constant on the left:
        LEFT = np.ones(Ex)*S[0]
        # Prepare the constant on the right:
        RIGHT = np.ones(Ex)*S[len(S)-1]
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend)-Ex:len(S_extend)] = RIGHT
    elif boundaries == "wrap":
        # Wrap on the Left:
        LEFT = S[len(S)-Ex:len(S)]
        # Wrap on the Right:
        RIGHT = S[0:Ex]
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend)-Ex:len(S_extend)] = RIGHT
    elif boundaries == "extrap":
        # Linear extrapolation on the left
        # Take the first slope.
        ds = S[1]-S[0]
        # First and last added values on the left:
        In = S[0]-Ex*ds; Fin = S[0]-ds
        LEFT = np.linspace(In,Fin,Ex)
        # Linear extrapolation on the right
        ds = S[-1]-S[-2] # Take the last slope
        In = S[-1]+ds; Fin = S[-1]+Ex*ds
        # Prepare the constant on the Right:
        RIGHT = np.linspace(In,Fin,Ex)
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend)-Ex:len(S_extend)] = RIGHT
    return S_extend



def filt_X(H,ORD,boundaries = "extrap",s = 0.1):
    """
    This function filters the matrix h along the rows
    (assuming the x is there)
    It can filter the other direction if you simply
    give the transpose of it.
    :param h: Input matrix for the thickness
    :param s: cut off frequency in the digital settings.
    The lower, the smoother
    :param ORD: Order of the Filter
    :param boundaries: The type of extension:
        ‘reflect’ (d c b a | a b c d | d c b a)
        The input is extended by reflecting about
        the edge of the last pixel.
        ‘nearest’ (a a a a | a b c d | d d d d)
        The input is extended by replicating the last pixel.
        ‘wrap’ (a b c d | a b c d | a b c d)
        The input is extended by wrapping around
        to the opposite edge.
        ‘extrap’ Extrapolation (not yet available)
        The input is extended via linear extrapolation.
    """
    # Filter along the raws
    n_x = np.shape(H)[1] # Number of points to filter
    H_F = np.zeros(np.shape(H)) # Initialize the filtered result
    kernel = firwin(ORD, s, window = 'hamming')
    # You could the transfer function like this if you want:
    #w, H_T  =  signal.freqz(kernel)

    # K_F = np.zeros(np.shape(K))
    for k in range(0,n_x):
        S = H[:,k]
        S_Ext = Bound_EXT(S,ORD,boundaries)
        S_Filt_1 = signal.fftconvolve(S_Ext, kernel, mode = 'valid')
        S_Filt = np.flip(signal.fftconvolve(np.flip(S_Filt_1),
                                           kernel, mode = 'valid'))
        # Check
        # plt.plot(S_Ext);plt.plot((S_Filt))
        # Compute where to take the signal
        Ex1 = int((len(S_Filt)-len(S))/2)

        H_F[:,k] = S_Filt[Ex1:(len(S_Filt)-Ex1)]

    return H_F