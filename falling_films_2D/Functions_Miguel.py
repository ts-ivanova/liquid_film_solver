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
    n_x = np.shape(H)[0] # Number of points to filter
    H_F = np.zeros(np.shape(H)) # Initialize the filtered result
    kernel = firwin(ORD, s, window = 'hamming')
    # You could the transfer function like this if you want:
    #w, H_T  =  signal.freqz(kernel)

    # K_F = np.zeros(np.shape(K))
    # for k in range(0,n_x):
    S = H[:]
    S_Ext = Bound_EXT(S,ORD,boundaries)
    S_Filt_1 = signal.fftconvolve(S_Ext, kernel, mode = 'valid')
    S_Filt = np.flip(signal.fftconvolve(np.flip(S_Filt_1),
                                       kernel, mode = 'valid'))
    # Check
    # plt.plot(S_Ext);plt.plot((S_Filt))
    # Compute where to take the signal
    Ex1 = int((len(S_Filt)-len(S))/2)

    H_F[:] = S_Filt[Ex1:(len(S_Filt)-Ex1)]

    return H_F



from scipy import interpolate


def FFT_diff_X(h,Xg,P=400,Frac=4):
    # Usual smoothing: P 
    # Grid info
    x=Xg[0][:];  dx=x[2]-x[1]; nzz,n_x=np.shape(Xg);
    # Odd number for the extension (a fraction Frac of the original signal)
    n_EXT=int(2*np.floor(n_x/(2*Frac))+1)
    # Initialize the partial_x:
    partial_x=np.zeros((n_x))  
    # Prepare the extension for the grid
    x_ext=Bound_EXT(x,n_EXT,boundaries = "extrap"); n_x_e=len(x_ext)
    # Prepare the information for the Frequency doain
    k=np.fft.fftfreq(n_x_e)*2*np.pi/dx; k_M=k.max()    
    # filter Transfer function (fourth order smoothing)
    H=1/(1+P*(k/k_M)**4)
    
    
    f=h[:]  # assign the function
    INTERP = interpolate.interp1d(x,f,kind='cubic',fill_value='extrapolate')
    f_ext=INTERP(x_ext)   
    
    # Create the mask and the extended signal
    STEP_L=smoothstep(x_ext,x_min=x_ext[0],x_max=x[0],N=4)
    STEP_R=smoothstep(x_ext,x_min=x[-1],x_max=x_ext[-1],N=4)
    MASK=STEP_L-STEP_R
    # Masked Signal
    f_M=f_ext*MASK
    # FFT based derivatives with a smooth
    F_p=1j*k*np.fft.fft(f_M)*H;
    fp=np.fft.ifft(F_p).real 
    partial_x[:]=fp[n_EXT:f_ext.size-n_EXT]
      
    return partial_x




def FFT_diff_Z(h,Xg,Zg,P=200,Frac=4):
    # Usual smoothing: P 
    # Grid info
    z=Zg[:,1];  dz=z[2]-z[1]; n_z,n_x=np.shape(Xg);
    # Odd number for the extension (a fraction Frac of the original signal)
    n_EXT=int(2*np.floor(n_z/(2*Frac))+1)
    # Initialize the partial_x:
    partial_z=np.zeros((n_z,n_x))  
    # Prepare the extension for the grid
    z_ext=Bound_EXT(z,n_EXT,boundaries = "extrap"); n_z_e=len(z_ext)
    # Prepare the information for the Frequency doain
    k=np.fft.fftfreq(n_z_e)*2*np.pi/dz; k_M=k.max()    
    # filter Transfer function (fourth order smoothing)
    H=1/(1+P*(k/k_M)**4)
    
    for i in range(n_x):
      f=h[:,i] # assign the function
      # Extend the function and the grid
      INTERP = interpolate.interp1d(z,f,kind='cubic',fill_value='extrapolate')
      f_ext=INTERP(z_ext) 
      
      # Create the mask and the extended signal
      STEP_L=smoothstep(z_ext,x_min=z_ext[0],x_max=z[0],N=4)
      STEP_R=smoothstep(z_ext,x_min=z[-1],x_max=z_ext[-1],N=4)
      MASK=STEP_L-STEP_R
      # Masked Signal
      f_M=f_ext*MASK
      # FFT based derivatives with a smooth
      F_p=1j*k*np.fft.fft(f_M)*H;
      fp=np.fft.ifft(F_p).real 
      partial_z[:,i]=fp[n_EXT:f_ext.size-n_EXT]
      
    return partial_z



def FFT_LAP_X(h,Xg,P=200,Frac=4):
    #%% Computing partial_xx
    x=Xg[0][:];  
    print(x)
    dx=x[2]-x[1]; nzz,n_x=np.shape(Xg);
    # Odd number for the extension (a fraction Frac of the original signal)
    n_EXT=int(2*np.floor(n_x/(2*Frac))+1)
    # Initialize the partial_x:
    partial_xx=np.zeros((n_x))  
    # Prepare the extension for the grid
    x_ext=Bound_EXT(x,n_EXT,boundaries = "extrap"); n_x_e=len(x_ext)
    # Prepare the information for the Frequency doain
    k=np.fft.fftfreq(n_x_e)*2*np.pi/dx; k_M=k.max()    
    # filter Transfer function (fourth order smoothing)
    H=1/(1+P*(k/k_M)**4)
    
    # for i in range(n_z):
    f=h[:]  # assign the function
    # Extend the function and the grid
    INTERP = interpolate.interp1d(x,f,kind='cubic',fill_value='extrapolate')
    f_ext=INTERP(x_ext)     
    # Create the mask and the extended signal
    STEP_L=smoothstep(x_ext,x_min=x_ext[0],x_max=x[0],N=4)
    STEP_R=smoothstep(x_ext,x_min=x[-1],x_max=x_ext[-1],N=4)
    MASK=STEP_L-STEP_R
    # Masked Signal
    f_M=f_ext*MASK
    # FFT based derivatives with a smooth
    F_p=-k**2*np.fft.fft(f_M)*H;
    fp=np.fft.ifft(F_p).real 
    partial_xx[:]=fp[n_EXT:f_ext.size-n_EXT]       
      
    return partial_xx



def FFT_LAP_Z(h,Xg,Zg,P=200,Frac=4):
      #%% Computing partial_zz
      z=Zg[:,1];  dz=z[2]-z[1]; n_z,n_x=np.shape(Xg);
      # Odd number for the extension (a fraction Frac of the original signal)
      n_EXT=int(2*np.floor(n_z/(2*Frac))+1)
      # Initialize the partial_x:
      partial_zz=np.zeros((n_z,n_x))  
      # Prepare the extension for the grid
      z_ext=Bound_EXT(z,n_EXT,boundaries = "extrap"); n_z_e=len(z_ext)
      # Prepare the information for the Frequency doain
      k=np.fft.fftfreq(n_z_e)*2*np.pi/dz; k_M=k.max()    
      # filter Transfer function (fourth order smoothing)
      H=1/(1+P*(k/k_M)**4)
       
      for i in range(n_x):
        f=h[:,i] # assign the function
        # Extend the function and the grid
        INTERP = interpolate.interp1d(z,f,kind='cubic',fill_value='extrapolate')
        f_ext=INTERP(z_ext) 
         
        # Create the mask and the extended signal
        STEP_L=smoothstep(z_ext,x_min=z_ext[0],x_max=z[0],N=4)
        STEP_R=smoothstep(z_ext,x_min=z[-1],x_max=z_ext[-1],N=4)
        MASK=STEP_L-STEP_R
        # Masked Signal
        f_M=f_ext*MASK
        # FFT based derivatives with a smooth
        F_p=-k**2*np.fft.fft(f_M)*H;
        fp=np.fft.ifft(F_p).real 
        partial_zz[:,i]=fp[n_EXT:f_ext.size-n_EXT]
        
      return partial_zz


#%% 3. Surface tension implementation


def Surf_T(h,Xg,P=500,Frac=2):
    # Compute the surface tension term from Tsveti's paper
    h_xx=FFT_LAP_X(h,Xg,P=P,Frac=Frac)
    # First Terms
    h_xxx=FFT_diff_X(h_xx,Xg,P=P,Frac=Frac)
    
    
    return h_xxx

