# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:13:54 2022

@author: Mendez
"""

import numpy as np

import matplotlib.pyplot as plt

#%% Introduction

# In this file we play use the trick from the file A3 to compute the surface
# tension terms. The 3D data is provided by the analytic function used by David
# in the codes A1-A2.
# The workflow is as follow:
    
# 1. Define the functions in 3D and show the derivatives along three lines.
# 2. Create a function Partial_deriv which computes the derivatives along a direction.
     # This include first derivatives and second derivatives
# 3. Compute the surface tension term ( gradient of the laplacian) for this 
#    configuration and compare with the analytical solution


#%% Create the Analytic function for validation purposes
n_x=1000; n_z=500 # Number of points along x and y.
# Define the 1D grids
x=np.linspace(0,80,n_x); z=np.linspace(-25,25,n_z)
# Define the mesh grid
Xg,Zg=np.meshgrid(x,z)

# Define the analytic thickness distributions
h0=0.2; A=0.1; k_x=2*np.pi/(26); k_z=2*np.pi/(19)
h = h0*(1 + A*np.sin(k_x*Xg)*np.sin(k_z*Zg)) # thickness distribution
plt.contourf(Zg,Xg,h)

# Important note: te kx and kz are taken so as NOT to have periodic BCs.


# Show first derivatives along two positions
I_X=20; z_v1=Zg[:,I_X]
I_Z=20; x_v1=Xg[I_Z,:]

# Partial_X along x_v1
partial_x_h=k_x*h0*A*np.cos(k_x*Xg)*np.sin(k_z*Zg)
partial_z_h=k_z*h0*A*np.sin(k_x*Xg)*np.cos(k_z*Zg)

# Second Derivatives
partial_xx_h=-k_x**2*h0*A*np.sin(k_x*Xg)*np.sin(k_z*Zg)
partial_zz_h=-k_z**2*h0*A*np.sin(k_x*Xg)*np.sin(k_z*Zg)

# Third Derivatives
partial_xxx_h=-k_x**3*h0*A*np.cos(k_x*Xg)*np.sin(k_z*Zg)
partial_zxx_h=-k_z*k_x**2*h0*A*np.sin(k_x*Xg)*np.cos(k_z*Zg)

# Show second and third derivative terms along both coordinates

# Plot the function
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(z_v1,partial_xx_h[:,I_X]/np.max(partial_xx_h[:,I_X]),'k--',label='h_xx')
plt.plot(z_v1,partial_zxx_h[:,I_X]/np.max(partial_zxx_h[:,I_X]),'b--',label='h_zxx')
plt.xlabel('z')
plt.legend()
plt.title('Normalized derivatives along z')

# Plot the function
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x_v1,partial_xx_h[I_Z,:]/np.max(partial_xx_h[I_Z,:]),'k--',label='h_xx')
plt.plot(x_v1,partial_zxx_h[I_Z,:]/np.max(partial_zxx_h[I_Z,:]),'b--',label='h_zxx')
plt.xlabel('x')
plt.legend()
plt.title('Normalized derivatives along x')



#%% Numerical Computation of derivative along x and z

from Functions_Miguel import Bound_EXT
from Functions_Miguel import smoothstep
from scipy import interpolate


def FFT_diff_X(h,Xg,Zg,P=400,Frac=4):
    # Usual smoothing: P 
    # Grid info
    x=Xg[1,:];  dx=x[2]-x[1]; n_z,n_x=np.shape(Xg);
    # Odd number for the extension (a fraction Frac of the original signal)
    n_EXT=int(2*np.floor(n_x/(2*Frac))+1)
    # Initialize the partial_x:
    partial_x=np.zeros((n_z,n_x))  
    # Prepare the extension for the grid
    x_ext=Bound_EXT(x,n_EXT,boundaries = "extrap"); n_x_e=len(x_ext)
    # Prepare the information for the Frequency doain
    k=np.fft.fftfreq(n_x_e)*2*np.pi/dx; k_M=k.max()    
    # filter Transfer function (fourth order smoothing)
    H=1/(1+P*(k/k_M)**4)
    
    for i in range(n_z):
      f=h[i,:]  # assign the function
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
      partial_x[i,:]=fp[n_EXT:f_ext.size-n_EXT]
      
    return partial_x



# Test the gradient
partial_x_h_N=FFT_diff_X(h,Xg,Zg,P=400,Frac=2)
# Check the overal error
Err=partial_x_h_N-partial_x_h

# Plot the function (this seems to be working very well)
I_Z=20
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x_v1,partial_x_h_N[I_Z,:],'k--',label='Analytic')
plt.plot(x_v1,partial_x_h[I_Z,:],'b--',label='Numerics')
plt.xlabel('x')
plt.legend()
plt.title('Partial derivatives along x')



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


# Test the gradient
partial_z_h_N=FFT_diff_Z(h,Xg,Zg,P=400,Frac=2)
# Check the overal error
Err=partial_z_h_N-partial_z_h

# Plot the function (this seems to be working very well)
I_Z=20
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(z_v1,partial_z_h[:,I_Z],'k--',label='Analytic')
plt.plot(z_v1,partial_z_h_N[:,I_Z],'b--',label='Numerics')
plt.xlabel('z')
plt.legend()
plt.title('Partial derivatives along z')


def FFT_LAP_X(h,Xg,Zg,P=200,Frac=4):
    #%% Computing partial_xx
    x=Xg[1,:];  dx=x[2]-x[1]; n_z,n_x=np.shape(Xg);
    # Odd number for the extension (a fraction Frac of the original signal)
    n_EXT=int(2*np.floor(n_x/(2*Frac))+1)
    # Initialize the partial_x:
    partial_xx=np.zeros((n_z,n_x))  
    # Prepare the extension for the grid
    x_ext=Bound_EXT(x,n_EXT,boundaries = "extrap"); n_x_e=len(x_ext)
    # Prepare the information for the Frequency doain
    k=np.fft.fftfreq(n_x_e)*2*np.pi/dx; k_M=k.max()    
    # filter Transfer function (fourth order smoothing)
    H=1/(1+P*(k/k_M)**4)
    
    for i in range(n_z):
      f=h[i,:]  # assign the function
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
      partial_xx[i,:]=fp[n_EXT:f_ext.size-n_EXT]       
      
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


# Test the Laplacian
partial_xx_h_N=FFT_LAP_X(h,Xg,Zg,P=400,Frac=2)
partial_zz_h_N=FFT_LAP_Z(h,Xg,Zg,P=400,Frac=2)


# Check the overal error
Err_x=partial_xx_h_N-partial_xx_h
Err_z=partial_zz_h_N-partial_zz_h


# Plot the function (this seems to be working very well)
I_Z=499
fig, ax = plt.subplots(figsize=(10,4))
plt.plot(z_v1,partial_zz_h[:,I_Z],'k--',label='Analytic')
plt.plot(z_v1,partial_zz_h_N[:,I_Z],'b--',label='Numerics')
plt.xlabel('z')
plt.legend()
plt.title('Partial derivatives along z')



#%% 3. Surface tension implementation


def Surf_T(h,Xg,Zg,P=500,Frac=2):
    # Compute the surface tension term from Tsveti's paper
    h_xx=FFT_LAP_X(h,Xg,Zg,P=P,Frac=Frac)
    h_zz=FFT_LAP_Z(h,Xg,Zg,P=P,Frac=Frac)
    # First Terms
    h_xxx=FFT_diff_X(h_xx,Xg,Zg,P=P,Frac=Frac)
    h_xzz=FFT_diff_X(h_zz,Xg,Zg,P=P,Frac=Frac)
    # Second terms
    h_zxx=FFT_diff_Z(h_xx,Xg,Zg,P=P,Frac=Frac)
    h_zzz=FFT_diff_Z(h_zz,Xg,Zg,P=P,Frac=Frac)   
    
    return h_xxx,h_xzz, h_zxx,h_zzz


h_xxx,h_xzz, h_zxx,h_zzz=Surf_T(h,Xg,Zg,P=500,Frac=2)




















