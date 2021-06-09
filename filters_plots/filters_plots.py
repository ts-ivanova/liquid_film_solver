# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from findiff import FinDiff, coefficients, Coefficient
from Functions_Miguel import filt_X
import os

# Load computed heights over the substrate
data = \
    np.load('h_np_FPXZ1_BLmi_dx0.2000_T100_dt0.020_n04800.npy')
h = data
#data.shape gives (250)

os.chdir('plots')

# number of cells along x:
nx = 250
# number of cells along z:
nz = 100
print('data.shape gives (nx, nz)', data.shape)
print('therefore x are rows, z are columns')

# cell sizes (determined in the solver by respecting
# the frequencies)
dx = 0.2
dz = 0.01

# First derivatives operators:
d_dx = FinDiff((0, dx, 1))
d_dz = FinDiff((1, dz, 1))

# Filter the height along x and along z:
H_Xf = filt_X(h[1:-1,1:-1],31,boundaries="extrap",s=0.2)
H_Zf = filt_X(h[1:-1,1:-1].T,31,boundaries="extrap",s=0.2)
H_Zf = H_Zf.T

#%%
###############
# plot a slice along x in an arbitrary chosen cell index
fig = plt.figure(figsize=(8, 4))
# plt.contourf(h)
plt.contourf(H_Xf)
# plt.plot(h[:,37],'.-')
# plt.plot(H_Xf[:,37])
plt.title('Wave profile section along x')
plt.savefig('wavex.png', dpi=200)
# plt.show()
# plot a slice along x in an arbitrary chosen cell index
fig = plt.figure(figsize=(8, 4))
plt.contourf(H_Zf)
# plt.plot(h[37,:],'.-')
# plt.plot(H_Xf[37,:])
plt.title('Wave profile section along z')
plt.savefig('wavez.png', dpi=200)
# plt.show()

plt.close("all")

#%%
###############
# Computations of the third derivatives
# that are included in sources S2 and S3:

# d3_dx3 h:
# take the first derivative:
hx = d_dx(H_Xf)
# filter it along x:
hx_Xf = filt_X(hx,31,boundaries="extrap",s=0.2)
# differentiate again:
hxx = d_dx(hx_Xf)
# filter again:
hxx_Xf = filt_X(hxx,31,boundaries="extrap",s=0.2)
# take the third derivative:
hxxx0 = d_dx(hxx_Xf)
# and filter it:
hxxx = filt_X(hxxx0,31,boundaries="extrap",s=0.2)

# d3_dz3 h:
# hzzz = d_dz(hzz)
#new attemp:
hz = d_dz(H_Zf)
hz_Zf = filt_X(hz.T,31,boundaries="extrap",s=0.2)
hzz = d_dz(hz_Zf.T)
hzz_Zf = filt_X(hzz.T,31,boundaries="extrap",s=0.2)
hzzz0 = d_dz(hzz_Zf.T)
hzzz = filt_X(hzzz0,31,boundaries="extrap",s=0.2)

# d3_dxdz2 h:
hxzz0 = d_dx(hzz_Zf.T)
hxzz = filt_X(hxzz0,31,boundaries="extrap",s=0.2)

# d3_dzdx2 h:
# (some computations repeaded as for d3_dx3)
hxx_Zf = filt_X(hxx.T,31,boundaries="extrap",s=0.2)
hzxx0 = d_dz(hxx_Zf.T)
hzxx = filt_X(hzxx0,31,boundaries="extrap",s=0.2)


#%%
###############
# PLOTS

# Directly taking the derivatives to roughly compare
# without filtering on each differetiation:
# 1st and 2nd derivs along x:
fig = plt.figure(figsize=(8, 4))
# plt.plot((d_dx(h))[:,55],'.-')
# plt.plot((d_dx(H_Xf))[:,55])
plt.title('First Derivative hx')
plt.savefig('hx.png', dpi=200)
# plt.show()

fig = plt.figure(figsize=(8, 4))
plt.plot(d_dx(d_dx(h))[:,55],'.-')
plt.plot(d_dx((d_dx(H_Xf)))[:,55])
plt.title('Second Derivative hxx')
plt.savefig('hxx.png', dpi=200)
# plt.show()

# 1st and 2nd derivs along z:
fig = plt.figure(figsize=(8, 4))
plt.plot((d_dz(h))[55,:],'.-')
plt.plot((d_dz(H_Zf))[55,:])
plt.title('First Derivative hz')
plt.savefig('hz.png', dpi=200)
# plt.show()

fig = plt.figure(figsize=(8, 4))
plt.plot(d_dz((d_dz(h)))[55,:],'.-')
plt.plot(d_dz((d_dz(H_Zf)))[55,:])
plt.title('Second Derivative hzz')
plt.savefig('hzz.png', dpi=200)
# plt.show()

plt.close("all")

#%%
###############
# Filtered versions:
# 1st and 2nd derivs along x:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hx_Xf)
# plt.plot((d_dx(h))[:,55],'.-')
# plt.plot(hx_Xf[:,55])
plt.title('First Derivative hx (filt)')
plt.savefig('hx_f.png', dpi=200)
# plt.show()

fig = plt.figure(figsize=(8, 4))
plt.contourf(hxx_Xf)
# plt.plot(d_dx(d_dx(h))[:,55],'.-')
# plt.plot(hxx_Xf[:,55])
plt.title('Second Derivative hxx (filt)')
plt.savefig('hxx_f.png', dpi=200)
# plt.show()

# 1st and 2nd derivs along z:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hz)
# plt.plot((d_dz(h))[55,:],'.-')
# plt.plot(hz[55,:])
plt.title('First Derivative hz (filt)')
plt.savefig('hz_f.png', dpi=200)
# plt.show()

fig = plt.figure(figsize=(8, 4))
plt.contourf(hzz)
# plt.plot(d_dz((d_dz(h)))[55,:],'.-')
# plt.plot(hzz[55,:])
plt.title('Second Derivative hzz (filt)')
plt.savefig('hzz_f.png', dpi=200)
# plt.show()


#%%
###############
# Plot hxxx:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hxxx)
# plt.plot(d_dx(d_dx((d_dx(h))))[55,:],'.-')
# plt.plot(hxxx[55,:])
plt.title('Third Derivative hxxx')
plt.savefig('hxxx.png', dpi=200)
# plt.show()

# Plot hxzz:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hxzz)
# plt.plot(d_dx(d_dz((d_dz(h))))[55,:],'.-')
# plt.plot(hxzz[55,:])
plt.title('Third Derivative hxzz')
plt.savefig('hxzz.png', dpi=200)
# plt.show()

# Plot hzzz:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hzzz)
# plt.plot(d_dz(d_dz((d_dz(h))))[55,:],'.-')
# plt.plot(hzzz[55,:])
plt.title('Third Derivative hzzz')
plt.savefig('hzzz.png', dpi=200)
# plt.show()

# Plot hzxx:
fig = plt.figure(figsize=(8, 4))
plt.contourf(hzxx)
# plt.plot(d_dz(d_dx((d_dx(h))))[55,:],'.-')
# plt.plot(hzxx[55,:])
plt.title('Third Derivative hzxx')
plt.savefig('hzxx.png', dpi=200)
# plt.show()


#%%
###############

plt.close("all")

fig.clear()
plt.clf()
plt.close(fig)


print(hxxx.shape)
print(hzzz.shape)
print(hzxx.shape)
print(hxzz.shape)



os.chdir('../')
