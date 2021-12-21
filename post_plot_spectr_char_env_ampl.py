#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:50:30 2021

@author: ivanova

Post-processing the liquid film waves:
Spectrograms in space,
characteristics,
wave envelopes,
and exponential fitting of the amplitudes.

The spectrograms give information on whether
the wave evolves towards other frequencies.
The characteristic lines indicate whether
the nonlinear effects are weak or not.
The wave envelopes show the
minimum and maximum values over time.
The amplitudes of the waves and their
exponential fit indicate the decay rate.
"""

import numpy as np

# plots:
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

# spectrogram
# from scipy.signal import spectrogram
# import pandas as pd
from scipy.fft import fftshift

# for the envelopes:
from scipy.signal import find_peaks

# file processing:
import os
from natsort import natsorted
# (to use proper sorting)
import glob
# Import libraries for paths and saving:
from pathlib import Path


# for the exponential fit:
from scipy.optimize import curve_fit

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c
# b is the decay rate in which we are interested.
Fitting = True

# PLOT CUSTOMIZATIONS:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)

# transparency level
alpha1 = 0.9


####################
# os.chdir('RESULTS-processed/')
os.chdir('RESULTS_December/')
# Gather all computed configurations:
LIQUIDS = natsorted(glob.glob('2D_WATER_without_surf_ten_Re319'))
# LIQUIDS = natsorted(glob.glob('3*'))

print('Going to process ', LIQUIDS)

for LIQUID in LIQUIDS:
    print('Case ', LIQUID)

    os.chdir(LIQUID + os.sep + 'SOLUTIONS_n')
    # FOLDERS = natsorted(glob.glob('dx*'))
    FOLDERS = natsorted(glob.glob('R*'))

    for FOLDER in FOLDERS:
        # Change working directory:
        os.chdir(FOLDER)
        subfolder = natsorted(glob.glob("P*"))

        # to save frequencies and decay rates:
        freqs = []
        decay_rates = []

        for i in range(len(subfolder)):
            print('Processing ', subfolder[i])
            
            # Extract information 
            # from the naming conventions:
            
            h0 = float(FOLDER[7:11])
            print('h0 = ', h0)
            # delta = Epsilon*Re:
            delta = FOLDER[-7:]
            # Freq:
            f = float(subfolder[i][-4:])
            print('f = ', f)
            # Re:
            Re = float(subfolder[i][-10:-7])
            print('Re = {:.0f}'.format(Re))
            
            conf_key = subfolder[i][:4]

            dx = float(subfolder[i][20:26])
            nx = int(subfolder[i][29:33])
            dz = float(subfolder[i][36:40])
            nz = int(subfolder[i][44:48])
            

            # Spatial dimensions:
            x = np.mgrid[0:nx]*dx
            z = np.mgrid[0:nz]*dz
            # Create matrices of the coordinate variables
            [Z,X] = np.meshgrid(z,x)

            os.chdir(subfolder[i])
            filelist = natsorted(glob.glob('h_np' + os.sep \
                                            + '*.npy'))

            # To save the contourf's:
            directory = "../../../POSTPROCESSED/" \
                        + "spectrogr_charact_env_ampl" \
                        + "_Re{:.0f}".format(Re) \
                        + "_h{:.1f}".format(h0) \
                        + "_delta" + delta
             
            Path(directory).mkdir(parents=True,
                                  exist_ok=True)

            # initialize the height along x in time list:
            H_XT_list = []

            # an arbitrary chosen xslice of the domain:
            # xslice = int(0.5*nz/8)
            xslice = int(0.35*nz)

            # Loop over all data files to extract the height
            # and attach them to the dictionary:
            for ii in range(len(filelist)):
                h_np = np.load(filelist[ii])
                h = h_np[:,xslice]

                # store in matrix the height along x 
                # for each time
                H_XT_list.append([])
                H_XT_list[ii].append(h)





            print('Computing spectrograms ... ')

            # extract timestep dt:
            dt = float(filelist[-1][-16:-11])
            # final_time:
            T  = int(filelist[-1][-22:-19])

            H_XT = np.asarray(H_XT_list).reshape(len(filelist), nx)
            print('H_XT.shape = ', H_XT.shape)

            # Construct space and time axes:
            # x-axis:
            x_axis = -x
            # (flip because the inlet is at the last index -1.)

            # t-axis:
            n_files_in_time = len(filelist)
            t_axis = 100*dt*np.linspace(0,
                                        len(filelist),
                                        len(filelist))
            # ^ multiplication by 100 because the saved solutions
            # are chosen to be every 100 steps (can be modified).

            H_XF = np.zeros(H_XT.shape)

            H_XT_max = np.zeros(H_XT.shape)
            H_XT_min = np.zeros(H_XT.shape)

            # This is the numpy tool for freq axis:
            Freqs = fftshift(np.fft.fftfreq(len(filelist)))*1/dt

            for j in range(1,nx):
                # Profile at location j in time:
                Prof_time = H_XT[:,j]
                Prof_Freq = np.abs(fftshift(np.fft.fft(
                                    Prof_time - Prof_time.mean())))
                H_XF[:,j] = Prof_Freq

                # Evaluate maximum and minimum heights in time 
                # along x:
                H_XT_max[:,j] = np.amax(Prof_time)
                H_XT_min[:,j] = np.amin(Prof_time)

            plt.close()

            fig, ax = plt.subplots()
            contourplot_sp = plt.contourf(Freqs/100,
                                          x_axis,#-x_axis.min(),
                                          H_XF.T,
                                          cmap = 'PuBu')
            # plt.colorbar(contourplot_sp,
            #              format = '%.2f',
            #              anchor = (0.5, 0.5))
            # print('H_XF.T', H_XF.T)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('frequency, [-]')
            plt.ylabel('length x, [-]')
            plt.xlim(0,)
            plt.title('Spectrogram')
            # plt.title(subfolder[i])
            plt.savefig(directory + os.sep \
                        + 'spect_' + subfolder[i] \
                        + '.png',
                        format      = 'png',
                        dpi         = 200,
                        bbox_inches = 'tight')





            print('Plotting characteristics ... ')
            plt.close()
            contourplot_ch = plt.contourf(t_axis,
                                          x_axis,#-x_axis.min(),
                                          H_XT.T,
                                          cmap = 'PuBu')
            plt.colorbar(contourplot_ch,
                          format = '%.2f',
                          anchor = (0.5, 0.5))
            plt.xlabel('time, [-]')
            plt.ylabel('length x, [-]')
            plt.title('Characteristics')
            # plt.title(subfolder[i])
            plt.savefig(directory + os.sep \
                        + 'char_' + subfolder[i] \
                        + '.png',
                        format      = 'png',
                        dpi         = 200,
                        bbox_inches = 'tight')





            print('Plotting envelopes ... ')
            plt.close()
            plt.figure(figsize = (12,4))
            plt.ylim(0.1,0.4)

            plt.plot(x_axis,
                      H_XT.T[:,-1],
                      linestyle = 'solid',
                      color = '#574d57',
                      linewidth = 2,
                      alpha = alpha1,
                      label = 'height')

            # identify for which indices there are peaks
            # of the wave maxima:
            peaks_max, _ = find_peaks(H_XT_max[xslice,:])

            plt.plot(-dx*peaks_max,
                      H_XT_max[xslice,:][peaks_max],
                      linestyle = 'dashed',
                      color = '#9e2ec9',
                      linewidth = 2,
                      alpha = alpha1,
                      label = 'max')

            # identify for which indices there are peaks
            # of the wave minima:
            peaks_min, _ = find_peaks(H_XT_min[xslice,:])

            plt.plot(-dx*peaks_min,
                      H_XT_min[xslice,:][peaks_min],
                      linestyle = 'dashdot',
                      color = '#2878b5',
                      linewidth = 2,
                      alpha = alpha1,
                      label = 'min')

            plt.xlabel('length x, [-]')
            plt.ylabel('film thickness h, [-]')
            plt.title('Envelopes in time for each position x')
            plt.legend(loc = 'upper right')
            # plt.ylim(h0 - 0.2, h0 + 0.4)
            plt.grid()
            # plt.title(subfolder[i])
            plt.savefig(directory + os.sep \
                        + 'env_' + subfolder[i] \
                        + '.png',
                        format      = 'png',
                        dpi         = 200,
                        bbox_inches = 'tight')



            if Fitting:
                print('Plotting exponential fit to amplitudes ... ')
                plt.close()
                plt.figure(figsize = (12,4))

                H_XT_ampl = H_XT_max[xslice,:] - H_XT_min[xslice,:]
                peaks_ampl, _ = find_peaks(H_XT_ampl, distance = 10)

                # Initial guess of the exp fit:
                c_guess = 0.1*H_XT_ampl[peaks_ampl[-19]]
                b_guess = np.log((H_XT_ampl[peaks_ampl[-20]]\
                                  - c_guess)/\
                                  (H_XT_ampl[peaks_ampl[-21]]\
                                  - c_guess))*\
                                  (1/(peaks_ampl[-21]\
                                      -peaks_ampl[-20]))
                a_guess = (H_XT_ampl[peaks_ampl[-20]] - \
                          H_XT_ampl[peaks_ampl[-21]]) /\
                          (np.exp(-b_guess*peaks_ampl[-20]) - \
                            np.exp(-b_guess*peaks_ampl[-21]))
                
                
                yn = exp_func(-dx*peaks_ampl[30:-30],
                          a_guess, b_guess, c_guess)
                
                
                popt, pcov = curve_fit(exp_func,
                                        -dx*peaks_ampl[30:-30],
                                        H_XT_ampl[peaks_ampl[30:-30]],
                                        (a_guess, b_guess, c_guess),
                                        maxfev=5000)
                
                a_1, b_1, c_1 = popt
                
                
                #
                # L2 errors:
                fitting_error = np.linalg.norm(
                                    exp_func(-dx*peaks_ampl[30:-30],
                                    a_1, b_1, c_1) \
                                    - H_XT_ampl[peaks_ampl[30:-30]])
                print('Exponential fitting parameters: \
                      \n a = {:.4f} \n b = {:.4f} \n c = {:.4f} \
                      \n fitting error L2: {:.5f}'
                      .format(a_1, b_1, c_1, fitting_error))
                #
                
                
                # plot the amplitudes from the wave solutions:
                plt.plot(-dx*peaks_ampl[30:-30],
                          H_XT_ampl[peaks_ampl[30:-30]],
                          'k*',
                          markersize = 6,
                          label = 'Amplitude',
                          alpha = alpha1)
                # plot the exponential fit:
                plt.plot(-dx*peaks_ampl[30:-30],
                          exp_func(np.array(-dx*peaks_ampl[30:-30]),
                                  a_1, b_1, c_1),
                          'm-.',
                          linewidth = 2,
                          label = 'Fit: {:.4f}*exp({:.4f} x) + {:.4f}'
                                  .format(a_1, b_1, c_1) \
                                  + '\n $L^2$ error: {:.4f}'
                                  .format(fitting_error),
                          alpha = alpha1)
                
                
                plt.xlabel('length x, [-]')
                plt.ylabel('amplitude, [-]')
                # plt.title('Amplitudes in time for each position x')
                plt.legend(loc = 'upper right')
                plt.grid()
                # plt.title(subfolder[i])
                
                plt.savefig(directory + os.sep \
                            + 'ampl_' + subfolder[i] \
                            + '.png',
                            format      = 'png',
                            dpi         = 200,
                            bbox_inches = 'tight')
                plt.show()
                plt.close()
                
                # append the frequencies and
                # the decay rates to lists
                # for plotting later:
                freqs.append(f)
                decay_rates.append(b_1)

            os.chdir('../')
            
        if Fitting:
            directory_fit = "../../../" \
                        + "decay"
            Path(directory_fit).mkdir(parents=True,
                                      exist_ok=True)
            
            # save the frequencies and the decay rate b_1 
            # to .npy's:
            Re_path = directory_fit + os.sep \
                      + 'Re{:.0f}'.format(Re) + os.sep \
                      + LIQUID
            Path(Re_path).mkdir(parents=True,
                                              exist_ok=True)
             
            freqs_file  = Re_path + os.sep \
                          + 'freqs_h' + str(h0) \
                          +'_delta' + delta
             
            decays_file = Re_path + os.sep \
                          + 'decays_h' + str(h0) \
                          +'_delta' + delta

            np.save(freqs_file,
                    np.asarray(freqs))
            np.save(decays_file,
                    np.asarray(decay_rates))
        
        
        os.chdir('../')
    os.chdir('../../')
    
os.chdir('../')
