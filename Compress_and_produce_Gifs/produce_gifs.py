#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:14:24 2021

Produce gifs from the .png frames

@author: tsveti
"""

import imageio
import os
from natsort import natsorted
# (to use proper sorting)
import glob

# Change working directory:
os.chdir('../RESULTS/case_of_interest/')

subfolder = natsorted(glob.glob("P*"))

# Extract frame names:
files0 = natsorted(os.listdir(subfolder[0]))
files1 = natsorted(os.listdir(subfolder[1]))
files2 = natsorted(os.listdir(subfolder[2]))
files3 = natsorted(os.listdir(subfolder[3]))
files4 = natsorted(os.listdir(subfolder[4]))
files5 = natsorted(os.listdir(subfolder[5]))
# Assign them to a list:
files = [files0, files1, files2, files3, files4, files5]


# Loop to produce the animated .gif:
for i in range(len(files)):
    images = [imageio.imread(subfolder[i]+os.sep+file) \
              for file in files[i]]
    imageio.mimwrite('../../produced_gifs/Movie_{}.gif'
                      .format(subfolder[i]),
                      images, fps=4)

os.chdir('../../../')
