#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 21:07:25 2021

@author: ivanova
"""

# Compress the png's

from PIL import Image
import PIL
import os
import glob
from natsort import natsorted
# (to use proper sorting)

def compress_images(directory=False, quality=30):
    # 1. If there is a directory then change into it,
    # else perform the next operations
    # inside of the current working directory:
    if directory:
        os.chdir(directory)

    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir()

    # 3. Extract all of the images:
    images = [file for file in files
              if file.endswith('png')]

    # 4. Loop over every image:
    for image in images:
        print(image)

        # 5. Open every image:
        img = Image.open(image)
        dim = img.size
        dim2 = tuple(int(dimi/2) for dimi in dim)
        img = img.resize(dim2,Image.ANTIALIAS)

        # 5. Compress every image and save it
        # with a new name:
        img.save("Compressed_"+image,
                 optimize=True, quality=quality)



# Working directory:
# folder = glob.glob('PLOTS/FRAMES/')
# subfolder = natsorted(glob.glob("h*"))

# compress_images(directory='h_dx0.046_T80_C1/',
#                 quality=10)
