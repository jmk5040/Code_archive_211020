#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:17:46 2020

@author: sonic
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
#%%
path        = '~/RASA36/lab/merge'
os.chdir(os.path.expanduser(path))
os.getcwd()
#%%
lolist  = glob.glob('*low.fits')
hilist  = glob.glob('*high.fits')
melist  = glob.glob('*merged.fits')
lolist.sort()
hilist.sort()
melist.sort()

# 0 : bias
# 1 : dark
# 2 : light

#%%
#thres   = 3800
thres   = 0
scale   = 16.06/0.71
ratio   = 20.5
offset  = 1245
#%%
#
#image_file = get_pkg_data_filename(hilist[2])
#imgdata     = fits.getdata(image_file, ext=0)
flat    = fits.open('nr.fits')[0].data

high    = fits.open(hilist[2])[0].data
low     = fits.open(lolist[2])[0].data
merge   = fits.open(melist[2])[0].data

highb   = fits.open(hilist[0])[0].data
lowb    = fits.open(lolist[0])[0].data
mergeb  = fits.open(melist[0])[0].data

final   = np.zeros(shape=high.shape)

for i in range(final.shape[0]):
    for j in range(final.shape[1]):
        if high[i][j] < thres:
            final[i][j] = (high[i][j] - highb[i][j]) / flat[i][j]
        elif high[i][j] >= thres:
#            final[i][j] = min(low[i][j] * scale, 65535)
#            final[i][j] = min((low[i][j] - lowb[i][j]) * ratio - offset, 65535)
#            final[i][j] = min((low[i][j] - lowb[i][j]) * scale, 65535)
#            final[i][j] = min(((low[i][j] - lowb[i][j]) * scale) / flat[i][j], 65535/flat[i][j]) # merge_cal
#            final[i][j] = min(((low[i][j] - lowb[i][j]) * ratio - offset) / flat[i][j], 65535/flat[i][j]) #merge_cal2
            final[i][j] = min(((low[i][j] - lowb[i][j])) / flat[i][j], 65535/flat[i][j]) #merge_low
            
#%%
outfile = 'merge_low_simulated.fits'
#resfile = 'residual_low_simulated.fits'

simage  = fits.PrimaryHDU(final)
#residu  = fits.PrimaryHDU(final-merge)
simage.writeto(outfile, overwrite=True)
#residu.writeto(resfile, overwrite=True)

#%%

path        = '~/research/observation/RASA36/masterdark/'
os.chdir(os.path.expanduser(path))
os.getcwd()


#%%

high1   = fits.open('dark-60_high1.fits')[0].data
high2   = fits.open('dark-60_high2.fits')[0].data
hdr1    = fits.open('dark-60_hdr1.fits')[0].data
hdr2    = fits.open('dark-60_hdr2.fits')[0].data

highhigh    = np.zeros(shape=high1.shape)
highhdr1    = np.zeros(shape=high1.shape)
highhdr2    = np.zeros(shape=high1.shape)
hdrhdr      = np.zeros(shape=high1.shape)

for i in range(highhigh.shape[0]):
    for j in range(highhigh.shape[1]):
        highhigh[i][j]   = high1[i][j]-high2[i][j]
        highhdr1[i][j]   = high1[i][j]-hdr1[i][j]
        highhdr2[i][j]   = high2[i][j]-hdr2[i][j]
        hdrhdr[i][j]     = hdr1[i][j]-hdr2[i][j]

fits.PrimaryHDU(highhigh).writeto('high-high.fits', overwrite=True)
fits.PrimaryHDU(highhdr1).writeto('high-hdr1.fits', overwrite=True)
fits.PrimaryHDU(highhdr2).writeto('high-hdr2.fits', overwrite=True)
fits.PrimaryHDU(hdrhdr).writeto('hdr-hdr.fits', overwrite=True)

