#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:23:39 2020

@author: sonic
"""

#%%
import time
import sfdmap
import os, glob
import extinction
import numpy as np
from numpy import median
import astropy.units as u
from gppy.phot import phot
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from mastapi.queryps1 import ps1cone #https://ps1images.stsci.edu/ps1_dr2_api.html
from astropy.coordinates import SkyCoord  # High-level coordinates
from gppy.util.changehdr import changehdr as chdr # e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles

path        = '~/RASA36/dark_frames'
os.chdir(os.path.expanduser(path))
os.getcwd()

alldat      = glob.glob('*')

dark, flat, bias, light     = [], [], [], []

for dat in alldat:
    if dat.split('.')[0][-4:]=='dark':
        dark.append(dat)
    elif dat.split('.')[0][-4:]=='flat':
        flat.append(dat)
    elif dat.split('.')[0][-4:]=='bias':
        bias.append(dat)
    else:
        light.append(dat)

dark.sort()
flat.sort()
bias.sort()
light.sort()
#%%
fits_file = get_pkg_data_filename(dark[0])
fits.info(fits_file)

#fits_image_filename = fits.util.get_testdata_filepath('test.fits')

hdul = fits.open(dark[0])

data = hdul[0].data
mean    = np.mean(data)
hd = fits.getheader(dark[0], ext = 0)
hd['EXPTIME']

meanv   = []
for df in dark:
    if fits.getheader(df, ext=0)['EXPTIME'] == 10.0:
        pixval  = np.mean(fits.open(df)[0].data)
        exp     = fits.getheader(df)['EXPTIME']
        meanv.append(pixval)
    #%%
plt.figure(figsize=(6,6))
plt.scatter(np.arange(len(meanv))+1, meanv)
plt.xlim(1,10)
plt.ylim(0,200)
plt.title('The Tendency of Dark Frames after Light Exposure')
plt.xlabel('Number of Frames')
plt.ylabel('Mean Value of 10sec Dark Frames [count]')
plt.show()

#%%
#%%
os.chdir('/home/sonic/research/observation/RASA36')
a = ascii.read('image.dat')
b = ascii.read('image_single.dat')

depth_hdr = []
depth_high = []
seeing_hdr = []
seeing_high = []

#single
depth_hdrs = []
depth_highs = []
seeing_hdrs = []
seeing_highs = []


hdr = '0503 0504 0506 0507 0508 0510 0511 0512 0513 0514 0517 0518 0601 0602 0603 0607 0609 0610 0613 0615 0617 0618 0630 0702 0704 0705 0706 0708 0709 0710 0711 0712 0713 0715 0716 0719 0720'
# high = ''
# hdr = '0503 0504 0504 0505 0506 0508 0509 0510 0511 0512 0515 0516 0530 0531 0601 0602 0605 0607 0608 0611 0613 0615 0616 0628 0630 0702'.split()
# high = '0426 0427 0428 0429 0523 0524 0525 0527 0529 0619 0621 0626 0627'.split()

for i in range(len(a)):
    if '720.com.' in a['col1'][i]:
        if a['col1'][i].split('-')[3][-4:] in hdr:
            seeing_hdr.append(a['col3'][i])
            if a['col3'][i] < 12:
                depth_hdr.append(a['col2'][i])
        else:
        # elif a['col1'][i].split('-')[3][-4:] in high:
            seeing_high.append(a['col3'][i])
            if a['col3'][i] < 12:
                depth_high.append(a['col2'][i])
                
for j in range(len(b)):
    if b['image'][j].split('-')[3][-4:] in hdr:
        depth_hdrs.append(b['depth'][j])
        seeing_hdrs.append(b['seeing'][j])
    else:
        depth_highs.append(b['depth'][j])
        seeing_highs.append(b['seeing'][j])
        
#%%
plt.figure(figsize=(14,7))
plt.suptitle(r'RASA36 Combined Image Seeing Distribution (exptime=60s$\times$12)')
plt.subplot(121)
plt.title('HDR mode (n={})'.format(len(seeing_hdr)))
hdrsweight = np.ones_like(seeing_hdr)/len(seeing_hdr)
plt.hist(seeing_hdr, weights=hdrsweight, bins=np.arange(0,22,1.0), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(seeing_hdr), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'Seeing [arcsec]')
plt.ylabel('Proportion')
plt.xlim(0,21)
plt.ylim(0,0.25)
plt.grid(alpha=0.5, ls='--')
plt.subplot(122)
plt.title('High Gain Only (n={})'.format(len(seeing_high)))
highsweight = np.ones_like(seeing_high)/len(seeing_high)
plt.hist(seeing_high, weights=highsweight, bins=np.arange(0,22,1.0), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(seeing_high), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'Seeing [arcsec]')
plt.xlim(0,21)
plt.ylim(0,0.25)
plt.grid(alpha=0.5, ls='--')
plt.savefig('com_seeing.png')
#%%
plt.figure(figsize=(14,7))
plt.suptitle(r'RASA36 Combined Image Depth Distribution (exptime=60s$\times$12, seeing<12")')
plt.subplot(121)
plt.title('HDR mode (n={})'.format(len(depth_hdr)))
hdrweight = np.ones_like(depth_hdr)/len(depth_hdr)
plt.hist(depth_hdr, weights=hdrweight, bins=np.arange(0,20.5,0.5), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(depth_hdr), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'$5\sigma$ limiting magnitude')
plt.ylabel('Proportion')
plt.xlim(13.5,20.5)
plt.ylim(0,0.50)
plt.grid(alpha=0.5, ls='--')
plt.subplot(122)
plt.title('High Gain Only (n={})'.format(len(depth_high)))
highweight = np.ones_like(depth_high)/len(depth_high)
plt.hist(depth_high, weights=highweight, bins=np.arange(0,20.5,0.5), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(depth_high), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'$5\sigma$ limiting magnitude')
plt.xlim(13.5,20.5)
plt.ylim(0,0.50)
plt.grid(alpha=0.5, ls='--')
plt.savefig('com_depth.png')
#%%
plt.figure(figsize=(14,7))
plt.suptitle(r'RASA36 Single Image Seeing Distribution (exptime=60s)')
plt.subplot(121)
plt.title('HDR mode (n={})'.format(len(seeing_hdrs)))
hdrsweight = np.ones_like(seeing_hdrs)/len(seeing_hdrs)
plt.hist(seeing_hdrs, weights=hdrsweight, bins=np.arange(0,22,1.0), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(seeing_hdrs), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'Seeing [arcsec]')
plt.ylabel('Proportion')
plt.xlim(0,21)
plt.ylim(0,0.25)
plt.grid(alpha=0.5, ls='--')
plt.subplot(122)
plt.title('High Gain Only (n={})'.format(len(seeing_highs)))
highsweight = np.ones_like(seeing_highs)/len(seeing_highs)
plt.hist(seeing_highs, weights=highsweight, bins=np.arange(0,22,1.0), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(seeing_highs), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'Seeing [arcsec]')
plt.xlim(0,21)
plt.ylim(0,0.25)
plt.grid(alpha=0.5, ls='--')
plt.savefig('sing_seeing.png')
#%%
plt.figure(figsize=(14,7))
plt.suptitle(r'RASA36 Single Image Depth Distribution (exptime=60s)')
plt.subplot(121)
plt.title('HDR mode (n={})'.format(len(depth_hdrs)))
hdrweights = np.ones_like(depth_hdrs)/len(depth_hdrs)
plt.hist(depth_hdrs, weights=hdrweights, bins=np.arange(0,20.5,0.5), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(depth_hdrs), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'$5\sigma$ limiting magnitude')
plt.ylabel('Proportion')
plt.xlim(13.5,20.5)
plt.ylim(0,0.50)
plt.grid(alpha=0.5, ls='--')
plt.subplot(122)
plt.title('High Gain Only (n={})'.format(len(depth_highs)))
highweights = np.ones_like(depth_highs)/len(depth_highs)
plt.hist(depth_highs, weights=highweights, bins=np.arange(0,20.5,0.5), color='dodgerblue', edgecolor='skyblue')
plt.axvline(np.median(depth_highs), c='crimson', ls='--', linewidth=2.5)
plt.xlabel(r'$5\sigma$ limiting magnitude')
plt.xlim(13.5,20.5)
plt.ylim(0,0.50)
plt.grid(alpha=0.5, ls='--')
plt.savefig('sing_depth.png')
#%%
bad     = []
good    = []

for i in range(len(b)):
    if b['image'][i].split('-')[3][-4:] in hdr:
        if b['seeing'][i] > 12:
            bad.append(b['image'][i].split('-')[3][-4:])
        else:
            good.append(b['image'][i].split('-')[3][-4:])

days    = list(set(bad+good)); days.sort()

aimgcount   = []
bimgcount   = []
gimgcount   = []

for day in days:
    aimgcount.append(good.count(day)+bad.count(day))
    bimgcount.append(bad.count(day))
    gimgcount.append(good.count(day))
    
#%%
plt.figure(figsize=(20,14))
plt.title('RASA36 Image Count (HDR mode, 60sec single)')
plt.bar(days, aimgcount, color='dodgerblue', label='All images')
plt.bar(days, bimgcount, color='crimson', label='seeing>12" images')
# plt.bar(days, gimgcount)
# plt.hist(good, alpha=0.7)
plt.ylabel('Counts')
plt.legend()
plt.xticks(days, rotation=90)
plt.savefig('bad_seeing_image_hdr.png')        