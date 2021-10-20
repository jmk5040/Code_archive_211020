#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:50:25 2020

@author: sonic
"""
#%% modules
import copy
import os, glob
import numpy as np
import pandas as pd
from gppy.phot import phot #DPM
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from ankepy.phot import anketool as atl
from astropy.coordinates import SkyCoord
from gppy.util.changehdr import changehdr as chdr #DPM e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.table import Table, vstack, hstack, join, Column
#%% working directories

# pathes
path_base   = '/home/sonic/research/photometry/'
path_img    = '/home/sonic/research/photometry/image/'
path_dust   = '/home/sonic/research/photometry/sfdmap/'
path_cat    = '/home/sonic/research/photometry/cat/'
path_cfg    = '/home/sonic/research/photometry/config/'
path_result = '/home/sonic/research/photometry/result/'

os.chdir(os.path.expanduser(path_base))

#%% survey science information

os.chdir(path_base)
# surveys in concern
chart       = 'survey'
#chart       = input()
obstable    = ascii.read('table/{}.txt'.format(chart))
surveys     = obstable['Survey']

os.chdir(path_base)
#%% target list

os.chdir(path_base)
# project target list input
print('Enter the name of running project (projectname.txt file should be in your project directory): ')
project     = 'hogwarts'
#project     = input()

os.chdir(path_result)
respro      = glob.glob('*')
if project not in respro:
    os.system('mkdir {}'.format(project))
os.chdir(path_base)

# target confirmation
try:
    info    = ascii.read('project/{}.txt'.format(project))
    targets = info['Target']
    print('='*42)
    print("Target list is read successfully. \nCheck if the headers are 'Target, RA, Dec'")
    print('='*42)
    print(info)
except:
    print('{}.txt is not found in your project directory.'.format(project))

#%% image download

os.chdir(path_img)

# SDSS
qtbl    = atl.queryform(info)
print('='*40)
print('Coord of Project Targets for RADEC Search')
print('='*40)
for i in range(len(qtbl)):
#    print(str(qtbl[i][0])+' '+str(qtbl[i][1])+' '+info['Target'][i]) # print with target names
    print(str(qtbl[i][0])+' '+str(qtbl[i][1]))
print('='*40)

os.chdir('SDSS')
try:
    os.system('wget -i {}_sdssquery.txt'.format(project)) # https://dr12.sdss.org/bulkFields/raDecs
    os.system('mv {0}_sdssquery.txt {0}_sdssquery_DONE.txt'.format(project))
    os.system('bzip2 -d *.bz2')
except:
    pass

sdssimgs    = glob.glob('*.fits')
sdssimgs.sort()

for sdssimg in sdssimgs:
    hdul    = fits.open(sdssimg)
    hdr     = hdul[0].header
    
    ra      = hdr['RA']
    dec     = hdr['DEC']
    
    dlist   = [atl.sqsum([((qtbl['ra'][i]-ra)*np.cos(qtbl['dec'][i])),(qtbl['dec'][i]-dec)]) for i in range(len(qtbl))]
    tarlabel= np.where(np.array(dlist) == min(dlist))[0][0]
    target  = info['Target'][tarlabel]
    
    os.system('mv {} {}'.format(sdssimg, target))
    
os.chdir(path_base)
#%%
os.chdir(path_img)

os.chdir('HSC')

hscimgs    = glob.glob('*.fits')
hscimgs.sort()

for hscimg in hscimgs:
    hdul    = fits.open(hscimg)
    hdr     = hdul[0].header
    
    print(hscimg, hdr['TARGNAME'])
    
os.chdir(path_base)
#%%