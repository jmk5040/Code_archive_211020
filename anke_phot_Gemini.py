#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Writer: Mankeun Jeong (Ankemun)

Description:
A photometry code for GMOS-N or GMOS-S images from Gemini Data Archive.
This code is especially for extended sources photometry. (returning Kron-like magnitude in ABmag system)
You should download image files for certain targets and save them to image directory.  
This code is compatible with ankephot.py.

How to use:
environments - astropy 4, dragons packages from https://github.com/GeminiDRSoftware/DRAGONS
input - images, dustmap module, observatory table, project target list (name, ra, dec),
output - Result table

References:
- gppy.py (Gregory S. H. Paek)
- ankephot.py
- anketool.py
"""
#%% modules
import time
import sfdmap #DPM
import os, glob
import extinction #DPM
import numpy as np
import pandas as pd
import astropy.units as u
from gppy.phot import phot #DPM
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from ankepy.phot import anketool as atl
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, hstack
from gppy.util.changehdr import changehdr as chdr #DPM e.g. chdr(img, where, what)

#%% target list (name, ra, dec)
obs         = 'Gemini'
# Data Source

path_base   = '/home/sonic/research/photometry/'
path_dust   = '/home/sonic/research/photometry/sfdmap/'
path_cfg    = '/home/sonic/research/photometry/config/'
path_img    = '/home/sonic/research/photometry/image/{}/'.format(obs)
path_cat    = '/home/sonic/research/photometry/cat/{}/'.format(obs)
path_result = '/home/sonic/research/photometry/result/{}/'.format(obs)

os.chdir(path_base)
# project target list input
#print('Enter the name of running project (projectname.txt file should be in your project directory): ')
#project     = input()
project     = 'alltargets'

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

#%% main: GMOS data photometry
#%% Choosing the target
# Image download: https://archive.gemini.edu/

os.chdir(path_img)
allims = glob.glob('*'); allims.sort()

# choosing the target & check the image list 
print('='*5+'Target List'+'='*5)
for ims in allims: print("{}".format(ims))
print('='*21+'\n')

#target  = 'GRB050509B'
target  = input('Input the target to process:\t')
ra      = info[info['Target'] == target]['RA'][0]
dec     = info[info['Target'] == target]['Dec'][0]
if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)

#%%
os.chdir(target)
allgem  = glob.glob('*.fits'); allgem.sort()
print('='*5+'Image List'+'='*6)
for gem in allgem: print(gem)
print('='*21+'\n')
img     = input("Select the image to process:\t")
imlist  = glob.glob('*.fits'); imlist.sort()

#%%

#image check
hdul    = fits.open(img)
hdr0    = hdul[0].header
hdr1    = hdul[1].header

instr   = hdr0['INSTRUME']
# ftr     = hdr0['FILTER1'][0]
ftr     = hdr0['FILTER2'][0]
pixscale= hdr1['PIXSCALE']
# gain    = hdr1['GAIN']
# exptime = int(img.split('_')[-1][:2])
gain    = 2/3*hdr1['GAIN']*hdr0['NCOMBINE']
exptime = hdr0['NCOMBINE']*hdr0['EXPOSURE']

xscale  = hdr1['NAXIS1'] * pixscale # arcsec
yscale  = hdr1['NAXIS2'] * pixscale # arcsec
radius  = np.mean([xscale/2, yscale/2])/3600 # searching radius in deg


# ABmag zeropoint

# reftbls    = atl.sdss_query(target, ra, dec, radius)
reftbls    = atl.ps1_query(target, ra, dec, radius)   
# reftbls    = atl.des_query(ra, dec, radius)

#cfg opt
fwhm        = 0.5
bkgsize     = 64
thres       = 2
detarea     = 10
# config
cfg         = path_cfg+'ankephot.sex'
param       = path_cfg+'ankephot.param'
conv        = path_cfg+'ankephot.conv'
nnw         = path_cfg+'ankephot.nnw'    
# output name
catname     = path_cat+instr+'_'+ftr+'_'+target+'.cat'
segname     = instr+'_'+ftr+'_'+target+'.seg'
apername    = instr+'_'+ftr+'_'+target+'.aper'
bkgname     = instr+'_'+ftr+'_'+target+'.bkg'
# prompt
prompt_cat  = ' -CATALOG_NAME '+catname
prompt_cfg  = ' -c '+cfg+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+conv+' -STARNNW_NAME '+nnw
prompt_opt  = ' -GAIN %.2f'%(gain)+' -PIXEL_SCALE %.2f'%(pixscale)+' -SEEING_FWHM %.2f'%(fwhm)+' -DETECT_THRESH %.2f'%(thres)+' -ANALYSIS_THRESH %.2f'%(thres)
prompt_bkg  = ' -BACK_SIZE %d'%(bkgsize)+' -DETECT_MINAREA %d'%(detarea)
prompt_chk  = ' -CHECKIMAGE_TYPE SEGMENTATION,APERTURES,BACKGROUND -CHECKIMAGE_NAME '+segname+','+apername+','+bkgname
prompt      = 'sex '+img+'[1]'+prompt_cfg+prompt_opt+prompt_cat+prompt_bkg+prompt_chk
os.system(prompt)
    
intbls       = ascii.read(path_cat+instr+'_'+ftr+'_'+target+'.cat')   

# matching
param_match = dict(intbl = intbls, 
                   reftbl = reftbls,
                   inra=intbls['ALPHA_J2000'], 
                   indec=intbls['DELTA_J2000'],
                   refra=reftbls['RA_ICRS'], 
                   refdec=reftbls['DE_ICRS'],
                   sep=2)
mtbl        = phot.matching(**param_match)
    
# calculation
aperture    = 'MAG_AUTO'
inmagkey	= 'MAG_AUTO'
inmagerkey	= 'MAGERR_AUTO'
refmagkey   = '{}Kmag'.format(ftr)
refmagerkey = 'e_{}Kmag'.format(ftr)
# refmagkey   = '{}MeanKronMag'.format(ftr)
# refmagerkey = ftr+'MeanKronMagErr'

param_st4zp	= dict(intbl=mtbl,
                   inmagerkey=aperture,
                   refmagkey=refmagkey,
                   refmagerkey=refmagerkey,
                   refmaglower=14, # saturation limit
                   refmagupper=21, # depth limit
                   refmagerupper=0.05,
                   inmagerupper=0.05)
param_zpcal	= dict(intbl=phot.star4zp(**param_st4zp),
                   inmagkey=inmagkey, 
                   inmagerkey=inmagerkey,
                   refmagkey=refmagkey, 
                   refmagerkey=refmagerkey,
                   sigma=3.0)
zp, zper, otbl, xtbl	= phot.zpcal(**param_zpcal)    

# plot (you can skip this part)
outname    = path_result+'{}_{}.zpcal.png'.format(ftr, target)
phot.zpplot(outname=outname,
            otbl=otbl, xtbl=xtbl,
            inmagkey=inmagkey, inmagerkey=inmagerkey,
            refmagkey=refmagkey, refmagerkey=refmagerkey,
            zp=zp, zper=zper)
param_plot  = dict(inim = img, numb_list = otbl['NUMBER'], xim_list = otbl['X_IMAGE'], yim_list = otbl['Y_IMAGE'], add = True, numb_addlist = xtbl['NUMBER'], xim_addlist = xtbl['X_IMAGE'], yim_addlist = xtbl['Y_IMAGE'])
try:
    phot.plotshow(**param_plot)
except:
    print('FAIL TO DRAW ZEROPOINT GRAPH')
    pass

dlist       = [atl.sqsum([((intbls['ALPHA_J2000'][i]-ra)*np.cos(intbls['DELTA_J2000'][i])),(intbls['DELTA_J2000'][i]-dec)]) for i in range(len(intbls['NUMBER']))]
tarlabel    = np.where(np.array(dlist) == min(dlist))[0][0]
tarxwin         = intbls['X_IMAGE'][tarlabel]
tarywin         = intbls['Y_IMAGE'][tarlabel]

#    # galactic extinction correction
lambmean    = {'g':4849.11, 'r':6289.22, 'i':7534.96, 'z':8674.20, 'y':9627.79} # lambda_ref
wave        = np.array([lambmean[ftr]])
ebv         = sfdmap.ebv(ra, dec, mapdir=path_dust)
ext         = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
        
target_mag      = atl.r2(intbls['MAG_AUTO'][tarlabel] + zp - ext)
syserr          = 0
# syserr          = atl.eJy2AB(atl.AB2Jy(target_mag), 0.03*atl.AB2Jy(target_mag))
target_magerr   = atl.r2(atl.sqsum([intbls['MAGERR_AUTO'][tarlabel], zper, syserr]))
#target_magerr   = atl.r2(intbls['MAGERR_AUTO'][tarlabel])
flags           = intbls['FLAGS'][tarlabel]
classtar        = intbls['CLASS_STAR'][tarlabel]
sep             = atl.r3(min(dlist)*3600)
    
print("""
      target    = {}
      mag       = {}
      magerr    = {}
      flags     = {}
      class_star= {}
      tarnum    = {}
      tarx      = {}
      tary      = {}
      sep       = {}
      """.format(img, target_mag, target_magerr, flags, classtar, tarlabel+1, tarxwin, tarywin, sep))
