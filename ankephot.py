#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Update: 2020.12.17.
Writer: Mankeun Jeong (Ankemun)

Description:
A photometry code for astronomical image data from survey science projects.
SDSS, PanSTARRS, KS4, DES, DeCaLs and HST are currently available.
Photometry for VST, SkyMapper and WOW data will be prepared in the future.
This code is especially for extended sources photometry. (returning Kron-like magnitude in ABmag system)
You should download image files for certain targets and save them to image directory.  

How to use:
environments - 
input - images, dustmap module, observatory table, project target list (name, ra, dec),
output - Result table

References:
- gppy.py (Gregory S. H. Paek)

"""
#%%
"""
Update Note
- 2020.12.17: New Target [GRB160624A]
- 2020.12.30: DECaLS photometry updated
- 2021.02.08: New Target [GRB201221D], Excluded target [GRB150424A, GRB061217]

TTD
- Separate HST & Gemini
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
path_table  = '/home/sonic/research/photometry/table'

os.chdir(os.path.expanduser(path_base))

# required directories
current     = glob.glob('*')
require     = 'cat code config image project result sfdmap table'.split(' ')

for folder in require:
    if folder not in current:
        os.system('mkdir {}'.format(folder))
    else:
        pass

#%% survey science information

os.chdir(path_base)
# surveys in concern
chart       = 'survey'
#chart       = input()
obstable    = ascii.read('table/{}.txt'.format(chart))
surveys     = obstable['Survey']

# making directories
os.chdir(path_img)
imgsur  = glob.glob('*')
os.chdir(path_cat)
catsur  = glob.glob('*')
os.chdir(path_result)
ressur  = glob.glob('*')

for survey in surveys:
    if survey not in imgsur:
        os.chdir(path_img)
        os.system('mkdir {}'.format(survey))
    if survey not in catsur:
        os.chdir(path_cat)
        os.system('mkdir {}'.format(survey))
    if survey not in ressur:
        os.chdir(path_result)
        os.system('mkdir {}'.format(survey))
    else:
        pass

os.chdir(path_base)

#%% target list

os.chdir(path_base)
# project target list input
#print('Enter the name of running project (projectname.txt file should be in your project directory): ')
#project     = input()
project     = 'hogwarts'

os.chdir(path_result)
respro      = glob.glob('*')
if project not in respro:
    os.system('mkdir {}'.format(project))
os.chdir(path_base)

# target confirmation
try:
#    info    = ascii.read('project/{}.txt'.format(project+' (0704)'))
#    info    = ascii.read('project/{}.txt'.format(project+' (0816)'))
    info    = ascii.read('project/{}.txt'.format(project))
    targets = info['Target']
    print('='*42)
    print("Target list is read successfully. \nCheck if the headers are 'Target, RA, Dec'")
    print('='*42)
    print(info)
except:
    print('{}.txt is not found in your project directory.'.format(project))

#%% image directories

# making directories for each target
os.chdir(path_img)

for survey in surveys:
    os.chdir('{}'.format(survey))
    tarlist     = glob.glob('*')
    
    for target in targets:
        if target not in tarlist:
            os.system('mkdir {}'.format(target))
        else:
            pass
        
    os.chdir(path_img)
    
os.chdir(path_base)

#%% file name changer
os.chdir(path_img)
# survey specification
PS1_file    = True
DES_file    = True
DECaLS_file = True

if PS1_file:
    os.chdir('PS1')
    for target in targets:
        ps1imgs     = glob.glob('{}/*.fits'.format(target))
        if len(ps1imgs) != 0:
            if 'cutout' in ps1imgs[0]:
                for img in ps1imgs:
                    atl.namechanger(img, target, 'PS1')
os.chdir(path_img)

if DES_file:
    os.chdir('DES')
    for target in targets:
        desimgs     = glob.glob('{}/*.fits'.format(target))
        if len(desimgs) != 0:
            if '.' in desimgs[0]:
                for img in desimgs:
                    atl.namechanger(img, target, 'DES')
os.chdir(path_img)
                
if DECaLS_file:
    os.chdir('DECaLS')
    for target in targets:
        decalsimgs      = glob.glob('{}/*.fits'.format(target))
        if len(decalsimgs) != 0:
            if ',' in decalsimgs[0]:
                for img in decalsimgs:
                    atl.namechanger(img, target, 'DECaLS')
os.chdir(path_img)

#%% main: survey data photometry (total samples)

os.chdir(path_img)
start   = time.time()
# survey specification
PS1_phot    = False
SDSS_phot   = False
DES_phot    = False
DECaLS_phot = False
# KS4_phot    = False
# VST_phot    = False
# WOW_phot    = False

# PS1 photometry: Cutout images can be downloaded from https://ps1images.stsci.edu/cgi-bin/ps1cutouts (image size: 600" x 600" recommended)
if PS1_phot:
    os.chdir('PS1')
    ps1phot     = Table()
    for target in targets:
        ra      = info[info['Target'] == target]['RA'][0]
        dec     = info[info['Target'] == target]['Dec'][0]
        if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)
        ps1imgs     = glob.glob('{}/*.fits'.format(target)); ps1imgs.sort()
        if len(ps1imgs) != 0:
            ps1params   = dict(imlist       = ps1imgs,
                               target       = target,
                               ra           = ra,
                               dec          = dec,
                               path_cfg     = path_cfg,
                               path_cat     = path_cat,
                               path_dust    = path_dust,
                               path_result  = path_result)
            ps1phot     = vstack([ps1phot, atl.ps1_photometry(**ps1params)])
        else:
            pass
    ps1phot.write(path_result+project+'/PS1.csv', format='csv', overwrite=True)
os.chdir(path_img)

# SDSS photometry: Cutout images can be downloaded from https://dr12.sdss.org/bulkFields/raDecs
if SDSS_phot:
    os.chdir('SDSS')
    sdssphot    = Table()
    for target in targets:
        ra      = info[info['Target'] == target]['RA'][0]
        dec     = info[info['Target'] == target]['Dec'][0]
        if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)
        sdssimgs    = glob.glob('{}/*.fits'.format(target)); sdssimgs.sort()
        if len(sdssimgs) != 0:
            sdssparams  = dict(imlist       = sdssimgs,
                               target       = target,
                               ra           = ra,
                               dec          = dec,
                               path_cfg     = path_cfg,
                               path_cat     = path_cat,
                               path_dust    = path_dust,
                               path_result  = path_result)
            sdssphot     = vstack([sdssphot, atl.sdss_photometry(**sdssparams)])
        else:
            pass
    sdssphot.write(path_result+project+'/SDSS.csv', format='csv', overwrite=True)
os.chdir(path_img)

# DES photometry: Cutout images can be downloaded from https://datalab.noao.edu/sia.php (default size 0.1'x0.1')
if DES_phot:
    os.chdir('DES')
    desphot    = Table()
    for target in targets:
        ra      = info[info['Target'] == target]['RA'][0]
        dec     = info[info['Target'] == target]['Dec'][0]
        if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)
        desimgs     = glob.glob('{}/*.fits'.format(target)); desimgs.sort()
        if len(desimgs) != 0:
            desparams   = dict(imlist       = desimgs,
                               target       = target,
                               ra           = ra,
                               dec          = dec,
                               path_cfg     = path_cfg,
                               path_cat     = path_cat,
                               path_dust    = path_dust,
                               path_result  = path_result)
            desphot     = vstack([desphot, atl.des_photometry(**desparams)])
        else:
            pass
    desphot.write(path_result+project+'/DES.csv', format='csv', overwrite=True)
os.chdir(path_img)

# DECaLS photometry: Cutout images can be downloaded from https://datalab.noao.edu/sia.php (default size 0.1'x0.1')
if DECaLS_phot:
    os.chdir('DECaLS')
    decalsphot    = Table()
    for target in targets:
        ra      = info[info['Target'] == target]['RA'][0]
        dec     = info[info['Target'] == target]['Dec'][0]
        if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)
        decalsimgs      = glob.glob('{}/*.fits'.format(target)); decalsimgs.sort()
        if len(decalsimgs) != 0:
            decalsparams= dict(imlist       = decalsimgs,
                               target       = target,
                               ra           = ra,
                               dec          = dec,
                               path_cfg     = path_cfg,
                               path_cat     = path_cat,
                               path_dust    = path_dust,
                               path_result  = path_result)
            decalsphot  = vstack([decalsphot, atl.decals_photometry(**decalsparams)])
        else:
            pass
    decalsphot.write(path_result+project+'/DECaLS.csv', format='csv', overwrite=True)
os.chdir(path_img)

finish  = time.time()

time.sleep(2)
print("="*70)
print("Image data photometry for [{}] project has finished. Total running time: {} sec".format(project, round(finish-start,1)))
print("="*70)

os.chdir(path_base)

#%% sub: Survey data photometry (single target)

os.chdir(path_img)

print("="*15)
print("Target List")
print("="*15)
for target in targets:
    print(target)
print("="*15+"\n")
target  = input("Target to process:\t")

print("Image List for the Target")

for survey in surveys:
    os.chdir(survey)
    os.chdir(target)
    
    imgs    = glob.glob("*.fits")
    imgs.sort()
    if len(imgs) != 0:
        print("="*10)
        print(survey)
        print("="*10)
        for img in imgs:
            print(img)
        print("="*10)
        
    os.chdir(path_img)

survey  = input("Survey to process:\t")
os.chdir(survey)
ra      = info[info['Target'] == target]['RA'][0]
dec     = info[info['Target'] == target]['Dec'][0]

if type(ra) == np.str_ and type(dec) == np.str_:
    ra, dec     = atl.wcs2deg(ra, dec)
imgs        = glob.glob('{}/*.fits'.format(target))
imgs.sort()

# settings
refband     = 'r'
params      = dict(imlist       = imgs,
                   target       = target,
                   ra           = ra,
                   dec          = dec,
                   path_cfg     = path_cfg,
                   path_cat     = path_cat,
                   path_dust    = path_dust,
                   path_result  = path_result,
                   refband      = refband,
                   fwhm         = 2,
                   bkgsize      = 128,
                   catmatch     = False,
                   thres        = 2)

# saving
if survey == 'PS1':
    newtbl      = atl.ps1_photometry(**params)
    if refband  == 'none':
        photbl  = ascii.read(path_result+project+'/ps1_single.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/ps1_single.csv', format='csv', overwrite=True)
    else:
        photbl  = ascii.read(path_result+project+'/ps1_dual.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/ps1_dual.csv', format='csv', overwrite=True)
elif survey == 'DES':
    newtbl      = atl.des_photometry(**params)
    if refband  == 'none':
        photbl  = ascii.read(path_result+project+'/des_single.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/des_single.csv', format='csv', overwrite=True)
    else:
        photbl  = ascii.read(path_result+project+'/des_dual.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/des_dual.csv', format='csv', overwrite=True)
elif survey == 'DECaLS':
    newtbl      = atl.decals_photometry(**params)
    if refband  == 'none':
        photbl  = ascii.read(path_result+project+'/decals_single.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/decals_single.csv', format='csv', overwrite=True)
    else:
        photbl  = ascii.read(path_result+project+'/decals_dual.csv')
        photbl  = vstack([photbl, newtbl])
        photbl.write(path_result+project+'/decals_dual.csv', format='csv', overwrite=True)

time.sleep(2)
for i in range(len(photbl.colnames)):
    print(photbl.colnames[i],'\t',photbl[photbl.colnames[i]][-1])

os.chdir(path_img)

#%% sub1: HST data photometry
# directory to HST data
# download data from Hubble Legacy Survey Archive or HST MAST

os.chdir(path_img)
os.chdir('HST')

allhst  = glob.glob('*'); allhst.sort()
print('='*5+'Target List'+'='*5)
for hst in allhst: print(hst)
print('='*21+'\n')

#target  = 'GRB050509B'
target  = input('Input the target to process:\t')
ra      = info[info['Target'] == target]['RA'][0]
dec     = info[info['Target'] == target]['Dec'][0]
if type(ra) == np.str_ and type(dec) == np.str_:
            ra, dec     = atl.wcs2deg(ra, dec)
            
os.chdir(target)
allimg  = glob.glob('*.fits'); allimg.sort()
print('='*5+'Image List'+'='*6)
for img in allimg: print(img)
# imlist  = glob.glob('*.fits')
# imlist.sort()
print('='*21+'\n')

img     = input("Select the image to process:\t")
#%%
#image check
hdul    = fits.open(img)
hdr0    = hdul[0].header
hdr1    = hdul[1].header

instr   = hdr0['INSTRUME'] # ACS, WFPC2

if instr=='ACS':
    detect  = hdr0['DETECTOR'] # WFC, HRC, SBC
    if 'CLEAR' not in hdr0['FILTER1']:
        ftr = hdr0['FILTER1']
    else:
        ftr = hdr0['FILTER2']
    gain    = hdr0['CCDGAIN'] * hdr0['EXPTIME']
    
elif instr=='WFPC2':
    ftr     = hdr0['FILTNAM1']
    gain    = hdr0['ATODGAIN'] * hdr0['EXPTIME']

elif instr=='WFC3':
    ftr     = hdr0['FILTER']
    gain    = hdr0['CCDGAIN'] * hdr0['EXPTIME']
    detect  = hdr0['DETECTOR']


pixscale= hdr0['D001SCAL']

# file name change
if target not in img:
    os.system("mv {} {}_{}_{}.fits".format(img, instr, ftr, target))
    img     = "{}_{}_{}.fits".format(instr, ftr, target)
else:
    pass

# ABmag zeropoint
try:
    photflam= hdr1['PHOTFLAM']
    photplam= hdr1['PHOTPLAM']
    photzpt = hdr1['PHOTZPT']
except:
    photflam= hdr0['PHOTFLAM']
    photplam= hdr0['PHOTPLAM']
    photzpt = hdr0['PHOTZPT']

zp      = -2.5*np.log10(photflam)+photzpt-5*np.log10(photplam)+18.6921
zper    = 0.0

#cfg opt
if detect=='IR':
    fwhm    = 0.3
    bkgsize = 128
    thres   = 5
else:
    fwhm    = 0.2
    bkgsize = 128
    thres   = 1
detarea     = 50

# config
cfg         = path_cfg+'ankephot.sex'
param       = path_cfg+'ankephot.param'
conv        = path_cfg+'ankephot.conv'
nnw         = path_cfg+'ankephot.nnw'    
# output name
catname     = path_cat+'HST/'+instr+'_'+ftr+'_'+target+'.cat'
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
    
intbls       = ascii.read(path_cat+'HST/'+instr+'_'+ftr+'_'+target+'.cat')   

dlist       = [atl.sqsum([((intbls['ALPHA_J2000'][i]-ra)*np.cos(intbls['DELTA_J2000'][i])),(intbls['DELTA_J2000'][i]-dec)]) for i in range(len(intbls['NUMBER']))]
tarlabel    = np.where(np.array(dlist) == min(dlist))[0][0]
tarxwin         = intbls['X_IMAGE'][tarlabel]
tarywin         = intbls['Y_IMAGE'][tarlabel]

# wavelength check
with open('{}/FILTER.RES.latest.info'.format(path_table), 'r') as f:
        t   = f.readlines()
        for l in t:
            if instr.lower() in l.lower():
                if ftr.lower() in l.lower():
                    wav = float(l.split('lambda_c=')[-1].strip().split(' ')[0])
f.close()

#    # galactic extinction correction
# lambmean    = {'F450W':4556.0, 'F606W': 5921.94, 'F814W': 8044.99, 'F160W':15369.18} # ACS_WFC, lamb_ref
# lambmean    = {'F475W': 4772.11, 'F606W': 5886.47, 'F814W': 8052.52, 'F125W': 12486.07, 'F160W': 15370.35} # WFC3, lamb_ref
wave        = np.array([wav])
ebv         = sfdmap.ebv(ra, dec, mapdir=path_dust)
ext         = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
        
target_mag      = atl.r2(intbls['MAG_AUTO'][tarlabel] + zp - ext)
syserr          = atl.eJy2AB(atl.AB2Jy(target_mag), 0.03*atl.AB2Jy(target_mag))
target_magerr   = atl.r2(atl.sqsum([intbls['MAGERR_AUTO'][tarlabel], zper, syserr]))
#target_magerr   = atl.r2(intbls['MAGERR_AUTO'][tarlabel])
flags           = intbls['FLAGS'][tarlabel]
classtar        = intbls['CLASS_STAR'][tarlabel]
sep             = atl.r3(min(dlist)*3600)

time.sleep(2)

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
      bkgsize   = {}
      """.format(img, target_mag, target_magerr, flags, classtar, tarlabel+1, tarxwin, tarywin, sep, bkgsize))
    
#    # target mag & magerr
#    if intbls['MAG_AUTO'][tarlabel] != 99.0:
#        target_mag      = intbls['MAG_AUTO'][tarlabel] + zp - ext
#        syserr          = eJy2AB(AB2Jy(target_mag), 0.03*AB2Jy(target_mag))
#        target_magerr   = sqsum([intbls['MAGERR_AUTO'][tarlabel], zper, syserr])
#        restbl['{}mag'.format(band[0])]     = r2(target_mag)
#        restbl['{}magerr'.format(band[0])]  = r2(target_magerr)
#        restbl['{}_zp'.format(band[0])]     = r2(zp)
#        restbl['{}_zper'.format(band[0])]   = r2(zper)
#            
#    if markimg==imtbl[band][0]:
#        restbl['target']        = restbl['target'].astype(str)
#        restbl['target']        = target
#        restbl['CLASS_STAR']    = intbls['CLASS_STAR'][tarlabel]
#        restbl['FLAGS']         = intbls['FLAGS'][tarlabel]
#        restbl['XWIN']          = tarxwin
#        restbl['YWIN']          = tarywin
#        restbl['sep']           = r3(min(dlist)*3600)
#        
#    restbl.write(path_result+'DES/'+target+'.cat', format='ascii', overwrite=True)
