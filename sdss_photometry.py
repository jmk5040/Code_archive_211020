#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:58:39 2020

@author: sonic
"""
#%%
import time
import sfdmap
import os, glob
import extinction
import numpy as np
import pandas as pd
from numpy import median
import astropy.units as u
from gppy.phot import phot
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord  # High-level coordinates
from gppy.util.changehdr import changehdr as chdr # e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
#%%
def matching(intbl, reftbl, inra, indec, refra, refdec, sep=2.0):
	"""
	MATCHING TWO CATALOG WITH RA, Dec COORD.
	INPUT   :   SE catalog, SDSS catalog file name, sepertation [arcsec]
	OUTPUT  :   MATCED CATALOG FILE & TABLE
	"""
	incoord		= SkyCoord(inra, indec, unit=(u.deg, u.deg))
	refcoord	= SkyCoord(refra, refdec, unit=(u.deg, u.deg))

	#   INDEX FOR REF.TABLE
	indx, d2d, d3d  = incoord.match_to_catalog_sky(refcoord)
	mreftbl			= reftbl[indx]
	mreftbl['sep']	= d2d
	mergetbl		= intbl
    
	for col in mreftbl.colnames:
		mergetbl[col]	= mreftbl[col]
	indx_sep		= np.where(mergetbl['sep']*3600. < sep)
	mtbl			= mergetbl[indx_sep]
	#mtbl.write(mergename, format='ascii', overwrite=True)
    
	return mtbl

def sdss_query(target, ra, dec, radius, path='cat/'): # ra [deg], dec [deg], radius [deg]
    Vizier.ROW_LIMIT    = -1
    query       = Vizier.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), width=str(radius*60)+'m', catalog=['SDSS12'])
    querycat    = query[0]
    outname     = 'sdss_'+target+'.cat'
    querycat.write(path+outname, format='ascii', overwrite=True)
    return querycat

def sdss_photometry(inim):
    
    #1. image file & header info open
    
    file    = get_pkg_data_filename(inim)
    hdul    = fits.open(file)
    hdr     = hdul[0].header
    
    pixscale= 0.396
    ftr     = hdr['FILTER']
    xscale  = hdr['NAXIS1'] * pixscale # arcsec
    yscale  = hdr['NAXIS2'] * pixscale # arcsec    
    radius  = np.sqrt((xscale/2)**2 + (yscale/2)**2)/3600 # searching radius
    
    #2. image identification
        
    sdss    = ascii.read('searchlist2.cat') # modified sdss output including 'target', 'run', 'ra', 'dec', etc. (need to be simplified) 
    
    run     = int(inim.split('-')[2][2:])
    camcol  = int(inim.split('-')[3])
    target  = sdss[sdss['run']==run]['target'][0]
    ra      = sdss[sdss['run']==run]['ra'][0]
    dec     = sdss[sdss['run']==run]['dec'][0]

    gaintbl = Table([[1.62, 1.825, 1.59, 1.6, 1.47, 2.17],
                     [3.32, 3.855, 3.845, 3.995, 4.05, 4.035],
                     [4.71, 4.6, 4.72, 4.76, 4.725, 4.895],
                     [5.165, 6.565, 4.86, 4.885, 4.64, 4.76],
                     [4.745, 5.155, 4.885, 4.775, 3.48, 4.69]], names = ['u', 'g', 'r', 'i', 'z'])

    #3. SExtractor run
    
    try:
        intbl   = ascii.read('cat/'+ftr+'_'+target+'.cat')
    
    except:
        
        gain    = gaintbl[ftr][camcol-1]
        
        configfile  = 'default.sex' # configuration files should be in the same directory with the image file
        catname = 'cat/'+ftr+'_'+target+'.cat'
        apopt   = ' -GAIN '+'%.3f'%(gain)+' -PIXEL_SCALE '+'%.3f'%(pixscale)
        prompt  = 'sex -c '+configfile+' '+inim+' -CATALOG_NAME '+catname+' '+apopt	
        os.system(prompt)

        intbl   = ascii.read('cat/'+ftr+'_'+target+'.cat')
        
    #4. SDSS catalog query

    reftbl  = sdss_query(target, ra, dec, radius)
    
    #5. matching
    
    param_match     = dict(intbl = intbl, reftbl = reftbl, 
                           inra=intbl['ALPHA_J2000'], indec=intbl['DELTA_J2000'], 
                           refra=reftbl['RA_ICRS'], refdec=reftbl['DE_ICRS'], 
                           sep=0.3)
    mtbl    = matching(**param_match)
    
    #6. quality cut
    
    stbl    = mtbl[mtbl['CLASS_STAR'] > 0.5]
    stbl    = stbl[stbl['FLAGS']==0]
    stbl    = stbl[stbl['e_'+ftr+'mag'] < 0.1]
    
    #7. zero-point calculation
    
    zptbl           = stbl[ftr+'mag'] - stbl['MAG_AUTO']
    zetbl           = phot.sqsum(stbl['e_'+ftr+'mag'], stbl['MAGERR_AUTO'])
    zptbl.name      = 'zp'
    zetbl.name      = 'zper'

    dummytbl        = hstack([Table(), zptbl])

    zplist_clip     = sigma_clip(zptbl, sigma=3, maxiters=None, cenfunc=median, copy=False) # 3-sigma clipping
    indx_alive      = np.where(zplist_clip.mask == False)
    indx_exile      = np.where(zplist_clip.mask == True)

    intbl_alive     = dummytbl[indx_alive]
    intbl_exile     = dummytbl[indx_exile]

    Zp              = np.median(np.copy(intbl_alive['zp']))
    Zper			= np.std(np.copy(intbl_alive['zp']))/np.sqrt(len(intbl_alive))
    
    #8. target inspection
    
    dlist           = [np.sqrt(((intbl['ALPHA_J2000'][i]-ra)*np.cos(intbl['DELTA_J2000'][i]))**2 + (intbl['DELTA_J2000'][i]-dec)**2) for i in range(len(intbl['NUMBER']))]

    target_label    = np.where(np.array(dlist) == min(dlist))[0][0]
    if min(dlist) * 3600 > 2:
        false   = 1
    else:
        false   = 0
        
    #9. galactic extinction correction

    lambmean    = {'u':3561.8, 'g':4718.9, 'r':6185.2, 'i':7499.7, 'z':8961.5} # in angstrom
    wave        = np.array([lambmean[ftr]])
    dustmap     = '../../photometry/sfdmap/'

    ebv = sfdmap.ebv(ra, dec, mapdir=dustmap)
    ext = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
    
    target_mag      = intbl['MAG_AUTO'][target_label] + Zp - ext
    target_mager    = phot.sqsum(intbl['MAGERR_AUTO'][target_label], Zper)
    
    result  = Table([[target], [ftr], [round(target_mag, 3)], [round(target_mager,3)], [round(Zp,3)], [round(Zper,3)], [false]], names=['target', 'filter', 'MAG_AUTO', 'MAGERR_AUTO', 'zp', 'zper', 'false_target'])
    
    return result
#%%
path        = '~/research/data/sdss/' 
os.chdir(os.path.expanduser(path))
os.system('mkdir config cat result')
os.getcwd()

sdss        = ascii.read('searchlist2.cat')
#%%
inim    = glob.glob('img/*.fits')
inim.sort()
result      = sdss_photometry(inim[0])
for img in inim:
    result  = vstack([result, sdss_photometry(img)])
result.write('result/result.cat', format='ascii', overwrite=True)
#%%
alltbl  = ascii.read('result/result.cat')
#newtbl  = Table([['samples'], ['0.0%'] ,[0.0] ,[0.0], [0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0]], names=['target', 'false', 'umag', 'uerr', 'gmag', 'gerr', 'rmag', 'rerr', 'imag', 'ierr', 'zmag', 'zerr'])
newtbl  = Table([['samples'] ,[0.0] ,[0.0], [0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0]], names=['target', 'umag', 'uerr', 'gmag', 'gerr', 'rmag', 'rerr', 'imag', 'ierr', 'zmag', 'zerr'])

targets = list(set(alltbl['target']))
targets.sort()
for i in range(len(targets)):
    calltbl = alltbl[alltbl['target']==targets[i]]
    newrow  = Table([[targets[i]],
                     [calltbl['MAG_AUTO'][3]], [calltbl['MAGERR_AUTO'][3]], 
                     [calltbl['MAG_AUTO'][0]], [calltbl['MAGERR_AUTO'][0]], 
                     [calltbl['MAG_AUTO'][2]], [calltbl['MAGERR_AUTO'][2]],
                     [calltbl['MAG_AUTO'][1]], [calltbl['MAGERR_AUTO'][1]],
                     [calltbl['MAG_AUTO'][4]], [calltbl['MAGERR_AUTO'][4]]], 
                    names=['target', 'umag', 'uerr', 'gmag', 'gerr', 'rmag', 'rerr', 'imag', 'ierr', 'zmag', 'zerr'])
    newtbl  = vstack([newtbl, newrow])
#targets.sort()
#for i in range(len(targets)):
#    calltbl = alltbl[alltbl['target']==targets[i]]
#    newrow  = Table([[targets[i]], [str(np.mean(calltbl['false_target'])*100)+'%'], 
#                     [calltbl['MAG_AUTO'][3]], [calltbl['MAGERR_AUTO'][3]], 
#                     [calltbl['MAG_AUTO'][0]], [calltbl['MAGERR_AUTO'][0]], 
#                     [calltbl['MAG_AUTO'][2]], [calltbl['MAGERR_AUTO'][2]],
#                     [calltbl['MAG_AUTO'][1]], [calltbl['MAGERR_AUTO'][1]],
#                     [calltbl['MAG_AUTO'][4]], [calltbl['MAGERR_AUTO'][4]]], 
#                    names=['target', 'false', 'umag', 'uerr', 'gmag', 'gerr', 'rmag', 'rerr', 'imag', 'ierr', 'zmag', 'zerr'])
#    newtbl  = vstack([newtbl, newrow])
newtbl.write('result/mag.cat', format='ascii', overwrite=True)
#%%
target = '111117A'
Ra  = '00:50:46.26'
Dec = '23:00:40.97'

ra  = SkyCoord(Ra+' '+Dec, unit=(u.hourangle, u.deg)).ra.degree
dec = SkyCoord(Ra+' '+Dec, unit=(u.hourangle, u.deg)).dec.degree

#ftr     = 'g'
#mag     = 24.08
#mager   = 0.09
#ftr     = 'r'
#mag     = 23.93
#mager   = 0.08
#ftr     = 'R'
#mag     = 23.95
#mager   = 0.23
ftr     = 'K'
mag     = 23.07
mager   = 0.23
#ftr     = 'J'
#mag     = 23.89
#mager   = 0.23
#ftr     = 'H'
#mag     = 23.89
#mager   = 0.23
#ftr     = 'K'
#mag     = 23.89
#mager   = 0.23


lambmean = {'R':6305.1, 'I':8377.6, 'J':12350.0, 'H':16620.0, 'K':21590.0} # in angstrom
wave    = np.array([lambmean[ftr]])
dustmap = '../../photometry/sfdmap/'

ebv = sfdmap.ebv(ra, dec, mapdir=dustmap)
ext = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
    
target_mag      = mag - ext
target_mager    = mager

print(target_mag, target_mager)


