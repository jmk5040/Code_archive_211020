#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Update: 2020.07.22.
Writer: Mankeun Jeong (Ankemun)

Description:
Some useful functions for ankephot.py
"""
#%%
import time
import copy
import sfdmap #DPM
import os, glob
import extinction #DPM
import numpy as np
import pandas as pd
from numpy import median
import astropy.units as u
from gppy.phot import phot #DPM
from astropy.io import fits
from astropy.io import ascii
from astroquery.ned import Ned
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from gppy.util.changehdr import changehdr as chdr #DPM e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5
#%% sdss
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
def ps1_photometry(inim, target, ra, dec, gain=1.072, pixscale=0.25, detsig=3.0):
    
    """
    input:
        ps1 cutout image
        target information
    output:
        target magnitude in ABmag
        target flux in *FAST default unit
        zero-point of the image

    # This routine is programmed for photometry of single extended source with Kron-like aperture magnitude.
    """
    # 0. Header information
    
    file    = get_pkg_data_filename(inim)
    hdul    = fits.open(file)
    hdr     = hdul[0].header
    
    ftr     = hdr['FILTER']
    xscale  = hdr['NAXIS1'] * pixscale # arcsec
    yscale  = hdr['NAXIS2'] * pixscale # arcsec
    radius  = np.sqrt((xscale**2 + yscale**2)/2)/2/3600 # searching radius
    
    # 1. SExtractor application on the image

    try:

        intbl   = ascii.read('cat/'+ftr+'_'+target+'.cat')
    
    except:
        
        configfile  = 'default.sex' # configuration files should be in the same directory with the image file
        catname = 'cat/'+ftr+'_'+target+'.cat'
        apopt   = ' -GAIN '+'%.2f'%(gain)+' -PIXEL_SCALE '+'%.2f'%(pixscale)
        prompt  = 'sex -c '+configfile+' '+inim+' -CATALOG_NAME '+catname+' '+apopt	
        os.system(prompt)

        intbl   = ascii.read('cat/'+ftr+'_'+target+'.cat')

    # 2. Zero-point calculation
    
    ## 2.1. Loading Pan-STARRS DR2 catalog data
    
    try:

        reftbl  = ascii.read('csv/ps1_'+target+'.csv')
    
    except:

        constraints = {'nDetections.gt':1}
            
        columns = """raMean,decMean,
        gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,yMeanPSFMag,yMeanPSFMagErr,
        gMeanKronMag,gMeanKronMagErr,rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr,zMeanKronMag,zMeanKronMagErr,yMeanKronMag,yMeanKronMagErr""".split(',')
        
        # strip blanks and weed out blank and commented-out values
        
        columns = [x.strip() for x in columns]
        columns = [x for x in columns if x and not x.startswith('#')]

        results = ps1cone(ra, dec, radius, release='dr2', columns=columns, verbose=True, **constraints)

        reftbl = ascii.read(results) # ps1 catalog photometry data around the target
        reftbl.write('csv/ps1_'+target+'.csv', format='csv', overwrite=True)
        
    ## 2.2. Matching the SExtracted sources (intbl) and catalog sources (reftbl) & sampling zeropoints with the magnitude difference 
    
    zplist  = []
    
    for i in range(len(intbl)):

        dist    = []

        for j in range(len(reftbl)):

            dist.append(phot.sqsum((intbl['ALPHA_J2000'][i]-reftbl['raMean'][j])*np.cos(intbl['DELTA_J2000'][i]), intbl['DELTA_J2000'][i]-reftbl['decMean'][j]))

        if min(dist) < 0.5/3600: # The position should be matched in less than 0.5 arcsec

            mlabel      = np.where(np.array(dist)==min(dist))[0][0]
            refmag      = reftbl[ftr+'MeanKronMag'][mlabel]
            inmag       = intbl['MAG_AUTO'][i]
            zeropoint   = refmag - inmag

            if (14 < refmag < 19) and (reftbl['iMeanPSFMag'][mlabel] - reftbl['iMeanKronMag'][mlabel] < 0.05): # (Quality cut) & (Point source determination)
                
                zplist.append(zeropoint)

    ## 2.3. Zero-point & error calculation
    
    dummytbl        = Table()
    dummytbl['zp']  = zplist
    
    zplist_clip     = sigma_clip(zplist, sigma=3, maxiters=None, cenfunc=median, copy=False) # 3-sigma clipping
    indx_alive      = np.where(zplist_clip.mask == False)
    indx_exile      = np.where(zplist_clip.mask == True)
    
    intbl_alive     = dummytbl[indx_alive]
    intbl_exile     = dummytbl[indx_exile]

    Zp              = np.median(np.copy(intbl_alive['zp']))
    Zper			= np.std(np.copy(intbl_alive['zp']))/np.sqrt(len(intbl_alive))
    
    # 3. Identifying the target

    dlist           = [np.sqrt(((intbl['ALPHA_J2000'][i]-ra)*np.cos(intbl['DELTA_J2000'][i]))**2 + (intbl['DELTA_J2000'][i]-dec)**2) for i in range(len(intbl['NUMBER']))]

    target_label    = np.where(np.array(dlist) == min(dlist))[0][0]
    
    target_mag      = intbl['MAG_AUTO'][target_label] + Zp
    target_mager    = phot.sqsum(intbl['MAGERR_AUTO'][target_label], Zper)

    # 4. Calibrating galactic extinction
    
    ## 4.1. Band effective wavelength
    
    lambmean        = {'g':4900.1, 'r':6241.3, 'i':7563.8, 'z':8690.1, 'y':9644.6} # in angstrom
    wave            = np.array([lambmean[ftr]])
    dustmap         = '../../../../photometry/sfdmap/'
    
    ## 4.2. Calculating the extinction (Rv = 3.1, Fitzpatrick99)
    
    ebv             = sfdmap.ebv(ra, dec, mapdir=dustmap)
    ext             = extinction.fitzpatrick99(wave, 3.1 * ebv)

    ## 4.3. Dereddening the magnitude
    
    target_mag      = target_mag - ext
    target_mag      = target_mag[0] # np.array to scalar

    # 5. Converting magnitude to flux unit
    
    target_flux     = AB2Jy(target_mag)
    target_fluxer   = eAB2Jy(target_mag, target_mager)

    return target_mag, target_mager, target_flux, target_fluxer, Zp, Zper
#%%
