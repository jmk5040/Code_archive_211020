#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:36:34 2019

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
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from mastapi.queryps1 import ps1cone #https://ps1images.stsci.edu/ps1_dr2_api.html
from astropy.coordinates import SkyCoord  # High-level coordinates
from gppy.util.changehdr import changehdr as chdr # e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
#%% functions
def photometry(inim, target, ra, dec, gain=1.072, pixscale=0.25, detsig=3.0):
    
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

def AB2Jy(ABmag):
    return 10**((25-ABmag)/2.5) # FAST SED FITTING PROGRAM DEFAULT UNIT

def eAB2Jy(ABmag, ABerr):
    return ABerr * np.abs(10**(25/2.5)*(-np.log(10)/2.5)*10**(ABmag/-2.5))

#%% Target List (User Setting)
path        = '~/research/data/PS1/'
os.chdir(os.path.expanduser(path))
os.getcwd()

tarlist     = ascii.read('tarlist2.cat')
tarname     = tarlist['target']
tarRA       = tarlist['RAJ2000']
tarDec      = tarlist['DEJ2000']
tarz        = tarlist['redshift']

gain        = 1.07272           # e-/ADU
pixscale    = 0.25              # arcsec/pixel
detsig      = 3.0
#%% Main Command
start   = time.time()

for indx in range(len(tarlist)):

    # Target Selection
    
    target      = tarname[indx]
    host_RA     = SkyCoord(tarRA[indx]+' '+tarDec[indx], unit=(u.hourangle, u.deg)).ra.degree
    host_Dec    = SkyCoord(tarRA[indx]+' '+tarDec[indx], unit=(u.hourangle, u.deg)).dec.degree
    host_z      = tarz[indx]
    
    # Working Directory
    
    path        = '~/research/data/PS1/'+target+'/sextractor'
    os.chdir(os.path.expanduser(path))
    os.getcwd()

    # Image files

    imlist      = glob.glob('../img/cutout*.fits')
    imlist.sort()

    # Header modification (in case FILTER not defined)

    for i in range(len(imlist)):
        chdr(imlist[i], 'FILTER', imlist[i].split('.')[-3])
        hdul = fits.getheader(imlist[i])
#        print('Header information updated as FILTER:{}'.format(hdul['FILTER']))

    # Starting process
    
    n = 20
    print('Starting a photometry process on GRB{} host galaxy'.format(target))
    print('-'*n+'images to process'+'-'*n)
    for inim in imlist: print(inim)
    print('-'*(2*n + 17))

    # Photometry for each band image
    
    maglist     = []
    fluxlist    = []
    zplist      = []
    
    for i in range(len(imlist)):
        photdat = photometry(imlist[i], target, ra=host_RA, dec=host_Dec)
        maglist.append(photdat[0])
        maglist.append(photdat[1])
        fluxlist.append(photdat[2])
        fluxlist.append(photdat[3])
        zplist.append(photdat[4])
        zplist.append(photdat[5])

    # Saving output in form of table

    MagTable    = Table([[indx+1], [maglist[0]], [maglist[1]], [maglist[4]], [maglist[5]], [maglist[2]], [maglist[3]], [maglist[8]], [maglist[9]], [maglist[6]], [maglist[7]]],
                         names = ['target', 'mag_g', 'magerr_g', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i', 'mag_z', 'magerr_z', 'mag_y', 'magerr_y'])

    FluxTable   = Table([[indx+1], [fluxlist[0]], [fluxlist[1]], [fluxlist[4]], [fluxlist[5]], [fluxlist[2]], [fluxlist[3]], [fluxlist[8]], [fluxlist[9]], [fluxlist[6]], [fluxlist[7]]], 
                         names = ['id', 'f_PS1_g', 'e_PS1_g', 'f_PS1_r', 'e_PS1_r', 'f_PS1_i', 'e_PS1_i', 'f_PS1_z', 'e_PS1_z', 'f_PS1_y', 'e_PS1_y'])

    ZpTable     = Table([[indx+1], [zplist[0]], [zplist[1]], [zplist[4]], [zplist[5]], [zplist[2]], [zplist[3]], [zplist[8]], [zplist[9]], [zplist[6]], [zplist[7]]],
                         names = ['target', 'mag_g', 'magerr_g', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i', 'mag_z', 'magerr_z', 'mag_y', 'magerr_y'])

    MagTable.write('result/mag_'+target+'.cat', format='ascii', overwrite=True)
    FluxTable.write('result/flux_'+target+'.cat', format='ascii', overwrite=True)
    ZpTable.write('result/zp_'+target+'.cat', format='ascii', overwrite=True)

    os.system('cp result/flux_'+target+'.cat ../../SED/sGRB'+target+'_host.cat')

# Merging the output (Fast input catalog file form)
    
path        = '~/research/data/PS1/SED'
os.chdir(os.path.expanduser(path))
os.getcwd()

fluxfiles   = glob.glob('sGRB*.cat')
fluxfiles.sort()
merge       = Table()

for cat in fluxfiles:
    cattbl  = ascii.read(cat)
    merge   = vstack([merge, cattbl]) 

merge.write('hdfn99.cat', format='ascii', overwrite=True)
os.system('rm sGRB*.cat')

finish  = time.time()

print('Photometry process on the image list finished. Total time: {} sec'.format(round(finish-start,1))) 

#%%
#a = host_RA[0]
#b = host_Dec[0]
#c = SkyCoord(a+' '+b, unit=(u.hourangle, u.deg))
#type(c.ra.degree)
a = Table()

for tar in tarname:
    path        = '~/research/data/PS1/'+tar+'/sextractor/result'
    os.chdir(os.path.expanduser(path))
    os.getcwd()
    
    b   = ascii.read('mag_'+tar+'.cat')
    a   = vstack([a, b])

path        = '~/research/data/PS1/'
os.chdir(os.path.expanduser(path))   
a.write('magmerge.cat', format='ascii', overwrite=True)
#    os.system('cp ../../default.* .')
#tarlist     = [['130603B',      '150101B',      '160821B',      '170817A'],     # target name
#               [172.2009458,    188.0207083,    279.9748667,    197.44875],     # RA
#               [17.0717833,     -10.9334722,    62.392875,      -23.3838889],   # Dec
#               [0.3568,         0.1343,         0.162,          0.009783]]      # redshift
#    
#%%
q = ascii.read('magmerge.cat')

for i in range(len(q)):
    for j in range(len(q.colnames)-1):
        q[i][j+1] = round(q[i][j+1], 2)
q.write('magmerge.cat', format='ascii', overwrite=True)
#%%
tarRA       = tarlist['RAJ2000']
tarDec      = tarlist['DEJ2000']
host_RA     = SkyCoord(tarRA[indx]+' '+tarDec[indx], unit=(u.hourangle, u.deg)).ra.degree
host_Dec    = SkyCoord(tarRA[indx]+' '+tarDec[indx], unit=(u.hourangle, u.deg)).dec.degree