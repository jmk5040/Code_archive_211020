#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:30:38 2019

@author: sonic
"""
#%%
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from astropy.io import ascii
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from numpy import median
#from multiprocessing import Process, Pool
#import multiprocessing as mp
from gppy.phot import phot
#import time
#%%
def apmag(inim, gain=1.07272, pixscale=0.258, detsig=3.0, host_ra=0, host_dec=0, photdat=Table()): #default values for PS1
    
    file            = get_pkg_data_filename(inim)
    hdul            = fits.open(file)
    hdr             = hdul[0].header
    ftr             = hdr['FILTER']
#    img             = hdul[0].data   
    
    peeing, seeing	= phot.psfex(inim, pixscale)
    param_secom     = dict(inim=inim, gain=gain, pixscale=pixscale, \
                           seeing=seeing, det_sigma=detsig, backsize=str(64), \
                           backfiltersize=str(3), psf=True, check=False)	
    intbl0, incat	= phot.secom(**param_secom)
    
    dlist           = [np.sqrt((intbl0['ALPHA_J2000'][j]-host_ra)**2 + \
                               (intbl0['DELTA_J2000'][j]-host_dec)**2) for j in range(len(intbl0['NUMBER']))]

    host_num        = np.where(np.array(dlist) == min(dlist))[0][0]
    host_Apmag      = intbl0['MAG_AUTO'][host_num] + photdat['zp'][photdat['filter']==ftr][0] - 2.5*np.log10(95.44/100) # the 3rd term: aperture correction for Zp estimation. Zp estimated with apertures 2*seeing
    host_Aperr      = phot.sqsum(intbl0['MAGERR_AUTO'][host_num],photdat['zper'][photdat['filter']==ftr][0])

#    plt.figure(figsize=(8,8))
#    plt.title(target + ftr)
#    plt.imshow(img, origin='lower', cmap='gray', vmin=0, vmax=5e2)
#    plt.colorbar()
#    plt.plot(intbl0['X_IMAGE'][host_num],intbl0['Y_IMAGE'][host_num],'rh')
#    plt.show()
    
    return [host_Apmag, host_Aperr, ftr]
#
#def AB2Jy(ABmag):
#    return 10**((16.4-ABmag)/2.5) #mJy
#
#def eAB2Jy(ABmag, ABerr):
#    return ABerr * np.abs(10**(16.4/2.5)*(-np.log(10)/2.5)*10**(ABmag/-2.5))
    
def AB2Jy(ABmag):
    return 10**((25-ABmag)/2.5) #FAST default

def eAB2Jy(ABmag, ABerr):
    return ABerr * np.abs(10**(25/2.5)*(-np.log(10)/2.5)*10**(ABmag/-2.5))
#%%
def aplist(idex):

    tarlist = [['130603B','150101B','150424A','160821B','170817A'], #target name
               [172.2009458,188.0207083,152.305620,279.9748667,197.44875], # RA
               [17.0717833,-10.9334722,-26.630920,62.392875,-23.3838889], #Dec
               [0.3568,0.1343,0.30,0.162,0.009783]] #redshift
    
#    extinction = [[0.073, 0.052, 0.039, 0.030, 0.025], 
#                  [0.130, 0.093, 0.069, 0.054, 0.045],
#                  [0.142, 0.102, 0.075, 0.059, 0.049],
#                  [0.391, 0.280, 0.207, 0.163, 0.134]]

    gain        = 1.07272           #e-/ADU
    pixscale    = 0.258             #arcsec/pixel
    detsig      = 3.0
    
    path = '~/research/data/PS1/'+tarlist[0][idex]
    os.chdir(os.path.expanduser(path))
    
    inim        = glob.glob('c*.fits')
    photdat     = ascii.read('phot.dat')

    target      = 'GRB'+tarlist[0][idex]
    host_ra     = tarlist[1][idex]      #host R.A.
    host_dec    = tarlist[2][idex]       #host Decl.
        
    photable    = []

    for i in range(len(inim)):
        photable.append(apmag(inim[i], gain = gain, pixscale = pixscale, detsig = detsig, host_ra=host_ra, host_dec=host_dec, photdat=photdat))
    
    dummytbl = Table()

    tables = np.array(photable)

    header = ['id', 'z_phot',
              'PS1_g', 'PS1_g_err',
              'PS1_r', 'PS1_r_err',
              'PS1_i', 'PS1_i_err',
              'PS1_z', 'PS1_z_err',
              'PS1_y', 'PS1_y_err']

    dummytbl[header[0]] =   [target]
    dummytbl[header[1]] =   [tarlist[3][idex]]
    dummytbl[header[2]] =   [AB2Jy(float(tables[np.where(tables=='g')[0][0],0]))]
    dummytbl[header[3]] =   [eAB2Jy(float(tables[np.where(tables=='g')[0][0],0]), float(tables[np.where(tables=='g')[0][0],1]))]
    dummytbl[header[4]] =   [AB2Jy(float(tables[np.where(tables=='r')[0][0],0]))]
    dummytbl[header[5]] =   [eAB2Jy(float(tables[np.where(tables=='r')[0][0],0]), float(tables[np.where(tables=='r')[0][0],1]))]
    dummytbl[header[6]] =   [AB2Jy(float(tables[np.where(tables=='i')[0][0],0]))]
    dummytbl[header[7]] =   [eAB2Jy(float(tables[np.where(tables=='i')[0][0],0]), float(tables[np.where(tables=='i')[0][0],1]))]
    dummytbl[header[8]] =   [AB2Jy(float(tables[np.where(tables=='z')[0][0],0]))]
    dummytbl[header[9]] =   [eAB2Jy(float(tables[np.where(tables=='z')[0][0],0]), float(tables[np.where(tables=='z')[0][0],1]))]
    dummytbl[header[10]] =  [AB2Jy(float(tables[np.where(tables=='y')[0][0],0]))]
    dummytbl[header[11]] =  [eAB2Jy(float(tables[np.where(tables=='y')[0][0],0]), float(tables[np.where(tables=='y')[0][0],1]))]
    
    return dummytbl
#%%
grblist = []

for i in range(5):
    grblist.append(aplist(i))
    
grbcat = vstack(grblist)

path = '~/research/data/SED/'
os.chdir(os.path.expanduser(path))

grbcat.write('./grb.cat', format='ascii', overwrite=True)