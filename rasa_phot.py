#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:58:39 2020

@author: sonic
"""
#%%
import time
import copy
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
from astroquery.ned import Ned
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from mastapi.queryps1 import ps1cone #DPM https://ps1images.stsci.edu/ps1_dr2_api.html
from astropy.stats import sigma_clip
from ankepy.phot import anketool as atl # anke package
from astropy.coordinates import SkyCoord  # High-level coordinates
from gppy.util.changehdr import changehdr as chdr # e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
#%%
def sdss_query(target, ra, dec, radius, path='cat/'): # ra [deg], dec [deg], radius [deg]
    
    Vizier.ROW_LIMIT    = -1
    query       = Vizier.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), width=str(radius*60)+'m', catalog=['SDSS12'])
    querycat    = query[0]
    
    outname     = 'sdss_'+target+'.cat'
    querycat.write(path+outname, format='ascii', overwrite=True)
    
    return querycat

def ps1_query(target, ra, dec, radius, path='cat/'):

    from mastapi.queryps1 import ps1cone

    constraints = {'nDetections.gt':1}
    columns     = "raMean,decMean,gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,yMeanPSFMag,yMeanPSFMagErr,gMeanKronMag,gMeanKronMagErr,rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr,zMeanKronMag,zMeanKronMagErr,yMeanKronMag,yMeanKronMagErr".split(',')
    columns     = [x.strip() for x in columns]
    columns     = [x for x in columns if x and not x.startswith('#')]

    results     = ps1cone(ra, dec, radius, release='dr2', columns=columns, verbose=True, **constraints)

    reftbl      = ascii.read(results) # ps1 catalog photometry data around the target
    outname     = 'ps1_'+target+'.cat'
    reftbl.write(path+outname, format='ascii', overwrite=True)

    return reftbl

def panstarrs_query(ra, dec, radius):
    
    _params = dict(ra=ra, dec=dec, radius=radius)

    for k, v in _params.items():
        if v is None:
            continue
        if not isinstance(v, u.Quantity):
            _params[k] = v * u.deg
            
    vquery = Vizier(columns=["**", "+objID"],
                    column_filters={"rmag": "10.0..20.0"},
                    row_limit=-1)

    field = SkyCoord(ra=_params['ra'], dec=_params['dec'], frame='icrs')

    queried = vquery.query_region(field,
                                  radius=_params['radius'],
                                  catalog="II/349/ps1")[0]
    return queried


def rasa_photometry(inim, path_config, path_cat, path_fig, detsig=3.0, frac=0.95):
    
    # path_config   = '/home/sonic/RASA36/lab/fwhm/config/'
    # path_cat      = '/home/sonic/RASA36/lab/fwhm/cat/'
    # path_fig      = '/home/sonic/RASA36/lab/fwhm/fig/'
    
    #1. image file & header info open
    
    # file    = get_pkg_data_filename(inim)
    hdul    = fits.open(inim)
    hdr     = hdul[0].header
    
    pixscale= 2.35 # arcsec per pix
    gain    = hdr['EGAIN'] # e-/ADU, why is it 1.5 not 0.71?

    ftr     = hdr['FILTER']
    xscale  = hdr['NAXIS1'] * pixscale # arcsec
    yscale  = hdr['NAXIS2'] * pixscale # arcsec    
    xcent, ycent= hdr['NAXIS1']/2., hdr['NAXIS2']/2.
    radius  = np.sqrt((xscale/2)**2 + (yscale/2)**2)/3600 # query radius in deg
    
    #2. image identification
        
    obs     = inim.split('-')[1]
    target  = inim.split('-')[2]
    date    = int(inim.split('-')[3])
    exptime = float(inim.split('-')[6][:-5])
    

    tartable= Ned.query_object(target)
    
    if len(tartable) != 0:
        ra      = tartable['RA'][0]
        dec     = tartable['DEC'][0]
    elif target == 'AT2020yxz':
        ra      = 42.1846
        dec     = 12.1372
    else:
        ra      = float(input("Enter RA in decimal form: "))
        dec     = float(input("Enter Dec in decimal form: "))

    #3. catalog query
    
    try:
        reftbl     = ascii.read(path_cat+'PS1/'+target+'.csv')    
    except:
        reftbl     = panstarrs_query(ra, dec, radius)
        reftbl.write(path_cat+'PS1/'+target+'.csv', format='csv', overwrite=True)
    
    #4. PSF & SExtractor run
    
    peeing, seeing	= phot.psfex(inim, pixscale)
    param_secom	= dict(	inim=inim,
						gain=gain, pixscale=pixscale, seeing=seeing,
						det_sigma=detsig,
						backsize=str(128), backfiltersize=str(3),
						psf=True, check=False)
    
    intbl0, incat	= phot.secom(**param_secom)
    
    deldist		= phot.sqsum((xcent-intbl0['X_IMAGE']), (ycent-intbl0['Y_IMAGE'])) #	CENTER POS. & DIST CUT
    indx_dist	= np.where(deldist < np.sqrt(frac)*(xcent+ycent)/2.)
    intbl		= intbl0[indx_dist]
    # incat       = path_cat+incat#.split('/')[1]
    # intbl.write(incat, format='ascii', overwrite=True)

    # SExtractor Seeing #
    
    seeing  = round(np.median(intbl['FWHM_WORLD'])*3600, 3)
    peeing  = round(seeing / pixscale, 3)

    param_secom	= dict(	inim=inim,
						gain=gain, pixscale=pixscale, seeing=seeing,
						det_sigma=detsig,
						backsize=str(128), backfiltersize=str(3),
						psf=True, check=False)
    
    intbl0, incat	= phot.secom(**param_secom)
    
    deldist		= phot.sqsum((xcent-intbl0['X_IMAGE']), (ycent-intbl0['Y_IMAGE'])) #	CENTER POS. & DIST CUT
    indx_dist	= np.where(deldist < np.sqrt(frac)*(xcent+ycent)/2.)
    intbl		= intbl0[indx_dist]
    incat       = path_cat+incat#.split('/')[1]
    intbl.write(incat, format='ascii', overwrite=True)

    #5. matching
    
    param_match     = dict(intbl = intbl, reftbl = reftbl, 
                           inra=intbl['ALPHA_J2000'], indec=intbl['DELTA_J2000'], 
                           refra=reftbl['RAJ2000'], refdec=reftbl['DEJ2000'], 
                           sep=2)
    mtbl    = phot.matching(**param_match)
    
    #6. zero-point calculation
    
    aperture    = 'MAG_APER_7'
    inmagkey	= 'MAG_APER_7'
    inmagerkey	= 'MAGERR_APER_7'
    refmagkey   = ftr+'mag'
    refmagerkey = 'e_'+ftr+'mag'    
    # aperture    = 'MAG_AUTO'
    # inmagkey	  = 'MAG_AUTO'
    # inmagerkey	= 'MAGERR_AUTO'
    # refmagkey   = ftr+'mag'
    # refmagerkey = 'e_'+ftr+'mag'    

    param_st4zp	= dict(	intbl=mtbl,
						inmagerkey=aperture,
						refmagkey=refmagkey,
						refmagerkey=refmagerkey,
						refmaglower=13,   # SDSS DR12 Saturation mag [13, 14, 14, 14, 12]
						refmagupper=17,
						refmagerupper=0.05,
						inmagerupper=0.05)
    
    param_zpcal	= dict(	intbl=phot.star4zp(**param_st4zp),
						inmagkey=inmagkey, inmagerkey=inmagerkey,
						refmagkey=refmagkey, refmagerkey=refmagerkey,
						sigma=3.0)
    
    zp, zper, otbl, xtbl	= phot.zpcal(**param_zpcal)
    
    outname	= path_fig+'{0}.{1}.zpcal.png'.format(inim[:-5], inmagkey)
    phot.zpplot(	outname=outname,
					otbl=otbl, xtbl=xtbl,
					inmagkey=inmagkey, inmagerkey=inmagerkey,
					refmagkey=refmagkey, refmagerkey=refmagerkey,
					zp=zp, zper=zper)
    param_plot	= dict(	inim		= inim,
						numb_list	= otbl['NUMBER'],
						xim_list	= otbl['X_IMAGE'],
						yim_list	= otbl['Y_IMAGE'],
						add			= True,
						numb_addlist= xtbl['NUMBER'],
						xim_addlist	= xtbl['X_IMAGE'],
						yim_addlist	= xtbl['Y_IMAGE'])
    try:
        phot.plotshow(**param_plot)
    except:
        print('FAIL TO DRAW ZEROPOINT GRAPH')
        pass
    
    #7. target inspection
    
    # dlist           = [np.sqrt(((intbl['ALPHA_J2000'][i]-ra)*np.cos(intbl['DELTA_J2000'][i]))**2 + (intbl['DELTA_J2000'][i]-dec)**2) for i in range(len(intbl['NUMBER']))]

    # target_label    = np.where(np.array(dlist) == min(dlist))[0][0]
    
    # if intbl['CLASS_STAR'][target_label] > 0.5:
    #     print('The matched target is not a galaxy')
    # else:
    #     pass
    
    #9. galactic extinction correction

    # lambmean    = {'u':3561.8, 'g':4718.9, 'r':6185.2, 'i':7499.7, 'z':8961.5} # in angstrom
    # wave        = np.array([lambmean[ftr]])
    # dustmap     = '/home/sonic/research/photometry/sfdmap/'

    # ebv = sfdmap.ebv(ra, dec, mapdir=dustmap)
    # ext = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
    
    # #10. Target magnitude estimation & limiting magnitude calculation
    
    # target_mag      = intbl['MAG_AUTO'][target_label] + zp - ext
    # target_mager    = phot.sqsum(intbl['MAGERR_AUTO'][target_label], zper)
    
    skymean, skymed, skysig		= phot.bkgest_mask(inim)
    aper	= 2*peeing
    ul		= phot.limitmag(detsig, zp, aper, skysig)
    
    #11. result return
    
    result  = Table([[target], [ftr], [round(zp,3)], [round(zper,3)], [ul], [seeing], [date], [exptime]], names=['target', 'filter', 'zp', 'zper', 'limitmag', 'seeing', 'date', 'exptime'])
    # result  = Table([[target], [ftr], [round(target_mag, 3)], [round(target_mager,3)], [round(zp,3)], [round(zper,3)], [ul], [seeing], [round(min(dlist)*3600,3)], [date], [exptime]], names=['target', 'filter', 'MAG_AUTO', 'MAGERR_AUTO', 'zp', 'zper', 'limitmag', 'seeing', 'sep', 'date', 'exptime'])
    
    return result
#%%
path        = '~/research/observation/RASA36/lab/fwhm'
os.chdir(os.path.expanduser(path))
os.getcwd()

path_imgs   = '/home/sonic/research/observation/RASA36/lab/fwhm/imgs/'
path_config = '/home/sonic/research/observation/RASA36/lab/fwhm/config/'
path_cat    = '/home/sonic/research/observation/RASA36/lab/fwhm/cat/'
path_fig    = '/home/sonic/research/observation/RASA36/lab/fwhm/fig/'

os.chdir(path_imgs)
imlist  = glob.glob('*.fits')
imlist.sort()
#inim    = imlist[0]
#%%
start   = time.time()

for inim in imlist:
    
    hdul    = fits.open(inim)
    hdr     = hdul[0].header

    obs     = inim.split('-')[1]
    target  = inim.split('-')[2]
    date    = inim.split('-')[3]
    exptime = float(inim.split('-')[6][:-5])
    
    pixscale= 2.35 # arcsec/pix
    ftr     = hdr['FILTER']
    gain    = hdr['EGAIN']

    os.system('mv *psf* xml')
    os.system('mv snap* xml')
    os.system('mv seg* xml')
    os.system('mv *cat xml')
    os.system('mv *png xml')
    os.system('mv *.xml xml')
    # seeingpsf   = photre['seeing'][0]

    photbl  = ascii.read(path_cat+'/Alltarget.csv')
    newtbl  = rasa_photometry(inim, path_config, path_cat, path_fig, detsig=5.0, frac=0.95)
    photbl  = vstack([photbl, newtbl])
    photbl.write(path_cat+'/Alltarget.csv', format='csv', overwrite=True)

finish  = time.time()

print('Photometry process on the image list finished. Total time: {} sec'.format(round(finish-start,1))) 

# photresult = vstack(result)
# photresult.write(path_cat+'/Alltarget.csv', format='csv', overwrite=True)
#%% aperture photometry
apsize  = np.linspace(1, 20, 20)
#aperture    = str(round(1/pixscale,2))
#for ap in apsize:
#    aperture    += ','+str(round(ap,2))
for aper in apsize:

    aperture    = str(round(aper/pixscale,2))
    
    configfile  = path_config+'default.sex'
    paramfile   = path_config+'default.param'
    nnwfile     = path_config+'default.nnw'
    convfile    = path_config+'default.conv'

    config      = ' -c '+configfile+' '
    catname     = ' -CATALOG_NAME '+'cat/'+target+'_ap_'+str(aper).zfill(4)+'.cat'
    params      = ' -PARAMETERS_NAME '+paramfile+' -STARNNW_NAME '+nnwfile+' -FILTER Y'+' -FILTER_NAME '+convfile
    apopt       = ' -PHOT_APERTURES '+aperture+' -MAG_ZEROPOINT '+'26.021'
    prompt      = 'sex'+config+inim+catname+params+apopt
    os.system(prompt)
#%% result
allcat  = glob.glob('cat/'+target+'_ap_*.cat')
allcat.sort()
#%%
plt.figure(figsize=(10,10))
for i in range(len(allcat)):
    classcut= 0.95
    acat    = ascii.read(allcat[i])
    acatf   = acat[acat['FLAGS']==0]
    acatc   = acatf[acatf['CLASS_STAR']>classcut]
    aper    = float(allcat[i].split('_')[2][:4])
    plt.scatter([aper]*len(acatc), acatc['MAGERR_APER'], s=5, alpha=0.3, c=acatc['MAG_APER'], cmap='seismic', label="CLASS_STAR > {}".format(classcut))
    if i==0:
        cbar = plt.colorbar()
        cbar.set_label('MAG_APER [ABmag]')
        plt.clim(np.quantile(np.array(acatc['MAG_APER']),0.90),np.quantile(np.array(acatc['MAG_APER']),0.10))
        plt.legend()
    plt.scatter(aper, np.median(acatc['MAGERR_APER']), c='gold')
plt.title('RASA36/'+target+': Error vs. Aperture Size')
plt.xlabel('Aperture Diameter [arcsec]')
plt.ylabel('MAGERR_APER [ABmag]')
plt.xticks(np.arange(1, 21, 1))
plt.ylim(0, 0.5)
plt.savefig(path_fig+'error_plot'+target+'.png', dpi=500, facecolor='w', edgecolor='w')
#plt.show()
#%%
plt.figure(figsize=(10,10))
for i in range(len(allcat)):
    acat    = ascii.read(allcat[i])
    acatf   = acat[acat['FLAGS']==0]
    acatc   = acatf[acatf['CLASS_STAR']>0.95]
    aper    = float(allcat[i].split('_')[2][:4])
    plt.scatter([aper]*len(acatc), acatc['FWHM_WORLD']*3600, s=5, alpha=0.3, label="CLASS_STAR > 0.95", c='dodgerblue')#, c=acatc['MAG_APER'], cmap='seismic', label="CLASS_STAR > 0.99")
    plt.scatter(aper, np.median(acatc['FWHM_WORLD'])*3600, c='gold', label='Median FWHM: {} arcsec'.format(round(np.median(acatc['FWHM_WORLD'])*3600,2)))
    seeingsex   = round(np.median(acatc['FWHM_WORLD'])*3600,2)
    if i==0:
#        cbar = plt.colorbar()
#        cbar.set_label('MAG_APER [ABmag]')
#        plt.clim(14,17)
        plt.legend()
plt.title('RASA36/'+target+': FWHM vs. Aperture Size')
plt.xlabel('Aperture Size [arcsec]')
plt.ylabel('FWHM [arcsec]')
plt.xticks(np.arange(1, 21, 1))
plt.ylim(0, max(acatc['FWHM_WORLD']*3600))
plt.show()
#%%
#plt.figure()
flux    = []
for i in range(len(allcat)):
#    ap  = float(allcat[i].split('_')[1][:-4])
    a   = ascii.read(allcat[i])
    b   = a[a['FLAGS']==0]
    c   = b[b['CLASS_STAR'] > 0.99]
    flux.append(list(c['FLUX_APER']))
#    plt.scatter([ap]*len(d), d, s=0.1, alpha=0.5)
#plt.show()
#%%
plt.figure(figsize=(10,10))
plt.vlines(seeingpsf, 0, 105, ls=":", colors='m', label='PSFEx Seeing {} arcsec'.format(seeingpsf))
plt.vlines(seeingsex, 0, 105, ls="-.", colors='g', label='SEx Seeing {} arcsec'.format(seeingsex))
for i in range(len(allcat)):
    ap  = float(allcat[i].split('_')[2][:-4])
    cover = 100*(np.array(flux[i]) / np.array(flux[-1]))
#    print(np.array(flux[i]) / np.array(flux[-1]))
    plt.scatter([ap]*len(cover), cover, s=0.5, c='dodgerblue', alpha=0.5, label="FLUX_APER / 20\" aperture")
    plt.scatter([ap], np.median(cover), s=10, c='crimson', label='Median ratio for the field stars')
    if i==0:
        plt.legend()
plt.ylim(0,105)
plt.xlim(0.5, 20.5)
plt.title('RASA36/'+target+': PSF Growth Curve')
#plt.hlines(50, xmin=1, xmax=10, color='limegreen', linestyles='-.')
plt.xlabel('Aperture Diameter [arcsec]')
plt.ylabel('Flux Coverage [%]')
plt.xticks(np.arange(1, 21, 1))
plt.yticks(np.arange(0, 101, 10))
plt.grid(linestyle='--', color='grey', alpha=0.3)
plt.savefig(path_fig+'flux_plot'+target+'.png', dpi=500, facecolor='w', edgecolor='w')
#plt.show()

#%%
#plt.figure()
#for i in range(len(allcat)):
#    acat    = ascii.read(allcat[i])
#    acatf   = acat[acat['FLAGS']==0]
#    acatc   = acatf[acatf['CLASS_STAR']>0.9]
#    aper    = float(allcat[i].split('_')[1][:3])
#    plt.scatter([aper]*len(acatc), acatc['MAGERR_APER'], s=5, alpha=0.1, c=acatc['MAG_APER'], cmap='seismic')
#    plt.scatter(aper, np.median(acatc['MAGERR_APER']), s=5, c='gold')
#plt.xlabel('Aperture Size [arcsec]')
#plt.ylabel('MAGERR_APER [mag]')
#plt.xticks(np.arange(1, 11, 1))
#plt.ylim(0, 0.5)
#plt.show()
#%%
a   = ascii.read(allcat[0])
b   = a[a['FLAGS']==0]
c   = b[b['CLASS_STAR']>0.7]
d   = c[c['FLUX_AUTO']>5000]
e   = d[d['FLUX_AUTO']<6000]
starnum     = e['NUMBER']
#%%
#for i in range(len(allcat)):
#    acat    = ascii.read(allcat[i])
#    aper    = float(allcat[i].split('_')[1][:3])
#    magerr  = []
#    for j in range(len(starnum)):
#        magerr.append(acat[acat['NUMBER']==starnum[j]]['MAGERR_APER'][0])
#        print(magerr)


for i in range(len(starnum)):
    magerr  = []
    for j in range(len(allcat)):
        acat    = ascii.read(allcat[j])
        magerr.append(acat[acat['NUMBER']==starnum[i]]['MAGERR_APER'][0])
    print(magerr)


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
##%%
#target = '111117A'
#Ra  = '00:50:46.26'
#Dec = '23:00:40.97'
#
#ra  = SkyCoord(Ra+' '+Dec, unit=(u.hourangle, u.deg)).ra.degree
#dec = SkyCoord(Ra+' '+Dec, unit=(u.hourangle, u.deg)).dec.degree
#
##ftr     = 'g'
##mag     = 24.08
##mager   = 0.09
##ftr     = 'r'
##mag     = 23.93
##mager   = 0.08
##ftr     = 'R'
##mag     = 23.95
##mager   = 0.23
#ftr     = 'K'
#mag     = 23.07
#mager   = 0.23
##ftr     = 'J'
##mag     = 23.89
##mager   = 0.23
##ftr     = 'H'
##mag     = 23.89
##mager   = 0.23
##ftr     = 'K'
##mag     = 23.89
##mager   = 0.23
#
#
#lambmean = {'R':6305.1, 'I':8377.6, 'J':12350.0, 'H':16620.0, 'K':21590.0} # in angstrom
#wave    = np.array([lambmean[ftr]])
#dustmap = '../../photometry/sfdmap/'
#
#ebv = sfdmap.ebv(ra, dec, mapdir=dustmap)
#ext = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
#    
#target_mag      = mag - ext
#target_mager    = mager
#
#print(target_mag, target_mager)


