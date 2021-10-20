#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:13:54 2020

KS4 astrometry & photometry validation
Created by Mankeun Jeong
"""
#%% modules
import time
import os, glob
import numpy as np
import pandas as pd
import astropy.units as u
from gppy.phot import phot
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier
from astropy.stats import sigma_clip
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
#%% functions
def ucac3_query(ra, dec, maglim, radius=1.0): # ra, dec, radius in degree unit
    """
    ----------#----------
    |         |         |
    |    k    |    m    |
    |         |<-1 deg->|
    |         |         |
    ----------#----------
    |         |         |
    |    n    |    t    |
    |         |         |
    |         |         |
    ----------#----------
    four 9k*9k CCDs mosaick
    pixel scale: 0.4"/pix
    """  
    query   = Vizier(columns=['RAJ2000', 'DEJ2000'], column_filters={"Bmag":">"+str(maglim)})
    query.ROW_LIMIT = -1 # no row limit
    result  = query.query_region(coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), width=str(radius*60)+'m', catalog=["UCAC3"])
    qtable  = result[0]
    
    return qtable

def apass_query(ra, dec, radius=1.0):
    """
    APASS QUERY
	INPUT   :   RA [deg], Dec [deg], radius
	OUTPUT  :   QUERY TABLE
    #   Vega    : B, V
    #   AB      : g, r, i
    #   Vega - AB Magnitude Conversion (Blanton+07)
    #   U       : m_AB - m_Vega = 0.79
    #   B       : m_AB - m_Vega =-0.09
    #   V       : m_AB - m_Vega = 0.02
    #   R       : m_AB - m_Vega = 0.21
    #   I       : m_AB - m_Vega = 0.45
    #   J       : m_AB - m_Vega = 0.91
    #   H       : m_AB - m_Vega = 1.39
    #   K       : m_AB - m_Vega = 1.85
    """
    Vizier.ROW_LIMIT    = -1
    query       = Vizier.query_region(coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), width=str(radius*60)+'m', catalog=["APASS9"])
    dum         = query[0]
    colnames    = dum.colnames
    
    for col in colnames:
        indx    = np.where(dum[col].mask == False)
        dum     = dum[indx]
        
    querycat    = Table()
    querycat['NUMBER']  = dum['recno']
    querycat['RA_ICRS'] = dum['RAJ2000']
    querycat['DE_ICRS'] = dum['DEJ2000']
    querycat['Numb_obs']= dum['nobs']
    querycat['Numb_img']= dum['mobs']
    querycat['B-V']     = dum['B-V']+ (-0.09 - 0.02)
    querycat['e_B-V']   = dum['e_B-V']
    querycat['Bmag']    = dum['Bmag']   - 0.09  # [Vega] to [AB]
    querycat['e_Bmag']  = dum['e_Bmag']
    querycat['Vmag']    = dum['Vmag']   + 0.02  # [Vega] to [AB]
    querycat['e_Vmag']  = dum['e_Vmag']
    querycat['Rmag']    = dum['r_mag'] - 0.0576 - 0.3718 * (dum['r_mag'] - dum['i_mag'] - 0.2589) # Blanton+07, sigma = 0.0072
    querycat['e_Rmag']  = dum['e_r_mag']
    querycat['Imag']    = dum['r_mag'] - 1.2444 * (dum['r_mag'] - dum['i_mag']) - 0.3820 + 0.45 # Lupton+05, sigma = 0.0078
    querycat['e_Imag']  = dum['e_i_mag']
#    querycat['gmag']    = dum['g_mag']
#    querycat['e_gmag']  = dum['e_g_mag']
#    querycat['rmag']    = dum['r_mag']
#    querycat['e_rmag']  = dum['e_r_mag']
#    querycat['imag']    = dum['i_mag']
#    querycat['e_imag']  = dum['e_i_mag']

    return querycat    


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

def patest(date):
    
    # 1
    
    path        = '~/research/data/kmtnet/'+date 
    os.chdir(os.path.expanduser(path))
    os.system('mkdir astrometry photometry ref result')
    os.getcwd()
    
    paset       = glob.glob('pa*.cat')
    paset.sort()
    info        = ascii.read('ks4check.cat', Reader = ascii.NoHeader) # col1: file name, col2: filter, col4: ZP, col5: ZP error, col7: seeing, col8: depth
    seeingmed   = np.median(np.nan_to_num(info['col7']))
    seeingstd   = np.std(np.nan_to_num(info['col7']))
    zperrmed    = np.median(np.nan_to_num(info['col5']))
    zperrstd    = np.std(np.nan_to_num(info['col5']))

    print('Images: {} \nSeeing: {}±{} [arcsec] \nZP error: {}±{} [mag]'.format(len(info), round(seeingmed,2), round(seeingstd,2), round(zperrmed,3), round(zperrstd,3)))

    # 2    

    start   = time.time()

    malist  = [] # misalignment list for all the images
    malimg0, malimg1, malimg2, malimg3 = [], [], [], [] # doubtful image list (0: imname, 1: misalignment, 2: seeing, 3: matched stars)

    mdlist  = [] # magnitude difference list for all the images
    mdfimg0, mdfimg1, mdfimg2, mdfimg3, mdfimg4, mdfimg5, mdfimg6 = [], [], [], [], [] ,[], [] # doubtful image list (0: imname, 1: band, 2: mag difference, 3: mag error, 4: zero point error, 5: zeropoint stars, 6: matched stars)

    for padata in paset:
    
        # Data reading & filter identification
    
        data    = ascii.read(padata)
        imname  = padata[1:-4]
        seeing  = info[info['col1'] == date+'/'+imname]['col7'][0] # arcsec
        band    = info[info['col1'] == date+'/'+imname]['col2'][0] # B,V,R,I
        zperr   = info[info['col1'] == date+'/'+imname]['col5'][0] # mag
        zpstar  = info[info['col1'] == date+'/'+imname]['col6'][0] # number of stars used for estimating the zeropoint
    
        # Image center position
        
        cra     = np.median(data['ALPHA_J2000'])
        cdec    = np.median(data['DELTA_J2000']) 
    
        """
        Astrometry Test
        """

        # ucac3 catalog query
    
        try:
            cata    = ascii.read('ref/ucac_'+imname+'.cat')
        
        except:
            cata    = ucac3_query(cra, cdec, maglim=13, radius=1.0) # maglim: B band mag low limit
            cata.write('ref/ucac_'+imname+'.cat', format='ascii', overwrite=True)

        # Star selection & magnitude cut
    
        star    = data[data['CLASS_STAR'] > 0.8][data[data['CLASS_STAR'] > 0.8]['FLAGS']==0]
        uppcut  = 14
        lowcut  = 17
    
        udata   = star[star['MAG_AUTO'] > uppcut]
        uldat   = udata[udata['MAG_AUTO'] < lowcut]

        # Matching & estimating misalignment
    
        param_match	= dict(	intbl=uldat, reftbl=cata,
                           inra=uldat['ALPHA_J2000'], indec=uldat['DELTA_J2000'],
                           refra=cata['RAJ2000'], refdec=cata['DEJ2000'], sep=2)
        mtbl	= phot.matching(**param_match)
        rmsalign= np.sqrt(np.mean((mtbl['sep']*3600)**2))

        # Astrometry residual mapping
    
        # residual graph (x-axis: delta(R.A.), y-axis: delta(Decl.))
        plt.figure(figsize=(6,6))
        plt.scatter((mtbl['RAJ2000']-mtbl['ALPHA_J2000'])*np.abs(np.cos(mtbl['DEJ2000']))*3600, (mtbl['DEJ2000']-mtbl['DELTA_J2000'])*3600, c='blue')
        plt.title('KS4 Astrometry Residual Map for {} \n Misalignment RMSE: {} arcsec'.format(imname, round(rmsalign, 2)))
        plt.xlabel(r'$\Delta \ R.A. cos(\delta)$ [arcsec]')
        plt.ylabel(r'$\Delta$ Dec. [arcsec]')
        if len(mtbl) == 0:
            maxdra  = 0
            maxdec  = 0
        else:
            maxdra  = np.max(np.abs((mtbl['RAJ2000']-mtbl['ALPHA_J2000'])*np.cos(mtbl['DEJ2000']))*3600)
            maxdec  = np.max(np.abs(mtbl['DEJ2000']-mtbl['DELTA_J2000'])*3600)
        maxmis  = np.max([maxdra, maxdec, 5])
        plt.xlim(-maxmis, maxmis)
        plt.ylim(-maxmis, maxmis)
        plt.minorticks_on()
        plt.grid(which='major',linestyle='-')
        plt.grid(which='minor',linestyle='--',linewidth=0.1)
        #plt.show()
        plt.savefig('astrometry/'+imname+'_graph.png')
        plt.close()
    
        # residual map (x-axis: XWIN_IMAGE, y-axis: YWIN_IMAGE)
        plt.figure(figsize=(6,6))
        plt.scatter(mtbl['XWIN_IMAGE'], mtbl['YWIN_IMAGE'], c=mtbl['sep']*3600, cmap='seismic')
        cbar    = plt.colorbar()
        cbar.set_label('Misalignment [arcsec]')
        plt.clim(0, 0.8) # 0.4" = 1 pixel
        plt.title('KS4 Astrometry Residual Map for {} \n Misalignment RMSE: {} arcsec'.format(imname, round(rmsalign, 2)))
        plt.xlabel('X axis [pixel]')
        plt.ylabel('Y axis [pixel]')
        #plt.show()
        plt.savefig('astrometry/'+imname+'_mapping.png')
        plt.close()

        malist.append(rmsalign)

        # Examining quality
    
        malimg0.append(imname)
        malimg1.append(round(rmsalign,3))
        malimg2.append(round(seeing,3))
        malimg3.append(len(mtbl))


        """
        Photometry Test
        """

        # apass catalog query
    
        try:
            bata    = ascii.read('ref/apass_'+imname+'.cat')
        
        except:
            bata    = apass_query(cra, cdec, radius=1.0)
            bata.write('ref/apass_'+imname+'.cat', format='ascii', overwrite=True)
    
        # Matching & estimating misalignment
    
        param_match	= dict(	intbl=star, reftbl=bata,
                           inra=star['ALPHA_J2000'], indec=star['DELTA_J2000'],
                           refra=bata['RA_ICRS'], refdec=bata['DE_ICRS'], sep=0.3)
        mtbl        = phot.matching(**param_match)
        mtbl        = mtbl[mtbl['MAG_AUTO'] != 99]
    
        if band == 'B':
            magdif  = mtbl['MAG_AUTO'] - mtbl['Bmag']
            magerr  = phot.sqsum(mtbl['e_Bmag'], mtbl['MAGERR_AUTO'])
    
        elif band == 'V':
            magdif  = mtbl['MAG_AUTO'] - mtbl['Vmag']
            magerr  = phot.sqsum(mtbl['e_Vmag'], mtbl['MAGERR_AUTO'])
        
        elif band == 'R':
            magdif  = mtbl['MAG_AUTO'] - mtbl['Rmag']
            magerr  = phot.sqsum(mtbl['e_Rmag'], mtbl['MAGERR_AUTO'])
        
        elif band == 'I':
            magdif  = mtbl['MAG_AUTO'] - mtbl['Imag']
            magerr  = phot.sqsum(mtbl['e_Imag'], mtbl['MAGERR_AUTO'])
    
        magdif_sat  = magdif[np.where(mtbl['MAG_AUTO'] < 13)]
        magdif_uns  = magdif[np.where(mtbl['MAG_AUTO'] > 13)]
        rmsmagdif   = np.sqrt(np.mean(magdif_uns**2))
        mediff      = np.median(magdif_uns)
        
        # Photometry accuracy mapping
    
        # residual graph (x-axis: magnitude, y-axis: delta(mag))
        plt.figure(figsize=(6,6))
        plt.scatter(mtbl[mtbl['MAG_AUTO']>13]['MAG_AUTO'], magdif_uns, c='red', marker='o')
        plt.scatter(mtbl[mtbl['MAG_AUTO']<13]['MAG_AUTO'], magdif_sat, c='red', marker='x')
        plt.axhline(y=mediff, color='crimson', linestyle='-')
        plt.title('KS4 Photometry Precision Map for {} ({}) \n Median Magnitude Difference: {} ABmag \n Magnitude Difference RMSE: {} ABmag'.format(imname, band, round(mediff, 3), round(rmsmagdif, 3)))
        plt.xlabel(r'$m_{KS4}$ [ABmag]')
        plt.ylabel(r'$m_{KS4} - m_{APASS}$ [ABmag]')
        if len(mtbl) == 0:
            maxmag  = 0
        else:
            maxmag  = np.abs(np.mean(magdif)) + 3.*np.std(magdif)
        maxmis  = np.max([maxmag, 1])
        plt.xlim(18, 10)
        plt.ylim(-maxmis, maxmis)
        plt.minorticks_on()
        plt.grid(which='major',linestyle='-')
        plt.grid(which='minor',linestyle='--',linewidth=0.1)
        #plt.show()
        plt.savefig('photometry/'+imname+'_graph.png')
        plt.close()

        # residual map (x-axis: XWIN_IMAGE, y-axis: YWIN_IMAGE)
        plt.figure(figsize=(6,6))
        plt.scatter(mtbl[mtbl['MAG_AUTO']>13]['XWIN_IMAGE'], mtbl[mtbl['MAG_AUTO']>13]['YWIN_IMAGE'], marker='o', c=magdif_uns, cmap='Spectral')
        #    plt.scatter(mtbl[mtbl['MAG_AUTO']<13]['XWIN_IMAGE'], mtbl[mtbl['MAG_AUTO']<13]['YWIN_IMAGE'], marker='x', c=magdif_sat, cmap='Spectral')
        cbar    = plt.colorbar()
        cbar.set_label(r'$m_{KS4} - m_{APASS}$ [mag]')
        plt.clim(-0.1, 0.1)
        plt.title('KS4 Photometry Precision Map for {} ({}) \n Median Magnitude Difference: {} ABmag \n Magnitude Difference RMSE: {} ABmag'.format(imname, band, round(mediff, 3), round(rmsmagdif, 3)))
        plt.xlabel('X axis [pixel]')
        plt.ylabel('Y axis [pixel]')
        #plt.show()
        plt.savefig('photometry/'+imname+'_mapping.png')
        plt.close()

        mdlist.append(rmsmagdif)

    # Examining quality
    
        mdfimg0.append(imname)
        mdfimg1.append(band)
        mdfimg2.append(round(mediff,3))
        mdfimg3.append(round(rmsmagdif,3))
        mdfimg4.append(round(zperr,3))
        mdfimg5.append(int(zpstar))
        mdfimg6.append(len(mtbl))

    # Writing doubtful image information

    mimtbl  = Table([malimg0, malimg1, malimg2, malimg3], names=['files', 'misalignment', 'seeing', 'stars'])
    mimtbl.write('result/mastroimg.cat', format='ascii', overwrite=True)
    mimtbl  = Table([mdfimg0, mdfimg1, mdfimg2, mdfimg3, mdfimg6, mdfimg4, mdfimg5], names=['files', 'band', 'magdifference', 'RMSE', 'matchstar', 'zperror', 'zpstar'])
    mimtbl.write('result/mphotoimg.cat', format='ascii', overwrite=True)

    # Plotting histogram

    # astrometry
    plt.figure(figsize=(6,6))
    plt.hist(malist, bins=len(paset), facecolor='blue')
    plt.title('KS4 '+date+' Astrometry Accuracy Statistics')
    plt.xlabel('Misalignment RMSE [arcsec]')
    plt.ylabel('Images')
    plt.minorticks_on()
    plt.grid(linestyle='--')
    plt.savefig('result/astro_'+date+'.png')
    plt.close()

    #photometry
    plt.figure(figsize=(6,6))
    plt.hist(mdlist, bins=len(paset), facecolor='red')
    plt.title('KS4 '+date+' Photometry Precision Statistics')
    plt.xlabel(r'$m_{APASS}-m_{KS4}$ RMSE [mag]')
    plt.ylabel('Images')
    plt.minorticks_on()
    plt.grid(linestyle='--')
    plt.savefig('result/photo_'+date+'.png')
    plt.close()

    finish  = time.time()

    print('KS4 astrometry & photometry precision test finished. Total time: {} sec'.format(round(finish-start,1)))
    
    return
#%%
#nights  = ['20191024', '20191025', '20191026', '20191029', '20191030', '20191031', '20191128', '20191129', '20191130', '20191201', '20191202', '20191203', '20191204', '20191205', '20191206', '20191207']
#
#for i in range(len(nights)):
#    patest(nights[i])
##%%
#%%
##%%
#plt.figure()
#plt.scatter(cata['RAJ2000'], cata['DEJ2000'], label='cata')
#plt.scatter(uldat['ALPHA_J2000'], uldat['DELTA_J2000'], label='data')
#plt.legend()
##plt.xlim(31.600, 31.612)
##plt.ylim(-61.736, -61.720)
##plt.show()
##%%
#plt.figure()
#plt.hist(malist, bins=50)
#plt.xlim(0, 5)
##plt.show()
##%%
#for m in malimg:
#    print(info[info['col1']==m]['col7'][0])
#for n in mdfimg:
#    print(info[info['col1']==n]['col5'][0])
##%%
#cata    = ucac3_query(cra, cdec, maglim=16, radius=1)
#
#param_match	= dict(	intbl=intbl, reftbl=reftbl,
#					inra=intbl['ALPHA_J2000'], indec=intbl['DELTA_J2000'],
#					refra=reftbl['ra'], refdec=reftbl['dec'])
#mtbl		= phot.matching(**param_match)
#%%
def astrosift(date):
    path        = '~/research/data/kmtnet/'+date 
    os.chdir(os.path.expanduser(path))

    mastroimg   = ascii.read('result/mastroimg.cat')
    ks4check    = ascii.read('ks4check.cat')
    aresult     = hstack([mastroimg, ks4check['col6']])
    
    aRMSEmed    = np.median(np.nan_to_num(aresult['misalignment']))
    seeingmed   = np.median(np.nan_to_num(aresult['seeing']))
    aRMSEstd    = np.std(np.nan_to_num(aresult['misalignment']))
    seeingstd   = np.std(np.nan_to_num(aresult['seeing']))
    
    summary     = "Summary \n Date: {} \n Seeing: {}+{} \n Astrometry RMSE: {}+{}"\
    .format(date, round(seeingmed,3), round(seeingstd,3), round(aRMSEmed, 3), round(aRMSEstd,3))
    print(summary)
    print('\n')
    
    misalign    = []
    
    for i in range(len(aresult)):
        if aresult['misalignment'][i] > 1.0:
            misalign.append(aresult['files'][i])
        elif (aresult['misalignment'][i] > 0.6) and (aresult['seeing'][i] > seeingmed + 3.*seeingstd):
            misalign.append(aresult['files'][i])
        elif np.isnan(aresult['misalignment'][i]) and aresult['col6'][i] == 0:
            misalign.append(aresult['files'][i])
    
    print('astrometry: {}'.format(len(misalign)))
    for j in range(len(misalign)):
        print(misalign[j], end=' ')
    
    print('\n')
    return
#%%    
nights  = ['20191022','20191024', '20191025', '20191026', '20191029', '20191030', '20191031', '20191128', '20191129', '20191130', '20191201', '20191202', '20191203', '20191204', '20191205', '20191206', '20191207']
#nights = ['20191130', '20191201']
for i in range(len(nights)):
    astrosift(nights[i])
#%%    
# visual inspection
date        = '20191022'
path        = '~/research/data/kmtnet/'+date 
os.chdir(os.path.expanduser(path))

presult     = ascii.read('result/mphotoimg.cat')
aresult     = ascii.read('result/mastroimg.cat')

pRMSE       = np.median(np.nan_to_num(presult['RMSE']))
aRMSE       = np.median(np.nan_to_num(aresult['misalignment']))
seeing      = np.median(np.nan_to_num(aresult['seeing']))
pRMSEstd    = np.std(np.nan_to_num(presult['RMSE']))
aRMSEstd    = np.std(np.nan_to_num(aresult['misalignment']))
seeingstd   = np.std(np.nan_to_num(aresult['seeing']))

summary     = "Summary \n Date: {} \n Seeing: {}+{} \n Photometry RMSE: {}+{} \n Astrometry RMSE: {}+{}".format(date, round(seeing,3), round(seeingstd,3), pRMSE, round(pRMSEstd,3), aRMSE, round(aRMSEstd,3))
print(summary)
print('\n')
magdiffer   = presult[presult['RMSE'] > 0.1]['files']
misalign    = aresult[aresult['misalignment'] > 0.8]['files']
misormag    = list(set(magdiffer) | set(misalign))
misandmag   = list(set(magdiffer).intersection(set(misalign)))
print('photometry: {}'.format(len(magdiffer)))
for i in range(len(magdiffer)):
    print(magdiffer[i], end=' ')
print('\n')
print('astrometry: {}'.format(len(misalign)))
for j in range(len(misalign)):
    print(misalign[j], end=' ')
print('\n')
print('intersection: {}'.format(len(misandmag)))
for l in range(len(misandmag)):
    print(misandmag[l], end=' ')
print('\n')
print('union: {}'.format(len(misormag)))
for k in range(len(misormag)):
    print(misormag[k], end=' ')
