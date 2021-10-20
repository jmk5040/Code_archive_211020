#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:19:45 2021

@author: sonic
"""
#%%
import time
import os, glob
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, hstack
from astropy.utils.data import get_pkg_data_filename
# from gppy.util.changehdr import changehdr as chdr #DPM e.g. chdr(img, where, what)
#%% define path
path_base   = '/home/sonic/data5/ks4/data/'
path_img    = '/home/sonic/data5/ks4/data/stack1/'
path_dust   = '/home/sonic/data5/ks4/data/sfdmap/'
path_cat    = '/home/sonic/data5/ks4/data/cat/'
path_cfg    = '/home/sonic/data5/ks4/data/config/'
path_result = '/home/sonic/data5/ks4/data/result/'
path_plot   = '/home/sonic/data5/ks4/data/plot/'
path_table  = '/home/sonic/data5/ks4/data/table/'

os.chdir(os.path.expanduser(path_base))

#%% define functions
# def sqsum(a, b):
#     '''
#     SQUARE SUM
#     USEFUL TO CALC. ERROR
#     '''
#     import numpy as np
#     return np.sqrt(a**2.+b**2.)
def sqsum(numlist):
    S   = 0
    for i in range(len(numlist)):
        S   += numlist[i]**2
    sqS     = np.sqrt(S)
    return sqS

def apass_query(ra, dec, radius=1.0): # unit=(deg, deg, arcsec)
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
    querycat['RAJ2000'] = dum['RAJ2000']
    querycat['DEJ2000'] = dum['DEJ2000']
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

    return querycat

def imlistsort_ks4(imlist):
        newlist     = []
        ks4ftr  = ['B', 'V', 'R', 'I']
        for ftr in ks4ftr:
            for i in range(len(imlist)):
                if ftr in imlist[i]:
                    newlist.append(imlist[i])
        return newlist

def matching(intbl, reftbl, inra, indec, refra, refdec, sep=2.0):
    """
    MATCHING TWO CATALOG WITH RA, Dec COORD. WITH python
    INPUT   :   SE catalog, SDSS catalog file name, sepertation [arcsec]
    OUTPUT  :   MATCED CATALOG FILE & TABLE
    """
    import numpy as np
    import astropy.units as u
    from astropy.table import Table, Column
    from astropy.coordinates import SkyCoord
    from astropy.io import ascii

    incoord        = SkyCoord(inra, indec, unit=(u.deg, u.deg))
    refcoord    = SkyCoord(refra, refdec, unit=(u.deg, u.deg))

    #   INDEX FOR REF.TABLE
    indx, d2d, d3d  = incoord.match_to_catalog_sky(refcoord)
    mreftbl            = reftbl[indx]
    mreftbl['sep']    = d2d
    mergetbl        = intbl
    for col in mreftbl.colnames:
        mergetbl[col]    = mreftbl[col]
    indx_sep        = np.where(mergetbl['sep']*3600.<sep)
    mtbl            = mergetbl[indx_sep]
    #mtbl.write(mergename, format='ascii', overwrite=True)
    return mtbl

def star4zp(intbl, inmagerkey, refmagkey, refmagerkey, refmaglower=14., refmagupper=17., refmagerupper=0.05, inmagerupper=0.1, flagcut=0):
    """
    SELECT STARS FOR USING ZEROPOINT CALCULATION
    INPUT   :   TABLE, IMAGE MAG.ERR KEYWORD, REF.MAG. KEYWORD, REF.MAG.ERR KEYWORD
    OUTPUT  :   NEW TABLE
    """
    import numpy as np
    indx    = np.where( (intbl['FLAGS'] <= flagcut) & 
                        (intbl[refmagkey] < refmagupper) & 
                        (intbl[refmagkey] > refmaglower) & 
                        (intbl[refmagerkey] < refmagerupper) &
                        (intbl[inmagerkey] < inmagerupper) 
                        )
    indx0   = np.where( (intbl['FLAGS'] <= flagcut) )
    indx2   = np.where( (intbl[refmagkey] < refmagupper) & 
                        (intbl[refmagkey] > refmaglower) & 
                        (intbl[refmagerkey] < refmagerupper) 
                        )
    indx3   = np.where( (intbl[inmagerkey] < inmagerupper) )
    newtbl  = intbl[indx]
    comment = '-'*60+'\n' \
            + 'ALL\t\t\t\t: '+str(len(intbl))+'\n' \
            + '-'*60+'\n' \
            + 'FLAG(<{})\t\t\t: '.format(flagcut)+str(len(indx0[0]))+'\n' \
            + refmagkey+' REF. MAGCUT ('+str(refmaglower)+'-'+str(refmagupper)+')'+'\t\t: '+str(len(indx2[0]))+'\n' \
            + refmagerkey+' REF. MAGERR CUT < '+str(refmagerupper)+'\n' \
            + inmagerkey+' OF IMAGE CUT < '+str(inmagerupper)+'\t: '+str(len(indx3[0]))+'\n' \
            + '-'*60+'\n' \
            + 'TOTAL #\t\t\t\t: '+str(len(indx[0]))+'\n' \
            + '-'*60
    print(comment)
    return newtbl

def zpcal(intbl, inmagkey, inmagerkey, refmagkey, refmagerkey, sigma=2.0):
    """
    ZERO POINT CALCULATION
    3 SIGMA CLIPPING (MEDIAN)
    """
    import matplotlib.pyplot as plt
    from numpy import median
    from astropy.stats import sigma_clip
    import numpy as np
    #    REMOVE BLANK ROW (=99)    
    indx_avail      = np.where( (intbl[inmagkey] != 99) & (intbl[refmagkey] != 99) )
    intbl           = intbl[indx_avail]
    zplist          = np.copy(intbl[refmagkey] - intbl[inmagkey])
    intbl['zp']     = zplist
    #    SIGMA CLIPPING
    zplist_clip     = sigma_clip(zplist, sigma=sigma, maxiters=None, cenfunc=median, copy=False)
    indx_alive      = np.where( zplist_clip.mask == False )
    indx_exile      = np.where( zplist_clip.mask == True )
    #    RE-DEF. ZP LIST AND INDEXING CLIPPED & NON-CLIPPED
    intbl_alive     = intbl[indx_alive]
    intbl_exile     = intbl[indx_exile]
    #    ZP & ZP ERR. CALC.
    zp              = np.median(np.copy(intbl_alive['zp']))
    zper            = np.std(np.copy(intbl_alive['zp']))
    return zp, zper, intbl_alive, intbl_exile

def zpplot(outname, otbl, xtbl, inmagkey, inmagerkey, refmagkey, refmagerkey, zp, zper, savepath):
    import numpy as np
    import matplotlib.pyplot as plt
    #   FILE NAME
    plt.close('all')
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 16})
    plt.axhline(zp, linewidth=1, linestyle='--', color='gray', label= 'ZP {}'.format(str(round(zp, 3))) )
    plt.axhline(zp+zper, linewidth=1, linestyle='-', color='gray', alpha=0.5, label='ZPER {}'.format(str(round(zper, 3))) )
    plt.axhline(zp-zper, linewidth=1, linestyle='-', color='gray', alpha=0.5 )#, label=str(round(zp, 3)) )
    plt.fill_between([np.min(otbl[refmagkey])-0.05, np.max(otbl[refmagkey])+0.05],
                     zp-zper, zp+zper,
                     color='silver', alpha=0.3)
    plt.errorbar(otbl[refmagkey], otbl['zp'],
                 yerr=sqsum([otbl[inmagerkey], otbl[refmagerkey]]),
                 c='dodgerblue', ms=6, marker='o', ls='',
                 capsize=5, capthick=1,
                 label='NOT CLIPPED ({})'.format(len(otbl)), alpha=0.75)
    plt.scatter( xtbl[refmagkey], xtbl['zp'], color='tomato', s=50, marker='x', linewidth=1, alpha=1.0, label='CLIPPED ({})'.format(len(xtbl)) )
    plt.xlim(np.min(otbl[refmagkey])-0.05, np.max(otbl[refmagkey])+0.05)
    plt.ylim(zp-0.5, zp+0.5)
    #    SETTING
    plt.title(outname, {'fontsize': 16})
    plt.gca().invert_yaxis()
    plt.xlabel('REF.MAG.', {'color': 'black', 'fontsize': 20})
    plt.ylabel('ZERO POINT [AB]', {'color': 'black', 'fontsize': 20})
    plt.legend(loc='best', prop={'size': 14}, edgecolor=None)
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(savepath+outname)
    #    PRINT
    print('MAG TYP     : '+inmagkey)
    print('ZP          : '+str(round(zp, 3)))
    print('ZP ERR      : '+str(round(zper, 3)))
    print('STD.NUMB    : '+str(int(len(otbl))))
    print('REJ.NUMB    : '+str(int(len(xtbl))))
    return 0
    
#%% image check

os.chdir(path_img)
field  = '2685.112-60'

imlist  = imlistsort_ks4(glob.glob('ks4.{}.*.stack.fits'.format(field)))
weight  = imlistsort_ks4(glob.glob('ks4.{}.*.weight.fits'.format(field)))

#%%
# wimg    = weight[1]
# whdul    = fits.open(wimg)
# wdata   = whdul[0].data
# a=[]
# for i in range(len(wdata)//100):
#     for j in range(len(wdata[i])//100):
#         if wdata[100*i][100*j]>0:
#             a.append(wdata[100*i][100*j])
# whdr     = whdul[0].header

#%%
# a =a 
# plt.figure(figsize=(6,6))
# plt.title('Weight Map (n={})'.format(len(a)))
# plt.xlabel('Pixel Values')
# plt.ylabel('Counts')
# plt.minorticks_on()
# plt.hist(a, color='dodgerblue', edgecolor='blue', bins=100)
# plt.axvline(np.min(a), linewidth=3, color='forestgreen', ls='--')
# plt.axvline(np.min(a)*2, linewidth=3, color='forestgreen', ls='--')
# plt.axvline(np.min(a)*3, linewidth=3, color='forestgreen', ls='--')
# plt.axvline(np.min(a)*4, linewidth=3, color='forestgreen', ls='--')
# # plt.axvline(np.max(a), linewidth=3, color='crimson', ls='--')
# plt.axvline(np.max(a)*1/4, linewidth=3, color='crimson', ls='--')
# plt.axvline(np.max(a)*2/4, linewidth=3, color='crimson', ls='--')
# plt.axvline(np.max(a)*3/4, linewidth=3, color='crimson', ls='--')

# plt.axvline(np.quantile(a, 0.25), label='Q1={}'.format(np.quantile(a, 0.25)), linewidth=3, color='crimson', ls='--')
# plt.axvline(np.median(a), label='Median={}'.format(np.median(a)), linewidth=3, color='crimson')
# plt.axvline(np.quantile(a, 0.75), label='Q3={}'.format(np.quantile(a, 0.75)), linewidth=3, color='crimson', ls='--')
# plt.xlim(0, 100)
# plt.ylim(0,10)
# plt.legend()
# plt.grid('--')
# plt.show()
#%% header infos
img     = imlist[0]
hdul    = fits.open(img)
hdr     = hdul[0].header
# sinfo  = ascii.read('ks4_seeing.txt')
# seeing = sinfo[sinfo['col1']==img]['col3']
pixscale=0.4
numimg  = hdr['NUMIMAGE']
fwhm    = []
for i in range(numimg):
    fwhm.append(float(hdr['SEEING{}'.format(i)]))
seeing  = max(fwhm); peeing = seeing/pixscale
band    = hdr['FILTER']
exptime = hdr['EXPTIME']
gain    = hdr['GAIN']

radec   = [hdr['CRVAL1'], hdr['CRVAL2']]
center  = [hdr['NAXIS1']/2, hdr['NAXIS2']/2]

xscale  = hdr['NAXIS1'] * pixscale # arcsec
yscale  = hdr['NAXIS2'] * pixscale # arcsec
radius  = np.mean([xscale/2, yscale/2])/3600 # searching radius in deg

# zps calculated for each images stacked
# zp0     = float(hdr['ZEROP0'])
# zp1     = float(hdr['ZEROP1'])
# zp2     = float(hdr['ZEROP2'])
# zp3     = float(hdr['ZEROP3'])
#%% zeropoint stars to compare
frac        = 1.2
#%%

catname     = '{}{}_{}.cat'.format(path_cat, field, band)
segname     = '{}{}_{}.seg'.format(path_cat, field, band)
apername    = '{}{}_{}.aper'.format(path_cat, field, band)
bkgname     = '{}{}_{}.bkg'.format(path_cat, field, band)

# config            
cfg         = path_cfg+'default.sex'
param       = path_cfg+'default.param'
conv        = path_cfg+'default.conv'
nnw         = path_cfg+'default.nnw'
bkgsize     = 64

# prompt
inim_single = imlist[0]
# inim_dual   = markimg+','+imtbl[band][0]
prompt_cat  = ' -CATALOG_NAME {}'.format(catname)
prompt_aper = ' -PHOT_APERTURES {}'.format(peeing)
prompt_cfg  = ' -c '+cfg+' -PARAMETERS_NAME '+param+' -FILTER_NAME '+conv+' -STARNNW_NAME '+nnw
prompt_opt  = ' -GAIN '+'%.2f'%(gain)+' -PIXEL_SCALE '+'%.2f'%(pixscale)+' -SEEING_FWHM %.2f'%(seeing)
prompt_bkg  = ' -BACK_SIZE {}'.format(bkgsize)
prompt_chk  = ' -CHECKIMAGE_TYPE SEGMENTATION,APERTURES,BACKGROUND -CHECKIMAGE_NAME '+segname+','+apername+','+bkgname
# if refband == 'none':
prompt      = 'sex '+inim_single+prompt_cfg+prompt_aper+prompt_opt+prompt_cat+prompt_bkg#+prompt_chk
# else:
    # prompt      = 'sex '+inim_dual+prompt_cfg+prompt_opt+prompt_cat+prompt_bkg+prompt_chk
os.system(prompt)

# a = ascii.read('{}'.format(catname))

# for i in a:
#     if i['X_IMAGE'] > center[0] and i['Y_IMAGE'] > center[1]:
#         print(i['MAG_AUTO']+zp3, i['MAGERR_AUTO'])
#%%
intbl   = ascii.read(catname)
reftbl  = apass_query(radec[0], radec[1], radius)
#%%
param_matching  = dict(intbl    = intbl,
                       reftbl   = reftbl,
                       inra     = intbl['ALPHA_J2000'], 
                       indec    = intbl['DELTA_J2000'],
                       refra    = reftbl['RAJ2000'], 
                       refdec   = reftbl['DEJ2000'],
                       sep      = 1.0)

mtbl    = matching(**param_matching)
mtbl    = mtbl[mtbl['MAG_AUTO']!=99.0]

inmagkey    = 'MAG_AUTO'
inmagerkey  = 'MAGERR_AUTO'
# inmagkey    = 'MAG_APER'
# inmagerkey  = 'MAGERR_APER'
refmaglower = 14
refmagupper = 20
refmagerupper   = 0.05
inmagerupper    = 0.05
flagcut     = 0

refmagkey   = '{}mag'.format(band)
refmagerkey = 'e_{}mag'.format(band)

param_st4zp = dict(intbl=mtbl,
                   inmagerkey=inmagkey,
                   refmagkey=refmagkey,
                   refmagerkey=refmagerkey,
                   refmaglower=refmaglower,
                   refmagupper=refmagupper,
                   refmagerupper=refmagerupper,
                   inmagerupper=inmagerupper,
                   flagcut=flagcut,
                   )

param_zpcal    = dict(intbl=star4zp(**param_st4zp),
                      inmagkey=inmagkey, inmagerkey=inmagerkey,
                      refmagkey=refmagkey, refmagerkey=refmagerkey,
                      sigma=2.0)

zp, zper, otbl, xtbl = zpcal(**param_zpcal)
alltbl      = vstack([otbl, xtbl]); alltbl      = alltbl[alltbl['NUMBER'].argsort()]
seeing_out  = np.median(alltbl['FWHM_WORLD'])*3600

zpplot(outname='{}_{}_{}.zpcal.png'.format(field, band, inmagkey), otbl=otbl, xtbl=xtbl, inmagkey=inmagkey, inmagerkey=inmagerkey,refmagkey=refmagkey, refmagerkey=refmagerkey, zp=zp, zper=zper, savepath=path_plot)
#%%


if band == 'B':
    magdif  = mtbl['MAG_AUTO'] + zp - mtbl['Bmag']
    magerr  = sqsum([mtbl['e_Bmag'], mtbl['MAGERR_AUTO'], zper])

# elif band == 'V':
#     magdif  = mtbl['MAG_AUTO'] - mtbl['Vmag']
#     magerr  = phot.sqsum(mtbl['e_Vmag'], mtbl['MAGERR_AUTO'])

# elif band == 'R':
#     magdif  = mtbl['MAG_AUTO'] - mtbl['Rmag']
#     magerr  = phot.sqsum(mtbl['e_Rmag'], mtbl['MAGERR_AUTO'])

# elif band == 'I':
#     magdif  = mtbl['MAG_AUTO'] - mtbl['Imag']
#     magerr  = phot.sqsum(mtbl['e_Imag'], mtbl['MAGERR_AUTO'])

# magdif_sat  = magdif[np.where(mtbl['MAG_AUTO']+zp < 13)]
# magdif_uns  = magdif[np.where(mtbl['MAG_AUTO']+zp > 13)]
from astropy.stats import sigma_clip
rmsmagdif   = np.sqrt(np.mean(sigma_clip(magdif,5)**2))
stdmagdif   = np.std(sigma_clip(magdif,5))
mediff      = np.median(magdif)

# Photometry accuracy mapping

# residual graph (x-axis: magnitude, y-axis: delta(mag))
#%%
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3, 3)

ax_main = plt.subplot(gs[0:3, :2])
plt.suptitle('KS4 Photometry Precision Map for {} ({})'.format(field, band))
ax_yDist = plt.subplot(gs[0:3, 2])#,sharey=ax_main)

ax_main.scatter(mtbl['MAG_AUTO']+zp, magdif, c='crimson', marker='o', alpha=0.3)
ax_main.set(xlabel=r'$m_{KS4}$ [ABmag]', ylabel=r'$m_{KS4} - m_{APASS}$ [ABmag]')
ax_main.set_xlim(18,11)
ax_main.set_ylim(-1,1)
ax_main.grid(which='major',linestyle='-', alpha=0.5)
ax_yDist.grid(which='major',linestyle='-', alpha=0.5)
weights   = np.ones_like(magdif)/len(magdif)
# ax_yDist.fill_between([0,0.3], mediff-rmsmagdif, mediff+rmsmagdif, alpha=0.3)
ax_yDist.hist(magdif,weights=weights, bins=np.arange(-1,1,0.05), color='crimson', orientation='horizontal',align='mid')
ax_yDist.set(xlabel='proportion')
ax_yDist.axhline(y=mediff, color='dodgerblue', linestyle='-', alpha=0.75, label='{:6}={:6.3f}mag'.format('Median', mediff))
ax_yDist.axhline(y=mediff-rmsmagdif, color='dodgerblue', linestyle='--', alpha=0.75, label='{:6}={:6.3f}mag'.format('RMSE', rmsmagdif))
ax_yDist.axhline(y=mediff+rmsmagdif, color='dodgerblue', linestyle='--', alpha=0.75)
ax_yDist.axes.yaxis.set_ticklabels([])
ax_yDist.set_ylim(-1,1)
ax_yDist.legend(fontsize=12)
# plt.tight_layout()
plt.savefig('{}{}_{}_graph.png'.format(path_plot, field, band))
plt.close()
#%%
# plt.figure(figsize=(10, 10))
# plt.scatter(mtbl['MAG_AUTO']+zp, magdif, c='crimson', marker='o', alpha=0.3)
# # plt.scatter(mtbl[mtbl['MAG_AUTO']+zp<13]['MAG_AUTO']+zp, magdif_sat, c='grey', marker='x', alpha=0.3)
# plt.xlabel(r'$m_{KS4}$ [ABmag]')
# plt.ylabel(r'$m_{KS4} - m_{APASS}$ [ABmag]')
# if len(mtbl) == 0:
#     maxmag  = 0
# else:
#     maxmag  = np.abs(np.mean(magdif)) + 3.*np.std(magdif)
# maxmis  = np.max([maxmag, 1])
# plt.xlim(18, 10)
# plt.ylim(-1, 1)
# plt.minorticks_on()
# plt.grid(which='major',linestyle='-')
# plt.grid(which='minor',linestyle='--',linewidth=0.1)
#plt.show()
# zp= np.mean([zp0, zp1, zp2, zp3])
# residual map (x-axis: XWIN_IMAGE, y-axis: YWIN_IMAGE)
#%%
from scipy.stats.kde import gaussian_kde
plt.figure()
Z, xedges, yedges = np.histogram2d(mtbl['X_IMAGE'], mtbl['Y_IMAGE'])
plt.pcolormesh(xedges, yedges, Z.T)
plt.show()
#%%
plt.figure(figsize=(10,9))
plt.scatter(mtbl['X_IMAGE'], mtbl['Y_IMAGE'], marker='o', c=magdif, cmap='seismic', alpha=0.7)
#    plt.scatter(mtbl[mtbl['MAG_AUTO']+zp<13]['XWIN_IMAGE'], mtbl[mtbl['MAG_AUTO'+zp]<13]['YWIN_IMAGE'], marker='x', c=magdif_sat, cmap='Spectral')
cbar    = plt.colorbar()
cbar.set_label(r'$m_{KS4} - m_{APASS}$ [mag]')
plt.clim(-0.5, 0.5)
plt.title('KS4 Photometry Precision Map for {} ({}) \n Median Magnitude Difference: {} ABmag \n Magnitude Difference RMSE: {} ABmag'.format(field, band, round(mediff, 3), round(rmsmagdif, 3)))
plt.xlabel('X axis [pixel]')
plt.ylabel('Y axis [pixel]')
# plt.axhline(center[0], ls='--', color='dodgerblue', alpha=0.5)
# plt.axvline(center[1], ls='--', color='dodgerblue', alpha=0.5)
#plt.show()
plt.savefig('{}{}_{}_plot.png'.format(path_plot, field, band))
plt.close()

# mdlist.append(rmsmagdif)

# # Examining quality

# mdfimg0.append(imname)
# mdfimg1.append(band)
# mdfimg2.append(round(mediff,3))
# mdfimg3.append(round(rmsmagdif,3))
# mdfimg4.append(round(zperr,3))
# mdfimg5.append(int(zpstar))
# mdfimg6.append(len(mtbl))
# # result
# # intbls       = ascii.read(path_cat+'KS4/'+band[0]+'_'+field+'.cat') 

# restbl  = Table(np.full((1,29), -99.0), 
#                 names = ['field', 'RA', 'DEC', 
#                          'Bmag_AUTO', 'Bmagerr_AUTO', 'Vmag_AUTO', 'Vmagerr_AUTO', 'Rmag_AUTO', 'Rmagerr_AUTO', 'Imag_AUTO', 'Imagerr_AUTO', 
#                          'Bmag_APER', 'Bmagerr_APER', 'Vmag_APER', 'Vmagerr_APER', 'Rmag_APER', 'Rmagerr_APER', 'Imag_APER', 'Imagerr_APER', 
#                          'CLASS_STAR_B','CLASS_STAR_V','CLASS_STAR_R','CLASS_STAR_I',
#                          'FLAGS',
#                          'bkgsize'])
# restbl['field']    = restbl['field'].astype(str)
# restbl['refband']   = restbl['refband'].astype(str)