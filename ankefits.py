#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:28:15 2021

@author: sonic
"""
import os, glob
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from ankepy.phot import anketool as atl
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, hstack
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.visualization import ZScaleInterval, LinearStretch
from astropy.wcs.utils import proj_plane_pixel_scales as pixscale
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize
from astropy.cosmology import FlatLambdaCDM
#%%
def open_image(imagename):
    
    image, header	= fits.getdata(img, header=True)
    wcs     = WCS(header)

    return image, header, wcs

def open_image_aligned(imagename):

    import montage_wrapper as montage

    hdu     = fits.open(imagename)
    if 'HST' in imagename or 'GMOS' in imagename:
        hdu     = montage.reproject_hdu(hdu[1], north_aligned=True) 
    else:
        hdu     = montage.reproject_hdu(hdu[0], north_aligned=True) 
    image   = hdu.data#; image[np.isnan(image)] = 0
    header  = hdu.header
    wcs     = WCS(header)

    return image, header, wcs
#%%
# pathes
path_base   = '/home/sonic/research/photometry/refimg/'
path_img    = '/home/sonic/research/photometry/refimg/hogwarts/'
path_table  = '/home/sonic/research/photometry/table/'
path_csv    = '/home/sonic/research/sed_modeling/csv/'
path_result = '/home/sonic/research/sed_modeling/result/hogwarts_0728/'
# path_dust   = '/home/sonic/research/photometry/sfdmap/'
# path_cat    = '/home/sonic/research/photometry/cat/'
# path_cfg    = '/home/sonic/research/photometry/config/'
version     = 'hogwarts'
#%% data load
os.chdir(os.path.expanduser(path_img))

imlist      = glob.glob('*.fits'); imlist.sort()
infotbl     = ascii.read(path_csv+'afterglow.csv')
masstbl     = ascii.read(path_csv+'host_mass.csv')
agetbl      = ascii.read(path_csv+'host_age.csv')
targets     = infotbl['GRB']

#%% test

i = 0
plt.figure()
target  = targets[i]
tartbl  = infotbl[infotbl['GRB']==target]
ra_host, de_host = atl.wcs2deg(tartbl['ra_host'][0], tartbl['de_host'][0])
ra_glow, de_glow = atl.wcs2deg(tartbl['ra_glow'][0], tartbl['de_glow'][0])
OA      = tartbl['OA'][0]
error   = tartbl['error'][0]

try:
    img     = [im for im in imlist if target in im][0]
except:
    print('No refimg for {}'.format(target))
# data, hdr	= fits.getdata(img, header=True)
# wcs			= WCS(hdr)
data, hdr, wcs = open_image_aligned(img)
pix         = pixscale(wcs)[0]*3600
obs         = img.split('_')[1]

center      = SkyCoord(ra_host, de_host, unit='deg') # frame='icrs'
circle      = SkyCoord(ra_glow, de_glow, unit='deg')
center_p    = wcs.world_to_pixel(center)
circle_p    = wcs.world_to_pixel(circle)
sep         = center.separation(circle).arcsec


# # nx, ny  = hdr['naxis1'], hdr['naxis2']
# i0, j0 = float(center_p[0]), float(center_p[1])
# [ra0, dec0], = wcs.wcs_pix2world([[i0, j0]], 1)
# hdr.update(crpix1=i0, crpix2=j0, crval1=ra0, crval2=dec0)

# hdr2 = hdr.copy()
# hdr2.update(cd1_1=-np.hypot(hdr["CD1_1"], hdr["CD1_2"]), 
#             cd1_2=0.0, 
#             cd2_1=0.0, 
#             cd2_2=np.hypot(hdr["CD2_1"], hdr["CD2_2"]), 
#             orientat=0.0)


# wcs     = WCS(hdr2)


if sep <= 1.5:
        length  = 3
elif sep > 1.5 and sep <= 3:
    length  = 3
elif sep > 3 and sep <= 5:
    length  = 5
elif sep > 5 and sep <= 8:
    length  = 8
else:
    length  = max(center.separation(circle).arcsec, 10)

if error > 100:
    length  = 75

x1, x2      = int(center_p[0]-length/pix*2), int(center_p[0]+length/pix*2)
y1, y2      = int(center_p[1]-length/pix*2), int(center_p[1]+length/pix*2)

cdata       = data[y1:y2, x1:x2]

plt.subplot(111, projection=wcs)
plt.title('{} ({})'.format(target,obs), fontsize=12)
norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
# norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=3000, vmax=8000)

plt.imshow(cdata, cmap='gray', norm=norm_zscale, extent=[-cdata.shape[1]/2, cdata.shape[1]/2, -cdata.shape[0]/2, cdata.shape[0]/2], origin='lower')
plt.hlines(0, -cdata.shape[1]/2, cdata.shape[1]/2, colors='white', ls=':', alpha=0.5)
plt.vlines(0, -cdata.shape[0]/2, cdata.shape[0]/2, colors='white', ls=':', alpha=0.5)
if OA==1:
    plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='limegreen', label='r = {}"'.format(error)))
elif OA==0:
    plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='yellow', label='r = {}"'.format(error)))
else:
    plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='crimson',label='r = {}"'.format(error)) )
    
# plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=length/pix, fill=False, edgecolor='blue',label='r = {}"'.format(error)) )
# plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=2*length/pix, fill=False, edgecolor='crimson',label='r = {}"'.format(error)) )

plt.xlim(-cdata.shape[1]/2, cdata.shape[1]/2)
plt.ylim(-cdata.shape[0]/2, cdata.shape[0]/2)
# plt.legend(fontsize=10, loc='upper right')
# plt.tick_params(axis='x', fontsize=6)
# plt.tick_params(axis='y', fontsize=6)
# plt.tick_params('x', labelleft=False, labelbottom=False)
# plt.tick_params('y', labelleft=False, labelbottom=False)

plt.show()

#%%
#part I
plt.figure(figsize=(15, 18)) 
plt.rcParams.update({'font.size': 22})
# plt.suptitle('sGRB Host Galaxies Reference Images', fontsize=20)
suspect = ['GRB050906', 'GRB070810B', 'GRB100216A', 'GRB080121', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', # X-ray
           'GRB070809'] # optical
# suspect = ['GRB050906', 'GRB070810B', 'GRB100216A', 'GRB080121', # gamma-ray only
#            'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
#            'GRB070809', 'GRB150424A'] # optical
for i in range(15):

    target  = targets[i]
    tartbl  = infotbl[infotbl['GRB']==target]
    
    ra_host, de_host = atl.wcs2deg(tartbl['ra_host'][0], tartbl['de_host'][0])
    ra_glow, de_glow = atl.wcs2deg(tartbl['ra_glow'][0], tartbl['de_glow'][0])
    OA      = tartbl['OA'][0]
    error   = tartbl['error'][0]
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    dist    = cosmo.angular_diameter_distance(z=tartbl['redshift'][0])
    
    try:
        img     = [im for im in imlist if target in im][0]
    except:
        print('No refimg for {}'.format(target))
    
    
    if target=='GRB050813' or target=='GRB070429B' or target=='GRB070714B' or target=='GRB060502B' or target=='GRB070809':
        data, hdr, wcs  = open_image_aligned(img)
    else:
        data, hdr, wcs  = open_image(img)
    pix         = pixscale(wcs)[0]*3600
    obs         = img.split('_')[1]
    
    center      = SkyCoord(ra_host, de_host, unit='deg') # frame='icrs'
    circle      = SkyCoord(ra_glow, de_glow, unit='deg')
    center_p    = wcs.world_to_pixel(center)
    circle_p    = wcs.world_to_pixel(circle)
    sep         = center.separation(circle).arcsec
    
    if sep <= 1.5:
        length  = 1.25
    elif sep > 1.5 and sep <= 3:
        length  = 2.5
    elif sep > 3 and sep <= 5:
        length  = 5
    elif sep > 5 and sep <= 8:
        length  = 7.5
    else:
        length  = round(max(center.separation(circle).arcsec, 10),-1)
    
    if error > 100:
        length  = 75
    
    x1, x2      = int(center_p[0]-length/pix*2), int(center_p[0]+length/pix*2)
    y1, y2      = int(center_p[1]-length/pix*2), int(center_p[1]+length/pix*2)
    
    cdata       = data[y1:y2, x1:x2]
    
    plt.subplot(5, 3, i+1, projection=wcs)
    if target in suspect:
        plt.title('{}* ({})'.format(target,obs), fontsize=24)
    else:
        plt.title('{} ({})'.format(target,obs), fontsize=24)
    norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(cdata, cmap='gray', norm=norm_zscale, extent=[-cdata.shape[1]/2, cdata.shape[1]/2, -cdata.shape[0]/2, cdata.shape[0]/2])
    plt.hlines(0, -cdata.shape[1]/2, cdata.shape[1]/2, colors='white', ls=':', alpha=0.5)
    plt.vlines(0, -cdata.shape[0]/2, cdata.shape[0]/2, colors='white', ls=':', alpha=0.5)
    if OA==1:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='limegreen'))
        plt.scatter([], [], s=20, edgecolor='limegreen', facecolor='none', label='r = {:.1f}"'.format(error))
    elif OA==0:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='yellow'))#, label='r = {}"'.format(error)))
        plt.scatter([], [], s=20, edgecolor='yellow', facecolor='none', label='r = {:.1f}"'.format(error))
    else:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='crimson'))#,label='r = {}"'.format(error)) )
        plt.scatter([], [], s=20, edgecolor='crimson', facecolor='none', label='r = {:.1f}"'.format(error))
    plt.text(0, -cdata.shape[0]/2*0.85, '{}" = {:.1f} kpc'.format(round(4*length), 1e3*(dist*4*length*u.arcsec).to(u.Mpc, u.dimensionless_angles()).value), bbox={'boxstyle':'round','facecolor':'white','alpha':0.8,'edgecolor':'grey','pad':0.3}, ha='center', va='center')
    plt.legend(fontsize=16, loc='upper right')
    plt.xlim(-cdata.shape[1]/2, cdata.shape[1]/2)
    plt.ylim(-cdata.shape[0]/2, cdata.shape[0]/2)
    plt.tick_params('x', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tick_params('y', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tight_layout(h_pad=1.5, rect=(0,0,1,0.99))
    # plt.show()
    plt.savefig('{}/{}_refimgs_11.pdf'.format(path_result, version))
    # plt.savefig('{}/{}/{}_refimgs_1.png'.format(path_result, version, version), dpi=300)

#%%
#Part II
plt.figure(figsize=(15,18)) 
plt.rcParams.update({'font.size': 22})
# plt.suptitle('sGRB Host Galaxies Reference Images', fontsize=20)

for i in range(15, 30):

    target  = targets[i]
    tartbl  = infotbl[infotbl['GRB']==target]
    
    ra_host, de_host = atl.wcs2deg(tartbl['ra_host'][0], tartbl['de_host'][0])
    ra_glow, de_glow = atl.wcs2deg(tartbl['ra_glow'][0], tartbl['de_glow'][0])
    OA      = tartbl['OA'][0]
    error   = tartbl['error'][0]
    error4c = tartbl['error_c'][0]
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    dist    = cosmo.angular_diameter_distance(z=tartbl['redshift'][0])
    
    try:
        img     = [im for im in imlist if target in im][0]
    except:
        print('No refimg for {}'.format(target))
    
    unaligned   = ['GRB090510', 'GRB100816A', 'GRB140903A', 'GRB160624A', 'GRB160821B']
    if target in unaligned:
        data, hdr, wcs  = open_image_aligned(img)
    else:
        data, hdr, wcs  = open_image(img)
    pix         = pixscale(wcs)[0]*3600
    obs         = img.split('_')[1]
    
    center      = SkyCoord(ra_host, de_host, unit='deg') # frame='icrs'
    circle      = SkyCoord(ra_glow, de_glow, unit='deg')
    center_p    = wcs.world_to_pixel(center)
    circle_p    = wcs.world_to_pixel(circle)
    sep         = center.separation(circle).arcsec
    
    if sep <= 1.:
        length  = 1.25
    elif sep > 1. and sep <= 3:
        length  = 2.5
    elif sep > 3 and sep <= 5:
        length  = 5
    elif sep > 5 and sep <= 8:
        length  = 7.5
    else:
        length  = round(max(center.separation(circle).arcsec, 10),-1)
    
    if error > 100:
        length  = 75
    
    x1, x2      = int(center_p[0]-length/pix*2), int(center_p[0]+length/pix*2)
    y1, y2      = int(center_p[1]-length/pix*2), int(center_p[1]+length/pix*2)
    
    cdata       = data[y1:y2, x1:x2]
    
    plt.subplot(5, 3, i-14, projection=wcs)
    if target in suspect:
        plt.title('{}* ({})'.format(target,obs), fontsize=24)
    else:
        plt.title('{} ({})'.format(target,obs), fontsize=24)
    if target=='GRB101219A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=27623, vmax=27823)
    elif target=='GRB111117A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=3590, vmax=3680)
    elif target=='GRB140903A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=3000, vmax=8900)
    elif target=='GRB181123B':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=9970, vmax=10125)
    else:
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(cdata, cmap='gray', norm=norm_zscale, extent=[-cdata.shape[1]/2, cdata.shape[1]/2, -cdata.shape[0]/2, cdata.shape[0]/2])
    plt.hlines(0, -cdata.shape[1]/2, cdata.shape[1]/2, colors='white', ls=':', alpha=0.5)
    plt.vlines(0, -cdata.shape[0]/2, cdata.shape[0]/2, colors='white', ls=':', alpha=0.5)
    if OA==1:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error4c/pix, fill=False, edgecolor='limegreen'))
        plt.scatter([], [], s=20, edgecolor='limegreen', facecolor='none', label='r = {:.1f}"'.format(error))
    elif OA==0:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='yellow'))#, label='r = {}"'.format(error)))
        plt.scatter([], [], s=20, edgecolor='yellow', facecolor='none', label='r = {:.1f}"'.format(error))
    else:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='crimson'))#,label='r = {}"'.format(error)) )
        plt.scatter([], [], s=20, edgecolor='crimson', facecolor='none', label='r = {:.1f}"'.format(error))
    
    plt.text(0, -cdata.shape[0]/2*0.85, '{}" = {:.1f} kpc'.format(round(4*length), 1e3*(dist*4*length*u.arcsec).to(u.Mpc, u.dimensionless_angles()).value), bbox={'boxstyle':'round','facecolor':'white','alpha':0.8,'edgecolor':'grey','pad':0.3}, ha='center', va='center')
    plt.legend(fontsize=16, loc='upper right')
    plt.xlim(-cdata.shape[1]/2, cdata.shape[1]/2)
    plt.ylim(-cdata.shape[0]/2, cdata.shape[0]/2)
    plt.tick_params('x', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tick_params('y', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tight_layout(h_pad=1.5, rect=(0,0,1,0.99))

    # plt.show()
    plt.savefig('{}/{}_refimgs_22.pdf'.format(path_result, version))
    # plt.savefig('{}/{}/{}_refimgs_2.png'.format(path_result, version, version), dpi=300)
#%% part III

plt.figure(figsize=(15,18)) 
plt.rcParams.update({'font.size': 22})
# plt.suptitle('sGRB Host Galaxies Reference Images', fontsize=20)

for i in range(30, 42):

    target  = targets[i]
    tartbl  = infotbl[infotbl['GRB']==target]
    
    ra_host, de_host = atl.wcs2deg(tartbl['ra_host'][0], tartbl['de_host'][0])
    ra_glow, de_glow = atl.wcs2deg(tartbl['ra_glow'][0], tartbl['de_glow'][0])
    OA      = tartbl['OA'][0]
    error   = tartbl['error'][0]
    error4c = tartbl['error_c'][0]
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    dist    = cosmo.angular_diameter_distance(z=tartbl['redshift'][0])
    
    try:
        img     = [im for im in imlist if target in im][0]
    except:
        print('No refimg for {}'.format(target))
    
    unaligned   = ['GRB090510', 'GRB100816A', 'GRB140903A', 'GRB160624A', 'GRB160821B']
    if target in unaligned:
        data, hdr, wcs  = open_image_aligned(img)
    else:
        data, hdr, wcs  = open_image(img)
    pix         = pixscale(wcs)[0]*3600
    obs         = img.split('_')[1]
    
    center      = SkyCoord(ra_host, de_host, unit='deg') # frame='icrs'
    circle      = SkyCoord(ra_glow, de_glow, unit='deg')
    center_p    = wcs.world_to_pixel(center)
    circle_p    = wcs.world_to_pixel(circle)
    sep         = center.separation(circle).arcsec
    
    if sep <= 1.:
        length  = 1.25
    elif sep > 1. and sep <= 3:
        length  = 2.5
    elif sep > 3 and sep <= 5:
        length  = 5
    elif sep > 5 and sep <= 8:
        length  = 7.5
    else:
        length  = round(max(center.separation(circle).arcsec, 10),-1)
    
    if error > 100:
        length  = 75
    
    x1, x2      = int(center_p[0]-length/pix*2), int(center_p[0]+length/pix*2)
    y1, y2      = int(center_p[1]-length/pix*2), int(center_p[1]+length/pix*2)
    
    cdata       = data[y1:y2, x1:x2]
    
    plt.subplot(5, 3, i-29, projection=wcs)
    if target in suspect:
        plt.title('{}* ({})'.format(target,obs), fontsize=24)
    else:
        plt.title('{} ({})'.format(target,obs), fontsize=24)
    if target=='GRB101219A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=27623, vmax=27823)
    elif target=='GRB111117A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=3590, vmax=3680)
    elif target=='GRB140903A':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=3000, vmax=8900)
    elif target=='GRB181123B':
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), vmin=9970, vmax=10125)
    else:
        norm_zscale	= ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(cdata, cmap='gray', norm=norm_zscale, extent=[-cdata.shape[1]/2, cdata.shape[1]/2, -cdata.shape[0]/2, cdata.shape[0]/2])
    plt.hlines(0, -cdata.shape[1]/2, cdata.shape[1]/2, colors='white', ls=':', alpha=0.5)
    plt.vlines(0, -cdata.shape[0]/2, cdata.shape[0]/2, colors='white', ls=':', alpha=0.5)
    if OA==1:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error4c/pix, fill=False, edgecolor='limegreen'))
        plt.scatter([], [], s=20, edgecolor='limegreen', facecolor='none', label='r = {:.1f}"'.format(error))
    elif OA==0:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='yellow'))#, label='r = {}"'.format(error)))
        plt.scatter([], [], s=20, edgecolor='yellow', facecolor='none', label='r = {:.1f}"'.format(error))
    else:
        plt.gca().add_patch(Circle(xy=(circle_p[0]-center_p[0], circle_p[1]-center_p[1]), radius=error/pix, fill=False, edgecolor='crimson'))#,label='r = {}"'.format(error)) )
        plt.scatter([], [], s=20, edgecolor='crimson', facecolor='none', label='r = {:.1f}"'.format(error))
    
    plt.text(0, -cdata.shape[0]/2*0.85, '{}" = {:.1f} kpc'.format(round(4*length), 1e3*(dist*4*length*u.arcsec).to(u.Mpc, u.dimensionless_angles()).value), bbox={'boxstyle':'round','facecolor':'white','alpha':0.8,'edgecolor':'grey','pad':0.3}, ha='center', va='center')
    plt.legend(fontsize=16, loc='upper right')
    plt.xlim(-cdata.shape[1]/2, cdata.shape[1]/2)
    plt.ylim(-cdata.shape[0]/2, cdata.shape[0]/2)
    plt.tick_params('x', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tick_params('y', left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.tight_layout(h_pad=1.5, rect=(0,0,1,0.99))

    # plt.show()
    plt.savefig('{}/{}_refimgs_33.pdf'.format(path_result, version))
#%% r mag plot

ps1z    = infotbl[infotbl['source']==1]['redshift']
ps1rmag = infotbl[infotbl['source']==1]['rmag']
ps1err  = infotbl[infotbl['source']==1]['rmagerr']

desz    = infotbl[infotbl['source']==2]['redshift']
desrmag = infotbl[infotbl['source']==2]['rmag']
deserr  = infotbl[infotbl['source']==2]['rmagerr']

gmosz   = infotbl[infotbl['source']==3]['redshift']
gmosrmag= infotbl[infotbl['source']==3]['rmag']
gmoserr = infotbl[infotbl['source']==3]['rmagerr']

forsz   = infotbl[infotbl['source']==5]['redshift']
forsrmag= infotbl[infotbl['source']==5]['rmag']
forserr = infotbl[infotbl['source']==5]['rmagerr']

etcz    = infotbl[infotbl['source']==4]['redshift']
etcrmag = infotbl[infotbl['source']==4]['rmag']
etcerr  = infotbl[infotbl['source']==4]['rmagerr']

plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 15})
plt.title('sGRB Hosts r-band Apparent Magnitudes')
plt.xlabel('redshift')
plt.ylabel(r'$m_{r}$ [ABmag]')
plt.ylim(26,10)
# plt.xlim(0, 2)
# plt.tick_params('both', labelsize=7)
plt.errorbar(ps1z, ps1rmag, yerr=ps1err, ms=6, marker='o', ls='', color='red', capsize=3, capthick=1, label='PanSTARRS', alpha=0.8)
plt.errorbar(desz, desrmag, yerr=deserr, ms=6, marker='s', ls='', color='crimson', capsize=3, capthick=1, label='DES / DECaLS', alpha=0.8)
plt.errorbar(gmosz, gmosrmag, yerr=gmoserr, ms=6, marker='^', ls='', color='maroon', capsize=3, capthick=1, label='GMOS', alpha=0.8)
plt.errorbar(forsz, forsrmag, yerr=forserr, ms=10, marker='*', ls='', color='salmon', capsize=3, capthick=1, label='FORS1 (R)', alpha=0.8)
plt.errorbar(etcz, etcrmag, yerr=etcerr, ms=6, marker='D', ls='', color='firebrick', capsize=3, capthick=1, label='The Others', alpha=0.8)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%% offset plot
from ankepy.phot import anketool as atl
from astropy.coordinates import SkyCoord
import astropy.units as u

seps        = []
seperrs     = []
cosmo   = FlatLambdaCDM(H0=70, Om0=0.27)
for i in range(len(infotbl)):
    z   = infotbl['redshift'][i]
    dist    = cosmo.angular_diameter_distance(z=z)
    c1  = SkyCoord(infotbl['ra_host'][i], infotbl['de_host'][i], unit=(u.hourangle, u.deg), frame='icrs', distance=dist)
    c2  = SkyCoord(infotbl['ra_glow'][i], infotbl['de_glow'][i], unit=(u.hourangle, u.deg), frame='icrs', distance=dist)
    sep = c1.separation_3d(c2).kpc
    seperr  = 1e3*(dist*infotbl['error'][i]*u.arcsec).to(u.Mpc, u.dimensionless_angles()).value
    seps.append(sep)
    seperrs.append(seperr)
    
seps1   = [] # high confidence
seps2   = [] # low confidence
mass1   = []
mass2   = []
age1    = []
age2    = []
z1  = []
z2  = []

for i in range(len(seps)):
    if infotbl['GRB'][i] in suspect:
        seps2.append(seps[i])
        mass2.append(masstbl['This Work'][i])
        z2.append(masstbl['z'][i])
        age2.append(agetbl['This Work'][i])
    else:
        seps1.append(seps[i])
        mass1.append(masstbl['This Work'][i])
        z1.append(masstbl['z'][i])
        age1.append(agetbl['This Work'][i])
        # print(infotbl['GRB'][i], seps[i], masstbl['This Work'][i])
        
#%%

plt.figure(figsize=(18,8))
plt.subplot(122)
plt.scatter([], [], c='k', marker='o', s=70, edgecolor='dimgrey', label='High Confidence')
plt.scatter([], [], c='k', marker='X', s=70, edgecolor='dimgrey', label='Low Confidence')
# plt.scatter(seps1, mass1, c=z1, marker='o', s=70, cmap='seismic', edgecolor='dimgrey', zorder=1); plt.clim(0, 1)
# plt.scatter(seps2, mass2, c=z2, marker='X', s=70, cmap='seismic', edgecolor='dimgrey', zorder=2); plt.clim(0, 1)
# plt.errorbar(seps, masstbl['This Work'], xerr=seperrs, yerr=[masstbl['l0'], masstbl['u0']], ms=1, marker='o', ls='', color='k', capsize=3, capthick=1, alpha=0.2, zorder=0)
plt.scatter(seps1, age1, c=mass1, marker='o', s=70, cmap='seismic', edgecolor='dimgrey', zorder=1); plt.clim(7, 12)
plt.scatter(seps2, age2, c=mass2, marker='X', s=70, cmap='seismic', edgecolor='dimgrey', zorder=2); plt.clim(7, 12)
plt.errorbar(seps, agetbl['This Work'], xerr=seperrs, yerr=[agetbl['l0'], agetbl['u0']], ms=1, marker='o', ls='', color='k', capsize=3, capthick=1, alpha=0.15, zorder=0)
# plt.semilogx(); plt.xlim(1, 110)
plt.colorbar().set_label(r'$log(M_{*} / M_{\odot})$')
plt.legend()
plt.xlim(0, 110)
# plt.semilogy()
plt.ylim(0.0, 12)
plt.grid(alpha=0.3, ls='--')
plt.xlabel('Projected Offset [kpc]')
plt.ylabel('Stellar Population Age [Gyr]')
# plt.ylabel(r'$log(M_{*} / M_{\odot}) $')
# plt.savefig('{}/{}_offset.png'.format(path_result, version))
# plt.close()
plt.subplot(121)
# plt.figure(figsize=(8,8))
plt.scatter([], [], c='limegreen', marker='o', s=70, edgecolor='dimgrey', label='Optical Afterglow')
plt.scatter([], [], c='yellow', marker='o', s=70, edgecolor='dimgrey', label='X-ray Afterglow')
plt.scatter([], [], c='crimson', marker='o', s=70, edgecolor='dimgrey', label=r'$\gamma$-ray Only')
for i in range(len(seps)):
    if infotbl['GRB'][i] in suspect:
        if infotbl['OA'][i] == 1:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='limegreen', marker='x', ms=1, ls='', capsize=3, capthick=1, alpha=0.6)
        elif infotbl['OA'][i] == 0:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='gold', marker='x', ms=1, ls='', capsize=3, capthick=1, alpha=0.6)
        else:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='crimson', marker='x', ms=1, ls='', capsize=3, capthick=1, alpha=0.6)
    else:
        if infotbl['OA'][i] == 1:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='limegreen', marker='o', ms=1, ls='', capsize=3, capthick=1, alpha=0.9)
        elif infotbl['OA'][i] == 0:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='gold', marker='o', ms=1, ls='', capsize=3, capthick=1, alpha=0.9)
        else:
            plt.errorbar(seps[i], masstbl['This Work'][i], xerr=seperrs[i], yerr=[[masstbl['l0'][i]], [masstbl['u0'][i]]], color='crimson', marker='o', ms=1, ls='', capsize=3, capthick=1, alpha=0.9)

for i in range(len(seps)):
    if infotbl['GRB'][i] in suspect:
        if infotbl['OA'][i] == 1:
            plt.scatter(seps[i], masstbl['This Work'][i], color='limegreen', marker='X', edgecolor='dimgrey', s=70, zorder=99, alpha=0.8)
        elif infotbl['OA'][i] == 0:
            plt.scatter(seps[i], masstbl['This Work'][i], color='gold', marker='X', edgecolor='dimgrey', s=70, zorder=99, alpha=0.8)
        else:
            plt.scatter(seps[i], masstbl['This Work'][i], color='crimson', marker='X', edgecolor='dimgrey', s=70, zorder=99, alpha=0.8)
    else:
        if infotbl['OA'][i] == 1:
            plt.scatter(seps[i], masstbl['This Work'][i], color='limegreen', marker='o', edgecolor='dimgrey', s=70, zorder=100, alpha=0.8)
        elif infotbl['OA'][i] == 0:
            plt.scatter(seps[i], masstbl['This Work'][i], color='gold', marker='o', edgecolor='dimgrey', s=70, zorder=100, alpha=0.8)
        else:
            plt.scatter(seps[i], masstbl['This Work'][i], color='crimson', marker='o', edgecolor='dimgrey', s=70, zorder=100, alpha=0.8)
        
# plt.scatter(seps2, mass2, c=z2, marker='X', s=70, cmap='seismic', edgecolor='dimgrey', zorder=2); plt.clim(0, 1)
# plt.errorbar(seps, masstbl['This Work'], xerr=seperrs, yerr=[masstbl['l0'], masstbl['u0']], ms=1, marker='o', ls='', color='k', capsize=3, capthick=1, alpha=0.2, zorder=0)
# plt.semilogx(); plt.xlim(1, 110)
# plt.colorbar().set_label('redshift')
plt.legend()
plt.xlim(0, 110)
plt.ylim(7, 12)
plt.grid(alpha=0.3, ls='--')
plt.xlabel('Projected Offset [kpc]')
# plt.ylabel('Host Galaxy Stellar Mass []')/
plt.ylabel(r'$log(M_{*} / M_{\odot}) $')
plt.savefig('{}/{}_offset2.png'.format(path_result, version))
plt.tight_layout()
plt.close()














