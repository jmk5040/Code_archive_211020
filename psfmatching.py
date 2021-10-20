#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:22:46 2019

@author: sonic
"""
#%%
import glob
import os, sys
import numpy as np
#from pyraf import iraf
from gppy.phot import phot
from gppy.util.changehdr import changehdr 
from astropy.io import fits
from astropy.io import ascii
from matplotlib import pyplot as plt
from photutils import create_matching_kernel, HanningWindow, TopHatWindow
#%%
path = '~/research/data/PS1/130603B'
os.chdir(os.path.expanduser(path))
os.getcwd()
#%%
inim        = glob.glob('c*.fits')
pixelscale  = 0.258 #arcsec/pixel
#%%
sigband = dict()

for i in range(len(inim)):
    img         = inim[i]
    data, hdr   = fits.getdata(img, header = True)
    ftr         = hdr['FILTER']
    FWHM        = phot.psfex(img, pixelscale)[0] #in pixel unit
    sigma       = FWHM / np.sqrt(8 * np.log(2)) 
    sigband[ftr]= sigma
#%%
fwhm            = []
for i in range(len(inim)):
    img         = inim[i]
    data, hdr   = fits.getdata(img, header = True)
    ftr         = hdr['FILTER']
    FWHM        = phot.psfex(img, pixelscale)[1]
    fwhm.append(ftr)
    fwhm.append(FWHM)
#%%
i, j        = 0, 1    

hdr1        = fits.getheader(inim[i])
hdr2        = fits.getheader(inim[j])

y, x        = np.mgrid[hdr1['NAXIS2'], hdr1['NAXIS1']]

gm1         = fits.getdata(inim[i])
gm2         = fits.getdata(inim[j])

g1          = gm1(x, y)
g2          = gm2(x, y)

g1          /= g1.sum()
g2          /= g2.sum()

window      = TopHatWindow(0.35)
kernel      = create_matching_kernel(g1, g2, window=window)

plt.imshow(kernel, cmap='Greys_r', origin='lower')
plt.colorbar()
#%%
import numpy as np
from astropy.modeling.models import Gaussian2D
from photutils import create_matching_kernel, TopHatWindow
import matplotlib.pyplot as plt

y, x = np.mgrid[0:51, 0:51]
gm1 = Gaussian2D(100, 25, 25, 3, 3)
gm2 = Gaussian2D(100, 25, 25, 5, 5)
g1 = gm1(x, y)
g2 = gm2(x, y)
g1 /= g1.sum()
g2 /= g2.sum()

window = TopHatWindow(0.35)
kernel = create_matching_kernel(g1, g2, window=window)
plt.imshow(kernel, cmap='Greys_r', origin='lower')
plt.colorbar()
#%% developed by G.Lim
def CFHT_convolution(CFHT_input, maoim='all'):
    """
    1. Description 
    : Match pixscale of CFHT (Small pixscale; 0.187"/pix) to that of MAO (Large pixscale; 0.267"/pix) using Swarp. Weight images need to be resampled! 

    CFHT_input : input resampled CFHT image.
    maoim      : MAO image to match

    (1) Load MAO & CFHT dataset : Seeing
    (2) Sigma calcuation

        Sigma ["]     = FWHM ["] / sqrt(8 * loge(2))
        Sigma [pixel] = Sigma ["] * (1./pixelscale)

        new Sigma = sigma_cfht + sqrt(sigma_mao^2 - sigma_cfht^2)
                    (intrinsic)              (sigma to add)

        iraf.images.imfilter.gauss task gives elliptical convolution to input images. (sigma to add) part should be entered to iraf input parameter using pixel unit. 

        input   =   W1+1+2.r.q2.resamp.fits  Input images to be fit
        output  = W1+1+2.r.q2.resamp.gauss.fits  Output images
        sigma   =      1.7988444871269  Sigma of Gaussian along major axis of ellipse
        (ratio  =                   1.) Ratio of sigma in y to x
        (theta  =                   0.) Position angle of ellipse
        (nsigma =                   4.) Extent of Gaussian kernel in sigma
        (bilinea=                  yes) Use bilinear approximation to Gaussian kernel
        (boundar=              nearest) Boundary (constant,nearest,reflect,wrap)
        (constan=                   0.) Constant for boundary extension
        (mode   =                   ql)

    2. Usage
    >>> run /data2/dwarf/code/give_noise/CFHT_convolution_03.py
    >>> CFHT_convolution('W*.r.q?.resamp.fits', maoim='all')
    >>> CFHT_convolution('W1+1+2.r.q2.resamp.fits', maoim='coadd.MAIDANAK.NGC0895.R.fscale.fits')

    3. History
    2019.03.18 Modified by G.Lim.
    """
    import glob
    import os, sys
    import numpy as np
    from pyraf import iraf
    from astropy.io import fits
    from astropy.io import ascii

    # (1) Loading dataset

    ### MAO dataset
    maocat_name = '/data2/dwarf/Maidanak/stacking_20180523/coadd_R/sdss_coadd/rms_mao.R.all.cat'
    maocat      = ascii.read(maocat_name)
    if maoim == 'all' :
        FWHM_MAO = np.median(maocat['seeing'])
    else :
        maoidx      = np.where(maoim == maocat['image'])[0]
        FWHM_MAO    = maocat[maoidx]['seeing'][0]
    # NGC0895 = 1.144
    print( 'Median FWHM = '+str(round(FWHM_MAO,3)) )

    ### CFHT dataset
    imlist = glob.glob(CFHT_input)
    for i in range(len(imlist)):
        inim           = imlist[i]
        data, hdr_CFHT = fits.getdata(inim, header = True)
        infoname       = 'cfht.info.r.resamp.cat'
        infotable      = ascii.read(infoname)
        cfhtidx        = np.where(infotable['image'] == inim)[0]
        FWHM_CFHT      = infotable['seeing'][cfhtidx][0]
        
        # (2) Sigma calculation 
        def fwhm2sigma(fwhm):
            return fwhm / np.sqrt(8 * np.log(2))

        pix_MAO, pix_CFHT = 0.267, 0.267 # For resampled CFHT
        sigma_MAO  = fwhm2sigma( FWHM_MAO *  (1./pix_MAO )  ) # pixel
        sigma_CFHT = fwhm2sigma( FWHM_CFHT * (1./pix_CFHT) )
        print( 'MAO  : ' +str(round(sigma_MAO,3))+ ' [pixel]')
        print( 'CFHT : '+str(round(sigma_CFHT,3))+ ' [piexl]')
        new_sigma = np.sqrt(sigma_MAO**2 - sigma_CFHT**2)
        print( 'new sigma :' + str(new_sigma)+ ' [pixel]')
        iraf.images.imfilter.gauss(input=inim, output=inim[:-5]+'.gauss.fits', sigma = new_sigma, ratio=1.)