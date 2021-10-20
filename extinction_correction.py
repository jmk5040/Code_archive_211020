#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:27:52 2021

@author: sonic
"""

def extcor(mag, ra, dec, band, path_dust='/home/sonic/research/photometry/sfdmap/'):
    
    import sfdmap #https://github.com/kbarbary/sfdmap
    import extinction #https://extinction.readthedocs.io/en/latest/api/extinction.fitzpatrick99.html
    import numpy as np
    
    band        = str(band)
    lambmean    = {'u':3561.8, 'g':4820.0, 'r':6423.0, 'i':7806.7, 'z':9158.5, 'Y':9866.8, 'J':12483.00, 'H':16313.00, 'K':21590.00} # in angstrom
    wave        = np.array([lambmean[band]])
    ebv         = sfdmap.ebv(ra, dec, mapdir=path_dust)
    ext         = extinction.fitzpatrick99(wave, 3.1 * ebv)[0]
    return round(mag-ext, 3)