#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:15:33 2021

@author: sonic
"""
#%% reduction - works in DRAGONS shell
def Gemini_Reduction(target, path_reduce, path_rawimg, detector):
    
    import os, glob
    from astropy.io import fits
    from gempy.utils import logutils
    from recipe_system import cal_service
    from gempy.adlibrary import dataselect
    from recipe_system.reduction.coreReduce import Reduce

    # unzip data

    os.chdir(path_rawimg)
    unzip_raw   = glob.glob('*.bz2')
    if len(unzip_raw) != 0:
        os.system('rm md5sums.txt README.txt')
        os.system('bzip2 -d *.bz2')

    # start
    
    os.chdir(path_reduce)
    
    # setting up the logger
    
    logutils.config(file_name='gmos_data_reduction.log')
    
    # setting up the configuration file
    
    with open('/home/sonic/.geminidr/rsys.cfg', 'r') as f:
        t   = f.read()
        c   = t.split('\n')
        for l in c:
            if l.startswith('database_dir ='):
                database = l.split('= ')[-1]
        t   = t.replace(database, path_reduce)
    f.close()
    with open('/home/sonic/.geminidr/rsys.cfg', 'w') as f:
        f.write(t)
    f.close()
    
    # setting up the calibration service
    
    caldb       = cal_service.CalibrationService()
    caldb.config()
    caldb.init()
    cal_service.set_calservice()
        
    # creating list of files
    
    if detector     =='GMOS-S':
        all_files   = glob.glob(path_rawimg+'S*.fits'); all_files.sort()
    elif detector   =='GMOS-N':
        all_files   = glob.glob(path_rawimg+'N*.fits'); all_files.sort()
    else:
        print('Please specify the detector: "GMOS-S" or "GMOS-N".')
    
    # creating master bias
    
    biases      = dataselect.select_data(all_files, ['BIAS'], [])
    reduce_bias = Reduce()
    reduce_bias.files.extend(biases)
    reduce_bias.runr()
    caldb.add_cal(reduce_bias.output_filenames[0])
    
    # reducing
    
    rpro    = [] # bands will not be reduced
    spro    = [] # bands will not be stacked
    
    for band in ['g', 'r', 'i', 'z']:
        
        flats  = dataselect.select_data(all_files, ['FLAT'], [], dataselect.expr_parser('filter_name=="{}"'.format(band)))
        sci    = dataselect.select_data(all_files, [], ['CAL'], dataselect.expr_parser('(observation_class=="science" and filter_name=="{}")'.format(band)))
        
        # image check
        
        if len(sci)==0:
            rpro.append(band)
            continue # reduction process skipped
        elif len(sci)==1:
            spro.append(band)
            pass
        
        reduce_flats = Reduce()
        reduce_flats.files.extend(flats)
        reduce_flats.runr()
        caldb.add_cal(reduce_flats.output_filenames[0])
        
        reduce_fringe = Reduce()
        reduce_fringe.files.extend(sci)
        reduce_fringe.recipename = 'makeProcessedFringe'
        reduce_fringe.runr()
        caldb.add_cal(reduce_fringe.output_filenames[0])

        reduce_science = Reduce()
        reduce_science.files.extend(sci)
        reduce_science.runr()

    # result check
    
    print('='*15+' Preprocessing Result '+'='*15)
    for band in ['g', 'r', 'i', 'z']:
        if band in rpro:
            print('No {}-band science image in the raw data directory.'.format(band))
        elif band in spro:
            print('{}-band science image was reduced but not stacked'.format(band))
        else:
            print('{}-band science images were reduced & stacked.'.format(band))
    print('='*52)

    # saving
    
    out_files   = glob.glob(path_reduce+'*_s*.fits'); out_files.sort()
    for out in out_files:
        
        hdul    = fits.open(out)
        hdr     = hdul[0].header
        
        instru  = hdr['INSTRUME']
        ftr     = hdr['FILTER2'][0]
        
        if 'stack' in out:
            exptime = round(hdr['EXPTIME']*hdr['NCOMBINE'])
        elif 'sourcesDetected' in out:
            exptime = round(hdr['EXPTIME'])
            data, nhdr = fits.getdata(out, header = True)
            nhdr['NCOMBINE'] = 1
            fits.writeto(out, data, nhdr, overwrite=True)

        newname = '../{}_{}_{}_{}s.fits'.format(target, instru, ftr, exptime)
        os.system('cp {} {}'.format(out, newname))

    # remove cache
    os.system('rm *.db *.fits *.log')
    os.system('rm -r calibrations')
    os.system('rm -r .reducecache')

    return 0

# =============================================================================
# user setting
# =============================================================================

import os, glob, time

# 1. pathes

obs         = 'Gemini'
path_img    = '/home/sonic/research/photometry/image/{}/'.format(obs)

# 2. target

os.chdir(path_img)
targets     = glob.glob('*'); targets.sort()
for t in targets:
    print(t)
target      = input('Enter the target to preprocess:\t')

path_reduce = '/home/sonic/research/photometry/image/{}/{}/process/'.format(obs, target)
path_rawimg = '/home/sonic/research/photometry/image/{}/{}/rawdata/'.format(obs, target)

# 3. process

gmos_s      = glob.glob('{}/S*.fits*'.format(path_rawimg))
gmos_n      = glob.glob('{}/N*.fits*'.format(path_rawimg))

start   = time.time()

if len(gmos_s) != 0:
    print('GMOS-S data reduction is in progress.')
    Gemini_Reduction(target, path_reduce, path_rawimg, "GMOS-S")
    print('GMOS-S data reduction is done.')
    time.sleep(5)

if len(gmos_n) != 0:
    print('GMOS-N data reduction is in progress.')
    Gemini_Reduction(target, path_reduce, path_rawimg, "GMOS-N")
    print('GMOS-N data reduction is done.')

end     = time.time()

print('DRAGONS reduction process finished. Total running time: {}sec'.format(round(end-start)))

#%%
import os, glob, time

# 1. pathes

obs         = 'Gemini'
path_img    = '/home/sonic/research/photometry/image/{}/'.format(obs)

# 2. target

os.chdir(path_img)
targets     = glob.glob('*'); targets.sort()
for t in targets:
    print(t)
target      = input('Enter the target to preprocess:\t')

path_reduce = '/home/sonic/research/photometry/image/{}/{}/process/'.format(obs, target)
path_rawimg = '/home/sonic/research/photometry/image/{}/{}/rawdata/'.format(obs, target)

from astropy.io import fits
from gempy.utils import logutils
from recipe_system import cal_service
from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce

# unzip data

os.chdir(path_rawimg)
unzip_raw   = glob.glob('*.bz2')
if len(unzip_raw) != 0:
    os.system('rm md5sums.txt README.txt')
    os.system('bzip2 -d *.bz2')

# start

os.chdir(path_reduce)

# setting up the logger

logutils.config(file_name='gmos_data_reduction.log')

# setting up the configuration file

with open('/home/sonic/.geminidr/rsys.cfg', 'r') as f:
    t   = f.read()
    c   = t.split('\n')
    for l in c:
        if l.startswith('database_dir ='):
            database = l.split('= ')[-1]
    t   = t.replace(database, path_reduce)
f.close()
with open('/home/sonic/.geminidr/rsys.cfg', 'w') as f:
    f.write(t)
f.close()

# setting up the calibration service

caldb       = cal_service.CalibrationService()
caldb.config()
caldb.init()
cal_service.set_calservice()
    
# creating list of files
detector    = 'GMOS-N'
if detector     =='GMOS-S':
    all_files   = glob.glob(path_rawimg+'S*.fits'); all_files.sort()
elif detector   =='GMOS-N':
    all_files   = glob.glob(path_rawimg+'N*.fits'); all_files.sort()
else:
    print('Please specify the detector: "GMOS-S" or "GMOS-N".')

# creating master bias

biases      = dataselect.select_data(all_files, ['BIAS'], [])
reduce_bias = Reduce()
reduce_bias.files.extend(biases)
reduce_bias.runr()
caldb.add_cal(reduce_bias.output_filenames[0])

# reducing

# rpro    = [] # bands will not be reduced
# spro    = [] # bands will not be stacked

# for band in ['g', 'r', 'i', 'z']:
band    = 'z'
flats  = dataselect.select_data(all_files, ['FLAT'], [], dataselect.expr_parser('filter_name=="{}"'.format(band)))
sci    = dataselect.select_data(all_files, [], ['CAL'], dataselect.expr_parser('(observation_class=="science" and filter_name=="{}")'.format(band)))
    
    # image check
    
# if len(sci)==0:
#     rpro.append(band)
#     # continue # reduction process skipped
# elif len(sci)==1:
#     spro.append(band)
#     pass

reduce_flats = Reduce()
reduce_flats.files.extend(flats)
reduce_flats.runr()
caldb.add_cal(reduce_flats.output_filenames[0])

reduce_fringe = Reduce()
reduce_fringe.files.extend(sci)
reduce_fringe.recipename = 'makeProcessedFringe'
reduce_fringe.runr()
caldb.add_cal(reduce_fringe.output_filenames[0])

reduce_science = Reduce()
reduce_science.files.extend(sci)
reduce_science.runr()

# result check

# print('='*15+' Preprocessing Result '+'='*15)
# for band in ['g', 'r', 'i', 'z']:
#     if band in rpro:
#         print('No {}-band science image in the raw data directory.'.format(band))
#     elif band in spro:
#         print('{}-band science image was reduced but not stacked'.format(band))
#     else:
#         print('{}-band science images were reduced & stacked.'.format(band))
# print('='*52)

# saving

out_files   = glob.glob(path_reduce+'*_s*.fits'); out_files.sort()
for out in out_files:
    
    hdul    = fits.open(out)
    hdr     = hdul[0].header
    
    instru  = hdr['INSTRUME']
    ftr     = hdr['FILTER2'][0]
    
    if 'stack' in out:
        exptime = round(hdr['EXPTIME']*hdr['NCOMBINE'])
    elif 'sourcesDetected' in out:
        exptime = round(hdr['EXPTIME'])
        data, nhdr = fits.getdata(out, header = True)
        nhdr['NCOMBINE'] = 1
        fits.writeto(out, data, nhdr, overwrite=True)

    newname = '../{}_{}_{}_{}s.fits'.format(target, instru, ftr, exptime)
    os.system('cp {} {}'.format(out, newname))

# remove cache
os.system('rm *.db *.fits *.log')
os.system('rm -r calibrations')
os.system('rm -r .reducecache')