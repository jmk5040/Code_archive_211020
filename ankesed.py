#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Update: 2021.01.05.
Writer: Mankeun Jeong (Ankemun)

Description:

How to use:

environments: 
input - 
output - 
filter information from https://github.com/gbrammer/eazy-photoz

References:

"""
#%% .csv file update log
"""
@20.12.30
filters not splited, 40 targets. Magdata not pruned.
failfit = [13, 21, 29, 30]

@21.01.04
All filter spilted, 40 targets, including 4 problematic samples.
Need to construct this code. Filter curve files not updated.

@21.01.12
Filter curves all updated.
Triming out some photometric points going adrift or overlapping with others.
There are 4 problematic samples. One of them is not included in previous failfit sample.
GRB051210 - using previous result.

@21.01.26
GRB150424A - wrong target

@21.01.28
GRB150424A - Fixed by HST data (True target detected)

@21.04.09
All photometry data have been checked.
Total: 47 samples
Suspected: GRB050906, GRB061217*, GRB070729, GRB070809, GRB080123, GRB090515, GRB150424A (7 samples)
Shortage of Data: GRB051210A, GRB080905A*, GRB121226A*, GRB201221D* (4 samples)
Phot-z only: GRB051210A, GRB070729, GRB120804A, GRB121226A (4 samples)
Fail-to-Fit: GRB050813, GRB061217, GRB070714B, GRB111117A, GRB120804A, GRB180418A (6 samples)

@21.04.15
GRB050813, GRB070714B, GRB111117A: Modified to solve failfits.
Problems in fitting shape: 051210a, 070714b, 070809, 101219a
Sample rejects: GRB051210A, GRB061217, GRB120804A, GRB180418A, GRB121226A, GRB201221D, GRB080905A
Total Sample: 40

@21.04.21
GRB090426A, GRB200826A: Collapsar Imposters

"""
#%%
import re
import time
import copy
import os, glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from ankepy.phot import anketool as atl
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table, vstack, hstack
#%% 0. Environments
#%% working directories

# pathes
path_base   = '/home/sonic/research/sed_modeling/'
path_csv    = '/home/sonic/research/sed_modeling/csv/'      # magnitude csv files
path_lib    = '/home/sonic/research/sed_modeling/lib/'      # filter, cat files
path_input  = '/home/sonic/research/sed_modeling/input/'    # FAST input 
path_output = '/home/sonic/research/sed_modeling/output/'   # SED raw output
path_result = '/home/sonic/research/sed_modeling/result/'   # Figures using the output

# required directories

# main
os.chdir(path_base)
current     = glob.glob('*')
require     = 'csv code config lib input output result'.split(' ')
default     = 'hdfn_fs99' # in lib directory

for folder in require:
    if folder not in current:
        os.system('mkdir {}'.format(folder))
    else:
        pass
    
# sub
os.chdir(path_lib)
current     = glob.glob('*')
require     = 'filter_curves filter_plots save'.split(' ')

for folder in require:
    if folder not in current:
        os.system('mkdir {}'.format(folder))
    else:
        pass

#%% checking the version of current process

project     = 'hogwarts'

os.chdir(path_csv)

allcsv      = glob.glob('{}_*.csv'.format(project)); allcsv.sort()
print('='*5+'Choose the csv file to process'+'='*5)
for csv in allcsv: print(csv)
print('='*40)

version     = input("Enter the file name w/o the extension :\t").split('.csv')[0]

os.chdir(path_base)

#%% making input, output directory for the process

verpath     = [path_csv, path_input, path_output, path_result]

for path in verpath:
    
    os.chdir(path)
    current     = glob.glob('*')
    
    if version not in current:
        os.system('mkdir {}'.format(version))
    else:
        pass

#%% 1. Formatting
#%% checking filter interpretation & curves

os.chdir(path_csv)

magtbl      = ascii.read('{}.csv'.format(version))
targets     = magtbl['Target']
cols        = magtbl.colnames

os.chdir(path_lib)

ftrinfo     = open('FILTER.RES.latest.info', 'r').readlines() # https://github.com/gbrammer/eazy-photoz
translate   = ascii.read('translate.cat') # write down manually

filters     = [f for f in cols if f.startswith('f_')]

ftrtbl      = Table()
for f in filters:
    if f not in translate['filter']:
        print("Warning: Filter name '{}' is not defined in your translate file.".format(f))
    else:
        linenum     = int(translate[translate['filter']==f]['lines'][0][1:])
        lambda_c    = float(ftrinfo[linenum-1].split('lambda_c= ')[-1].split(' ')[0])
        dummy       = Table([[f], [lambda_c]], names=['filter', 'lambda_c'])
        ftrtbl      = vstack([ftrtbl, dummy])

# This cell will be used for magnitude plotting as well.

# #%% filter translate file generator
# 
# os.chdir(path_lib)
# 
# with open('empty_translate.cat', 'w') as t:
#     for c in cols:
#         if c[0]=='f':
#             t.write(c+' F\n')
#         elif c[0]=='e':
#             t.write(c+' E\n')
#         else:
#             pass

#%% making catalog file from the csv table

# multi   = False # type 'True' for making all target's data in one cat file
# single  = not multi # for making a cat file for each target

multi   = False
single  = False

if multi:
    
    # catalog file
    os.chdir(path_input+version)
    
    magcat  = copy.deepcopy(magtbl)
    
    for i in range(len(magtbl)):
        for j in range(len(cols)):
            if j == 0:
                magcat[i][j]        = i+1
            elif 0 < j < 4:
                pass
            elif j%2 == 0:
                if np.ma.is_masked(magtbl[i][j]):
                    magcat[i][j]    = -99
                else:
                    magcat[i][j]    = atl.rsig(atl.AB2Jy(magtbl[i][j]), 5)
            elif j%2 == 1:
                if np.ma.is_masked(magtbl[i][j]):
                    magcat[i][j]    = -99
                elif magtbl[i][j-1] == 99: # when the data have only upperlimit
                    magcat[i][j]    = atl.rsig(atl.AB2Jy(magtbl[i][j]), 5)
                else:
                    magcat[i][j]    = atl.rsig(atl.eAB2Jy(magtbl[i][j-1], magtbl[i][j]), 5)
    
    try:
        magcat.rename_column('Target', '#id')
    except:
        pass
    
    try:
        magcat.remove_columns(['hostRA', 'hostDec'])
    except:
        pass

    magcat.write('{}.cat'.format(version), format='ascii', delimiter='\t', fill_values=[(-99.0,'-99')], overwrite=True)
    
    # parameter file
    with open('{}/{}.param'.format(path_lib, default), 'r') as f:
        contents    = f.readlines()
        with open('{}.param'.format(version), 'w') as p:
            for line in contents:
                if line.startswith('CATALOG'):
                    p.write(line.replace(default, version))
                elif line.startswith('N_SIM'):
                    p.write(line.replace('0', '100'))
                elif line.startswith('RESOLUTION'):
                    p.write(line.replace("= 'hr'", "= 'lr'"))
                elif line.startswith('NO_MAX_AGE'):
                    p.write(line.replace('= 0', '= 1'))
                else:
                    p.write(line)
    f.close()
    p.close()
    
    # translate file
    with open('{}/{}.translate'.format(path_lib, default), 'r') as f:
        contents    = f.readlines()
        with open('{}.translate'.format(version), 'w') as t:
            for line in contents:
                t.write(line)
    f.close()
    t.close()

elif single:

    # catalog file
    os.chdir(path_input+version)
    
    for i in range(len(magtbl)):
        band    = []
        flux    = []
        for j in range(len(cols)):
            if j == 0:
                band.append('#id')
                flux.append([1])
                target  = magtbl[i][j]
            elif 0 < j < 3:
                pass
            elif j == 3:
                band.append(cols[j]) # z_spec
                flux.append([magtbl[i][j]])
            elif j%2 == 0:
                if np.ma.is_masked(magtbl[i][j]):
                    pass
                else:
                    band.append(cols[j])
                    flux.append([atl.rsig(atl.AB2Jy(magtbl[i][j]), 5)])
            elif j%2 == 1:
                if np.ma.is_masked(magtbl[i][j]):
                    pass
                elif magtbl[i][j-1] == 99: # when the data have only upperlimit
                    band.append(cols[j])
                    flux.append([atl.rsig(atl.AB2Jy(magtbl[i][j]), 5)])
                else:
                    band.append(cols[j])
                    flux.append([atl.rsig(atl.eAB2Jy(magtbl[i][j-1], magtbl[i][j]), 5)])
        magline = Table(flux, names=band)
        magline.write('{}.cat'.format(target), format='ascii', delimiter='\t', overwrite=True)

        # parameter file
        with open('{}/{}.param'.format(path_lib, default), 'r') as f:
            contents    = f.readlines()
            with open('{}.param'.format(target), 'w') as p:
                for line in contents:
                    if line.startswith('CATALOG'):
                        p.write(line.replace(default, target))
                    elif line.startswith('N_SIM'):
                        p.write(line.replace('0', '100'))
                    elif line.startswith('OUTPUT_DIR'):
                        p.write(line.replace('result', '{}_result'.format(target)))
                    elif line.startswith('RESOLUTION'):
                        p.write(line.replace("= 'hr'", "= 'lr'"))
                    elif line.startswith('Z_MAX'):
                        p.write(line.replace("= 1.00", "= 2.00"))
                    # elif line.startswith('NO_MAX_AGE'):
                    #     p.write(line.replace('= 0', '= 1'))
                    else:
                        p.write(line)
        f.close()
        p.close()
        
        # translate file
        with open('{}/{}.translate'.format(path_lib, default), 'r') as f:
            contents    = f.readlines()
            with open('{}.translate'.format(target), 'w') as t:
                for line in contents:
                    t.write(line)
        f.close()
        t.close()

os.chdir(path_base)

#%% magnitude plot
os.chdir(path_csv)
subdir  = version

for i in range(len(targets)):
    
    # manual choice
    # i   = np.where(np.array(targets)==input('Enter the target:\t'))[0][0]
    
    # photometric points information
    target  = targets[i]
    zspec   = magtbl['z_spec'][i]
    maginfo = magtbl[magtbl['Target']==target]
    
    mag     = []
    magerr  = []
    wl      = []
    ftr     = []
    
    # when only upper limit is available
    magupp  = []
    wlupp   = []
    ftrupp  = []

    for j in range(len(maginfo.colnames)):
        col     = maginfo.colnames[j]
        try: nextcol = maginfo.colnames[j+1]
        except: pass
        if np.ma.is_masked(maginfo[col]):
            pass
        elif col[0] == 'f':
            if maginfo[col][0] != 99:
                mag.append(maginfo[col][0])
                magerr.append(maginfo[nextcol][0])
                wl.append(ftrtbl[ftrtbl['filter']==col]['lambda_c'][0])
                ftr.append(col[2:])
            else:
                magupp.append(maginfo[nextcol][0])
                wlupp.append(ftrtbl[ftrtbl['filter']==col]['lambda_c'][0])
                ftrupp.append(col[2:])
        else:
            pass

    # plot
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 16})
    
    plt.title('{} Host Galaxy SED Photometry Points (z={})'.format(target, zspec))
    
    plt.errorbar(wl, mag, yerr=magerr, ms=6, marker='s', ls='', c='dodgerblue', capsize=4, capthick=1)
    plt.scatter(wlupp, magupp, marker='v', c='dodgerblue', s=40)

    for k in range(len(ftr)):
        plt.annotate(ftr[k], (wl[k], mag[k]-magerr[k]-0.1), alpha=0.6)
    for l in range(len(ftrupp)):
        plt.annotate(ftrupp[l], (wlupp[l], magupp[l]-0.1), alpha=0.6)

    plt.xlabel('$\lambda_{obs} \ [\AA]$')
    plt.ylabel('$ABmag$')
    plt.ylim(np.round(np.median(mag))+4, np.round(np.median(mag))-4)
    plt.grid(ls='-', alpha=0.7)
    # plt.show()
    plt.savefig('{}/{}.png'.format(subdir, target), overwrite=True)
    plt.close()
#%% 2. Modeling
#%%
# This process would be done in QSO server w/ FAST. Transport the input files.
# When the fitting process is done, download the result into output directory.

# shell script file generator
os.chdir(path_input+version)
os.system('rm *.sh')

core    = int(input("How many cores do you want to use?:\t"))
each    = len(targets)//core
remn    = len(targets)%core

# dividing targets
for i in range(core):
    with open('{}_{}.sh'.format(version, i), 'w') as s:
        s.write('#!/bin/sh\n')
        for j in range(each):
            s.write('../fast {}.param\n'.format(targets[i*each+j]))
    s.close()

# appending remnants
with open('{}_{}.sh'.format(version, core-1), 'a') as r:
    for k in range(remn):
        r.write('../fast {}.param\n'.format(targets[core*each+k]))
r.close()


#%% 3. Plotting
#%% spectral energy distribution
os.chdir(path_output+'/'+version)

multi   = False # type 'True' for making all target's data in one cat file
single  = not multi # for making a cat file for each target

if multi:
    mulcat  = ascii.read('{}/{}/{}.cat'.format(path_input, version, version))

for i in range(len(targets)):
    
    # manual choice
    # i   = np.where(np.array(targets)==input('Enter the target:\t'))[0][0]
    
    # photometric points information (input flux [erg s-1 cm-2 Hz-1])
    target  = targets[i]

    if multi:
        zspec   = mulcat['z_spec'][i]
        flxinfo = mulcat[mulcat['id']==i+1]
    elif single:
        sincat  = ascii.read('{}/{}/{}.cat'.format(path_input, version, target))
        zspec   = sincat['z_spec'][0]
        flxinfo = sincat
    
    flx     = []
    flxerr  = []
    wl      = []
    ftr     = []
    
    # when only upper limit is available
    flxupp  = []
    wlupp   = []
    ftrupp  = []

    for j in range(len(flxinfo.colnames)):
        col     = flxinfo.colnames[j]
        try: nextcol = flxinfo.colnames[j+1]
        except: pass
        if flxinfo[col][0] == -99:
            pass
        elif col[0] == 'f':
            if flxinfo[col][0] > 1e-50:
                wavelength  = ftrtbl[ftrtbl['filter']==col]['lambda_c'][0]
                flx.append(atl.flux_f2w(flxinfo[col][0], wavelength))
                flxerr.append(atl.flux_f2w(flxinfo[nextcol][0], wavelength))
                wl.append(wavelength)
                ftr.append(col[2:])
            else:
                wavelength  = ftrtbl[ftrtbl['filter']==col]['lambda_c'][0]
                flxupp.append(atl.flux_f2w(flxinfo[nextcol][0], wavelength))
                wlupp.append(wavelength)
                ftrupp.append(col[2:])
        else:
            pass

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 16})
    
    plt.title('{} Host Galaxy SED (z={})'.format(target, zspec))
    
    plt.errorbar(wl, flx, yerr=flxerr, ms=6, marker='s', ls='', c='dodgerblue', capsize=4, capthick=1)
    plt.scatter(wlupp, flxupp, marker='v', c='dodgerblue', s=40)

    for k in range(len(ftr)):
        plt.annotate(ftr[k], (wl[k], flx[k]+flxerr[k]-0.1), alpha=0.6)
    for l in range(len(ftrupp)):
        plt.annotate(ftrupp[l], (wlupp[l], flxupp[l]-0.1), alpha=0.6)
    
    try:
        if multi:
            model   = ascii.read('result/BEST_FITS/{}_{}.fit'.format(version, i+1)) # multiple fitting
            plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
        elif single:
            model   = ascii.read('{}_result/BEST_FITS/{}_{}.fit'.format(target, target, 1)) # separate fitting
            # fits    = ascii.read('{}_result/BEST_FITS/{}_{}.input_res.fit'.format(target, target, 1))
            # factor  = model[model['col1']>fits['col1'][0]]['col2'][0]/fits['col2'][0]
            # fitsf   = [atl.flux_f2w(fits['col2'][k], fitsw[k]) for k in range(len(fits))]
            plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
            # plt.scatter(fits['col1'], fits['col2']*factor, c='crimson', marker='s')
    except:
        print('There are no SED fitting result in the output directory.')
    
    plt.xlabel('$\lambda_{obs} \ [\AA]$')
    plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
    plt.xlim(000, 50000)
    plt.ylim(000, max(flx)*1.5)
    plt.grid(ls='-', alpha=0.7)

    plt.savefig('{}/{}/{}.png'.format(path_result, version, target), overwrite=True)
    plt.close()

#%% filter curve plot

os.chdir(path_lib)
subdir  = 'filter_curves'
figdir  = 'filter_plots'

#check the task
split   = True
merge   = False
plot    = False

# split the data from 'FILTER.RES.latest'
if split:
    with open('FILTER.RES.latest') as f:
        for line in f:
            if re.search('lambda_c', line):
                band    = line.strip().replace(' ', '/').replace('/', '_')
                try: out.close()
                except: pass
                out     = open('{}/{}.txt'.format(subdir, band), 'w')
                out.write(line)
            else: out.write(line)
    out.close()
    f.close()

# merge the filter curves into 'FILTER.RES.anke'
if merge:
    allftr      = glob.glob('{}/*.txt'.format(subdir))
    allftr.sort()
    # merged info file
    with open('FILTER.RES.anke.info', 'w') as f:
        with open('FILTER.RES.latest.info') as r:
            for line in r:
                f.write(line.strip()+'\n')
        r.close()
    f.close()
    # merged curve file
    with open('FILTER.RES.anke', 'w') as f:
        with open('FILTER.RES.anke.info') as r:
            for line in r:
                # fnum    = line.split()[1]
                # lc      = line.split('lambda_c')[-1].strip('= ').split()[0]
                name    = line.strip().replace(' ', '/').replace('/', '_')[8:]
                lf      = glob.glob('{}/*{}*.txt'.format(subdir, name))[0]
                if len(glob.glob('{}/*{}*.txt'.format(subdir, name)))>1: 
                    print('Check if there are more than one filter named: *{}.'.format(name))
                with open('{}'.format(lf)) as i:
                    f.write(i.read().strip()+'\n')
                i.close()
        r.close()
    f.close()
        
# plot each filter curve
if plot:
    allftr      = glob.glob('{}/*.txt'.format(subdir))
    allftr.sort()
    for ftr in allftr:
        name    = ftr.split('/')[-1].split('lambda_c')[0].strip('_.dat.txt')
        curv    = ascii.read(ftr, data_start=1, names=['num', 'wl', 'tr'])
        wl      = curv['wl']
        tr      = curv['tr']
        
        plt.figure(figsize=(10,7))
        plt.rcParams.update({'font.size': 15})

        plt.title('Filter Transmission Curve \n({})'.format(name))
        plt.plot(wl, tr, c='dodgerblue')
        plt.xlabel('$\lambda_{obs} \ [\AA]$')
        plt.ylabel('$Transmission$')
        plt.grid(alpha=0.7)
        plt.show()
        plt.ylim(-0.05, max(1.05, max(tr)*1.05))
        plt.show()
        plt.savefig('{}/ftrcurv_{}.png'.format(figdir, name), overwrite=True)
        plt.close()
        
#%% filter data numbering & sorting - use ankesed_sub.py (filter data)
#%% 4. Resulting
#%% multiplot - use ankesed_sub.py (multiplot)
#%% histogram - use ankesed_sub.py (output merge)
vlspec = vlspec
vlphot = vlphot

z_sdss       = vlphot['z']
mass_sdssp   = vlphot['lmass']
sfr_sdssp    = vlphot['lsfr']
age_sdssp    = 10**vlphot['lage']/1e9
mass_sdsss   = vlspec['lmass']
sfr_sdsss    = vlspec['lsfr']
age_sdsss    = 10**vlspec['lage']/1e9

# portss  = ascii.read('{}/portsmouth_starforming.csv'.format(path_csv))
# portsp  = ascii.read('{}/portsmouth_passive.csv'.format(path_csv))

# mass_portss  = portss['logMass']
# sfr_portss   = np.log10(portss['SFR'])
# age_portss   = portss['age']
# mass_portsp  = portsp['logMass']
# sfr_portsp   = np.log10(portsp['SFR'])
# age_portsp   = portsp['age']

#%% field galaxy all
os.chdir(path_output+version)

result  = ascii.read('{}_result/merged.fout'.format(version))
failfit = [target for target in list(result['Targets']) if result[result['Targets']==target]['chi2'][0]==-1]
suspect = ['GRB050906', 'GRB070810B', 'GRB100216A', 'GRB080121', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
           'GRB070809']#, 'GRB150424A'] # optical

# mass

plt.figure(figsize=(36, 12))
plt.rcParams.update({'font.size': 24})
plt.suptitle('Histograms of sGRB Host Galaxy Properties (cf. SDSS Field Galaxies)')
plt.tight_layout()
mass_tot    = []
mass_app    = []

for i in range(len(result['lmass'])):
    mass_tot.append(result['lmass'][i])
    if result['Targets'][i] not in suspect:
        mass_app.append(result['lmass'][i])

medmass_tot = np.median(mass_tot)
medmass_app = np.median(mass_app)

plt.subplot(1,3,1)
plt.title('Stellar Mass')
plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.ylabel('Proportion')
mtweights   = np.ones_like(mass_tot)/len(mass_tot)
maweights   = np.ones_like(mass_app)/len(mass_tot)
plt.hist(mass_tot, weights=mtweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(mass_app, weights=maweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
msweights   = np.ones_like(mass_sdssp)/len(mass_sdssp)
plt.hist(mass_sdssp, weights=msweights, edgecolor='limegreen', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
msweights   = np.ones_like(mass_sdssp[-int(0.3*len(vlphot)):])/len(mass_sdssp)
plt.hist(mass_sdssp[-int(0.3*len(vlphot)):], weights=msweights, edgecolor='gold', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medmass_tot, label='{:.2f}'.format(medmass_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medmass_app, label='{:.2f}'.format(medmass_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(7, 13)
plt.ylim(0, 0.5)

# sfr

sfr_tot    = []
sfr_app    = []

for i in range(len(result['lsfr'])):
    sfr_tot.append(result['lsfr'][i])
    if result['Targets'][i] not in suspect:
        sfr_app.append(result['lsfr'][i])

medsfr_tot = np.median(sfr_tot)
medsfr_app = np.median(sfr_app)

plt.subplot(1, 3, 2)
plt.title('Star Formation Rate')
plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
plt.ylabel('Proportion')
stweights     = np.ones_like(sfr_tot)/len(sfr_tot)
saweights     = np.ones_like(sfr_app)/len(sfr_tot)
plt.hist(sfr_tot, weights=stweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(sfr_app, weights=saweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
ssweights     = np.ones_like(sfr_sdssp)/len(sfr_sdssp)
plt.hist(sfr_sdssp, weights=ssweights, edgecolor='limegreen', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
ssweights     = np.ones_like(sfr_sdssp[-int(0.3*len(vlphot)):])/len(sfr_sdssp)
plt.hist(sfr_sdssp[-int(0.3*len(vlphot)):], weights=ssweights, edgecolor='gold', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medsfr_tot, label='{:.2f}'.format(medsfr_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medsfr_app, label='{:.2f}'.format(medsfr_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(-8, 4)
plt.ylim(0, 0.5)
plt.grid(alpha=0.3)

# age

age_tot    = []
age_app    = []

for i in range(len(result['lage'])):
    age_tot.append(10**result['lage'][i]/1e9)
    if result['Targets'][i] not in suspect:
        age_app.append(10**result['lage'][i]/1e9)

medage_tot = np.median(age_tot)
medage_app = np.median(age_app)

plt.subplot(1, 3, 3)
plt.title('Mean Stellar Age')
plt.xlabel(r'$t_{age} \ [Gyr]$')
plt.ylabel('Proportion')
atweights     = np.ones_like(age_tot)/len(age_tot)
aaweights     = np.ones_like(age_app)/len(age_tot)
plt.hist(age_tot, weights=atweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='Total Samples (n={})'.format(len(sfr_tot)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(age_app, weights=aaweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='High Confidence Samples (n={})'.format(len(sfr_app)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
# plt.hist([], edgecolor='limegreen', fill=False, alpha=0.8, linewidth=5, label='SDSS DR12 Galaxies (All)')
asweights     = np.ones_like(age_sdssp)/len(age_sdssp)
plt.hist(age_sdssp, weights=asweights, edgecolor='limegreen', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Field Galaxies (n={})'.format(len(age_sdssp)), alpha=0.8, linewidth=5)
asweights     = np.ones_like(age_sdssp[-int(0.3*len(vlphot)):])/len(age_sdssp)
plt.hist(age_sdssp[-int(0.3*len(vlphot)):], weights=asweights, edgecolor='gold', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Higher Merger Rate (n={})'.format(len(age_sdssp[-int(0.3*len(vlphot)):])), alpha=0.8, linewidth=5)
plt.axvline(medage_tot, label='Median (Total) = {:.2f}'.format(medage_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medage_app, label='Median (High Confidence) = {:.2f}'.format(medage_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(0,12)
plt.ylim(0, 0.5)
plt.grid(alpha=0.3)
plt.savefig('{}/{}/histogram_field2.png'.format(path_result, version), overwrite=True)
plt.close()
#%% low-z only
os.chdir(path_output+version)

result  = ascii.read('{}_result/merged.fout'.format(version))
resultz  = result[result['z']<0.3]
failfit = [target for target in list(result['Targets']) if result[result['Targets']==target]['chi2'][0]==-1]
suspect = ['GRB050906', 'GRB070810B', 'GRB080121', 'GRB100216A', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
           'GRB070809']#, 'GRB150424A'] # optical

# mass

plt.figure(figsize=(36, 12))
plt.rcParams.update({'font.size': 24})
plt.suptitle('Histograms of Local (z<0.3) sGRB Host Galaxy Properties (cf. SDSS Field Galaxies)')
plt.tight_layout()
mass_totz    = []
mass_appz    = []

for i in range(len(resultz['lmass'])):
    mass_totz.append(resultz['lmass'][i])
    if resultz['Targets'][i] not in suspect:
        mass_appz.append(resultz['lmass'][i])

medmass_totz = np.median(mass_totz)
medmass_appz = np.median(mass_appz)

plt.subplot(1,3,1)
plt.title('Stellar Mass')
plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.ylabel('Proportion')
mtweights   = np.ones_like(mass_totz)/len(mass_totz)
maweights   = np.ones_like(mass_appz)/len(mass_totz)
plt.hist(mass_totz, weights=mtweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(mass_appz, weights=maweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
msweights   = np.ones_like(mass_sdssp)/len(mass_sdssp)
plt.hist(mass_sdssp, weights=msweights, edgecolor='limegreen', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
msweights   = np.ones_like(mass_sdssp[-int(0.3*len(vlphot)):])/len(mass_sdssp)
plt.hist(mass_sdssp[-int(0.3*len(vlphot)):], weights=msweights, edgecolor='gold', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medmass_totz, label='{:.2f}'.format(medmass_totz), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medmass_appz, label='{:.2f}'.format(medmass_appz), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(7, 13)
plt.ylim(0, 0.5)

# sfr

sfr_totz    = []
sfr_appz    = []

for i in range(len(resultz['lsfr'])):
    sfr_totz.append(resultz['lsfr'][i])
    if resultz['Targets'][i] not in suspect:
        sfr_appz.append(resultz['lsfr'][i])

medsfr_totz = np.median(sfr_totz)
medsfr_appz = np.median(sfr_appz)

plt.subplot(1, 3, 2)
plt.title('Star Formation Rate')
plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
plt.ylabel('Proportion')
stweights     = np.ones_like(sfr_totz)/len(sfr_totz)
saweights     = np.ones_like(sfr_appz)/len(sfr_totz)
plt.hist(sfr_totz, weights=stweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(sfr_appz, weights=saweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
ssweights     = np.ones_like(sfr_sdssp)/len(sfr_sdssp)
plt.hist(sfr_sdssp, weights=ssweights, edgecolor='limegreen', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
ssweights     = np.ones_like(sfr_sdssp[-int(0.3*len(vlphot)):])/len(sfr_sdssp)
plt.hist(sfr_sdssp[-int(0.3*len(vlphot)):], weights=ssweights, edgecolor='gold', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medsfr_totz, label='{:.2f}'.format(medsfr_totz), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medsfr_appz, label='{:.2f}'.format(medsfr_appz), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(-8, 4)
plt.ylim(0, 0.5)
plt.grid(alpha=0.3)

# age

age_totz    = []
age_appz    = []

for i in range(len(resultz['lage'])):
    age_totz.append(10**resultz['lage'][i]/1e9)
    if resultz['Targets'][i] not in suspect:
        age_appz.append(10**resultz['lage'][i]/1e9)

medage_totz = np.median(age_totz)
medage_appz = np.median(age_appz)

plt.subplot(1, 3, 3)
plt.title('Mean Stellar Age')
plt.xlabel(r'$t_{age} \ [Gyr]$')
plt.ylabel('Proportion')
atweights     = np.ones_like(age_totz)/len(age_totz)
aaweights     = np.ones_like(age_appz)/len(age_totz)
plt.hist(age_totz, weights=atweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='Total Low-z Samples (n={})'.format(len(sfr_totz)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(age_appz, weights=aaweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='High Confidence Samples (n={})'.format(len(sfr_appz)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
# plt.hist([], edgecolor='limegreen', fill=False, alpha=0.8, linewidth=5, label='SDSS DR12 Galaxies (All)')
asweights     = np.ones_like(age_sdssp)/len(age_sdssp)
plt.hist(age_sdssp, weights=asweights, edgecolor='limegreen', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Field Galaxies (n={})'.format(len(age_sdssp)), alpha=0.8, linewidth=5)
asweights     = np.ones_like(age_sdssp[-int(0.3*len(vlphot)):])/len(age_sdssp)
plt.hist(age_sdssp[-int(0.3*len(vlphot)):], weights=asweights, edgecolor='gold', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Higher Merger Rate (n={})'.format(len(age_sdssp[-int(0.3*len(vlphot)):])), alpha=0.8, linewidth=5)
plt.axvline(medage_totz, label='Median (Total) = {:.2f}'.format(medage_totz), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medage_appz, label='Median (High Confidence) = {:.2f}'.format(medage_appz), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(0,12)
plt.ylim(0, 0.5)
plt.grid(alpha=0.3)
plt.savefig('{}/{}/histogram_lowz.png'.format(path_result, version), overwrite=True)
plt.close()

#%% PASSIVE ONLY
os.chdir(path_output+version)

result  = ascii.read('{}_result/merged.fout'.format(version))
failfit = [target for target in list(result['Targets']) if result[result['Targets']==target]['chi2'][0]==-1]
suspect = ['GRB050906', 'GRB070810B', 'GRB080121', 'GRB100216A', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
           'GRB070809']#, 'GRB150424A'] # optical


plt.figure(figsize=(36, 12))
plt.suptitle('Histograms of sGRB Passive Host Galaxy Properties (cf. SDSS Passive Galaxies)')
plt.rcParams.update({'font.size': 24})
plt.tight_layout()

mass_ptot    = []
mass_papp    = []

for i in range(len(result['lmass'])):
    if 10**result['lssfr'][i]*1e9 < 1/(3*cosmo.age(result['z'][i]).value):
        mass_ptot.append(result['lmass'][i])
        if result['Targets'][i] not in suspect:
            mass_papp.append(result['lmass'][i])

medmass_ptot = np.median(mass_ptot)
medmass_papp = np.median(mass_papp)

plt.subplot(1,3,1)
plt.title('Stellar Mass')
plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.ylabel('Proportion')
mtweights   = np.ones_like(mass_ptot)/len(mass_ptot)
maweights   = np.ones_like(mass_papp)/len(mass_ptot)
plt.hist(mass_ptot, weights=mtweights, color='dodgerblue', bins=np.arange(0,14,0.50), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(mass_papp, weights=maweights, color='dodgerblue', bins=np.arange(0,14,0.50), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
msweights   = np.ones_like(mass_sdsss)/len(mass_sdsss)
plt.hist(mass_sdsss, weights=msweights, edgecolor='limegreen', fill=False, bins=np.arange(0,14,0.50), rwidth=1, alpha=0.8, linewidth=5)
msweights   = np.ones_like(mass_sdsss[-int(0.3*len(vlspec)):])/len(mass_sdsss)
plt.hist(mass_sdsss[-int(0.3*len(vlspec)):], weights=msweights, edgecolor='gold', fill=False, bins=np.arange(0,14,0.50), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medmass_ptot, label='{:.2f}'.format(medmass_ptot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medmass_papp, label='{:.2f}'.format(medmass_papp), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(7, 13)
plt.ylim(0, 1.0)

# sfr

sfr_ptot    = []
sfr_papp    = []

for i in range(len(result['lsfr'])):
    if 10**result['lssfr'][i]*1e9 < 1/(3*cosmo.age(result['z'][i]).value):
        sfr_ptot.append(result['lsfr'][i])
        if result['Targets'][i] not in suspect:
            sfr_papp.append(result['lsfr'][i])

medsfr_ptot = np.median(sfr_ptot)
medsfr_papp = np.median(sfr_papp)

plt.subplot(1, 3, 2)
plt.title('Star Formation Rate')
plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
plt.ylabel('Proportion')
stweights     = np.ones_like(sfr_ptot)/len(sfr_ptot)
saweights     = np.ones_like(sfr_papp)/len(sfr_ptot)
plt.hist(sfr_ptot, weights=stweights, color='dodgerblue', bins=np.arange(-8, 4), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(sfr_papp, weights=saweights, color='dodgerblue', bins=np.arange(-8, 4), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
ssweights     = np.ones_like(sfr_sdsss)/len(sfr_sdsss)
plt.hist(sfr_sdsss, weights=ssweights, edgecolor='limegreen', fill=False, bins=np.arange(-8, 4), rwidth=1, alpha=0.8, linewidth=5)
ssweights     = np.ones_like(sfr_sdsss[-int(0.3*len(vlspec)):])/len(sfr_sdsss)
plt.hist(sfr_sdsss[-int(0.3*len(vlspec)):], weights=ssweights, edgecolor='gold', fill=False, bins=np.arange(-8, 4), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medsfr_ptot, label='{:.2f}'.format(medsfr_ptot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medsfr_papp, label='{:.2f}'.format(medsfr_papp), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(-8, 4)
plt.ylim(0, 0.5)
plt.grid(alpha=0.3)

# age

age_ptot    = []
age_papp    = []

for i in range(len(result['lage'])):
    if 10**result['lssfr'][i]*1e9 < 1/(3*cosmo.age(result['z'][i]).value):
        age_ptot.append(10**result['lage'][i]/1e9)
        if result['Targets'][i] not in suspect:
            age_papp.append(10**result['lage'][i]/1e9)

medage_ptot = np.median(age_ptot)
medage_papp = np.median(age_papp)

plt.subplot(1, 3, 3)
plt.title('Mean Stellar Age')
plt.xlabel(r'$t_{age} \ [Gyr]$')
plt.ylabel('Proportion')
atweights     = np.ones_like(age_ptot)/len(age_ptot)
aaweights     = np.ones_like(age_papp)/len(age_ptot)
plt.hist(age_ptot, weights=atweights, color='dodgerblue', bins=np.arange(0,12), rwidth=1, label='Total Passive Samples (n={})'.format(len(sfr_ptot)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(age_papp, weights=aaweights, color='dodgerblue', bins=np.arange(0,12), rwidth=1, label='High Confidence Samples (n={})'.format(len(sfr_papp)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
asweights     = np.ones_like(age_sdsss)/len(age_sdsss)
plt.hist(age_sdsss, weights=asweights, edgecolor='limegreen', bins=np.arange(0,12), rwidth=1, fill=False, label='Passive Galaxies (n={})'.format(len(age_sdsss)), alpha=0.8, linewidth=5)
asweights     = np.ones_like(age_sdsss[-int(0.3*len(vlspec)):])/len(age_sdsss)
plt.hist(age_sdsss[-int(0.3*len(vlspec)):], weights=asweights, edgecolor='gold', bins=np.arange(0,12), rwidth=1, fill=False, label='Higher Merger Rate (n={})'.format(len(age_sdsss[-int(0.3*len(vlspec)):])), alpha=0.8, linewidth=5)
plt.axvline(medage_ptot, label='Median (Total) = {:.2f}'.format(medage_ptot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medage_papp, label='Median (High Confidence) = {:.2f}'.format(medage_papp), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(0,12)
plt.ylim(0, 1.0)
plt.grid(alpha=0.3)
plt.savefig('{}/{}/histogram_passive.png'.format(path_result, version), overwrite=True)
plt.close()

#%% PORTSMOUTH

# mass

plt.figure(figsize=(36, 12))
plt.suptitle('Histograms of sGRB Host Galaxy Properties (cf. Portsmouth Catalog)')
plt.rcParams.update({'font.size': 24})
plt.tight_layout()
mass_tot    = []
mass_app    = []

for i in range(len(result['lmass'])):
    mass_tot.append(result['lmass'][i])
    if result['Targets'][i] not in suspect:
        mass_app.append(result['lmass'][i])

medmass_tot = np.median(mass_tot)
medmass_app = np.median(mass_app)

plt.subplot(1,3,1)
plt.title('Stellar Mass')
plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.ylabel('Proportion')
mtweights   = np.ones_like(mass_tot)/len(mass_tot)
maweights   = np.ones_like(mass_app)/len(mass_tot)
plt.hist(mass_tot, weights=mtweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(mass_app, weights=maweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
msweights   = np.ones_like(mass_portsp)/len(mass_portsp)
plt.hist(mass_portsp, weights=msweights, edgecolor='gold', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
msweights   = np.ones_like(mass_portss)/len(mass_portss)
plt.hist(mass_portss, weights=msweights, edgecolor='limegreen', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medmass_tot, label='{:.2f}'.format(medmass_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medmass_app, label='{:.2f}'.format(medmass_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(7, 13)
plt.ylim(0, 0.6)

# sfr

sfr_tot    = []
sfr_app    = []

for i in range(len(result['lsfr'])):
    sfr_tot.append(result['lsfr'][i])
    if result['Targets'][i] not in suspect:
        sfr_app.append(result['lsfr'][i])

medsfr_tot = np.median(sfr_tot)
medsfr_app = np.median(sfr_app)

plt.subplot(1, 3, 2)
plt.title('Star Formation Rate')
plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
plt.ylabel('Proportion')
stweights     = np.ones_like(sfr_tot)/len(sfr_tot)
saweights     = np.ones_like(sfr_app)/len(sfr_tot)
plt.hist(sfr_tot, weights=stweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(sfr_app, weights=saweights, color='dodgerblue', bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
ssweights     = np.ones_like(sfr_portsp)/len(sfr_portsp)
plt.hist(sfr_portsp, weights=ssweights, edgecolor='gold', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
ssweights     = np.ones_like(sfr_portss)/len(sfr_portss)
plt.hist(sfr_portss, weights=ssweights, edgecolor='limegreen', fill=False, bins=np.arange(-8, 4, 1), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medsfr_tot, label='{:.2f}'.format(medsfr_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medsfr_app, label='{:.2f}'.format(medsfr_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(-8, 4)
plt.ylim(0, 0.6)
plt.grid(alpha=0.3)

# age

age_tot    = []
age_app    = []

for i in range(len(result['lage'])):
    age_tot.append(10**result['lage'][i]/1e9)
    if result['Targets'][i] not in suspect:
        age_app.append(10**result['lage'][i]/1e9)

medage_tot = np.median(age_tot)
medage_app = np.median(age_app)

plt.subplot(1, 3, 3)
plt.title('Mean Stellar Age')
plt.xlabel(r'$t_{age} \ [Gyr]$')
plt.ylabel('Proportion')
atweights     = np.ones_like(age_tot)/len(age_tot)
aaweights     = np.ones_like(age_app)/len(age_tot)
plt.hist(age_tot, weights=atweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='Total Samples (n={})'.format(len(sfr_tot)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(age_app, weights=aaweights, color='dodgerblue', bins=np.arange(0,12,1.00), rwidth=1, label='High Confidence Samples (n={})'.format(len(sfr_app)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
asweights     = np.ones_like(age_portss)/len(age_portss)
plt.hist(age_portss, weights=asweights, edgecolor='limegreen', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Portsmouth Starforming Model'.format(len(age_portss)), alpha=0.8, linewidth=5)
asweights     = np.ones_like(age_portsp)/len(age_portsp)
plt.hist(age_portsp, weights=asweights, edgecolor='gold', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='Portsmouth Passive Model', alpha=0.8, linewidth=5)
plt.axvline(medage_tot, label='Median (Total) = {:.2f}'.format(medage_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medage_app, label='Median (High Confidence) = {:.2f}'.format(medage_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(0,12)
plt.ylim(0, 0.6)
plt.grid(alpha=0.3)
plt.savefig('{}/{}/histogram_portsmouth.png'.format(path_result, version), overwrite=True)
plt.close()

#%% comparison plot
#%% comparison plot - data

os.chdir(path_csv)

# data collection: these csv data should be written manually in advance.

massdata    = ascii.read('host_mass.csv') # unit [log(M_sol)]
agedata     = ascii.read('host_age.csv')  # unit [M_sol / yr]
sfrdata     = ascii.read('host_sfr.csv')  # unit [Gyr]

# representative value of each sample

allmass     = []
allage      = []
allsfr      = []

for i in range(len(massdata)):
    masslist    = np.array([10**massdata['This Work'][i], 
                            10**massdata['Savaglio+09'][i], 
                            10**massdata['Leibler+10'][i], 
                            10**massdata['Berger+14'][i], 
                            10**massdata['Nugent+20'][i], 
                            10**massdata['Other Works'][i]])
    mass        = np.nanmean(masslist)
    if not np.isnan(mass): allmass.append(mass)
    
    agelist    = np.array([agedata['This Work'][i], 
                           agedata['Savaglio+09'][i], 
                           agedata['Leibler+10'][i], 
                           agedata['Berger+14'][i], 
                           agedata['Nugent+20'][i], 
                           agedata['Other Works'][i]])
    age        = np.nanmean(agelist)
    if not np.isnan(age): allage.append(age)
    
    sfrlist    = np.array([sfrdata['This Work'][i], 
                           sfrdata['Savaglio+09'][i], 
                            # sfrdata['Leibler+10'][i], 
                           sfrdata['Berger+14'][i], 
                            # sfrdata['Nugent+20'][i], 
                           sfrdata['Other Works'][i]])
    sfr        = np.nanmean(sfrlist)
    if not np.isnan(sfr): allsfr.append(sfr)

# representative values of the whole samples

# median of this work
medmass0    = np.ma.median(massdata['This Work'])
medage0     = np.ma.median(agedata['This Work'])
medsfr0     = np.ma.median(sfrdata['This Work'])

# total sample median
medmass1    = np.log10(np.median(allmass))
medage1     = np.median(allage)
medsfr1     = np.median(allsfr)

# simulational result (BNS merger, Artale+19)
medmass2    = [10.3, 10.4, 10.2, 8.4] # at z=0.1, 1, 2, 6          
medsfr2     = [10**x for x in [0.07, 0.72, 1.0, 0.02]] # at z=0.1, 1, 2, 6

#%% comparison plot - plotting

os.chdir(path_result)

plt.figure(figsize=(24,15))
# plt.suptitle('sGRB Host Galaxy Properties (Stellar Mass, Age and SFR) Distribution', fontsize=20)
plt.rcParams.update({'font.size': 24})
plt.subplot(3,1,1)
plt.errorbar(massdata['Targets'], massdata['Savaglio+09'], yerr=[massdata['l1'], massdata['u1']], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Savaglio+09', alpha=0.8)
plt.errorbar(massdata['Targets'], massdata['Leibler+10'], yerr=[massdata['l2'], massdata['u2']], ms=6, marker='X', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Leibler+10', alpha=0.8)
plt.errorbar(massdata['Targets'], massdata['Berger+14'], yerr=[massdata['l3'], massdata['u3']], ms=6, marker='D', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Berger+14', alpha=0.8)
plt.errorbar(massdata['Targets'], massdata['Nugent+20'], yerr=[massdata['l4'], massdata['u4']], ms=8, marker='*', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Nugent+20', alpha=0.8)
plt.errorbar(massdata['Targets'], massdata['Other Works'], yerr=[massdata['l5'], massdata['u5']], ms=6, marker='s', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'The Other Works', alpha=0.8)
plt.errorbar(massdata['Targets'], massdata['This Work'], yerr=[massdata['l0'], massdata['u0']], ms=6, marker='s', ls='', color='dodgerblue', capsize=3, capthick=1, alpha=0.8)#'This work', alpha=0.8)
plt.axhline(medmass0, label='Median = {:.2f} (This Work)'.format(medmass0), c='c', ls='--', lw=2)
plt.axhline(medmass1, label='Median = {:.2f} (In Total)'.format(medmass1), c='m', ls='-.', lw=2)
plt.axhspan(medmass2[0], medmass2[1], label='Median = {:.2f}-{:.2f} (z=0.1-1, Artale+19)'.format(medmass2[0], medmass2[1]), facecolor='gold', alpha=0.3)
plt.legend(bbox_to_anchor=(1.02,0), loc="lower left", borderaxespad=0, fontsize=16)
plt.ylim(6.5,12.5)
plt.grid(alpha=0.3)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel(r'$log(M_{*}) \ [M_{\odot}]$')

plt.subplot(3,1,2)
plt.errorbar(sfrdata['Targets'], sfrdata['Savaglio+09'], yerr=[sfrdata['l1'], sfrdata['u1']], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Savaglio+09', alpha=0.8)
# plt.errorbar(sfrdata['Targets'], sfrdata['Leibler+10'], yerr=[sfrdata['l2'], sfrdata['u2']], ms=6, marker='X', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Leibler+10', alpha=0.8)
plt.errorbar(sfrdata['Targets'], sfrdata['Berger+14'], yerr=[sfrdata['l3'], sfrdata['u3']], ms=6, marker='D', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Berger+14', alpha=0.8)
# plt.errorbar(sfrdata['Targets'], sfrdata['Nugent+20'], yerr=[sfrdata['l4'], sfrdata['u4']], ms=6, marker='*', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'Nugent+20', alpha=0.8)
plt.errorbar(sfrdata['Targets'], sfrdata['Other Works'], yerr=[sfrdata['l5'], sfrdata['u5']], ms=6, marker='s', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)#'The Other Works', alpha=0.8)
plt.errorbar(sfrdata['Targets'], sfrdata['This Work'], yerr=[sfrdata['l0'], sfrdata['u0']], ms=6, marker='s', ls='', color='dodgerblue', capsize=3, capthick=1, alpha=0.8)#'This work', alpha=0.8)
plt.axhline(medsfr0, label='Median = {:.2f} (This Work)'.format(medsfr0), c='c', ls='--', lw=2)
plt.axhline(medsfr1, label='Median = {:.2f} (In Total)'.format(medsfr1), c='m', ls='-.', lw=2)
plt.axhspan(medsfr2[0], medsfr2[1], label='Median = {:.2f}-{:.2f} (z=0.1-1, Artale+19)'.format(medsfr2[0], medsfr2[1]), facecolor='gold', alpha=0.3)
plt.grid(alpha=0.3)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel(r'$SFR \ [M_{\odot}/yr]$')
plt.legend(bbox_to_anchor=(1.02,0), loc="lower left", borderaxespad=0, fontsize=16)
plt.ylim(pow(10,-4.5), pow(10,4.5))
plt.semilogy()

plt.subplot(3,1,3)
plt.errorbar(agedata['Targets'], agedata['Savaglio+09'], yerr=[agedata['l1'], agedata['u1']], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, label='Savaglio+09', alpha=0.8)
plt.errorbar(agedata['Targets'], agedata['Leibler+10'], yerr=[agedata['l2'], agedata['u2']], ms=6, marker='X', ls='', color='crimson', capsize=3, capthick=1, label='Leibler+10', alpha=0.8)
plt.errorbar(agedata['Targets'], agedata['Berger+14'], yerr=[agedata['l3'], agedata['u3']], ms=6, marker='D', ls='', color='crimson', capsize=3, capthick=1, label='Berger+14', alpha=0.8)
plt.errorbar(agedata['Targets'], agedata['Nugent+20'], yerr=[agedata['l4'], agedata['u4']], ms=8, marker='*', ls='', color='crimson', capsize=3, capthick=1, label='Nugent+20', alpha=0.8)
plt.errorbar(agedata['Targets'], agedata['Other Works'], yerr=[agedata['l5'], agedata['u5']], ms=6, marker='s', ls='', color='crimson', capsize=3, capthick=1, label='The Other Works', alpha=0.8)
plt.errorbar(agedata['Targets'], agedata['This Work'], yerr=[agedata['l0'], agedata['u0']], ms=6, marker='s', ls='', color='dodgerblue', capsize=3, capthick=1, label='This work', alpha=0.8)
plt.axhline(medage0, label='Median = {:.2f} (This Work)'.format(medage0), c='c', ls='--', lw=2)
plt.axhline(medage1, label='Median = {:.2f} (In Total)'.format(medage1), c='m', ls='-.', lw=2)
plt.legend(bbox_to_anchor=(1.02,0), loc="lower left", borderaxespad=0, fontsize=16)
plt.ylim(-0.5,13.5)
plt.grid(alpha=0.3)
plt.xticks(size=16, rotation=90)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=10, hspace=0.1)
plt.ylabel(r'$t_{age} \ [Gyr]$')

plt.savefig('{}/{}/{}_comparison_plot.pdf'.format(path_result, version, version), dpi=300)
plt.close()

#%% redshift evolution

os.chdir(path_output+'/'+version)
output      = ascii.read('{}_result/merged.fout'.format(version))
ssfrdata    = ascii.read(path_csv+'host_ssfr.csv')

# morphological classification
ltgs        = ssfrdata[ssfrdata['type']==-1]
etgs        = ssfrdata[ssfrdata['type']== 1]
ukns        = ssfrdata[ssfrdata['type']== 0]

lssfr       = 10**ltgs['This Work']
essfr       = 10**etgs['This Work']
ussfr       = 10**ukns['This Work']
e_lssfr     = [10**ltgs['This Work']-10**ltgs['l0'], 10**ltgs['u0']-10**ltgs['This Work']]
e_essfr     = [10**etgs['This Work']-10**etgs['l0'], 10**etgs['u0']-10**etgs['This Work']]
e_ussfr     = [10**ukns['This Work']-10**ukns['l0'], 10**ukns['u0']-10**ukns['This Work']]

lmass       = ltgs['m']
emass       = etgs['m']
umass       = ukns['m']
e_lmass     = [ltgs['ml'], ltgs['mu']]
e_emass     = [etgs['ml'], etgs['mu']]
e_umass     = [ukns['ml'], ukns['mu']]

zrange      = np.arange(0,2.5,0.001)
zage        = cosmo.age(zrange).value*1e9

plt.figure(figsize=(15,15))
plt.rcParams.update({'font.size': 22})
# plt.suptitle('sGRB Host Galaxies Properties Redshift Tendency', fontsize=25)
# mass

plt.subplot(221)
plt.title('r-band Apparent Magnitudes')
plt.xlabel('redshift')
plt.ylabel(r'$m_{r}$ [ABmag]')
plt.ylim(26,10)
plt.xlim(-0.05, 2.55)
# plt.xlim(0, 2)
# plt.tick_params('both', labelsize=7)
plt.errorbar(ps1z, ps1rmag, yerr=ps1err, ms=6, marker='o', ls='', color='red', capsize=3, capthick=1, label='PanSTARRS', alpha=0.8)
plt.errorbar(desz, desrmag, yerr=deserr, ms=6, marker='s', ls='', color='crimson', capsize=3, capthick=1, label='DES / DECaLS', alpha=0.8)
plt.errorbar(gmosz, gmosrmag, yerr=gmoserr, ms=6, marker='^', ls='', color='maroon', capsize=3, capthick=1, label='GMOS', alpha=0.8)
plt.errorbar(forsz, forsrmag, yerr=forserr, ms=15, marker='*', ls='', color='salmon', capsize=3, capthick=1, label='FORS1 (R)', alpha=0.8)
plt.errorbar(etcz, etcrmag, yerr=etcerr, ms=6, marker='D', ls='', color='firebrick', capsize=3, capthick=1, label='The Others', alpha=0.8)
plt.legend()
plt.grid(alpha=0.3)
# plt.show()

plt.subplot(222)
plt.title('Stellar Mass')
plt.scatter(z_sdss, mass_sdssp, alpha=0.08, marker='s', c='limegreen')
plt.scatter(z_sdss, np.zeros_like(mass_sdssp), alpha=0.4, marker='s', c='limegreen', label='SDSS DR12 Galaxies')
plt.errorbar(ltgs['z'], lmass, yerr=e_lmass, c='dodgerblue', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.errorbar(etgs['z'], emass, yerr=e_emass, c='crimson', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.errorbar(ukns['z'], umass, yerr=e_umass, c='grey', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.scatter(ltgs['z'], lmass, c='dodgerblue', label='Late Type ({}%)'.format(atl.r2(len(ltgs)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.scatter(etgs['z'], emass, c='crimson', label='Early Type ({}%)'.format(atl.r2(len(etgs)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.scatter(ukns['z'], umass, c='grey', label='Unknown ({}%)'.format(atl.r2(len(ukns)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.xlim(-0.05, 2.55)
plt.ylim(6.5, 13.5)
plt.xlabel('redshift')
plt.ylabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.xticks(np.arange(0, 2.55, step=0.5))

# sSFR

plt.subplot(223)
plt.title('Specific Star Formation Rate')
plt.errorbar(ltgs['z'], lssfr, yerr=e_lssfr, c='dodgerblue', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.errorbar(etgs['z'], essfr, yerr=e_essfr, c='crimson', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.errorbar(ukns['z'], ussfr, yerr=e_ussfr, c='grey', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.scatter(ltgs['z'], lssfr, c='dodgerblue', label='Late Type ({}%)'.format(atl.r2(len(ltgs)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.scatter(etgs['z'], essfr, c='crimson', label='Early Type ({}%)'.format(atl.r2(len(etgs)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.scatter(ukns['z'], ussfr, c='grey', label='Unknown ({}%)'.format(atl.r2(len(ukns)/len(ssfrdata)*100)), alpha=0.90, marker='s', s=50)
plt.plot(zrange, 1/(3*zage), c='limegreen', ls='-.', label=r'1/$(3\tau_{universe \ age})$')
plt.semilogy()
plt.xlim(-0.05, 2.55)
plt.ylim(1e-30, 1e-5)
plt.xlabel('redshift')
plt.ylabel(r'$sSFR \ [yr^{-1}]$')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.xticks(np.arange(0, 2.55, step=0.5))

# redshift evolution (quiescent fraction)

bins        = [0.00, 0.50, 1.00, 1.50]  # redshift bins (3 sections)
# bins        = [0.00, 0.50, 1.00, 1.50, 2.00]  # redshift bins (4 sections)

quiescent   = []
b0          = []
b1          = []
b2          = []
# b3          = []

for i in range(len(ssfrdata)):
    if 10**ssfrdata['This Work'][i] < 1/(3*cosmo.age(ssfrdata['z'][i]).value*1e9):# and ssfrdata['m0'][i] > 9.10:
        quiescent.append(1)
    else:
        quiescent.append(0)

for i in range(len(quiescent)):
    if bins[0] <= ssfrdata['z'][i] < bins[1]:
        b0.append(quiescent[i])
    elif bins[1] <= ssfrdata['z'][i] < bins[2]:
        b1.append(quiescent[i])
    elif bins[2] <= ssfrdata['z'][i] < bins[3]+1.0:
        b2.append(quiescent[i])
    # elif bins[3] <= ssfrdata['z'][i] < bins[4]:
    #     b3.append(quiescent[i])
    else:
        print('?')

bs      = [b0, b1, b2]#, b3]
zb      = []
qf      = []
for i in range(len(bins)-1):
    zb.append((bins[i]+bins[i+1])/2)
    qf.append(sum(bs[i])/len(bs[i]))

plt.subplot(224)
plt.title('Quiescent Galaxy Fraction')
# plt.errorbar(zb, qf, xerr=0.25, c='crimson', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.plot(zb, qf, '--s', c='crimson', label='Quiescent')
# plt.errorbar(zb, [1-x for x in qf], xerr=0.25, c='dodgerblue', alpha=0.15, ms=1, marker='s', ls='', capsize=5, capthick=1)
plt.plot(zb, [1-x for x in qf], '--s', c='dodgerblue', label='Starforming')
# plt.scatter((bins[4]+bins[5])/2, sum(b4)/len(b4))
plt.legend()
# plt.xlim(-0.05, 2.05)
plt.xlim(-0.05, 1.55)
plt.ylim(-0.025, 1.025)
plt.xticks(np.arange(0, 1.45, step=0.5))
plt.xlabel('redshift')
plt.ylabel('Fraction')
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig('{}/{}/{}_zqf_plot.pdf'.format(path_result, version, version), dpi=300)
plt.close()



