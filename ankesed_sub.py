#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:30:12 2021

@author: sonic
"""
#%%
#%%
import re
import time
import copy
import os, glob
import numpy as np
import pandas as pd
import astropy.units as u
from gppy.phot import phot
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from ankepy.phot import anketool as atl # anke package
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, hstack, join
#%% 0. Environments
#%% working directories

# pathes
path_base   = '/home/sonic/research/sed_modeling/'
path_csv    = '/home/sonic/research/sed_modeling/csv/'
path_lib    = '/home/sonic/research/sed_modeling/lib/'
path_input  = '/home/sonic/research/sed_modeling/input/'
path_output = '/home/sonic/research/sed_modeling/output/'
path_result = '/home/sonic/research/sed_modeling/result/'
#%%
version     = 'hogwarts_0728'

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


#%% spectral energy distribution (multiplot)

os.chdir(path_output+'/'+version)

# suspected
sus     = ['GRB050906', 'GRB070810B', 'GRB100216A', 'GRB080121', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
           'GRB070809', 'GRB150424A'] # optical

#PART1
fig     = plt.figure(figsize=(15, 21))
plt.rcParams.update({'font.size': 10})
# fig.text(0.5, 0.93, 'SED of Short GRB Host Galaxies', ha='center', va='center', fontsize=30)
fig.text(0.5, 0.07, '$\lambda \ [\AA]$', ha='center', va='center', fontsize=30)
fig.text(0.07, 0.5, '$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=30)

for i in range(1, 22):

    target  = targets[i-1]
    plt.subplot(7, 3, i)

    fcat    = ascii.read('{}/{}/{}.cat'.format(path_input, version, target))
    zspec   = fcat['z_spec'][0]
    flxinfo = fcat
    
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

    if target not in sus:
        plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='dodgerblue', capsize=2, capthick=1)
        plt.scatter(wlupp, flxupp, marker='v', c='dodgerblue', s=40)
    else:
        plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='crimson', capsize=2, capthick=1)
        plt.scatter(wlupp, flxupp, marker='v', c='crimson', s=40)
        

    try:
        model   = ascii.read('{}_result/BEST_FITS/{}_{}.fit'.format(target, target, 1)) # separate fitting
        plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
    except:
        print('There are no SED fitting result in the output directory.')

    plt.text(0.7e4, max(flx)*1.3, '{:11} (z={:.3f})'.format(target, zspec), fontsize=17)
    if i <= 18:
        plt.tick_params(axis='x', labelbottom=False)
    plt.xlim(000, 50000)
    plt.ylim(000, max(flx)*1.5)

plt.savefig('{}/{}/{}.pdf'.format(path_result, version, 'multiplot_1'), overwrite=True)
plt.close()

#PART2
fig     = plt.figure(figsize=(15, 21))
plt.rcParams.update({'font.size': 10})
# fig.text(0.5, 0.93, 'SED of Short GRB Host Galaxies', ha='center', va='center', fontsize=30)
fig.text(0.5, 0.07, '$\lambda \ [\AA]$', ha='center', va='center', fontsize=30)
fig.text(0.07, 0.5, '$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=30)

for i in range(22, 43):

    target  = targets[i-1]
    plt.subplot(7, 3, i-21)

    fcat    = ascii.read('{}/{}/{}.cat'.format(path_input, version, target))
    zspec   = fcat['z_spec'][0]
    flxinfo = fcat
    
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

    if target not in sus:
        plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='dodgerblue', capsize=2, capthick=1)
        plt.scatter(wlupp, flxupp, marker='v', c='dodgerblue', s=40)
    else:
        plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='crimson', capsize=2, capthick=1)
        plt.scatter(wlupp, flxupp, marker='v', c='crimson', s=40)

    try:
        model   = ascii.read('{}_result/BEST_FITS/{}_{}.fit'.format(target, target, 1)) # separate fitting
        plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
    except:
        print('There are no SED fitting result in the output directory.')

    plt.text(0.7e4, max(flx)*1.3, '{:11} (z={:.3f})'.format(target, zspec), fontsize=17)
    if i <= 38:
        plt.tick_params(axis='x', labelbottom=False)
    plt.xlim(000, 50000)
    plt.ylim(000, max(flx)*1.5)

plt.savefig('{}/{}/{}.pdf'.format(path_result, version, 'multiplot_2'), overwrite=True)
plt.close()

#%% comparison plot2

os.chdir(path_csv)

# MASS
# data collection: these csv data should be written manually in advance.

massdata    = ascii.read('host_mass.csv') # unit [log(M_sol)]

# representative value of each sample

ourmass, ourmlow, ourmupp     = [], [], []
refmass, refmlow, refmupp     = [], [], []
mtargets, refmnum             = [], []


for i in range(len(massdata)):
    if np.ma.is_masked(massdata['This Work'][i]):
        pass
    else:
        dumlist = [massdata['Savaglio+09'][i],
                   massdata['Leibler+10'][i],
                   massdata['Berger+14'][i],
                   massdata['Nugent+20'][i],
                   massdata['Other Works'][i]]

        dumscore= []
        for j in range(len(dumlist)):
            if np.ma.is_masked(dumlist[j]):
                dumscore.append(0)
            else:
                if massdata['l{}'.format(j+1)][i] == 0:
                    dumscore.append(1)
                else:
                    dumscore.append(2)
        
        if max(dumscore)==0:
            pass
        elif len(np.where(np.array(dumscore)==2)[0]) > 1:
            dumscore[np.where(np.array(dumscore)==2)[0][-1]] = 3
        else:
            ourmass.append(massdata['This Work'][i])
            ourmlow.append(massdata['l0'][i])
            ourmupp.append(massdata['u0'][i])
            
            k   = dumscore.index(max(dumscore))
            refmass.append(dumlist[k])
            refmlow.append(massdata['l{}'.format(k+1)][i])
            refmupp.append(massdata['u{}'.format(k+1)][i])
            refmnum.append(k+1)
            mtargets.append(massdata['Targets'][i])
# SFR
# data collection: these csv data should be written manually in advance.

sfrdata    = ascii.read('host_sfr.csv') # unit [M_sol/yr]

# representative value of each sample

oursfr, ourslow, oursupp     = [], [], []
refsfr, refslow, refsupp     = [], [], []
stargets, refsnum             = [], []


for i in range(len(sfrdata)):
    if np.ma.is_masked(sfrdata['This Work'][i]):
        pass
    else:
        dumlist = [sfrdata['Savaglio+09'][i],
                   sfrdata['Leibler+10'][i],
                   sfrdata['Berger+14'][i],
                   sfrdata['Nugent+20'][i],
                   sfrdata['Other Works'][i]]

        dumscore= []
        for j in range(len(dumlist)):
            if np.ma.is_masked(dumlist[j]):
                dumscore.append(0)
            else:
                if sfrdata['l{}'.format(j+1)][i] == 0:
                    dumscore.append(1)
                else:
                    dumscore.append(2)
        
        if max(dumscore)==0:
            pass
        elif len(np.where(np.array(dumscore)==2)[0]) > 1:
            dumscore[np.where(np.array(dumscore)==2)[0][-1]] = 3
        else:
            oursfr.append(sfrdata['This Work'][i])
            ourslow.append(sfrdata['l0'][i])
            oursupp.append(sfrdata['u0'][i])
            
            k   = dumscore.index(max(dumscore))
            refsfr.append(dumlist[k])
            refslow.append(sfrdata['l{}'.format(k+1)][i])
            refsupp.append(sfrdata['u{}'.format(k+1)][i])
            refsnum.append(k+1)
            stargets.append(sfrdata['Targets'][i])

# data collection: these csv data should be written manually in advance.

agedata    = ascii.read('host_age.csv') # unit [M_sol/yr]

# representative value of each sample

ourage, ouralow, ouraupp     = [], [], []
refage, refalow, refaupp     = [], [], []
atargets, refanum             = [], []


for i in range(len(agedata)):
    if np.ma.is_masked(agedata['This Work'][i]):
        pass
    else:
        dumlist = [agedata['Savaglio+09'][i],
                   agedata['Leibler+10'][i],
                   agedata['Berger+14'][i],
                   agedata['Nugent+20'][i],
                   agedata['Other Works'][i]]

        dumscore= []
        for j in range(len(dumlist)):
            if np.ma.is_masked(dumlist[j]):
                dumscore.append(0)
            else:
                if agedata['l{}'.format(j+1)][i] == 0:
                    dumscore.append(1)
                else:
                    dumscore.append(2)
        
        if max(dumscore)==0:
            pass
        elif len(np.where(np.array(dumscore)==2)[0]) > 1:
            dumscore[np.where(np.array(dumscore)==2)[0][-1]] = 3
        else:
            ourage.append(agedata['This Work'][i])
            ouralow.append(agedata['l0'][i])
            ouraupp.append(agedata['u0'][i])
            
            k   = dumscore.index(max(dumscore))
            refage.append(dumlist[k])
            refalow.append(agedata['l{}'.format(k+1)][i])
            refaupp.append(agedata['u{}'.format(k+1)][i])
            refanum.append(k+1)
            atargets.append(agedata['Targets'][i])
            
#%%

plt.figure(figsize=(24,8))
plt.rcParams.update({'font.size': 18})
# plt.suptitle('SED Fitting Result Compared to the Refereces')

plt.subplot(1,3,1)
plt.title('Stellar Mass')
# plotting
for i in range(len(refmnum)):
    if refmnum[i] == 1:
        plt.errorbar(ourmass[i], refmass[i], xerr=[[ourmlow[i]], [ourmupp[i]]], yerr=[[refmlow[i]], [refmupp[i]]], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)
    elif refmnum[i] == 2:
        plt.errorbar(ourmass[i], refmass[i], xerr=[[ourmlow[i]], [ourmupp[i]]], yerr=[[refmlow[i]], [refmupp[i]]], ms=6, marker='X', ls='', color='dodgerblue', capsize=3, capthick=1, alpha=0.8)
    elif refmnum[i] == 3:
        plt.errorbar(ourmass[i], refmass[i], xerr=[[ourmlow[i]], [ourmupp[i]]], yerr=[[refmlow[i]], [refmupp[i]]], ms=6, marker='D', ls='', color='limegreen', capsize=3, capthick=1, alpha=0.8)
    elif refmnum[i] == 4:
        plt.errorbar(ourmass[i], refmass[i], xerr=[[ourmlow[i]], [ourmupp[i]]], yerr=[[refmlow[i]], [refmupp[i]]], ms=6, marker='*', ls='', color='magenta', capsize=3, capthick=1, alpha=0.8)
    elif refmnum[i] == 5:
        plt.errorbar(ourmass[i], refmass[i], xerr=[[ourmlow[i]], [ourmupp[i]]], yerr=[[refmlow[i]], [refmupp[i]]], ms=6, marker='s', ls='', color='gold', capsize=3, capthick=1, alpha=0.8)
plt.plot(np.arange(7,12,0.01), np.arange(7,12,0.01), alpha=0.2, c='grey', linewidth=1)

plt.xlabel(r'$log(M_{*} / M_{\odot})_{This \ Work} $')
plt.ylabel(r'$log(M_{*} / M_{\odot})_{Reference} $')
plt.xlim(7, 12)
plt.ylim(7, 12)
plt.grid(alpha=0.3)

plt.subplot(1,3,2)
plt.title('Star Formation Rate')
# plotting
for i in range(len(refsnum)):
    if refsnum[i] == 1:
        plt.errorbar(oursfr[i], refsfr[i], xerr=[[ourslow[i]], [oursupp[i]]], yerr=[[refslow[i]], [refsupp[i]]], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)
    elif refsnum[i] == 2:
        plt.errorbar(oursfr[i], refsfr[i], xerr=[[ourslow[i]], [oursupp[i]]], yerr=[[refslow[i]], [refsupp[i]]], ms=6, marker='X', ls='', color='dodgerblue', capsize=3, capthick=1, alpha=0.8)
    elif refsnum[i] == 3:
        plt.errorbar(oursfr[i], refsfr[i], xerr=[[ourslow[i]], [oursupp[i]]], yerr=[[refslow[i]], [refsupp[i]]], ms=6, marker='D', ls='', color='limegreen', capsize=3, capthick=1, alpha=0.8)
    elif refsnum[i] == 4:
        plt.errorbar(oursfr[i], refsfr[i], xerr=[[ourslow[i]], [oursupp[i]]], yerr=[[refslow[i]], [refsupp[i]]], ms=6, marker='*', ls='', color='magenta', capsize=3, capthick=1, alpha=0.8)
    elif refsnum[i] == 5:
        plt.errorbar(oursfr[i], refsfr[i], xerr=[[ourslow[i]], [oursupp[i]]], yerr=[[refslow[i]], [refsupp[i]]], ms=6, marker='s', ls='', color='gold', capsize=3, capthick=1, alpha=0.8)

plt.plot(np.arange(0,1e5,0.01), np.arange(0,1e5,0.01), alpha=0.2, c='grey', linewidth=1)

plt.xlabel(r'$SFR_{This \ Work} \ [M_{\odot}/yr]$')
plt.ylabel(r'$SFR_{Reference} \ [M_{\odot}/yr]$')
plt.grid(alpha=0.3)
plt.xlim(pow(10,-4.5), pow(10,4.5))
plt.ylim(pow(10,-4.5), pow(10,4.5))
plt.semilogx()
plt.semilogy()

plt.subplot(1,3,3)
plt.title('Mean Stellar Age')
# plotting
for i in range(len(refanum)):
    if refanum[i] == 1:
        plt.errorbar(ourage[i], refage[i], xerr=[[ouralow[i]], [ouraupp[i]]], yerr=[[refalow[i]], [refaupp[i]]], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, alpha=0.8)
    elif refanum[i] == 2:
        plt.errorbar(ourage[i], refage[i], xerr=[[ouralow[i]], [ouraupp[i]]], yerr=[[refalow[i]], [refaupp[i]]], ms=6, marker='X', ls='', color='dodgerblue', capsize=3, capthick=1, alpha=0.8)
    elif refanum[i] == 3:
        plt.errorbar(ourage[i], refage[i], xerr=[[ouralow[i]], [ouraupp[i]]], yerr=[[refalow[i]], [refaupp[i]]], ms=6, marker='D', ls='', color='limegreen', capsize=3, capthick=1, alpha=0.8)
    elif refanum[i] == 4:
        plt.errorbar(ourage[i], refage[i], xerr=[[ouralow[i]], [ouraupp[i]]], yerr=[[refalow[i]], [refaupp[i]]], ms=6, marker='*', ls='', color='magenta', capsize=3, capthick=1, alpha=0.8)
    elif refanum[i] == 5:
        plt.errorbar(ourage[i], refage[i], xerr=[[ouralow[i]], [ouraupp[i]]], yerr=[[refalow[i]], [refaupp[i]]], ms=6, marker='s', ls='', color='gold', capsize=3, capthick=1, alpha=0.8)
plt.plot(np.arange(0,1e2,0.01), np.arange(0,1e2,0.01), alpha=0.2, c='grey', linewidth=1)

plt.xlabel(r'$t_{age, This \ Work} \ [Gyr]$')
plt.ylabel(r'$t_{age, Reference} \ [Gyr]$')
# plt.xlim(0, 12)
# plt.ylim(0, 12)
plt.semilogx()
plt.semilogy()
plt.xlim(pow(10,-2), pow(10,2))
plt.ylim(pow(10,-2), pow(10,2))
plt.grid(alpha=0.3)

# for legend
plt.errorbar(0, 0, xerr=[[0], [0]], yerr=[[0], [0]], ms=6, marker='o', ls='', color='crimson', capsize=3, capthick=1, label='Savaglio+09')
plt.errorbar(0, 0, xerr=[[0], [0]], yerr=[[0], [0]], ms=6, marker='X', ls='', color='dodgerblue', capsize=3, capthick=1, label='Leibler+10')
plt.errorbar(0, 0, xerr=[[0], [0]], yerr=[[0], [0]], ms=6, marker='D', ls='', color='limegreen', capsize=3, capthick=1, label='Berger+14')
plt.errorbar(0, 0, xerr=[[0], [0]], yerr=[[0], [0]], ms=6, marker='*', ls='', color='magenta', capsize=3, capthick=1, label='Nugent+20')
plt.errorbar(0, 0, xerr=[[0], [0]], yerr=[[0], [0]], ms=6, marker='s', ls='', color='gold', capsize=3, capthick=1, label='The Other Works')
plt.legend(loc='lower right')

plt.tight_layout()

plt.savefig('{}/{}/refcomp.pdf'.format(path_result, version), overwrite=True)
plt.close()

#%% output merger

os.chdir(path_output)
os.chdir(version)
try: os.system('mkdir {}_result'.format(version))
except: pass
    

with open('{}_result/merged.fout'.format(version, version), 'w') as t:
    t.write('#Targets        id         z     l68_z     u68_z      ltau  l68_ltau  u68_ltau     metal l68_metal u68_metal      lage  l68_lage  u68_lage        Av    l68_Av    u68_Av     lmass l68_lmass u68_lmass      lsfr  l68_lsfr  u68_lsfr     lssfr l68_lssfr u68_lssfr      la2t  l68_la2t  u68_la2t      chi2\n')
    for target in targets:
        
        output  = open('{}_result/{}.fout'.format(target, target), 'r').readlines()
        t.write(target+output[-1])

t.close()

result  = ascii.read('{}_result/merged.fout'.format(version))
#%%
sfr    = list(result['lsfr'])
lsfr    = list(result['l68_lsfr'])
usfr    = list(result['u68_lsfr'])
age     = list(result['lage'])
lage    = list(result['l68_lage'])
uage    = list(result['u68_lage'])

for i in range(len(sfr)):
    print('{:13.3f}\t{:13.3f}\t{:13.3f}'.format(round(10**age[i]/1e9,4), 
                                                round(10**age[i]/1e9-10**lage[i]/1e9,4),
                                                round(10**uage[i]/1e9-10**age[i]/1e9,4)))

#%% filter data

os.chdir(path_lib+'/new_filters')
nftrs   = glob.glob('*')
nftrs.sort()

for nftr in nftrs:
    try:
        dat     = ascii.read(nftr)
    except:
        dat     = ascii.read(nftr, data_start=1)
    if len(dat.columns) == 2:
        atl.num_file(nftr)
        dat     = ascii.read(nftr[:-4]+'_num.txt')
        if dat[dat.colnames[1]][0] > dat[dat.colnames[1]][-1]:
            atl.reverse_file(nftr[:-4]+'_num.txt')
    if dat[dat.colnames[1]][0] > dat[dat.colnames[1]][-1]:
        atl.reverse_file(nftr)

# data sorting (optional)
# atl.num_file(file)
# atl.denum_file(file)
# atl.reverse_file(file)
#%%
# plt.figure(figsize=(12,12))
# plt.rcParams.update({'font.size': 16})
# plt.title('SGRB Host Galaxy Mass Distribution')#.format(len(targets)-len(failfit)))
# plt.hist(mass_tot, normed=True, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=100, label='Total Samples (n={})'.format(len(mass_tot)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
# plt.hist(mass_app, normed=True, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=100, label='Approved Samples (n={})'.format(len(mass_app)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
# plt.axvline(medmass_tot, label='Median (Total) = {:.3f}'.format(medmass_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
# plt.axvline(medmass_app, label='Median (Approved) = {:.3f}'.format(medmass_app), c='crimson', ls='-', linewidth=2.5)
# plt.savefig('{}/{}/histogram_mass.png'.format(path_result, version), overwrite=True)
# plt.close()
# plt.figure(figsize=(12,12))
# plt.rcParams.update({'font.size': 16})
# plt.title('SGRB Host Galaxy SFR Distribution')# (n = {})'.format(len(targets)-len(failfit)))
# plt.hist(sfr_tot, weights=weights, color='dodgerblue', bins=np.arange(-8, 8, 1), rwidth=1, label='Total Samples (n={})'.format(len(sfr_tot)), alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
# plt.hist(sfr_app, weights=weights, color='dodgerblue', bins=np.arange(-8, 8, 1), rwidth=1, label='Approved Samples (n={})'.format(len(sfr_app)), alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
# plt.axvline(medsfr_tot, label='Median (Total) = {:.3f}'.format(medsfr_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
# plt.axvline(medsfr_app, label='Median (Approved) = {:.3f}'.format(medsfr_app), c='crimson', ls='-', linewidth=2.5)
# plt.savefig('{}/{}/histogram_sfr.png'.format(path_result, version), overwrite=True)
# plt.close()
# plt.figure(figsize=(12,12))
# plt.rcParams.update({'font.size': 16})
# plt.title('SGRB Host Galaxy Mean Stellar Age Distribution')#' (n = {})'.format(len(targets)-len(failfit)))
# #%% General galaxies from SDSS DR12
# date    = '0502'
# sdssdate = 'sdssdr12_{}'.format(date)

# os.chdir(path_input)
# try:
#     os.system('mkdir {}'.format(sdssdate))
# except:
#     pass

# os.chdir(path_csv)

# sql     = ascii.read('{}.csv'.format(sdssdate))
# sqlc    = sql.colnames
# sql.rename_column(sqlc[0], '#id')
# sql.rename_column(sqlc[1], 'z_spec')
# sql.rename_column(sqlc[2], 'f_SDSS_u')
# sql.rename_column(sqlc[3], 'e_SDSS_u')
# sql.rename_column(sqlc[4], 'f_SDSS_g')
# sql.rename_column(sqlc[5], 'e_SDSS_g')
# sql.rename_column(sqlc[6], 'f_SDSS_r')
# sql.rename_column(sqlc[7], 'e_SDSS_r')
# sql.rename_column(sqlc[8], 'f_SDSS_i')
# sql.rename_column(sqlc[9], 'e_SDSS_i')
# sql.rename_column(sqlc[10], 'f_SDSS_z')
# sql.rename_column(sqlc[11], 'e_SDSS_z')
# sql['#id'] = sql['#id'].astype('str')

# targets = sql['#id']
# sqlc    = sql.colnames

# os.chdir(path_lib)

# ftrinfo     = open('FILTER.RES.latest.info', 'r').readlines() # https://github.com/gbrammer/eazy-photoz
# translate   = ascii.read('translate.cat') # write down manually

# filters     = [f for f in cols if f.startswith('f_')]

# ftrtbl      = Table()
# for f in filters:
#     if f not in translate['filter']:
#         print("Warning: Filter name '{}' is not defined in your translate file.".format(f))
#     else:
#         linenum     = int(translate[translate['filter']==f]['lines'][0][1:])
#         lambda_c    = float(ftrinfo[linenum-1].split('lambda_c= ')[-1].split(' ')[0])
#         dummy       = Table([[f], [lambda_c]], names=['filter', 'lambda_c'])
#         ftrtbl      = vstack([ftrtbl, dummy])

# # catalog file
# os.chdir('{}/{}'.format(path_input, sdssdate))

# sqlcat  = copy.deepcopy(sql)

# for i in range(len(sql)):
#     for j in range(len(sqlc)):
#         if j == 0:
#             sqlcat[i][j]    = str(i+1).zfill(5)
#         elif j==1:
#             sqlcat[i][j]    = sql[i][j]
#         elif j%2 == 0:
#             if sql[i][j] == -9999:
#                 sqlcat[i][j]    = -99
#             else:
#                 sqlcat[i][j]    = atl.rsig(atl.AB2Jy(sql[i][j]), 5)
#         elif j%2 == 1:
#             if sql[i][j] == -9999:
#                 sqlcat[i][j]    = -99
#             else:
#                 sqlcat[i][j]    = atl.rsig(atl.eAB2Jy(sql[i][j-1], sql[i][j]), 5)

# sqlcat.write('{}.cat'.format(sdssdate), format='ascii', delimiter='\t', overwrite=True)

# default     = 'hdfn_fs99'

# # parameter file
# with open('{}/{}.param'.format(path_lib, default), 'r') as f:
#     contents    = f.readlines()
#     with open('{}.param'.format(sdssdate), 'w') as p:
#         for line in contents:
#             if line.startswith('CATALOG'):
#                 p.write(line.replace(default, sdssdate))
#             # elif line.startswith('N_SIM'):
#             #     p.write(line.replace('0', '100'))
#             elif line.startswith('RESOLUTION'):
#                 p.write(line.replace("= 'hr'", "= 'lr'"))
#             # elif line.startswith('NO_MAX_AGE'):
#             #     p.write(line.replace('= 0', '= 1'))
#             else:
#                 p.write(line)
# f.close()
# p.close()

# # translate file
# with open('{}/{}.translate'.format(path_lib, default), 'r') as f:
#     contents    = f.readlines()
#     with open('{}.translate'.format(sdssdate), 'w') as t:
#         for line in contents:
#             t.write(line)
# f.close()
# t.close()

# #%% spliter

# os.chdir(path_input+sdssdate)
# sqlcat  = ascii.read('{}.cat'.format(sdssdate))
# sqlcat.rename_column('id', '#id')
# sqlcat['#id'] = sqlcat['#id'].astype(str)
# targets     = sqlcat['#id']

# core    = 5

# # dividing targets
# for i in range(core):

#     s   = int(len(sqlcat)/core * i)
#     e   = int(len(sqlcat)/core * (i+1))
    
#     sqlcat[s:e].write('multicore/{}_{}.cat'.format(sdssdate, i), format='ascii', delimiter='\t', overwrite=True)
    
#     with open('{}.param'.format(sdssdate), 'r') as f:
#         contents    = f.readlines()
#         with open('multicore/{}_{}.param'.format(sdssdate, i), 'w') as p:
#             for line in contents:
#                 if line.startswith('CATALOG'):
#                     p.write(line.replace(sdssdate, '{}_{}'.format(sdssdate, i)))
#                 # elif line.startswith('N_SIM'):
#                 #     p.write(line.replace('0', '100'))
#                 # elif line.startswith('RESOLUTION'):
#                 #     p.write(line.replace("= 'hr'", "= 'lr'"))
#                 # elif line.startswith('NO_MAX_AGE'):
#                 #     p.write(line.replace('= 0', '= 1'))
#                 else:
#                     p.write(line)
#     f.close()
#     p.close()

#     os.system('cp {}.translate multicore/{}_{}.translate'.format(sdssdate, sdssdate, i))
    
    
    
# #%%
# plt.figure()
# plt.scatter(sql['redshift'], sql['petroMag_r'], c='dodgerblue', alpha=0.01)
# # plt.xlim(0, 0.2)
# plt.show()
# #%%
# # os.chdir(path_csv)
# # VESPA   = ascii.read('GalProp.csv')
# # Vmass   = np.log10(VESPA['M_stellar'])

# # BOSS_P  = Table(fits.open('portsmouth_stellarmass_passive_salp-DR12.fits', memmap=True)[1].data)
# # BOSS_S  = Table(fits.open('portsmouth_stellarmass_starforming_salp-DR12.fits', memmap=True)[1].data)
# # mbweights_P = np.ones_like(BOSS_P['LOGMASS'])/len(BOSS_P['LOGMASS'])
# # mbweights_S = np.ones_like(BOSS_S['LOGMASS'])/len(BOSS_S['LOGMASS'])
# # mvweights   = np.ones_like(Vmass)/len(Vmass)
# # plt.hist(BOSS_P['LOGMASS'], weights=mbweights_P, fill=False, edgecolor='red', bins=np.arange(0,14,0.25), label='SDSS BOSS DR12 (Passive)')
# # plt.hist(BOSS_S['LOGMASS'], weights=mbweights_S, fill=False, edgecolor='blue', bins=np.arange(0,14,0.25), label='SDSS BOSS DR12 (Starforming)')
# # plt.hist(Vmass, weights=mvweights, color='grey', alpha=0.5, bins=np.arange(0,14,0.25), label='VESPA (Tojeiro+09)')
# # plt.hist(BOSS_P['SFR'], weights=mbweights_P, fill=False, edgecolor='red', bins=np.arange(0,14,0.25), label='SDSS BOSS DR12 (Passive)')
# # plt.hist(np.log10(BOSS_S['SFR']), weights=mbweights_S, fill=False, edgecolor='blue', bins=np.arange(-8, 8, 1), label='SDSS BOSS DR12 (Starforming)')
# # plt.hist(BOSS_P['AGE'], weights=mbweights_P, fill=False, edgecolor='red', bins=np.arange(0,12,0.5), label='SDSS BOSS DR12 (Passive)')
# # plt.hist(BOSS_S['AGE'], weights=mbweights_S, fill=False, edgecolor='blue', bins=np.arange(0,12,0.5), label='SDSS BOSS DR12 (Starforming)')

# #%%

# fcat    = ascii.read('sdssdr12_0502.cat')
# zspec   = fcat['spec_z'][0]
# flxinfo = fcat

# flx     = []
# flxerr  = []
# wl      = []
# ftr     = []

# # when only upper limit is available
# flxupp  = []
# wlupp   = []
# ftrupp  = []

# for j in range(len(flxinfo.colnames)):
#     col     = flxinfo.colnames[j]
#     try: nextcol = flxinfo.colnames[j+1]
#     except: pass
#     if flxinfo[col][0] == -99:
#         pass
#     elif col[0] == 'f':
#         if flxinfo[col][0] > 1e-50:
#             wavelength  = ftrtbl[ftrtbl['filter']==col]['lambda_c'][0]
#             flx.append(atl.flux_f2w(flxinfo[col][0], wavelength))
#             flxerr.append(atl.flux_f2w(flxinfo[nextcol][0], wavelength))
#             wl.append(wavelength)
#             ftr.append(col[2:])
#         else:
#             wavelength  = ftrtbl[ftrtbl['filter']==col]['lambda_c'][0]
#             flxupp.append(atl.flux_f2w(flxinfo[nextcol][0], wavelength))
#             wlupp.append(wavelength)
#             ftrupp.append(col[2:])
#     else:
#         pass

# if target not in sus:
#     plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='dodgerblue', capsize=2, capthick=1)
#     plt.scatter(wlupp, flxupp, marker='v', c='dodgerblue', s=40)
# else:
#     plt.errorbar(wl, flx, yerr=flxerr, ms=3, marker='s', ls='', c='crimson', capsize=2, capthick=1)
#     plt.scatter(wlupp, flxupp, marker='v', c='crimson', s=40)
    

# try:
#     model   = ascii.read('sdssdr12_0502_1.fit') # separate fitting
#     plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
# except:
#     print('There are no SED fitting result in the output directory.')

# # plt.text(1.7e4, max(flx)*1.3, '{:11} (z={:.3f})'.format(target, zspec), fontsize=12)
# # if i <= 15:
# #     plt.tick_params(axis='x', labelbottom=False)
# plt.xlim(000, 50000)
# plt.ylim(000, max(flx)*1.5)



#  #%%
 
# os.chdir(path_output+'sdssdr12_0502')
# a = ascii.read('sdssdr12_0502.fout')#, header_start=14)
# os.chdir(path_csv)
# b = ascii.read('sdssdr12_0502.csv')
# #%%
# plt.figure(figsize=(5,5))
# plt.scatter(a['col2'], b['redshift'], alpha = 0.05, c='dodgerblue')
# plt.plot(np.arange(0,1,0.001), np.arange(0,1,0.001))
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)
# plt.show()

#%%
#%%
os.chdir(path_output+version)

result  = ascii.read('{}_result/merged.fout'.format(version))
failfit = [target for target in list(result['Targets']) if result[result['Targets']==target]['chi2'][0]==-1]
# suspect = ['GRB050906', 'GRB070729', 'GRB070809', 'GRB080123', 'GRB090515', 'GRB150424A']
suspect = ['GRB050906', 'GRB070810B', 'GRB100216A', 'GRB080121', # gamma-ray only
           'GRB050813', 'GRB060502B', 'GRB060801', 'GRB070729', 'GRB090515', 'GRB080123', # X-ray
           'GRB070809', 'GRB150424A'] # optical
# mass

plt.figure(figsize=(36, 12))
plt.suptitle('sGRB Host Galaxy Properties Histograms (SED Fitting Result)', fontsize=20)
plt.rcParams.update({'font.size': 16})
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
msweights   = np.ones_like(mass_sdsss)/len(mass_sdsss)
plt.hist(mass_sdsss, weights=msweights, edgecolor='gold', fill=False, bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medmass_tot, label='{:.3f}'.format(medmass_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medmass_app, label='{:.3f}'.format(medmass_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(7, 13)
plt.ylim(0, 0.7)

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
plt.hist(sfr_tot, weights=stweights, color='dodgerblue', bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.5, edgecolor='dodgerblue', linewidth=2.5)
plt.hist(sfr_app, weights=saweights, color='dodgerblue', bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.8, edgecolor='dodgerblue', linewidth=2.5)
ssweights     = np.ones_like(sfr_sdssp)/len(sfr_sdssp)
plt.hist(sfr_sdssp, weights=ssweights, edgecolor='limegreen', fill=False, bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.8, linewidth=5)
ssweights     = np.ones_like(sfr_sdsss)/len(sfr_sdsss)
plt.hist(sfr_sdsss, weights=ssweights, edgecolor='gold', fill=False, bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.8, linewidth=5)
plt.axvline(medsfr_tot, label='{:.3f}'.format(medsfr_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medsfr_app, label='{:.3f}'.format(medsfr_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(-8, 8)
plt.ylim(0, 0.7)
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
plt.hist(age_sdssp, weights=asweights, edgecolor='limegreen', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='SDSS DR12 Galaxies (Field)', alpha=0.8, linewidth=5)
asweights     = np.ones_like(age_sdsss)/len(age_sdsss)
plt.hist(age_sdsss, weights=asweights, edgecolor='gold', bins=np.arange(0,12,1.00), rwidth=1, fill=False, label='SDSS DR12 Galaxies (Passive)', alpha=0.8, linewidth=5)
plt.axvline(medage_tot, label='Median (Total) = {:.3f}'.format(medage_tot), c='crimson', ls='--', linewidth=2.5, alpha=0.5)
plt.axvline(medage_app, label='Median (High Confidence) = {:.3f}'.format(medage_app), c='crimson', ls='-', linewidth=2.5)
plt.legend()
plt.xlim(0,12)
plt.ylim(0, 0.7)
plt.grid(alpha=0.3)
plt.savefig('{}/{}/histogram_sedresult.png'.format(path_result, version), overwrite=True)
plt.close()
#%% histogram (averaged samples) ; This block should be run after the next block to define allmass, allsfr, allage.
# os.chdir(path_output+'/'+version)

# # mass

# plt.figure(figsize=(36, 12))
# plt.suptitle('sGRB Host Galaxy Properties Histograms (Averaged with Published Values)', fontsize=20)
# plt.rcParams.update({'font.size': 16})
# plt.tight_layout()

# plt.subplot(1,3,1)
# plt.title('Stellar Mass')
# plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
# plt.ylabel('Proportion')
# mweights     = np.ones_like(allmass)/len(allmass)
# plt.hist(np.log10(allmass), weights=mweights, color='dodgerblue', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.7, edgecolor='dodgerblue', linewidth=2.5)
# msweights   = np.ones_like(mass_sdss)/len(mass_sdss)
# plt.hist(mass_sdss, weights=msweights, edgecolor='limegreen', bins=np.arange(0,14,0.5), rwidth=1, alpha=0.8, linewidth=5, fill=False)
# plt.axvline(medmass1, label='{:.3f}'.format(medmass1), c='crimson', ls='--', linewidth=2.5, alpha=0.7)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.xlim(7, 13)
# plt.ylim(0, 0.5)
# # sfr

# plt.subplot(1, 3, 2)
# plt.title('Star Formation Rate')
# plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
# plt.ylabel('Proportion')
# sweights     = np.ones_like(allsfr)/len(allsfr)
# plt.hist(np.log10(allsfr), weights=sweights, color='dodgerblue', bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.7, edgecolor='dodgerblue', linewidth=2.5)
# ssweights     = np.ones_like(sfr_sdss)/len(sfr_sdss)
# plt.hist(sfr_sdss, weights=ssweights, edgecolor='limegreen', bins=np.arange(-8, 8, 1), rwidth=1, alpha=0.8, linewidth=5, fill=False)
# plt.axvline(np.log10(medsfr1), label='{:.3f}'.format(np.log10(medsfr1)), c='crimson', ls='--', linewidth=2.5, alpha=0.7)
# plt.legend()
# plt.xlim(-8, 8)
# plt.ylim(0, 0.5)
# plt.grid(alpha=0.3)

# # age

# plt.subplot(1, 3, 3)
# plt.title('Mean Stellar Age')
# plt.xlabel(r'$t_{age} \ [Gyr]$')
# plt.ylabel('Proportion')
# aweights     = np.ones_like(allage)/len(allage)
# plt.hist(allage, weights=aweights, color='dodgerblue', bins=np.arange(0,12,0.5), rwidth=1, label='Total Samples (n={})'.format(len(allmass)), alpha=0.7, edgecolor='dodgerblue', linewidth=2.5)
# asweights     = np.ones_like(age_sdss)/len(age_sdss)
# plt.hist([], edgecolor='limegreen', fill=False, alpha=0.8, linewidth=5, label='SDSS DR12 Galaxies (All)')
# plt.hist(age_sdss, weights=asweights, edgecolor='gold', bins=np.arange(0,12,0.50), rwidth=1, label='SDSS DR12 Galaxies (Passive)', alpha=0.8, linewidth=5, fill=False)
# plt.axvline(medage1, label='Median (Total) = {:.3f}'.format(medage1), c='crimson', ls='--', linewidth=2.5, alpha=0.7)
# plt.legend()
# plt.xlim(0,12)
# plt.ylim(0, 0.5)
# plt.grid(alpha=0.3)
# plt.savefig('{}/{}/histogram_averaged.png'.format(path_result, version), overwrite=True)
# plt.close()
