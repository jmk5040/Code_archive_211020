#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:50:23 2020

@author: sonic
"""

#%%
import time
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
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astroquery.vizier import Vizier 
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord  # High-level coordinates
from gppy.util.changehdr import changehdr as chdr # e.g. chdr(img, where, what)
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, vstack, hstack, join
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
#%% Calculators
def r2(a):
    return round(a,2)

def r3(a):
    return round(a,3)

def rsig(x, sig=2):
    dex     = int(np.floor(np.log10(x)))
    fac     = round(x*10**(-dex), sig)
    num     = str(fac)+'E'+str(dex)
    out     = np.double(num)
    return out

def wcs2deg(RA, DEC): # 00:00:00, 00:00:00 ---> 000.00, 000.00
    ra  = SkyCoord(RA+' '+DEC, unit=(u.hourangle, u.deg)).ra.degree
    dec = SkyCoord(RA+' '+DEC, unit=(u.hourangle, u.deg)).dec.degree
    return ra, dec

def AB2Jy(ABmag):
    zeropoint   = -48.57 # erg s-1 cm-2 Hz-1
    return 10**((zeropoint-ABmag)/2.5) 

def eAB2Jy(ABmag, ABerr):
    return 2.5/np.log(10)*ABerr*AB2Jy(ABmag)
#    zeropoint   = -48.57 # erg s-1 cm-2 Hz-1
#    return ABerr * np.abs(10**(zeropoint-ABmag/2.5)*(-np.log(10)/2.5))

def flux_f2w(flux, wav): # [flux] = erg s-1 cm-2 Hz-1 & [wav] = Angstrom
    erg2Jy      = flux * 1e23
    Jy2f_lambda = erg2Jy / (3.34e4 * (wav**2)) * 1e19
    return Jy2f_lambda
#%% WD
path        = '~/research/SED/' 
os.chdir(os.path.expanduser(path))
#os.system('mkdir cat')
os.getcwd()

sgrb        = ascii.read('sgrb_host.csv', format='csv')
fgrb        = ascii.read('sgrb_host.csv', format='csv')
cols        = sgrb.colnames

#projectbl   = Table([sgrb['GRB'], sgrb['hostRA'], sgrb['hostDec']], names=['target', 'RA', 'Dec'])
#projectbl.write('/home/sonic/research/photometry/project/hogwarts.txt', format='ascii', overwrite=True)
#%% ABmag to flux (FAST)
for i in range(len(sgrb)):
    for j in range(len(cols)):
        if j < 4:
            pass
        elif j%2 == 0:
            if np.ma.is_masked(AB2Jy(sgrb[i][j])):
                fgrb[i][j]  = -99
            else:
                fgrb[i][j]  = AB2Jy(sgrb[i][j])
#                fgrb[i][j]  = rsig(AB2Jy(sgrb[i][j]), 6)
        elif j%2 == 1:
            if np.ma.is_masked(AB2Jy(sgrb[i][j])):
                fgrb[i][j]  = -99
            elif sgrb[i][j-1] == 99:
                fgrb[i][j]  = AB2Jy(sgrb[i][j])
#                fgrb[i][j]  = rsig(AB2Jy(sgrb[i][j]), 6)
            else:
                fgrb[i][j]  = eAB2Jy(sgrb[i][j-1], sgrb[i][j])
#                fgrb[i][j]  = rsig(eAB2Jy(sgrb[i][j-1], sgrb[i][j]), 6)

fgrb.write('flux_sgrb.csv', format='csv', overwrite=True)
#%%
wgrb        = ascii.read('flux_sgrb.csv')#, format='csv')
wgrb.remove_columns(['Target','hostRA', 'hostDec', 'z_spec'])
bandnam     = wgrb.colnames[0::2]
translate   = ascii.read('filters/translate.cat')
filinfo     = open('filters/filter_latest.cat', 'r').readlines()

bandwav     = []
for band in bandnam:
    num     = int(translate['lines'][np.where(translate['filter']==band)][0][1:])
    wavelen = float(filinfo[num-1].split('lambda_c=')[1].split(' ')[0])
    bandwav.append(wavelen)

#bandnam     = [x.split('_')[-1] for x in bandnam]
bandict     = {bandnam[i]: bandwav[i] for i in range(len(bandnam))}

allpoints   = []
allerrors   = []

for i in range(len(wgrb)):
    inpoints    = dict()
    inerrors    = dict()
    for j in range(len(wgrb.colnames)):
        if j%2  == 0:
            if wgrb[i][j] == -99:
                pass
            else:
                inpoints[wgrb.colnames[j]] = flux_f2w(wgrb[i][j], bandwav[j//2])
        elif j%2 == 1:
            if wgrb[i][j] == -99:
                pass
            else:
                inerrors[wgrb.colnames[j]] = flux_f2w(wgrb[i][j], bandwav[j//2])
    allpoints.append(inpoints)
    allerrors.append(inerrors)
#%% flux catalog & redshift file
fcat    = Table(fgrb)

for i in range(len(fcat)):
    for c in cols:
        if c == 'Target':
            fcat[c][i]  = i+1
        elif c == 'hostRA' or c == 'hostDec' or c == 'z_spec':
            pass
        elif fcat[c][i] == -99:
            pass
        else:
            fcat[c][i]  = rsig(fcat[c][i], 5)
         
fcat.rename_column('Target', '#id')
fcat.remove_columns(['hostRA', 'hostDec'])
fcat.write('cat/sgrb.cat', format='ascii', delimiter='\t', fill_values=[(-99.0,'-99')],overwrite=True)
#%% FAST
#%% result 
attempt     = '2{}sgrb'.format(str(7))
"""
0: 2020.07.04
1: 2020.08.06
4: 2020.11.09
5: 2020.11.10
7: 2020.11.16
"""
fout        = ascii.read('result/{}/{}.fout'.format(attempt, attempt), header_start=16)
tarnums     = fout['id']
tarnames    = sgrb['Target']
zspecs      = sgrb['z_spec']
#failfit     = [4, 5, 11, 13, 19, 20, 22, 25, 26, 27] #0
#failfit     = [11, 13, 25, 26] #1
failfit     = [13, 21, 29, 30] #4
#failfit     = [13, 21, 29, 30] #5

#%% SED plots
for i in tarnums:
    if i in failfit:
        pass
    else:
        plt.figure(figsize=(30,15))
        plt.rcParams.update({'font.size': 24})

        flux    = ascii.read('result/{}/BEST_FITS/{}_{}.fit'.format(attempt, attempt, str(i)))
        model   = ascii.read('result/{}/BEST_FITS/{}_{}.input_res.fit'.format(attempt, attempt, str(i)))
        
        points  = allpoints[i-1]
        errors  = allerrors[i-1]
        wavelen = [bandict[key] for key in points.keys()]
        filband = list(points.keys())
        
        plt.title('{} Host Galaxy SED (z={})'.format(tarnames[i-1], zspecs[i-1]))
        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
#        plt.scatter(wavelen, list(points.values()), s=0.1, c='blue')
        for j, txt in enumerate(filband):
            plt.annotate(txt, (wavelen[j], list(points.values())[j]), xytext = (wavelen[j]+100, list(points.values())[j]*1.1), alpha=0.7)
        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=6, marker='s', ls='', color='dodgerblue',capsize=5, capthick=1)
        plt.xlabel('$\lambda \ [\AA]$')
        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
        plt.xlim(000, 50000)
        plt.ylim(000, max(list(points.values()))*1.5)
        plt.grid()
#        plt.semilogx()
        plt.show()
        plt.savefig('result/plot_{}/{}.png'.format(attempt, tarnames[i-1]), overwrite=True)
        plt.close()
#%% SED multiplots

#PART1
fig     = plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 6})
fig.text(0.5, 0.93, 'SED of Short GRB Host Galaxies', ha='center', va='center', fontsize=20)
fig.text(0.5, 0.04, '$\lambda \ [\AA]$', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, '$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=20)

for i in range(1,21):
    if i in failfit:
        pass
    else:
        plt.subplot(4, 5, i)

        flux    = ascii.read('result/{}/BEST_FITS/{}_{}.fit'.format(attempt, attempt, str(i)))
        model   = ascii.read('result/{}/BEST_FITS/{}_{}.input_res.fit'.format(attempt, attempt, str(i)))
        
        points  = allpoints[i-1]
        errors  = allerrors[i-1]
        wavelen = [bandict[key] for key in points.keys()]
        filband = list(points.keys())
        
        plt.text(1.9e4, max(list(points.values()))*1.3, '{} (z={})'.format(tarnames[i-1], round(zspecs[i-1],2)), fontsize=8)
        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
        for j, txt in enumerate(filband):
            plt.annotate(txt.split('_')[-1], (wavelen[j], list(points.values())[j]), xytext = (wavelen[j]+250, list(points.values())[j]+list(errors.values())[j]), fontsize=6, alpha=0.6)
        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=2, marker='s', ls='', color='dodgerblue',capsize=2, capthick=1)
        if i <= 15:
            plt.tick_params(axis='x', labelbottom=False)
        plt.xlim(000, 50000)
        plt.ylim(000, max(list(points.values()))*1.5)

plt.savefig('result/plot_{}/{}.png'.format(attempt, 'multiplot_1'), overwrite=True)
plt.close()
#%%
# PART2
fig     = plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 6})
fig.text(0.5, 0.93, 'SED of Short GRB Host Galaxies', ha='center', va='center', fontsize=20)
fig.text(0.5, 0.04, '$\lambda \ [\AA]$', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, '$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=20)

for i in range(21,41):
    if i in failfit:
        pass
    else:
        plt.subplot(4, 5, i-20)

        flux    = ascii.read('result/{}/BEST_FITS/{}_{}.fit'.format(attempt, attempt, str(i)))
        model   = ascii.read('result/{}/BEST_FITS/{}_{}.input_res.fit'.format(attempt, attempt, str(i)))
        
        points  = allpoints[i-1]
        errors  = allerrors[i-1]
        wavelen = [bandict[key] for key in points.keys()]
        filband = list(points.keys())
        
        plt.text(1.9e4, max(list(points.values()))*1.3, '{} (z={})'.format(tarnames[i-1], round(zspecs[i-1],2)), fontsize=8)
        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
        for j, txt in enumerate(filband):
            plt.annotate(txt.split('_')[-1], (wavelen[j], list(points.values())[j]), xytext = (wavelen[j]+250, list(points.values())[j]+list(errors.values())[j]), fontsize=6, alpha=0.7)#, rotation=45, style='italic')
        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=2, marker='s', ls='', color='dodgerblue',capsize=2, capthick=1)
        if i <= 35:
            plt.tick_params(axis='x', labelbottom=False)
        plt.xlim(000, 50000)
        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid(alpha=0.3)

plt.savefig('result/plot_{}/{}.png'.format(attempt, 'multiplot_2'), overwrite=True)
plt.close()

#%% mass hist
mass    = list(fout['lmass'])
for i in range(len(failfit)):
    mass.remove(np.float64(-1.0))
medmass = median(mass)
plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 16})
plt.title('SGRB Host Galaxy Mass Distribution (n = {})'.format(len(tarnums)-len(failfit)))
plt.xlabel(r'$log(M_{*}) \ [M_{\odot}]$')
plt.ylabel('Counts')
plt.hist(mass, density=False, color='dodgerblue', bins=15, rwidth=1, label='SED fitting result')
plt.axvline(medmass, label='Median = {}'.format(medmass), c='crimson', ls='-.')
plt.legend()
plt.grid()
plt.xlim(5, 14)
plt.show()

#%%
sfr    = list(fout['lsfr'])
for i in range(len(failfit)):
    sfr.remove(np.float64(-1.0))
medsfr = median(sfr)
plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 16})
plt.title('SGRB Host Galaxy SFR Distribution (n = {})'.format(len(tarnums)-len(failfit)))
plt.xlabel(r'$log(SFR) \ [M_{\odot}/yr]$')
plt.ylabel('Counts')
plt.hist(sfr, density=False, color='dodgerblue', bins=25, rwidth=1, label='SED fitting result')
plt.axvline(medsfr, label='Median = {}'.format(medsfr), c='crimson', ls='-.')
plt.legend()
#plt.semilogx()
plt.grid()
#plt.xlim(5, 14)
plt.show()
#%%
#%% host properties
#%%
pro         = ascii.read('host_properties.csv')

# Stellar Mass
medmass0    = np.log10(pd.Series(10**pro['m0']).median()) # FAST output

hostmass    = []
for i in range(len(pro)):
    masslist    = np.array([10**pro['m0'][i], 10**pro['m1'][i], 10**pro['m2'][i], 10**pro['m3'][i], 10**pro['m4'][i]])
    mass        = np.nanmean(masslist)
    hostmass.append(mass)
medmass1    = np.log10(np.median(hostmass)) # Total sample median

medmass2    = [10.3, 10.4] # Simulational result

# SFR
medsfr0     = 10**(pd.Series(pro['s0']).median())

hostsfr     = []
for i in range(len(pro)):
    sfrlist     = np.array([pro['s0'][i], pro['s1'][i], pro['s2'][i], pro['s3'][i]])
    sfr         = np.nanmean(sfrlist)
    hostsfr.append(sfr)
medsfr1     = np.nanmedian(hostsfr)

medsfr2     = [0.07, 0.72]
sfr0    = np.where(np.array(pro['s0'])==0.0, np.nan, np.array(pro['s0']))

## Age
#medage0     = pd.Series(pro['a0']).median()
#
#hostage     = []
#for i in range(len(pro)):
#    agelist     = np.array([pro['a0'][i], pro['a1'][i], pro['a2'][i], pro['a3'][i], pro['a4'][i]])
#    age         = np.nanmean(agelist)
#    hostage.append(age)
#medage1     = np.nanmedian(hostage)
#
## Metallicity
#medmetal0   = pd.Series(pro['z0']).median()
#
#hostmetal   = []
#for i in range(len(pro)):
#    metalist    = np.array([pro['z0'][i], pro['z1'][i], pro['z2'][i]])
#    metal       = np.nanmean(metalist)
#    hostmetal.append(metal)
#medmetal1   = np.nanmedian(hostmetal)

#for i in range(len(pro)):
#    if np.ma.is_masked(pro['a0'][i]):
#        print(' ')
#    else:
#        print(10**pro['a0'][i]/1e9)

#%%
plt.figure(figsize=(16,10))
plt.suptitle('sGRB Host Galaxy Stellar Mass & Star Formation Rate Distribution')

plt.subplot(2,1,1)
#plt.scatter(pro['Target'], pro['m0'], marker='o', c='dodgerblue', label='This Work', zorder=3)
plt.errorbar(pro['Target'], pro['m1'], yerr=pro['em1'], ms=6, marker='s', ls='', color='crimson',capsize=3, capthick=1, label='Gompertz+20')
plt.scatter(pro['Target'], pro['m3'], c='crimson', label='Leibler+10', marker='D',zorder=3)
plt.scatter(pro['Target'], pro['m2'], c='crimson', label='Berger+14', marker='X',zorder=1)
plt.scatter(pro['Target'], pro['m4'], c='crimson', label='The Other Works', marker='*',zorder=2)
plt.errorbar(pro['Target'], pro['m0'], yerr=[pro['m0_u']-pro['m0'], pro['m0']-pro['m0_l']], ms=6, marker='o', ls='', color='dodgerblue',capsize=3, capthick=1, label='This Work',zorder=0)
plt.axhline(medmass0, label='Median = {} (This Work)'.format(medmass0), c='c', ls='--')
plt.axhline(medmass1, label='Median = {} (In Total)'.format(medmass1), c='m', ls='-.')
plt.axhspan(medmass2[0], medmass2[1], label='Median = {} (Artale+19, z=0)'.format(medmass2), facecolor='gold', alpha=0.3)
plt.ylim(6.5,12.5)
plt.grid(alpha=0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel(r'$log(M_{*}) \ [M_{\odot}]$')

plt.subplot(2,1,2)
plt.scatter(pro['Target'], pro['s0'], marker='o', c='dodgerblue', label='This Work', zorder=3)
#plt.errorbar(pro['Target'], pro['s0'], yerr=[pro['s0_u']-pro['s0'], pro['s0']-pro['s0_l']], ms=6, marker='o', ls='', color='dodgerblue',capsize=3, capthick=1, label='This Work',zorder=0)
plt.scatter(pro['Target'], (pro['s2']), c='crimson', label='Other Works', marker='^')
plt.scatter(pro['Target'], (pro['s1']), c='crimson', marker='*')
plt.scatter(pro['Target'], (pro['s3']), c='crimson', marker='*') #
#plt.scatter([np.nan],[np.nan], 's', c='crimson', label='Gompertz+20')
plt.axhline(medsfr0, label='Median (This Work)'.format(medmass0), c='c', ls='--')
plt.axhline(medsfr1, label='Median (In Total)'.format(medmass1), c='m', ls='-.')
plt.axhspan(medsfr2[0], medsfr2[1], label='Median (Artale+19, z=0-1)'.format(medmass2), facecolor='gold', alpha=0.3)
plt.xticks(size=12, rotation=60)
#plt.semilogy()
#plt.ylim(0,3)
plt.ylabel(r'$SFR \ [M_{\odot}/yr]$')
#plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
plt.grid(alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.ylim(pow(10,-4), pow(10,4))
plt.semilogy()
plt.savefig('b.png', dpi=300)
plt.close()
#%%
plt.figure(figsize=(16,10))
plt.suptitle('sGRB Host Galaxy Mean Stellar Age & Metallicity Distribution')

plt.subplot(2,1,1)
plt.scatter(pro['Target'], pro['a0'], marker='o', c='dodgerblue', label='This Work', zorder=3)
plt.scatter(pro['Target'], pro['a2'], c='crimson', label='Savaglio+09', marker='^')
plt.scatter(pro['Target'], pro['a1'], c='crimson', label='Leibler+10', marker='D')
plt.scatter(pro['Target'], pro['a3'], c='crimson', label='Berger+14', marker='X')
plt.scatter(pro['Target'], pro['a4'], c='crimson', label='The Other Works', marker='*')
plt.axhline(medage0, label='Median = {} (This Work)'.format(medage0), c='c', ls='--')
plt.axhline(medage1, label='Median = {} (In Total)'.format(medage1), c='m', ls='-.')
plt.semilogy()
plt.grid(alpha=0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel(r'$log(\tau) \ [Gyr]$')

plt.subplot(2,1,2)
plt.scatter(pro['Target'], sfr0, marker='o', c='dodgerblue', label='This Work', zorder=3)
plt.scatter(pro['Target'], (pro['s1']), c='crimson', label='Savaglio+09', marker='^')
plt.scatter(pro['Target'], [np.nan]*len(pro), c='crimson', label='Leibler+10', marker='D') #
plt.scatter(pro['Target'], (pro['s2']), c='crimson', label='Berger+14', marker='X')
#plt.errorbar(pro['Target'], [np.nan]*len(pro), yerr=pro['em1'], ms=6, marker='s', ls='', color='crimson',capsize=3, capthick=1, label='Gompertz+20')
plt.scatter(pro['Target'], [np.nan]*len(pro), marker='s', color='crimson', label='Gompertz+20')
plt.scatter(pro['Target'], (pro['s3']), c='crimson', label='The Other Works', marker='*') #

plt.axhline(medsfr0, label='Median (This Work)'.format(medmass0), c='c', ls='--')
plt.axhline(medsfr1, label='Median (In Total)'.format(medmass1), c='m', ls='-.')
plt.axhspan(medsfr2[0], medsfr2[1], label='Median (Artale+19, z=0-1)'.format(medmass2), facecolor='gold', alpha=0.3)
plt.xticks(size=12, rotation=60)
#plt.semilogy()
#plt.ylim(0,3)
plt.ylabel(r'$SFR \ [M_{\odot}/yr]$')
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
#plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
plt.grid(alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.ylim(pow(10,-4), pow(10,4))
plt.semilogy()
plt.show()
#plt.savefig('agemetal.png', dpi=300)
#plt.close()
#%%
ltgs    = pro[pro['type']==-1]
etgs    = pro[pro['type']==+1]
ukns    = pro[pro['type']==0]
#%%
plt.figure(figsize=(9,9))
plt.rcParams.update({'font.size': 14})
plt.title('sGRB Host Galaxies sSFR Tendency according to the Redshift')
plt.scatter(ltgs['z'], ltgs['log(SSFR)'], c='dodgerblue', label='Late Type ({}%)'.format(r2(len(ltgs)/len(pro)*100)), alpha=0.75, marker='s', s=50)
plt.scatter(etgs['z'], etgs['log(SSFR)'], c='crimson', label='Early Type ({}%)'.format(r2(len(etgs)/len(pro)*100)), alpha=0.75, marker='s', s=50)
plt.scatter(ukns['z'], ukns['log(SSFR)'], c='k', label='Unknown ({}%)'.format(r2(len(ukns)/len(pro)*100)), alpha=0.75, marker='s', s=50)
plt.axhline(-10, c='limegreen', ls='-.')
plt.xlabel('redshift')
plt.ylabel(r'$log(sSFR) \ [yr^{-1}]$')
plt.legend(loc='lower right')
plt.ylim(-40,0)
plt.grid(alpha=0.5)
#plt.subplots_adjust(wspace=10, hspace=0)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
#%%

#plt.figure()
#plt.subplot(2,1,1)
#plt.scatter(host, mass)
#plt.subplot(2,1,2)
#plt.scatter(host, sfr)
#plt.show()



##%%
#
#fout    = ascii.read('fout.csv')
#fout2   = ascii.read('fresult.csv')
#
#rfout   = fout[np.abs(fout['chi2']) < 10]
#ufout   = fout[np.abs(fout['chi2']) > 10]
#mass = []
#sfr = []
#for i in range(len(fout)):
#    mass.append(10**fout['lmass'][i])
#    sfr.append(10**fout['lsfr'][i])
#meanmass = np.log10(np.average(mass))
#meansfr = np.log10(np.average(sfr))
#Gmass = []
#for i in range(len(fout['lmass_G'])):
#    if np.ma.is_masked(fout['lmass_G'][i]):
#        pass
#    else:
#        Gmass.append(10**fout['lmass_G'][i])
#Gmeanmass = np.log10(np.average(Gmass))
##%%
#for i in allpoints:
#    print(len(i.values()))





#%%
path        = '~/research/SED/' 
os.chdir(os.path.expanduser(path))
#os.system('mkdir cat')
os.getcwd()
#%%

wgrb        = ascii.read('sgrb_host.csv')#, format='csv')
targets     = wgrb['Target']
zspecs      = wgrb['z_spec']
wgrb.remove_columns(['Target','hostRA', 'hostDec', 'z_spec'])
bandnam     = wgrb.colnames[0::2]
translate   = ascii.read('filters/translate.cat')
filinfo     = open('filters/filter_latest.cat', 'r').readlines()

bandwav     = []
for band in bandnam:
    num     = int(translate['lines'][np.where(translate['filter']==band)][0][1:])
    wavelen = float(filinfo[num-1].split('lambda_c=')[1].split(' ')[0])
    bandwav.append(wavelen)

bandict     = {bandnam[i]: bandwav[i] for i in range(len(bandnam))}

allpoints   = []
allerrors   = []

for i in range(len(wgrb)):
    inpoints    = dict()
    inerrors    = dict()
    for j in range(len(wgrb.colnames)):
        if j%2  == 0:
            if wgrb[i][j] == 99:
                pass
            else:
                inpoints[wgrb.colnames[j]] = wgrb[i][j]
#                inpoints[wgrb.colnames[j]] = (wgrb[i][j], bandwav[j//2])
        elif j%2 == 1:
            if wgrb[i][j-1] == 99:
                pass
            else:
                inerrors[wgrb.colnames[j]] = wgrb[i][j]
#                inerrors[wgrb.colnames[j]] = (wgrb[i][j], bandwav[j//2])
    allpoints.append(inpoints)
    allerrors.append(inerrors)

#%%
for i in range(len(targets)):
#    i   = np.where(np.array(targets)==input())[0][0]
    plt.figure(figsize=(30,15))
    plt.rcParams.update({'font.size': 24})
    
    target  = targets[i]
    zspec   = zspecs[i] 
    points  = allpoints[i]
    errors  = allerrors[i]
    
    wavelen = [bandict[key] for key in points.keys()]
    filband = list(points.keys())

    plt.title('{} Host Galaxy SED Photometry Points (z={})'.format(target, zspec))
#    plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
    plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=6, marker='s', ls='', color='dodgerblue',capsize=5, capthick=1)
    for j, txt in enumerate(filband):
        plt.annotate(txt, (wavelen[j], list(points.values())[j]), xytext = (wavelen[j]+100, list(points.values())[j]-0.1))
    plt.xlabel('$\lambda \ [\AA]$')
    plt.ylabel('$ABmag$')
#    plt.xlim(000)
    plt.ylim(np.max(np.ma.MaskedArray(list(points.values())))+0.5, np.min(np.ma.MaskedArray(list(points.values())))-0.5)
    plt.grid(ls='-', alpha=0.5)
    
    plt.savefig('lab/{}.png'.format(target), overwrite=True)
    plt.close()
#%% SED plot separate
#for i in tarnums:
#    plt.figure(figsize=(30,15))
#    plt.rcParams.update({'font.size': 16})
#    
#    if i in failfit:
#        plt.subplot(1,2,1)
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        filband = list(points.keys())
#    
#        plt.title(tarnames[i-1]+' Host Galaxy Phot Points')
#        plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
#        for j, txt in enumerate(filband):
#            plt.annotate(txt, (wavelen[j], list(points.values())[j]))
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 50000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid(ls='-', alpha=0.5)
#    
#    else:
#        plt.subplot(1,2,1)
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        filband = list(points.keys())
#    
#        plt.title(tarnames[i-1]+' Host Galaxy Phot Points')
#        plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
#        for j, txt in enumerate(filband):
#            plt.annotate(txt, (wavelen[j], list(points.values())[j]))
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 50000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid(ls='-', alpha=0.5)
#        
#        plt.subplot(1,2,2)    
#        flux    = ascii.read('result/BEST_FITS/{}_{}.fit'.format(attempt, str(i)))
#        model   = ascii.read('result/BEST_FITS/{}_{}.input_res.fit'.format(attempt, str(i)))
#        
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        
#        plt.title(tarnames[i-1]+' Host Galaxy SED')
#        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
##        plt.scatter(model['col1'],    model['col2'] / 3.714e-30,    c='crimson',      zorder=1)
#        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=6, marker='s', ls='', color='dodgerblue',capsize=5, capthick=1)
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 50000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid()
##        plt.show()
#    plt.savefig('result/plot_1111/'+tarnames[i-1]+'.png', overwrite=True)
#    plt.close()
#    
#%%
#    if i in failfit:
#        plt.subplot(1,2,1)
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        filband = list(points.keys())
#    
#        plt.title(tarnames[i-1]+' Host Galaxy Phot Points')
#        plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
#        for j, txt in enumerate(filband):
#            plt.annotate(txt, (wavelen[j], list(points.values())[j]))
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid(ls='-', alpha=0.5)
#    
#    else:
#        plt.subplot(1,2,1)
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        filband = list(points.keys())
#    
#        plt.title(tarnames[i-1]+' Host Galaxy Phot Points')
#        plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
#        for j, txt in enumerate(filband):
#            plt.annotate(txt, (wavelen[j], list(points.values())[j]))
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid(ls='-', alpha=0.5)
#        
#        plt.subplot(1,2,2)    
#        flux    = ascii.read('result/BEST_FITS/{}_{}.fit'.format(attempt, str(i)))
#        model   = ascii.read('result/BEST_FITS/{}_{}.input_res.fit'.format(attempt, str(i)))
#        
#        points  = allpoints[i-1]
#        errors  = allerrors[i-1]
#        wavelen = [bandict[key] for key in points.keys()]
#        
#        plt.title(tarnames[i-1]+' Host Galaxy SED')
#        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
##        plt.scatter(model['col1'],    model['col2'] / 3.714e-30,    c='crimson',      zorder=1)
#        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=6, marker='s', ls='', color='dodgerblue',capsize=5, capthick=1)
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.ylim(000, max(list(points.values()))*1.5)
#        plt.grid()
##        plt.show()
#    plt.savefig('result/plot_0806/'+tarnames[i-1]+'.png', overwrite=True)
#    plt.close()



#
#         #%%
#for c in cols:
#    if c=='GRB':
#        fcat    = Table([range(1, len(fgrb)+1)], names=['#id'])
#    elif c == 'hostRA' or c == 'hostDec':
#        pass
#    elif c == 'z_spec':
#        fcat    = hstack([fcat, Table([fgrb[c]], names=[c])])
#    else:
#        fcat    = hstack()
#
#
#for i in range(len(fgrb)):
#    fcat    = Table([range(1, len(fgrb)+1)], names=['#id'])
#    for c in cols:
#        if c == 'GRB':
#            target  = fgrb[i][c][1:]
#        elif c == 'hostRA' or c == 'hostDec':
#            pass
#        elif c == 'z':
#            redshift    = fgrb[i][c]
#            zcat    = Table([ [1], [redshift] ], names=['#id', 'z_phot'])
#        elif not np.ma.is_masked(fgrb[i][c]):
#            fdat    = Table([ [round(fgrb[i][c],3)] ], names = [c])
#            fcat    = hstack([fcat, fdat])
#    fcat.write('cat/'+target+'.cat', format='ascii', overwrite=True)
#    zcat.write('cat/'+target+'.zout', format='ascii', overwrite=True)
##%%
#for i in range(len(fgrb)):
#    trans   = ascii.read('cat/hdfn_fs99.translate')
#    target  = fgrb[i]['GRB'][1:]
#    trans.write('cat/'+target+'.translate', format='ascii', overwrite=True)
##%% Parameter file modification
#path        = '~/research/data/SED/param' 
#os.chdir(os.path.expanduser(path))
#
#tarnames     = ascii.read('../sgrb_host.csv', format='csv')['GRB']
#
#for target in tarnames:
#    with open('fast.param') as fp:
#        lines   = fp.read().splitlines()
#    with open(target[1:]+'.param', 'w+') as prm:
#        for i in range(len(lines)):
#            if i == 80:
#                e   = lines[i].split("=")
#                print(e[0]+"='cat/"+target[1:]+"'", file = prm)
#            elif i == 143:
#                e   = lines[i].split("=")
#                print(e[0]+"='result/"+target[1:]+"'", file = prm)
#            elif i == 145:
#                e   = lines[i].split("=")
#                print(e[0]+"=''", file = prm)
#            else:
#                print(lines[i], file = prm)
#    os.rename('fast.param', 'cat/'+target[1:]+'.param')
#    prm.close()
##%% command
#for target in tarnames:
#    print('../fast '+target[1:]+'.param')
#print('')
##%% plotting
#path        = '~/research/data/SED/result' 
#os.chdir(os.path.expanduser(path))
#os.system('mkdir plot')
##%%
#for target in tarnames:
#    if target[1:]=='061201' or target[1:]=='070429B' or target[1:]=='070714B':
#        pass
#    else:
#        flux    = ascii.read(target[1:]+'/BEST_FITS/'+target[1:]+'_1.fit')
#        model   = ascii.read(target[1:]+'/BEST_FITS/'+target[1:]+'_1.input_res.fit')
#    
#        plt.figure(figsize=(10,10))
#        plt.title(target[1:])
#        plt.plot(flux['col1'],       flux['col2'],    c='grey',   zorder=0)
#        plt.scatter(model['col1'],    model['col2'],    c='r',      zorder=1)
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.grid()
#        plt.savefig('plot/'+target[1:]+'.png', overwrite=True)
#        plt.close()
##%%
#target = tarnames[7]
#flux    = ascii.read(target[1:]+'/BEST_FITS/'+target[1:]+'_1.fit')
#model   = ascii.read(target[1:]+'/BEST_FITS/'+target[1:]+'_1.input_res.fit')
#
#mag     = ascii.read('../cat/'+target[1:]+'.cat')['f_SDSS_g']
#magerr  = ascii.read('../cat/'+target[1:]+'.cat')['e_SDSS_g']
#
#
#plt.figure(figsize=(8,8))
#plt.title(target[1:])
#plt.plot(flux['col1'],       flux['col2'],    c='grey',   zorder=0)
#plt.scatter(model['col1'],    model['col2'],    c='r',      zorder=1)
#plt.xlabel('$\lambda \ [\AA]$')
#plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#plt.xlim(000, 30000)
#plt.grid()
#plt.show()
##%%
#path        = '~/research/data/SED/' 
#os.chdir(os.path.expanduser(path))
#
#fout    = ascii.read('fout.csv')
#fout2   = ascii.read('fresult.csv')
#
#rfout   = fout[np.abs(fout['chi2']) < 10]
#ufout   = fout[np.abs(fout['chi2']) > 10]
#mass = []
#sfr = []
#for i in range(len(fout)):
#    mass.append(10**fout['lmass'][i])
#    sfr.append(10**fout['lsfr'][i])
#meanmass = np.log10(np.average(mass))
#meansfr = np.log10(np.average(sfr))
#Gmass = []
#for i in range(len(fout['lmass_G'])):
#    if np.ma.is_masked(fout['lmass_G'][i]):
#        pass
#    else:
#        Gmass.append(10**fout['lmass_G'][i])
#Gmeanmass = np.log10(np.average(Gmass))
##%%
#plt.figure(figsize=(6,6))
#plt.scatter(fout['z'], fout['lmass'], c=fout['chi2'], cmap='RdBu')
#cbar    = plt.colorbar()
#cbar.set_label(r'$\chi^2$')
#plt.clim(0, 10) # 0.4" = 1 pixel
##plt.scatter(rfout['z'], rfout['lmass'], marker='o', c='crimson')
##plt.scatter(ufout['z'], ufout['lmass'], marker='x', label=r'$\chi^2 > 10$', c='grey')
##plt.errorbar(fout['z'], fout['lmass_G'], yerr=fout['emass_G'], fmt='s', ms=5, capsize=2, color='blue', label='Gompertz+20')
#plt.axhline(meanmass, label='Average ({})'.format(round(meanmass, 2)), c='red')
#plt.axhline(10.4, label='Artale+19 (10.4 at z=0)'.format(round(meanmass, 2)), c='c')
#plt.title('sGRB Host Galaxies Mass Distribution')
#plt.xlabel('redshift')
#plt.ylabel(r'$log(M_{*}/M_{\odot})$')
#plt.ylim(7,12)
#plt.xlim(0,3)
#plt.legend()
#plt.show()
##%%
#plt.figure(figsize=(6,6))
#plt.errorbar(fout['z'], fout['lmass_G'], yerr=fout['emass_G'], fmt='s', ms=5, capsize=2, color='crimson', label='Gompertz+20')
#plt.axhline(Gmeanmass, label='Average ({})'.format(round(Gmeanmass, 2)), c='red')
#plt.axhline(10.4, label='Artale+19 (10.4 at z=0)'.format(round(meanmass, 2)), c='c')
#plt.title('sGRB Host Galaxies Mass Distribution')
#plt.xlabel('redshift')
#plt.ylabel(r'$log(M_{*}/M_{\odot})$')
#plt.legend()
#plt.ylim(7,12)
#plt.xlim(0,3)
#plt.show()
#
##%%
#plt.figure(figsize=(6,6))
#plt.scatter(fout[fout['type_G']==-1]['z'], fout[fout['type_G']==-1]['lsfr'], c='dodgerblue', label='Late type')
#plt.scatter(fout[fout['type_G']==0]['z'], fout[fout['type_G']==0]['lsfr'], c='grey', label='unknown')
#plt.scatter(fout[fout['type_G']==1]['z'], fout[fout['type_G']==1]['lsfr'], c='crimson', label='Early type')
#plt.axhline(meansfr, label='Average ({})'.format(round(meansfr, 2)), c='red')
#plt.axhline(0.07, c='c')
#plt.axhline(0.72, c='c')
#plt.fill_between(np.linspace(0,3,1e4), 0.07, 0.72, facecolor='c', alpha=0.1, label='Artale+19 (0.07 (z=0) - 0.72 (z=1))')
#plt.title('sGRB Host Galaxies SFR Distribution')
#plt.xlabel('redshift')
#plt.ylabel(r'$log(SFR) [M_{\odot}/yr]$')
##plt.ylim()
#plt.xlim(0,3)
#plt.legend()
#plt.show()
##%%
#plt.figure(figsize=(6,6))
#plt.title('sGRB Host Galaxies Morphology Distribution')
#size    = [len(fout[fout['type_G']==1])/len(fout)*100, len(fout[fout['type_G']==-1])/len(fout)*100, len(fout[fout['type_G']==0])/len(fout)*100]
#labels  = ['Early type ({}%)'.format(round(size[0],1)), 'Late type ({}%)'.format(round(size[1],1)), 'Unknown ({}%)'.format(round(size[2],1))]
#colors  = ['crimson', 'dodgerblue', 'grey']
#patches, texts = plt.pie(size, colors=colors, startangle=90)
#plt.legend(patches, labels, loc='lower right')
#plt.tight_layout()
#plt.show()
##%% Galactic extinction correction
#ra, dec     = 12.692,	23.011
#tarmag  = [24.08, 23.93, 23.89, 23.76, 23.95, 24.22, 23.13, 22.94, 23.07]
#lambmean= [4718.9, 6185.2, 7499.7, 8961.5, 6471.2, 7872.6, 12093.81, 16482.17, 21409.12] # in angstrom
#
#pathmap = '/home/sonic/research/photometry/sfdmap/'
#
#for i in range(len(tarmag)):
#    
#    ebv     = sfdmap.ebv(ra, dec, mapdir=pathmap)
#    ext     = extinction.fitzpatrick99(np.array([lambmean[i]]), 3.1*ebv)
##    print(ext[0])
#    cormag  = tarmag[i] - ext
#    print(tarmag[i], r2(cormag[0]), r3(ebv), r3(ext[0]))
##lambmean= {'g':4900.1, 'r':6241.3, 'i':7563.8, 'z':8690.1, 'y':9644.6} # in angstrom
##lambmean= {'g':4718.9, 'r':6185.2, 'i':7499.7, 'z':8961.5, 'J':12093.81, 'H':16482.17, 'K':21409.12} # in angstrom
##wave= np.array([lambmean[ftr]])
##
    #%%
### 4.2. Calculating the extinction (Rv = 3.1, Fitzpatrick99)
#
##ebv = sfdmap.ebv(ra, dec, mapdir=dustmap)
##ext = extinction.fitzpatrick99(wave, 3.1 * ebv)
##
#### 4.3. Dereddening the magnitude
##
##target_mag  = target_mag - ext
##target_mag  = target_mag[0] # np.array to scalar
#








##%% flux catalog & redshift file
#for i in range(len(fgrb)):
#    fcat    = Table([[1]], names=['#id'])
#    for c in cols:
#        if c == 'GRB':
#            target  = fgrb[i][c][1:]
#        elif c == 'hostRA' or c == 'hostDec':
#            pass
#        elif c == 'z':
#            redshift    = fgrb[i][c]
#            zcat    = Table([ [1], [redshift] ], names=['#id', 'z_phot'])
#        elif not np.ma.is_masked(fgrb[i][c]):
#            fdat    = Table([ [round(fgrb[i][c],3)] ], names = [c])
#            fcat    = hstack([fcat, fdat])
#    fcat.write('cat/'+target+'.cat', format='ascii', overwrite=True)
##%%
#mgrb        = ascii.read('magplot.csv', format='csv')
#mgrb.remove_columns(['hostRA', 'hostDec', 'z_spec'])
#targets     = mgrb['Target']
#bandnam     = mgrb.colnames[1::2]
##bandwav     = [4380, 4802.19, 6034.55, 8140.18, 15463.6, 3572.18, 4750.82, 6204.29, 7519.21, 8991.92, 4900.14, 6241.27, 7563.76, 8690.08, 9644.61, 3644.41, 4337.31, 5504.35, 6554.8, 7900.11, 10328.6, 12409.9, 16497.1, 21655.7, 3572.18, 4750.82, 6204.29, 7519.21, 8991.92, 4820.0, 6423.0, 7806.7, 9158.5, 9866.8]
#bandwav     = [4380, 4802.19, 6034.55, 8140.18, 15463.6, 3572.18, 4750.82, 6204.29, 7519.21, 8991.92, 4900.14, 6241.27, 7563.76, 8690.08, 9644.61, 4820.0, 6423.0, 7806.7, 9158.5, 9866.8, 3644.41, 4337.31, 5504.35, 6554.8, 7900.11, 10328.6, 12409.9, 16497.1, 21655.7]
#bandict     = {bandnam[i]: bandwav[i] for i in range(len(bandnam))}
##%%
#allpoints   = []
#allerrors   = []
#
#for i in range(len(mgrb)):
#    inpoints    = dict()
#    inerrors    = dict()
#    for j in range(len(mgrb.colnames)):
#        if j == 0:
#            pass
#        elif j%2 == 1:
#            inpoints[mgrb.colnames[j]] = flux_f2w(AB2Jy(mgrb[i][j]), bandwav[(j-1)//2])
#        elif j%2 == 0:
#            inerrors[mgrb.colnames[j]] = flux_f2w(eAB2Jy(mgrb[i][j-1], mgrb[i][j]), bandwav[(j-1)//2])
#    allpoints.append(inpoints)
#    allerrors.append(inerrors)
##%%
#for i in range(len(targets)):
#    plt.figure(figsize=(30,15))
#    plt.rcParams.update({'font.size': 16})
#    target  = targets[i]
#    if True:    
#        plt.subplot(1,2,1)
#        points  = allpoints[i]
#        errors  = allerrors[i]
#        wavelen = [bandict[key] for key in points.keys()]
#        filband = list(points.keys())
#    
#        plt.title(target+' Host Galaxy Phot Points')
#        plt.scatter(wavelen, list(points.values()), c=wavelen, cmap='RdBu')
#        for j, txt in enumerate(filband):
#            plt.annotate(txt, (wavelen[j], list(points.values())[j]))
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.ylim(000, np.nanmax(np.array(list(points.values())))*1.5)
#        plt.grid(ls='-', alpha=0.5)
#        
#        plt.subplot(1,2,2)    
##        flux    = ascii.read('result/BEST_FITS/{}_{}.fit'.format(attempt, str(i)))
##        model   = ascii.read('result/BEST_FITS/{}_{}.input_res.fit'.format(attempt, str(i)))
#        
#        points  = allpoints[i]
#        errors  = allerrors[i]
#        wavelen = [bandict[key] for key in points.keys()]
#        
#        plt.title(target+' Host Galaxy SED')
##        plt.plot(flux['col1'],       flux['col2'],    c='grey',  alpha=0.7, zorder=0)
##        plt.scatter(model['col1'],    model['col2'] / 3.714e-30,    c='crimson',      zorder=1)
#        plt.errorbar(wavelen, list(points.values()), yerr=list(errors.values()), ms=6, marker='s', ls='', color='dodgerblue',capsize=5, capthick=1)
#        plt.xlabel('$\lambda \ [\AA]$')
#        plt.ylabel('$F_{\lambda} \ [10^{-19} \ erg \ s^{-1} \ cm^{-2} \ \AA^{-1}]$')
#        plt.xlim(000, 30000)
#        plt.ylim(000, np.nanmax(np.array(list(points.values())))*1.5)
#        plt.grid()
#    plt.savefig('result/plot_0803/'+target+'.png', overwrite=True)
#    plt.close()