#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:12:47 2021

@author: sonic
"""
#%%
import copy
import os, glob
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from ankepy.phot import anketool as atl # anke package
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table, vstack, hstack
#%% working directories

# pathes
path_base   = '/home/sonic/research/sed_modeling/'
path_csv    = '/home/sonic/research/sed_modeling/csv/'
path_lib    = '/home/sonic/research/sed_modeling/lib/'
path_input  = '/home/sonic/research/sed_modeling/input/'
path_output = '/home/sonic/research/sed_modeling/output/'
path_result = '/home/sonic/research/sed_modeling/result/'

#%% sql data load

os.chdir(path_base)

os.chdir(path_csv)

# mode        = 'spec'
mode        = 'phot'
sdssdate    = 'sdssdr12_{}'.format(mode)
sql         = ascii.read('{}.csv'.format(sdssdate))

os.chdir(path_input)

try:
    os.system('mkdir {}'.format(sdssdate))
except:
    pass

os.chdir(path_csv)

if mode == 'spec':
        
    sqlc    = sql.colnames
    sql.rename_column(sqlc[0], '#id')
    sql.rename_column(sqlc[1], 'z_spec')
    sql.rename_column(sqlc[2], 'f_SDSS_u')
    sql.rename_column(sqlc[3], 'e_SDSS_u')
    sql.rename_column(sqlc[4], 'f_SDSS_g')
    sql.rename_column(sqlc[5], 'e_SDSS_g')
    sql.rename_column(sqlc[6], 'f_SDSS_r')
    sql.rename_column(sqlc[7], 'e_SDSS_r')
    sql.rename_column(sqlc[8], 'f_SDSS_i')
    sql.rename_column(sqlc[9], 'e_SDSS_i')
    sql.rename_column(sqlc[10], 'f_SDSS_z')
    sql.rename_column(sqlc[11], 'e_SDSS_z')
    sql['#id'] = sql['#id'].astype('str')
    targets = sql['#id']
    sqlc    = sql.colnames

else:
    
    sqlc    = sql.colnames
    sql.rename_column(sqlc[0], '#id')
    sql.rename_column(sqlc[2-1], 'f_SDSS_u')
    sql.rename_column(sqlc[3-1], 'e_SDSS_u')
    sql.rename_column(sqlc[4-1], 'f_SDSS_g')
    sql.rename_column(sqlc[5-1], 'e_SDSS_g')
    sql.rename_column(sqlc[6-1], 'f_SDSS_r')
    sql.rename_column(sqlc[7-1], 'e_SDSS_r')
    sql.rename_column(sqlc[8-1], 'f_SDSS_i')
    sql.rename_column(sqlc[9-1], 'e_SDSS_i')
    sql.rename_column(sqlc[10-1], 'f_SDSS_z')
    sql.rename_column(sqlc[11-1], 'e_SDSS_z')
    sql['#id'] = sql['#id'].astype('str')
    targets = sql['#id']
    sqlc    = sql.colnames
    
#%% FAST input

os.chdir(path_lib)

default     = 'hdfn_fs99'

ftrinfo     = open('FILTER.RES.latest.info', 'r').readlines() # https://github.com/gbrammer/eazy-photoz
translate   = ascii.read('translate.cat') # write down manually

filters     = [f for f in sqlc if f.startswith('f_')]
ftrtbl      = Table()

for f in filters:
    if f not in translate['filter']:
        print("Warning: Filter name '{}' is not defined in your translate file.".format(f))
    else:
        linenum     = int(translate[translate['filter']==f]['lines'][0][1:])
        lambda_c    = float(ftrinfo[linenum-1].split('lambda_c= ')[-1].split(' ')[0])
        dummy       = Table([[f], [lambda_c]], names=['filter', 'lambda_c'])
        ftrtbl      = vstack([ftrtbl, dummy])

# catalog file
os.chdir('{}/{}'.format(path_input, sdssdate))

sqlcat  = copy.deepcopy(sql)

if mode == 'spec':
    
    for i in range(len(sql)):
        for j in range(len(sqlc)):
            if j == 0:
                sqlcat[i][j]    = str(i+1).zfill(5)
            elif j==1:
                sqlcat[i][j]    = sql[i][j]
            elif j%2 == 0:
                if sql[i][j] == -9999:
                    sqlcat[i][j]    = -99
                else:
                    sqlcat[i][j]    = atl.rsig(atl.AB2Jy(sql[i][j]), 5)
            elif j%2 == 1:
                if sql[i][j] == -9999:
                    sqlcat[i][j]    = -99
                else:
                    sqlcat[i][j]    = atl.rsig(atl.eAB2Jy(sql[i][j-1], sql[i][j]), 5)

else:
    
    for i in range(len(sql)):
        for j in range(len(sqlc)):
            if j == 0:
                sqlcat[i][j]    = str(i+1).zfill(5)
            elif j%2 == 1:
                if sql[i][j] == -9999:
                    sqlcat[i][j]    = -99
                else:
                    sqlcat[i][j]    = atl.rsig(atl.AB2Jy(sql[i][j]), 5)
            elif j%2 == 0:
                if sql[i][j] == -9999:
                    sqlcat[i][j]    = -99
                else:
                    sqlcat[i][j]    = atl.rsig(atl.eAB2Jy(sql[i][j-1], sql[i][j]), 5)

sqlcat.write('{}.cat'.format(sdssdate), format='ascii', delimiter='\t', overwrite=True)

# parameter file
with open('{}/{}.param'.format(path_lib, default), 'r') as f:
    contents    = f.readlines()
    with open('{}.param'.format(sdssdate), 'w') as p:
        for line in contents:
            if line.startswith('CATALOG'):
                p.write(line.replace(default, sdssdate))
            # elif line.startswith('N_SIM'):
            #     p.write(line.replace('0', '100'))
            elif line.startswith('RESOLUTION'):
                p.write(line.replace("= 'hr'", "= 'lr'"))
            # elif line.startswith('NO_MAX_AGE'):
            #     p.write(line.replace('= 0', '= 1'))
            else:
                p.write(line)
f.close()
p.close()

# translate file
with open('{}/{}.translate'.format(path_lib, default), 'r') as f:
    contents    = f.readlines()
    with open('{}.translate'.format(sdssdate), 'w') as t:
        for line in contents:
            t.write(line)
f.close()
t.close()

#%% FAST output

def absolmag(m, z):
    dist    = np.array(cosmo.luminosity_distance(z))
    return m - 5*np.log10(dist) - 25

# input data = sql
os.chdir(path_output+sdssdate)

if mode == 'spec':
    
    specin      = copy.deepcopy(sql)
    specfout    = ascii.read('{}.fout'.format(sdssdate), header_start=16)
    passive     = specfout[10**specfout['lssfr']*1e9 < np.array(1/(3*cosmo.age(specfout['z'])))]
    passive_r   = []
    passive_z   = []
    for i in range(len(passive)):
        j   = passive['id'][i]
        passive_r.append(specin['f_SDSS_r'][j-1])
        passive_z.append(specin['z_spec'][j-1])
    passive_R   = absolmag(passive_r, passive_z)
    passive['rmag']     = passive_r
    passive['z_spec']   = passive_z
    passive['rmag(ab)'] = passive_R 
    
else:
    
    photin      = copy.deepcopy(sql)
    photzout    = ascii.read('{}.zout'.format(sdssdate))
    photfout    = ascii.read('{}.fout'.format(sdssdate), header_start=17)
    phot_R      = absolmag(photin['f_SDSS_r'], photzout['z_m1'])
    photfout['z_phot']  = photzout['z_m1']
    # photfout['z_phot']  = photzout[:37578]['z_m1']
    photfout['rmag(ab)']= phot_R#[:37578]

#%% volume limited sample

vlspec  = passive[passive['z_spec']>0.03]; vlspec  = vlspec[vlspec['z_spec']<0.05]; vlspec  = vlspec[vlspec['rmag(ab)']<-19]
vlphot  = photfout[photfout['z_phot']>0.06]; vlphot  = vlphot[vlphot['z_phot']<0.12]; vlphot  = vlphot[vlphot['rmag(ab)']<-17]

#%% priority
g1  = 0.701
g2  = 0.356
g3  = 0.411
g4  = -1.968 # Artale+19, BNS, z=0.1

vlspec['prior'] = g1*vlspec['lmass']+g2*vlspec['lsfr']+g3*vlspec['metal']+g4
vlphot['prior'] = g1*vlphot['lmass']+g2*vlphot['lsfr']+g3*vlphot['metal']+g4

vlphot.sort('prior')
vlspec.sort('prior')

#%%
def Mr2logM(Mr):
    gr  = 0.88
    return -0.39*Mr+1.05*gr+1.60


# Define a closure function to register as a callback
def convert(ax_Mr):
    """
    Update second axis according with first axis.
    """
    y1, y2 = ax_Mr.get_ylim()
    ax_lM.set_ylim(Mr2logM(y1), Mr2logM(y2))
    ax_lM.figure.canvas.draw()

# plt.rcParams.update({'font.size': 14})

fig, axs = plt.subplots(2, 1, figsize=(9,12))
plt.rcParams.update({'font.size': 18})
ax_Mr   = axs[1]
ax_lM   = ax_Mr.twinx()
# ax_Mr.set_title('SDSS DR12 Galaxies Distribution')
# automatically update ylim of ax2 when ylim of ax1 changes.
ax_Mr.callbacks.connect("ylim_changed", convert)
ax_Mr.scatter(specin['z_spec'], absolmag(specin['f_SDSS_r'], specin['z_spec']), c='grey', alpha=0.2, label='Galaxies (Spectroscopy)')# (n={})'.format(len(specin)))
ax_Mr.scatter(passive_z, absolmag(passive_r, passive_z), c='crimson', alpha=0.2, label='Passive Galaxies')# (n={})'.format(len(passive)))
# ax_Mr.hlines(-21, 0.07, 0.13, linestyle='--')#, label='Volume Limited Samples')
# ax_Mr.vlines(0.07, -21, -50, linestyle='--')
# ax_Mr.vlines(0.13, -21, -50, linestyle='--')
ax_Mr.hlines(-19, 0.03, 0.05, linestyle='--', label='Volume Limited Samples')
ax_Mr.vlines(0.03, -19, -50, linestyle='--')
ax_Mr.vlines(0.05, -19, -50, linestyle='--')
ax_Mr.set_xlim(0.0075, 0.20)
ax_Mr.set_ylim(-15, -24)
ax_Mr.legend(loc='lower right')
# ax_Mr.set_title('Spectroscopically Confirmed')
ax_Mr.set_ylabel('$M_r$')
ax_Mr.set_xlabel('Spectroscopic redshift')
ax_lM.tick_params(labelsize=12)
ax_Mr.tick_params(labelsize=12)
ax_lM.set_ylabel(r'log($M/M_{\odot}$)')

ax_P    = axs[0]
s = ax_P.scatter(photzout[:37578]['z_m1'], absolmag(photin[:37578]['f_SDSS_r'], photzout[:37578]['z_m1']), c=photfout['lmass'], cmap='jet', alpha=0.3, vmin=7, vmax=12, label='Galaxies (Photometry)')
ax_P.hlines(-17.0, 0.06, 0.17, linestyle='--', label='Volume Limited Samples')
ax_P.vlines(0.06, -17.0, -50, linestyle='--')
ax_P.vlines(0.17, -17.0, -50, linestyle='--')
ax_P.set_xlim(0.015, 0.40)
ax_P.set_ylim(-15, -24)
# ax_P.set_title('Photometric Data Only')
caxes   = fig.add_axes([0.891, 0.559, 0.01, 0.413])
fig.colorbar(s, ax=ax_P, label=r'log($M/M_{\odot}$)', orientation='vertical', cax=caxes)
ax_P.legend(loc='lower right')
ax_P.set_ylabel('$M_r$')
ax_P.set_xlabel('Photometric redshift')
ax_P.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('{}hogwarts_0728/{}_VLsets.png'.format(path_result, sdssdate), overwrite=True)
plt.close()

#%%
#%%

fcat    = ascii.read('sdssdr12_0502.cat')
zspec   = fcat['spec_z'][0]
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
    model   = ascii.read('sdssdr12_0502_1.fit') # separate fitting
    plt.plot(model['col1'], model['col2'], c='grey', alpha=0.7, zorder=0)
except:
    print('There are no SED fitting result in the output directory.')

# plt.text(1.7e4, max(flx)*1.3, '{:11} (z={:.3f})'.format(target, zspec), fontsize=12)
# if i <= 15:
#     plt.tick_params(axis='x', labelbottom=False)
plt.xlim(000, 50000)
plt.ylim(000, max(flx)*1.5)



 #%%
 
os.chdir(path_output+'sdssdr12_spec')
a = ascii.read('sdssdr12_0502.fout')#, header_start=14)
os.chdir(path_csv)
b = ascii.read('sdssdr12_0502.csv')

c = ascii.read('sdssdr12_spec.zout')
#%%
plt.figure(figsize=(7,7))
plt.scatter(c['z_m1'], c['z_spec'], alpha = 0.1, c='dodgerblue')
plt.plot(np.arange(0,1,0.001), np.arange(0,1,0.001))
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.title('phot-z test')
plt.xlabel('phot-z')
plt.ylabel('spec-z')
plt.grid(alpha=0.5)
plt.show()
#%% spliter

os.chdir(path_input+sdssdate)
sqlcat  = ascii.read('{}.cat'.format(sdssdate))
sqlcat.rename_column('id', '#id')
sqlcat['#id'] = sqlcat['#id'].astype(str)
targets     = sqlcat['#id']

core    = 5

# dividing targets
for i in range(core):

    s   = int(len(sqlcat)/core * i)
    e   = int(len(sqlcat)/core * (i+1))
    
    sqlcat[s:e].write('multicore/{}_{}.cat'.format(sdssdate, i), format='ascii', delimiter='\t', overwrite=True)
    
    with open('{}.param'.format(sdssdate), 'r') as f:
        contents    = f.readlines()
        with open('multicore/{}_{}.param'.format(sdssdate, i), 'w') as p:
            for line in contents:
                if line.startswith('CATALOG'):
                    p.write(line.replace(sdssdate, '{}_{}'.format(sdssdate, i)))
                # elif line.startswith('N_SIM'):
                #     p.write(line.replace('0', '100'))
                # elif line.startswith('RESOLUTION'):
                #     p.write(line.replace("= 'hr'", "= 'lr'"))
                # elif line.startswith('NO_MAX_AGE'):
                #     p.write(line.replace('= 0', '= 1'))
                else:
                    p.write(line)
    f.close()
    p.close()

    os.system('cp {}.translate multicore/{}_{}.translate'.format(sdssdate, sdssdate, i))
    
    
    