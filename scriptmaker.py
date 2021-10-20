#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:35:09 2021

@author: Mankeun Jeong
Last Update: 2021.05.04.

"""
#%%
import time
import copy
import calendar
import os, glob
import numpy as np
import pandas as pd
import datetime as dt
from astropy.io import ascii

#%%
def scriptmaker(obs, year, month, day, exp=60, cnt=12, temp=-15.0, path_input='./', path_output='./'):

    #0. time parameter
    
    if obs=='RASA36' or 'KCT':
        LT2UTC      = dt.timedelta(hours=3) # Chilian local time to UTC
    else:
        LT2UTC      = dt.timedelta(hours=int(input("Enter the time difference btw LT and UTC:")))
    TOMORROW    = dt.timedelta(days=1)
    
    #1. Reading tonight's RTS file from the input directory.
    
    project     = 'IMSNG'
    
    os.chdir(path_input)
    rts         = ascii.read('rts_vis_{}{}{}_{}_{}.txt'.format(year, month, day, project, obs))
    rts.sort('ra')

    #2. Checking the twilight time. (Starting & ending of an observation)
    
    with open('rts_vis_{}{}{}_{}_{}.txt'.format(year, month, day, project, obs), 'r') as f:
        contents    = f.readlines()
        for c in contents:
            if '#	-18 deg sunset	:' in c:
                sunset  = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, c.split('\t: ')[1][:-1]), '%Y-%m-%d %H:%M') + LT2UTC
            elif '#	-18 deg sunrise	:' in c:
                sunrise = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, c.split('\t: ')[1][:-1]), '%Y-%m-%d %H:%M') + TOMORROW + LT2UTC
                
    #3. Writing the script.
    
    os.chdir(path_output)
    with open('script_{}{}{}_{}_{}.txt'.format(year, month, day, project, obs), 'w') as f:
        
        # starting time
        f.write('; Start of the evening\n')
        sunset_form     = '{:02}/{:02}/{:04} {:02}:{:02}:{:02}'.format(sunset.month, sunset.day, sunset.year, sunset.hour, sunset.minute, sunset.second)
        sunrise_form    = '{:02}/{:02}/{:04} {:02}:{:02}:{:02}'.format(sunrise.month, sunrise.day, sunrise.year, sunrise.hour, sunrise.minute, sunrise.second)
        f.write('#WAITUNTIL 1, {}\n'.format(sunset_form))
    
        # cooling
        f.write('\n; Cooling\n')
        f.write('#CHILL {}, {}\n'.format(temp, 1.0))
        
        # calibration
        f.write('\n; Calibration frames\n')
        f.write("#COUNT {}\n#INTERVAL {}\n#DARK\n".format(cnt, exp)) # dark
        f.write("#COUNT {}\n#BIAS\n".format(cnt)) # bias
        
        # autofocus
        f.write('\n; Autofocus\n')
        f.write('#AFINTERVAL {}\n'.format(180)) # AF every 3 hrs
    
        # targeting
        f.write('\n; Targeting')
    
        total       = 0
        midnight    = 0
        
        for i in range(len(rts)):

            dumtime1    = int(rts['transit(LT)'][i-1][:2])
            dumtime2    = int(rts['transit(LT)'][i][:2])
            if dumtime1 - 20 > dumtime2: # date passed (ex. 23:59 --> 00:01)
                midnight += 1
     
            target  = rts['name'][i]
            ra      = rts['ra'][i]
            dec     = rts['dec'][i]
            
            if midnight == 0:
                transit     = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, rts['transit(LT)'][i]), '%Y-%m-%d %H:%M') + LT2UTC
                rise        = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, rts['rise(LT)'][i]), '%Y-%m-%d %H:%M') + LT2UTC
            elif midnight == 1:
                transit     = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, rts['transit(LT)'][i]), '%Y-%m-%d %H:%M') + TOMORROW + LT2UTC
                if int(rts['rise(LT)'][i][:2]) < 12:
                    rise        = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, rts['rise(LT)'][i]), '%Y-%m-%d %H:%M') + TOMORROW + LT2UTC
                else: #(ex. rises at 22:00 & transits at 01:00)
                    rise        = dt.datetime.strptime('{}-{}-{} {}'.format(year, month, day, rts['rise(LT)'][i]), '%Y-%m-%d %H:%M') + LT2UTC
            else:
                print('Date passed twice. Check the file or parameters.')
            
            # WAIT time = 2 hours before the target transits the meridan.
            # If the target does not rise 2 hrs before the transit time, the rising time is set to be the WAIT time.
            # For more description please check the "WAITUNTIL" directive from: http://solo.dc3.com/ar/RefDocs/HelpFiles/ACP81Help/planfmt.html#directives
            
            wait       = transit - dt.timedelta(hours=2)
            wait_form  = '{:02}/{:02}/{:04} {:02}:{:02}:{:02}'.format(wait.month, wait.day, wait.year, wait.hour, wait.minute, wait.second)
            # if (transit - dt.timedelta(hours=2)) > rise:
            #     wait       = transit - dt.timedelta(hours=2)
            #     wait_form  = '{:02}/{:02}/{:04} {:02}:{:02}:{:02}'.format(wait.month, wait.day, wait.year, wait.hour, wait.minute, wait.second)
            # else:
            #     wait       = rise
            #     wait_form  = '{:02}/{:02}/{:04} {:02}:{:02}:{:02}'.format(wait.month, wait.day, wait.year, wait.hour, wait.minute, wait.second)
    
            if wait < sunrise:
                f.write('\n#WAITUNTIL 1, {}\n'.format(wait_form))
                f.write('#COUNT {}\n#INTERVAL {}\n{}\t{}\t{}\n'.format(cnt, exp, target, ra, dec))
                total   = total + 1
            else:
                print("{} rises after moring".format(target))
                pass # Targets rising after the sunrise will not be included.
    
        # closing
        f.write('\n; Total Targets: ({}/{})\n'.format(total, len(rts)))
        f.write('\n; Closing\n')
        f.write('#QUITAT {}\n'.format(sunrise_form))

    return 0

#%% USER SETTING
obs         = 'RASA36'
project     = 'IMSNG'

# pathes
path_base   = '/home/sonic/research/observation/script'
path_rts    = '/home/sonic/research/observation/script/rts_vis_{}'.format(2021)
path_script = '/home/sonic/research/observation/script/script_{}'.format(2021)

#%% Daily script maker
now         = dt.datetime.now() - dt.timedelta(hours=13) # for Chilean Local Time
year        = str(now.year)
month       = str(now.month).zfill(2)
day         = str(now.day).zfill(2)

scriptmaker(obs, year, month, day, exp=60, cnt=12, path_input=path_rts, path_output=path_script)

#%% Monthly script maker
for date in calendar.Calendar().itermonthdates(2021, 9):

    year    = str(date.year)
    month   = str(date.month).zfill(2)
    day     = str(date.day).zfill(2)

    scriptmaker(obs, year, month, day, exp=60, cnt=12, path_input=path_rts, path_output=path_script)
    
#%% Script merger
os.chdir(path_script)
allscript   = glob.glob('script*.txt'); allscript.sort()

for i in range(len(allscript)-1):
    
    with open('merged/m{}'.format(allscript[i]), 'w') as out:
        
        tonight     = open(allscript[i], 'r').readlines()
        tomorrow    = open(allscript[i+1], 'r').readlines()
        
        for line1 in tonight:
            if line1.startswith('; Closing'):
                add    = False
                for line2 in tomorrow:
                    if line2.startswith('; Targeting'):
                        out.write(line2)
                        add     = True
                    elif line2.startswith('; Closing'):
                        break
                    else:
                        if not(add):
                            pass
                        else:
                            out.write(line2)
                out.write(line1)
            else:
                out.write(line1)
        
                    
        
                    


