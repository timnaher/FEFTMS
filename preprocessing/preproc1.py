#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import uneye
import csv
import glob
from src.preprocess import *
from src.neural_network import *
from src.saccade_tools import *
from PIL import Image
import math
from skimage.measure import label
import re


datapath = '/mnt/cs/projects/HEFEFTMS/data'
Pdirs    = glob.glob(os.path.join(datapath, "P[0-9][0-9][0-9]"))
Pfiles   = [file for Pdir in Pdirs for file in glob.glob(os.path.join(Pdir, "*.hdf5"))]
pids     = [int(path.split('/')[-1].split('.')[0][1:]) for path in Pfiles]


#%%

for path,pid in zip(Pfiles,pids): 
    print(pid)
    df,eventNames,eventTimes             =  create_df(path)
    df                                   =  zero_phase_lowpass_filter_df(df,cutoff_freq=100,sample_rate=1000,order=2)
    df                                   =  remove_blinks(df,w1=20,w2=20)
    df                                   =  parse_trials(df,eventNames,eventTimes,Fs=1000)
    df['RT']                             =  get_RT(eventNames,eventTimes)
    df['pid']                            =  pid
    df['Prediction']                     =  df.apply(lambda row: detect_saccades(row),axis=1)
    df['Prediction']                     =  df.apply(lambda row: remove_invalid_sacs(row.Prediction), axis = 1)
    df['Prediction']                     =  df.apply(lambda row: merge_saccades(row,samp_freq=1000,min_sacc_dist=30), axis = 1)
    df['sac_onsets'],df['sac_offsets']   =  zip(*df.apply(lambda row: get_sac_on_off_from_pred(row),axis=1))
    df['sac_onsets'],df['sac_offsets']   =  zip(*df.apply(lambda row: remove_short_sacs(row,min_dur=6,max_dur=100) if remove_short_sacs(row,min_dur=5,max_dur=100) is not None else (None,None), axis=1))
    #df['sac_onsets'],df['sac_offsets']   =  zip(*df.apply(lambda row: saccade_cleaning(row,tol=200)  if  saccade_cleaning(row,tol=200) is not None else (None,None), axis=1))
    df['sac_onsets'],df['sac_offsets']   =  zip(*df.apply(lambda row: remove_saccade_after_blink(row,tol=20) if remove_saccade_after_blink(row,tol=20) is not None else None, axis=1))
    df['vels'],df['amps'],df['dirs'],df['durs'] = zip(*df.apply(lambda row: get_sac_params(row) if get_sac_params(row) is not None else (None,None,None,None),axis=1) )

    # save the dataframe
    df.to_pickle(path[:-5]+'_df.pkl')


#%% TEMPORARY DIAGNOSTICS

# plot example scanpath
iTrial = 2
fig, ax = plt.subplots(figsize=(50,30))
ax.plot(df.iloc[iTrial].x[df.iloc[iTrial].ImageOnset:],df.iloc[iTrial].y[df.iloc[iTrial].ImageOnset:],color='r',linewidth=2,alpha=0.5)
ax.imshow(df.iloc[iTrial].image_matrix,cmap='gray')





#%%
# plot the main sequequence
velocities, amplitudes,ISIs = [],[],[]
for row in df.iterrows():
    row = row[1]
    velocities.append(list(row.vels))
    amplitudes.append(list(row.amps))
    ISIs.append(list(np.diff(row.sac_onsets)))

velocities = np.concatenate(velocities)
amplitudes = np.concatenate(amplitudes)
ISIs       = np.concatenate(ISIs)
ISIs       = ISIs[ISIs<1000]

fig, ax = plt.subplots(figsize=(10,5))
plt.scatter(amplitudes,velocities,alpha=0.5,s=2)

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(ISIs,bins=80)

#%%
# plot some event markers
fig, ax = plt.subplots(figsize=(20,5))
plt.plot(df.gaze_x[:250000],linewidth=2,alpha=0.5)
plt.plot(df.gaze_y[:250000],linewidth=2,alpha=0.5)

for i in range(1,10):
    plt.axvline(x=(eventTimes[i]-eventTimes[0]),color='r',linestyle='--')
    # put text at the top of the lines
    plt.text((eventTimes[i]-eventTimes[0]),0.9*plt.ylim()[1],eventNames[i],rotation=30,fontsize=15)



#%%


iTrial = 0

fig, ax = plt.subplots(figsize=(50,30))
ax.plot(df.iloc[iTrial].x[-10000:],df.iloc[iTrial].y[-10000:],color='r',linewidth=2,alpha=0.5)
ax.imshow(df.iloc[iTrial].image_matrix,cmap='gray')
# %%


# %%
