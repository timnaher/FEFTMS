
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import uneye
import csv
from src.preprocess import *
from src.neural_network import *
from PIL import Image
import math
from skimage.measure import label



def detect_saccades(row):
    weights_name = '/cs/projects/HEFEFTMS/FEFTMS/preprocessing/saccade_detection/training/weights'
    model = uneye.DNN(max_iter=500, sampfreq=1000,
                lr=0.001, weights_name=weights_name,
                min_sacc_dist=50,min_sacc_dur=6,augmentation=True)

    # check if the trial is long enough to run the model, otherwise return zeros
    if len(row.x[row.ImageOnset:]) < 100:
        return np.zeros(len(row.x[row.ImageOnset:]))
    else:
        Prediction, Probability = model.predict(row.x[row.ImageOnset:],row.y[row.ImageOnset:])
        return Prediction


def remove_invalid_sacs(Prediction):
    # check if prediction is an empty list
    if len(Prediction) == 0:
        return Prediction

    # check if Prediction starts with a 1 and if so, make all values before the first 0 a 0
    if Prediction[0] == 1:
        jj = 0
        while jj < len(Prediction) and Prediction[jj] == 1:
            Prediction[jj] = 0
            jj += 1

    # check if Prediction ends with a 1 and if so, make all values after the last 1 a 0
    if Prediction[-1] == 1:
        jj = 1
        while jj <= len(Prediction) and Prediction[-jj] == 1:
            Prediction[-jj] = 0
            jj += 1

    return Prediction


def merge_saccades(row, samp_freq=1000, min_sacc_dist=3):
    if len(row.Prediction) == 0:
        return row.Prediction
    Prediction = row.Prediction
    Prediction2 = (Prediction.copy()==0).astype(int)
    Prediction_new = Prediction.copy()

    # Create a function to encapsulate the label merging process
    def merge_labels(l):
        first_label = 1 + int(l[0]==1)
        last_label = np.max(l) - int(l[-1]==np.max(l))
        for i in range(first_label, last_label+1):
            if np.sum(l==i) < int(min_sacc_dist*(samp_freq/1000)):
                Prediction_new[l==i] = 1
    
    # Apply the merge_labels function for 1D and 2D cases
    if len(Prediction.shape) < 2:  # case where network output is a vector
        l = label(Prediction2)
        merge_labels(l)
    else:  # case where network output is a matrix
        for n in range(Prediction.shape[0]):
            l = label(Prediction2[n, :])
            merge_labels(l)
    
    return Prediction_new 


def get_sac_on_off_from_pred(row):
    """
    Get saccade onsets and offsets from a prediction vector.
    """
    sac_onsets  = [i+1 for i, x in enumerate(np.diff(row.Prediction)) if x==  1]
    sac_offsets = [i   for i, x in enumerate(np.diff(row.Prediction)) if x== -1]
    return sac_onsets, sac_offsets


def remove_short_sacs(row, min_dur=6, max_dur=100):
    sac_onsets  = row.sac_onsets
    sac_offsets = row.sac_offsets

    # check if sac_onsets and sac_offsets are the same length
    if len(sac_onsets) != len(sac_offsets):
        # check which one is longer
        if len(sac_onsets) > len(sac_offsets):
            sac_onsets = sac_onsets[:-1]
        else:
            sac_offsets = sac_offsets[:-1]

    sac_durs    = np.array(sac_offsets) - np.array(sac_onsets)
    sac_onsets  = np.array(sac_onsets)[sac_durs>min_dur]
    sac_offsets = np.array(sac_offsets)[sac_durs>min_dur]
    sac_durs    = sac_durs[sac_durs>min_dur]
    sac_onsets  = sac_onsets[sac_durs<max_dur]
    sac_offsets = sac_offsets[sac_durs<max_dur]
    sac_durs    = sac_durs[sac_durs<max_dur]

    if len(sac_onsets) == 0:
        return None
    else:
        return sac_onsets, sac_offsets



def saccade_cleaning(row, tol=200):

    onsets   = row.sac_onsets
    offsets  = row.sac_offsets
    triallen = len(row.x)

    # find indicies of saccades that are within the first 50 ms or the last 50 ms of the trial
    idx = np.where((onsets < tol) | (onsets > (triallen-tol) ))[0]
    if len(idx) > 0:
        onsets  = np.delete(onsets,  idx)
        offsets = np.delete(offsets, idx)

    # if the onsets are empty, return None
    if len(onsets) == 0:
        return None
    else:
        return onsets, offsets


def remove_saccade_after_blink(row, tol=50):
    eyes = row.x
    onsets = row['sac_onsets']
    offsets = row['sac_offsets']

    if onsets is None:
        return None, None

    # Convert to numpy arrays if they're not already
    onsets  = np.asarray(onsets)
    offsets = np.asarray(offsets)

    idx_to_remove = []
    for i, onset in enumerate(onsets):
        # Ensure that the index is within the valid range of the eye data
        start = max(0, onset - tol)
        end = min(len(eyes), onset + 1 + tol)

        # Check if there's a blink event within the time window around the saccade onset
        if np.isnan(eyes[start:end]).any():
            idx_to_remove.append(i)

    # Remove the saccades
    onsets  = np.delete(onsets,  idx_to_remove)
    offsets = np.delete(offsets, idx_to_remove)

    return onsets, offsets


def get_sac_params(row):

    if (row.sac_onsets is None) or (row.sac_offsets is None):
        return None
    if len(row.sac_onsets) != len(row.sac_offsets):
        return None
    if len(row.sac_onsets) == 0:
        return None
    if len(row.sac_offsets) == 0:
        return None

    else:

        sac_onsets  = row.sac_onsets
        sac_offsets = row.sac_offsets
        vels,amps,dirs,durs = np.empty((4,0))
        X = row.x
        Y = row.y

        for sac in zip(sac_onsets, sac_offsets):
                a        = sac[0]
                b        = sac[1]

                #if np.abs(b-a) < 3: # saccade too short
                #    continue

                dur      = b-a
                idx      = np.arange(a,b+1)
                edata    = np.vstack((X,Y))
                v        = np.diff(edata[:,idx],axis=1)
                peak_vel = np.max( np.sqrt(v[0,:]**2)+np.sqrt(v[1,:]**2) )
                ampl     = np.sqrt((edata[0,a]-edata[0,b])**2+(edata[1,a]-edata[1,b])**2)
                dx       = edata[0,b] - edata[0,a]
                dy       = edata[1,b] - edata[1,a]
                phi      = 180/np.pi*math.atan2(dy,dx) # saccade direction in degrees
                # round phi to 2 decimal places
                phi = round(phi,2)

                vels = np.append(vels,peak_vel)
                amps = np.append(amps,ampl)
                dirs = np.append(dirs,phi)
                durs = np.append(durs,dur)

        return vels,amps,dirs,durs