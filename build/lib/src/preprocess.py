#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import butter, filtfilt
import csv
from PIL import Image
import re


path = '/mnt/pns/scratch/4Tim/data/P005/P005_FEFTMS_pilot_rawBuilderBaseVersion_2023-08-10_11h16.36.629.hdf5'

def create_df(path):
    with h5py.File(path, 'r') as file:
        #open the file
        data = file['data_collection']['events']['eyetracker']['MonocularEyeSampleEvent']

        events = np.array(file['data_collection']['events']['experiment']['MessageEvent'])
        # make a pandas dataframe
        eventNames = [events[j][-1].decode('UTF-8') for j in range(len(events))]
        eventTimes = [events[j][5] for j in range(len(events))]
        # get the event times in milliseconds (it currently in seconds) and round to integer
        eventTimes = (np.array(eventTimes)*1000).astype(int)

        return pd.DataFrame(np.array(data)),eventNames, eventTimes

def remove_blinks(df,w1=20,w2=20):
    """
    Remove blinks from a DataFrame containing gaze data.
    
    This function identifies blinks based on non-zero values in the 'status' column of the input DataFrame.
    It then sets a window of 20 samples before and after each blink to NaN in the 'gaze_x' and 'gaze_y' columns.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing columns 'status', 'gaze_x', and 'gaze_y'. 
                             The 'status' column should have non-zero values indicating blinks.
                             
    Returns:
    - pandas.DataFrame: A new DataFrame with blinks removed (set to NaN) in 'gaze_x' and 'gaze_y' columns.
    
    Notes:
    - The function makes a copy of the input DataFrame and does not modify the original.
    """

    df        = df.copy()
    status    = np.zeros(len(df))
    blink_idx = np.unique(np.concatenate([np.where(df.status != 0)[0] - j for j in range(-w1,w2)]))
    blink_idx = blink_idx[blink_idx < len(status)] # remove any indices that are out of bounds
    # remove indicies which are negative
    blink_idx = blink_idx[blink_idx > 0]
    status[blink_idx] = 1
    df['status'] = status
    df.loc[blink_idx, 'gaze_x'] = np.nan
    df.loc[blink_idx, 'gaze_y'] = np.nan
    return df


def zero_phase_lowpass_filter_df(df, cutoff_freq, sample_rate, order=4):
    """
    Apply a zero-phase lowpass filter to the gaze columns of a DataFrame.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing columns 'gaze_x' and 'gaze_y'.
    - cutoff_freq: The cutoff frequency of the lowpass filter.
    - sample_rate: The sample rate of the data.
    - order: The order of the Butterworth filter (default is 4).
    
    Returns:
    - pandas.DataFrame: A new DataFrame with 'gaze_x' and 'gaze_y' columns filtered.
    """
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Create a copy of the input DataFrame
    df_filtered = df.copy()
    
    # Filter the gaze columns
    df_filtered['gaze_x'] = filtfilt(b, a, df['gaze_x'])
    df_filtered['gaze_y'] = filtfilt(b, a, df['gaze_y'])
    
    return df_filtered


def parse_trials(df, eventNames, eventTimes, Fs=1000, imagepath='/cs/projects/HEFEFTMS/data/images/Waldo/'):
    # Convert eventTimes to indices (with default Fs = 1000 Hz)
    eventTimes = (eventTimes * Fs) // 1000

    # Adjust event times relative to the start
    eventTimes -= eventTimes[0]

    # Get relevant event indices and corresponding image names
    ImageOn_idx = [i for i, s in enumerate(eventNames) if 'ImageOn' in s]
    Found_idx   = [i for i, s in enumerate(eventNames) if 'Found' in s]
    ImageNames  = [s.split("_", 1)[1] for s in eventNames if 'ImageOn' in s]
    FixOn_idx   = [i for i, s in enumerate(eventNames) if 'FixCrossOn' in s]
    FixOff_idx  = [i for i, s in enumerate(eventNames) if 'FixCrossOff' in s]


    # Extract sample times
    ImageOn_samples = eventTimes[ImageOn_idx]
    Found_samples   = eventTimes[Found_idx]
    FixOn_samples   = eventTimes[FixOn_idx]
    FixOff_samples  = eventTimes[FixOff_idx]

    # get the position of the fixcross
    fixPos = [[int(re.search(r'x(-?\d+)', s).group(1)), int(re.search(r'y(-?\d+)', s).group(1))] for s in eventNames if 'FixCrossOn' in s]

    # Collect trial data to later convert to a dataframe
    trial_data = []

    for trial in range(len(ImageOn_samples)):
        trial_start = FixOn_samples[trial]
        trial_end   = Found_samples[trial]
        ImageOnset  = ImageOn_samples[trial] - trial_start # relative to trial start

        trial_x = df.loc[trial_start:trial_end-1, 'gaze_x'].values + 960  # Adjust for screen center
        trial_y = -df.loc[trial_start:trial_end-1, 'gaze_y'].values + 600
        # adjust the fixpos for the screen center as well
        fixPos[trial][0] = fixPos[trial][0] + 960
        fixPos[trial][1] = -fixPos[trial][1] + 600

        full_image = imagepath + ImageNames[trial] + '.jpg'

        # get fullscreen canvas
        image_canvas = np.zeros((1200,1920))

        # load the image
        image = Image.open(full_image)

        # resize it
        image = image.resize((1400, 900))

        # put it in the canvas
        image_canvas[150:1050, 260:1660] = image


        trial_data.append({'x': trial_x, 'y': trial_y, 'image': full_image,
                            'image_matrix': image_canvas,'fixPos':fixPos[trial],
                            'ImageOnset':ImageOnset})

    # Convert list of trial data to dataframe
    df_trials = pd.DataFrame(trial_data)

    return df_trials




def get_RT(eventNames,eventTimes):
    image_on_indices = [index for index, marker in enumerate(eventNames) if "ImageOn" in marker]
    found_indices    = [index for index, marker in enumerate(eventNames) if "Found" in marker]
    return eventTimes[found_indices] - eventTimes[image_on_indices] 

