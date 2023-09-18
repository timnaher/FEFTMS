#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from src.preprocess import *
from src.neural_network import *

datapath = '/mnt/cs/projects/HEFEFTMS/data'
# find all dirs that start with P in datapath
Pdirs = [os.path.join(datapath,dir) for dir in os.listdir(datapath) if re.match('P[0-9]{3}',dir)]

# find all files that end with .hdf5 in each Pdir
Pfiles = []
for Pdir in Pdirs:
    Pfiles.append([os.path.join(Pdir,file) for file in os.listdir(Pdir) if file.endswith('.pkl')])
Pfiles = np.concatenate(Pfiles)

#%%
result_dict = {}
for path in Pfiles:
    # load data from path
    df = pd.read_pickle(path)


    def extract_image_name(path):
        return path.split('/')[-1].split('.')[0]

    # Create an empty dictionary to store results



    # Add a new column for image name
    df['image_name'] = df['image'].apply(extract_image_name)
    
    # Group by image name and collect RTs
    grouped = df.groupby('image_name')['RT'].apply(list).to_dict()
    
    # Merge with the main result dictionary
    for key, value in grouped.items():
        if key in result_dict:
            result_dict[key].extend(value)
        else:
            result_dict[key] = value

print(result_dict)
# %%
