import os
import pandas as pd
import numpy as np
import csv
import uneye


def make_labeling_data(df,path2save):
    # cut the data in 3 second chunks and save it in a list
    X = []; Y = []

    for j in range(0, len(df), 3*1000):
        # create a list of 3 second chunks
        X.append(list( df.gaze_x[j:j+3*1000]))
        Y.append(list( df.gaze_y[j:j+3*1000]))


    # make a directory
    try:
        os.mkdir(f'{path2save}')
    except:
        pass

    # opening the csv file in 'w+' mode
    file = open(f'{path2save}/X.csv', 'w+')
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(X)

    file = open(f'{path2save}/Y.csv', 'w+')
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(Y)



def train_network(data_path,Fs=1000):
    # load the data
    X = np.genfromtxt(data_path + 'human_eye_x.csv', delimiter=',',usecols=np.arange(0,1000))
    Y = np.genfromtxt(data_path + 'human_eye_y.csv', delimiter=',')
    Labels = np.genfromtxt(data_path + 'human_binary_labels.csv', delimiter=',')
    weights_name = 'fef_weights'

    # initialize the model with specified sampling frequency and weights
    model = uneye.DNN(sampfreq=Fs, weights_name=weights_name,val_samples=20)

    # train the model
    model.train(X, Y, Labels)

