#%%
from src.preprocess import *
from src.neural_network import *

# load the data
path = '/mnt/cs/projects/HEFEFTMS/data/P006/P006.hdf5'
df, eventNames, eventTimes = create_df(path)

# lowpassfilter x and y at 50 Hz

df = zero_phase_lowpass_filter_df(df,cutoff_freq=100,sample_rate=1000,order=4)

plt.plot(df.gaze_x[5000:10000],linewidth=2,alpha=0.5)
plt.plot(df.gaze_y[5000:10000],linewidth=2,alpha=0.5)


# create training data for the neural network
path2save = '/cs/projects/HEFEFTMS/data/P006/training_data'
make_labeling_data(df,path2save)

#%%


model = uneye.DNN(max_iter=500, sampfreq=1000,
             lr=0.001, weights_name='weights',
            min_sacc_dist=1,min_sacc_dur=6,augmentation=True,
             ks=5,mp=5,inf_correction=1.5,val_samples=30)

data_path = '/cs/projects/HEFEFTMS/uneye/data/'
X = np.genfromtxt(data_path + 'human_eye_x.csv', delimiter=',',usecols=np.arange(0,1000))
Y = np.genfromtxt(data_path + 'human_eye_y.csv', delimiter=',')
Labels = np.genfromtxt(data_path + 'human_binary_labels.csv', delimiter=',')
weights_name = 'fef_weights'


model.train(X, Y, Labels)
plt.plot(X[10,:])
plt.plot(Y[10,:])
plt.plot(Labels[10,:]*600)

#train_network(data_path='/cs/projects/HEFEFTMS/uneye/data/')


# %%
