#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
import os, pickle, time, glob, sys, copy
import numpy as np 
import scipy
import mdtraj as md 
import MDAnalysis as mda
import nglview as nv 
from ipywidgets import interactive, VBox
import sklearn.preprocessing as pre
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pyemma as py 
from pyemma.util.contexts import settings
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))


import sys
from hde import *#__init__, hde, propagator, molgen, analysis #



source_folds = "/project2/andrewferguson/mikejones/AT-all_SRV_data/individual_trajectories/"
full_source_folds = "/project2/andrewferguson/mikejones/AT-all_SRV_data/full_trajectories/"

#this list is which trajectories the system will be trained over. 
#Mike suggested sampling equally from the indexes 0-19 and 20-39 as they make up two different initial states 
#of the DNA molecules


avail_srvs = [0,1,2,3,20,21,22,23]






##traj names, in the same order of the SRV coordinates. SRV trajectories 0-19, 20-39 have different initial states.
##This list preserves the correct numbering of these states.

traj_names =[
"AT-all-implicit-dis-0-0_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-1_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-2_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-3_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-4_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-5_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-6_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-7_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-8_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-9_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-10_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-11_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-12_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-13_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-14_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-15_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-16_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-17_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-18_T-309_t-20_s-13e+09",
"AT-all-implicit-dis-0-19_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-0_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-1_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-2_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-3_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-4_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-5_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-6_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-7_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-8_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-9_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-10_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-11_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-12_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-13_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-14_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-15_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-16_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-17_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-18_T-309_t-20_s-13e+09",
"AT-all-implicit-hbr-0-19_T-309_t-20_s-13e+09"
]


num_srvs = len(avail_srvs)
traj_frame_length = 250000


#trajectories are actually 260001 frames, however the first 10.001 are removed from the SRV training data


#some colon wizardry to get the proper SRV coordinates and concatenate into a single array
hde_coords = np.zeros((num_srvs*traj_frame_length, 5))
for i in range(num_srvs):
    hde_coords[i*traj_frame_length:(i+1)*traj_frame_length, :] = np.load(source_folds+str(avail_srvs[i])+"_full_srv.npy")

#need the hde coordinates in a list so as not to train prop over the gaps

hde_coords2 = np.array([hde_coords[i*traj_frame_length:(i+1)*traj_frame_length] for i in range(num_srvs)])
dim_prop = 5


traj_prop = copy.deepcopy(hde_coords[:,:dim_prop])
prop_scaler = pre.MinMaxScaler(feature_range=(0,1))
traj_prop2 = copy.deepcopy(hde_coords2[:,:,:dim_prop])
prop_scaler2 = pre.MinMaxScaler(feature_range=(0,1))

if dim_prop==1:
    traj_prop_scaled = prop_scaler.fit_transform(traj_prop.reshape(-1, 1))
else:
    traj_prop_scaled = prop_scaler.fit_transform(traj_prop)
    traj_prop_scaled2 = [prop_scaler2.fit_transform(traj_prop2[i]) for i in range(num_srvs)]


lag_time = 12   # x 100 ps save rate = 1.2 ns
is_reversible = False

n_mix = 25
lag_prop = lag_time
lr_prop = 0.0001



callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
]


# In[45]:


prop = Propagator(
    traj_prop_scaled.shape[1], 
    n_components=n_mix, 
    lag_time=lag_prop, 
    batch_size=200000, 
    learning_rate=lr_prop, 
    n_epochs=20000,
    callbacks=callbacks,
    hidden_size=100,
    activation='relu'
)


# In[58]:


from hde.propagator import get_mixture_loss_func


# In[59]:


prop.model.compile(loss=get_mixture_loss_func(prop.input_dim, prop.n_components), optimizer=tf.keras.optimizers.Adam(lr=lr_prop))


# In[60]:

if is_reversible:
    prop.fit([traj_prop_scaled2, traj_prop_scaled2[::-1]]) # reversible -- data augmentation with inverted trajectory
else:
    prop.fit(traj_prop_scaled2)
    prop.fit(traj_prop_scaled2)


# In[62]:


pickle.dump(prop.model.get_weights(), open('prop_weights.pkl', 'wb'))

print("PROP SUCCESSFULLY TRAINED")

# In[46]:


#prop.model.set_weights(pickle.load(open('prop_weights.pkl', 'rb')))


# In[62]:
# ## (3) generator


# #### Setting Up Data
pdb_file = "dna.pdb"
traj_dir = "/project2/andrewferguson/mikejones/AT-all_SRV_data/full_trajectories"
trj_file = []
trj_file.append(traj_dir)
stride = 1 #can be used to remove some of training data and speed up convergence.

for i in range(num_srvs):
	f = full_source_folds +  str(traj_names[avail_srvs[i]])+"/traj.lammpstrj"
	temp = md.load_lammpstrj(f, stride = stride, top = pdb_file)
	temp = temp[(math.floor(10000/stride)+1):] #first 10001 not used
	if(i == 0):
		traj_obj = temp
	else:
		traj_obj = md.join(traj_obj, temp)
 
traj_obj.center_coordinates(mass_weighted=False)
traj_obj.superpose(traj_obj[0])

# ### mdtraj
traj_all = traj_obj
traj_all.superpose(traj_obj[0])
xyz = traj_obj.xyz.reshape(-1, traj_obj.n_atoms*3)
xyz_scaler = pre.MinMaxScaler(feature_range=(-1,1))
y_train = xyz_scaler.fit_transform(xyz)
x_train = traj_prop_scaled


molgen = MolGen(
    latent_dim=x_train.shape[1],
    output_dim=y_train.shape[1],
    batch_size=30000,
    noise_dim=50,
    n_epochs=2500,
    hidden_layer_depth=2,
    hidden_size=200,
    n_discriminator=5
)




print("fitting Molgen")
molgen.fit(x_train, y_train)
molgen.fit(x_train, y_train)
#potentially more calls required for convergence, stops at 2500 epochs automatically



molgen.generator.save('molgen_generator.h5')
molgen.discriminator.save('molgen_discriminator.h5')







