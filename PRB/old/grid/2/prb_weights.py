#!usr/bin/env python
# coding: utf-8

# # prototyping LSS pipeline

# ## setup

# In[1]:


# In[2]:


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


# In[3]:


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


# In[4]:


import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))


# In[5]:


import sys
from hde import *#__init__, hde, propagator, molgen, analysis #


# ## locating trajectory data

# In[6]:


# DESRES Protein B
# Lindorff-Larsen et al. Science 334 6055 517-520 (2011)
# 510ps time steps, 272 us traj, (10,000 * 53 frames)
timestep = 200 # ps

pdb_dir = "/home/jaulicino/scratch-midway2/PRB/DESRES-Trajectory_PRB-0-protein/PRB-0-protein"
pdb_file = os.path.join(pdb_dir, "system.pdb")

trj_dir = pdb_dir
trj_file = []
for i in np.arange(0,10):
    fname = "PRB-0-protein-" + str(i).zfill(3) + ".dcd"
    trj_file.append(os.path.join(trj_dir, fname))
print(pdb_file)


# In[7]:


traj_obj = md.load(trj_file, top=pdb_file)


# In[9]:


traj_obj.center_coordinates(mass_weighted=False)
traj_obj.superpose(traj_obj[0])


# ## featurization

# In[15]:


# featurizing
features = py.coordinates.featurizer(pdb_file)
features.add_backbone_torsions(cossin=True)
features.add_sidechain_torsions(which='all', cossin=True)
atom_idx = features.select_Backbone() # select_Heavy() select_Ca()
features.add_inverse_distances(atom_idx)

#print(features.describe())
print('dim = %d' % features.dimension())


# In[16]:


data = np.zeros((0,features.dimension()))
for i in range(len(trj_file)):
    q = py.coordinates.load(trj_file[i], features=features)
    data = np.concatenate((data,q), axis=0)
    print(i)
    print(data.shape)


# ## (1) latent space projection

# ### parameters

# In[17]:


lag=50
dim=1
is_reversible=True


# ### SRV

# In[26]:


#earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)

hde = HDE(
    features.dimension(), 
    n_components=dim, 
    lag_time=lag,
    reversible=is_reversible, 
    n_epochs=300,
    learning_rate=0.0005,
    hidden_layer_depth=2,
    hidden_size=100,
    activation='tanh', 
    batch_size=20000,
    batch_normalization=True,
    #callbacks=[earlyStopping], 
    verbose=True
)


# In[27]:


hde.fit(data)


# In[28]:


hde.callbacks = None
hde.history = None
pickle.dump(hde, open('hde.pkl', 'wb'), protocol=4)


# In[18]:



# In[20]:


hde_coords = hde.transform(data, side='left')
#hde_coords_right = hde.transform(data, side='right')
hde_timescales = hde.timescales_
print(hde_timescales)
print(hde_coords)
#print(hde_coords_right)


# ### SRV k-means clustering

# #### n_cluster optimization using silhouette score
# 
# - expect n_cluster = (dim_kmeans+1) similar to PCCA (# macrostates = 1 + # singular vectors)
# 
# - Idea of inner simplex (PCCA) clustering directly in singular vectors of transfer operator:
# F. Paul, H. Wu, M. Vossel, B.L. de Groot, and F. Noe "Identification of kinetic order parameters
# for non-equilibrium dynamics" J. Chem. Phys. 150, 164120 (2019); doi: 10.1063/1.5083627
# 
# - k-means simpler and only for understanding of macrostates not construction of MSM

# In[35]:


dim_kmeans = 2
hde_coords_kmeans = copy.deepcopy(hde_coords[:,:dim_kmeans])


# In[36]:


range_n_clusters = np.arange(2,5,1)
silhouette_avg_array = []

for n_clusters in range_n_clusters:
    
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=200184)
    cluster_labels = clusterer.fit_predict(hde_coords_kmeans)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(hde_coords_kmeans, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_avg_array.append(silhouette_avg)


# In[37]:


n_clusters = dim_kmeans+1 # usually, but check silhouette scores


# In[38]:


clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(hde_coords_kmeans)
centers = clusterer.cluster_centers_
sample_silhouette_values = silhouette_samples(hde_coords_kmeans, cluster_labels)


# #### visualizing configs proximate to k-means centers

# In[44]:


centers


# In[45]:


kmeans_viz = []
for i in range(n_clusters):
    idx_sort = np.argsort(np.sqrt(np.sum((hde_coords_kmeans - centers[i,:])**2, axis=1)))
    kmeans_viz.append(idx_sort[:10])


# In[47]:




# In[49]:




# In[50]:


clust_id = 2
view = nv.NGLWidget()
for k in range(len(kmeans_viz[clust_id])):
    view.add_trajectory(traj_obj[kmeans_viz[clust_id][k]])
view


# ## (2) propagator

# In[40]:


dim_prop = dim


# In[41]:


traj_prop = copy.deepcopy(hde_coords[:,:dim_prop])


# In[42]:


prop_scaler = pre.MinMaxScaler(feature_range=(0,1))
if dim_prop==1:
    traj_prop_scaled = prop_scaler.fit_transform(traj_prop.reshape(-1, 1))
else:
    traj_prop_scaled = prop_scaler.fit_transform(traj_prop)


# In[43]:


n_mix = 25
lag_prop = lag
lr_prop = 0.0001


# In[44]:


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
    prop.fit([traj_prop_scaled, traj_prop_scaled[::-1]]) # reversible -- data augmentation with inverted trajectory
else:
    prop.fit(traj_prop_scaled)


# In[62]:


pickle.dump(prop.model.get_weights(), open('prop_weights.pkl', 'wb'))


# In[46]:


#prop.model.set_weights(pickle.load(open('prop_weights.pkl', 'rb')))


# In[62]:


n_steps = np.int(np.floor(np.float(hde_coords.shape[0])/np.float(lag_prop)))
n_traj = 5
synth_trajs_scaled = [prop.propagate(traj_prop_scaled[0].reshape(1,-1).astype(np.float32), n_steps).reshape(n_steps, -1) for item in range(n_traj)]
synth_trajs = [prop_scaler.inverse_transform(synth_trajs_scaled[i]) for i in range(n_traj)]


# ## (3) generator

# #### x_train = scaled latent space coordinates

# In[65]:


x_train = traj_prop_scaled


# #### y_train = aligned molecular configurations

# In[66]:


ca_idx = traj_obj.top.select_atom_indices('alpha')
traj_ca = traj_obj.atom_slice(ca_idx)


# In[67]:


traj_ca.superpose(traj_ca[0])


# In[68]:


view = nv.show_mdtraj(traj_ca[::lag_prop])
#view.component_0.clear_representations()
view.component_0.add_ribbon(color='blue', opacity=0.6)
view
view


# In[69]:


xyz = traj_ca.xyz.reshape(-1, traj_ca.n_atoms*3)


# In[70]:


xyz_scaler = pre.MinMaxScaler(feature_range=(-1,1))


# In[71]:


y_train = xyz_scaler.fit_transform(xyz)


# In[72]:


n_atoms = traj_ca.n_atoms


# #### training cWGAN

# In[73]:


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


# In[7]:


molgen.fit(x_train, y_train)


# In[87]:


#molgen.fit(x_train, y_train) reason called twice?


# In[84]:


molgen.generator.save('molgen_generator.h5')


# In[85]:


molgen.discriminator.save('molgen_discriminator.h5')
