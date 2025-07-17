#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt



"""
Plots

Code written by Nile Coble

"""

plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                    'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')

#%% load data
# all_data = np.load('./data/with_friction.npy') Code for the non-compressed data
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

#%% plot of all k paths overlapped
plt.figure(figsize=(6.5,2.5))
plt.plot(t, k.T, linewidth=0.7)
plt.xlim((0, 120))
plt.ylim((450, 1550))
plt.title('stiffness paths over time')
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.tight_layout()
plt.savefig('./plots/k_paths.png', dpi=300)

#%% plot one acceleration signal
plt.figure(figsize=(6, 2.2))
plt.plot(t, a[0], linewidth=0.3)
plt.xlim((0, 120))
plt.xlabel('time (s)')
plt.ylabel(r'acceleration ($m/s^2$)')
plt.title('acceleration')
plt.tight_layout()
plt.savefig('./plots/one_accel.png', dpi=300)

#%% plot one pure physics prediction
all_data = np.load('./data/with_friction.npz')['data']

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

data_reconstructed = np.load('./model_predictions/pure_physics/k_pred.npy')[:,:,:-1]
k_pred = data_reconstructed[:,0,:]
a_pred = data_reconstructed[:,1,:]


i=81
plt.figure(figsize=(6, 2.3))
plt.plot(t[:-1], k_pred[i], c='tab:orange', label='predicted')
plt.plot(t, k[i], c='tab:blue', label='true')
plt.xlim((0, 120))
plt.ylim((450, 1550))
plt.title('pure physics predection')
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/pure_physics_pred.png', dpi=300)

#%% pure physics cumulative error with time plot
all_data = np.load('./data/with_friction.npz')['data'][:,:-1,:]
t = all_data[0,:,0]
rmse_k = np.load('./metric_results/pure_physics/rmse_k.npy')
rmse_a = np.load('./metric_results/pure_physics/rmse_a.npy')
residual = np.load('./metric_results/pure_physics/residual.npy')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 3.1), sharex=True)
ax1.set_title('pure physics predection error')
ax1.plot(t, rmse_k)
ax1.set_ylabel('RMSE ($N/s$)')
ax2.plot(t, rmse_a)
ax2.set_ylabel(r'RMSE ($m/s^2$)')
ax3.plot(t, residual, linewidth=0.5)
ax3.set_xlabel('time ($s$)')
ax3.set_ylabel('residual ($N$)')
ax3.set_xlim((0, 120))
fig.tight_layout()
fig.savefig('./plots/pure_physics_cumulative.png', dpi=300)

#%% plot one pure nn prediction
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

k_pred = np.load('./model_predictions/pure_nn/k_pred.npy')
i=0
plt.figure(figsize=(6,2.7))
plt.plot(t[49:], k_pred[i], c='tab:orange', label='predicted')
plt.plot(t, k[80+i], c='tab:blue', label='true')
plt.title('pure nerual network predection')
plt.xlim((0, 120))
plt.ylim((450, 1550))
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/pure_nn_pred.png', dpi=300)

#%% plot an indirect measurement, filtered test
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]

k_pred = np.load('./model_predictions/indirect/k_pred_filtered.npy')

i=5
plt.figure(figsize=(6,2.4))
plt.plot(t[58:], k_pred[i], c='tab:orange', label='predicted')
plt.plot(t, k[80+i], c='tab:blue', label='true')
plt.title('Indirect Measurement Predection')
plt.xlim((0, 120))
plt.ylim((375, 1800))
plt.xlabel('time (s)')
plt.ylabel('stiffness (N/m)')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/indirect_measurement_pred.png', dpi=300)

















