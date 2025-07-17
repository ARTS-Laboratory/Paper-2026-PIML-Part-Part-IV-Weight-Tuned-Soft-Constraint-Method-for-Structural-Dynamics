#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import os

"""
Run to convert .csv files to .npy.

This code creates three files in the "PIML examples\data" data folder, 
pinn_test_0.npy, with_friction.npz and no_friction.npz.

The comressed versions are used to get them under 500 MB so they can push to GitHub

"""

#%% load data
# n_tests = len(os.listdir('./test_data'))
# test1 = np.loadtxt('./test_data/test_1.csv', delimiter=',')
# all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
# for test in range(0, n_tests):
#     all_data[test] = np.loadtxt(f'./test_data/test_{test+1}.csv', delimiter=',')
# np.save('all_data.npy', all_data)


#%% for pinn tests
test0 = np.loadtxt('./data/pinn_data/test_0.csv', delimiter=',')
np.save('../PIML examples/data/pinn_test_0.npy', test0)


#%% with friction
n_tests = len(os.listdir('./data/with_friction'))-1 # -1 needed to account for the markdown file
test1 = np.loadtxt('./data/with_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./data/with_friction/test_{test+1}.csv', delimiter=',')
np.savez_compressed('../PIML examples/data/with_friction', data=all_data)

#%% no friction
n_tests = len(os.listdir('./data/no_friction')) -1 # -1 needed to account for the markdown file
test1 = np.loadtxt('./data/no_friction/test_1.csv', delimiter=',')
all_data = np.zeros((n_tests,) + test1.shape, dtype=float)
for test in range(0, n_tests):
    all_data[test] = np.loadtxt(f'./data/no_friction/test_{test+1}.csv', delimiter=',')
np.savez_compressed('../PIML examples/data/no_friction', data=all_data)

