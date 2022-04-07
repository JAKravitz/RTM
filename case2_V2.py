#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:55:57 2022

@author: jakravit
"""
#%%
from __future__ import division
import os
import pandas as pd
import numpy as np
# from lognorm import lognorm_params,lognorm_random
import matplotlib.pyplot as plt
import pickle
import shortuuid
import random

# phytoplankton SIOP spectral library
path = '/Users/jakravit/data/EAP_phytoplankton_dataset/'
phy_library = {'Haptophytes': {}, 
               'Diatoms': {},
               'Dinoflagellates': {},
               'Cryptophytes': {},
               'Green_algae': {},
               'Cyano_blue': {},
               'Heterokonts': {},
               'Cyano_red': {},
               'Rhodophytes': {}
               }

for phy in phy_library:
    print (phy)
    with open(path+phy+'.p', 'rb') as fp:
        phy_library[phy] = pickle.load(fp)

# NAP spectral libraries
minpath = '/Users/jakravit/data/EAP_NAP_dataset/minerals_V2.p'
with open(minpath, 'rb') as fp:
    datamin = pickle.load(fp)  

detpath = '/Users/jakravit/data/EAP_NAP_dataset/det_V1.p'
with open(detpath, 'rb') as fp:
    datadet = pickle.load(fp) 

# Benthic library
benthic_lib = pd.read_csv('/Users/jakravit/data/benthic_floating_truth/benthic_spec_libary_FT_V4.csv')

# adjacency library
adj_lib = pd.read_csv('/Users/jakravit/data/benthic_floating_truth/adjacency_spectra_V2.csv')

# aeronet library
aero_lib = pd.read_csv('/Users/jakravit/data/aeronet_invdata.csv')

#%%
#----------------------------- START BUILDING ---------------------------------#
from build_library import *

# Case
sname_title = 'case2_test'

# lambda
l = np.arange(400, 902.5, 2.5)  

# run names
snames = []
runlist = {}

# initiate df
df = pd.DataFrame()

# how many runs to build
runs = 1

for k in range(runs):
    print (k)
    
    
    
    
    