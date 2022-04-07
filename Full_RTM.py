#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:51:13 2022

@author: jakravit
"""
#%% SMARTG libraries 

%pylab inline
# next 2 lines allow to automatically reload modules that have been changed externally
%reload_ext autoreload
%autoreload 2
import os, sys
sys.path.insert(0, '/Users/jakravit/git/smartg/')
from luts.luts import LUT, MLUT, Idx, merge, read_mlut

from smartg.albedo import Albedo_spectrum, Albedo_speclib, Albedo_cst
from smartg.bandset import BandSet
from smartg.smartg import Smartg, Sensor, multi_profiles, reduce_diff
from smartg.smartg import RoughSurface, LambSurface, FlatSurface, Environment, multi_profiles
from smartg.atmosphere import AtmAFGL, AeroOPAC, CompOPAC, CloudOPAC, diff1, read_phase, Species, trapzinterp
from smartg.water import IOP_1, IOP, IOP_profile, IOP_Rw, IOP_base
from smartg.reptran import REPTRAN, reduce_reptran
#from smartg.kdis import KDIS, reduce_kdis
from smartg.tools.tools import SpherIrr, Irr, reduce_Irr
from smartg.tools.cdf import ICDF
from smartg.tools.phase import integ_phase, calc_iphase

from smartg.tools.smartg_view import compare, plot_polar, spectrum , mdesc 
from smartg.tools.smartg_view import spectrum_view,transect_view,profile_view,phase_view,smartg_view,input_view
import warnings

warnings.filterwarnings("ignore")

#%% Data paths and libraries

from scipy.interpolate import interp1d
import pickle
import pandas as pd

# Data paths 

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
aero_lib = pd.read_csv('/Users/jakravit/data/aeronet_invdata_match.csv')

#%%

#----------------------------- START BUILDING ---------------------------------#
from smartg_int_library import *
from build_Case2 import build_Case2

# Case
sname_title = 'case2_test'

# lambda
# l = np.arange(400, 902.5, 2.5)  
# Static data : IOPs wavelength grids
WAV = np.linspace(400, 900., num=201, dtype=np.float32)
WAV_CDOM = np.arange(240., 900., 2.5, dtype=np.float32)

# run names
snames = []
runlist = {}

# initiate df
df = pd.DataFrame()

# how many runs to build
runs = 3

for k in range(runs):
    print (k)
    # initiate iop dict
    iops = build_Case2(phy_library, datamin, datadet, benthic_lib, adj_lib, aero_lib)
    
    # get mixing IOP's
    ap1,ac1,bp1,vsf1,angles = mix(iops)
    
    # wavelength grid/slice for absorption and scattering coefficients
    # wavelength grid for outputs
    wavrange = slice(0,200,5)
    wav      = WAV[wavrange]
    # wavelength grid/slice for scattering matrices
    wavrange_vsf = slice(0,200,20)
    wav_vsf      = WAV[wavrange_vsf]
    # vertical grid
    Z = iops['Depth']['Depth']
    # Z   = np.array([0, -2.5, -5., -10.])
    xfactor = iops['Depth']['xfactor']
    zeros = np.zeros_like(wav)
    
    #IOP's profiles
    bp_list = [bp1[wavrange] * x for x in xfactor] 
    ap_list = [ap1[wavrange] * x for x in xfactor]
    ac_list = [ac1[wavrange] * x for x in xfactor]
    vsf_list = [vsf1[:,wavrange_vsf,:] * x for x in xfactor]
    aw_list = None # use SMART-G default pure water absorption
    bw_list = None # use SMART-G default pure water scattering
    
    # Albedos speclib library inputs
    
    bALB = Albedo_spectrum(iops['Benthic']['Tot'])
    aAlB = Albedo_speclib(iops['Adjacency']['Tot'])

    
    # water MLUT building
    # VSF truncation at 10 deg
    # seafloor Sand albedo
    water   = import_iop(ap_list, bp_list, ac_list, vsf_list, wav, wav_vsf, Z, angles, 
                        aw_list=aw_list, bw_list=bw_list, ang_trunc=5.,  ALB=bALB).describe()
