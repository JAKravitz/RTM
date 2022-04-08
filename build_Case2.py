#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:05:05 2022

@author: jakravit
"""
from build_library import *
import os
import pandas as pd
import numpy as np
import pickle
import random

def build_Case2(phy_library, datamin, datadet, benthic_lib, adj_lib, aero_lib):
    
    # initiate dictionary
    iops = {}
    
    # VSF angles
    theta901 = np.arange(0, 90.1, 0.1) # length 901
    nang=901
    angles=np.cos(np.deg2rad(theta901)) # length 901
    theta1 = np.deg2rad(theta901) 
    theta2 = np.flipud(np.pi-np.deg2rad(theta901[0:900]))
    theta=np.append(theta1,theta2)
    iops['VSF_angles'] = np.rad2deg(theta)
    
    # lambda
    iops['lambda'] = np.arange(.4, .9025, .0025).astype(np.float32) 
    
#################### PHYTOPLANKTON ##############################################    
    
    # assign class contributions
    alphas = [.5, 1, 5, 10]
    groups = ['Haptophytes','Diatoms','Dinoflagellates','Cryptophytes',
              'Green_algae','Cyano_blue','Heterokonts','Cyano_red','Rhodophytes']
    phyto_class_frxn, maxpft = dirichlet_phyto(alphas)

    # define species for each class first
    frxns = np.linspace(.05, .95, 50)
    for c, f in phyto_class_frxn.items():
        specs = np.random.choice(list(phy_library[c].keys()),2)
        fx = np.random.choice(frxns)
        for i, sp in enumerate(specs):
            f['sps'].append(sp+'_{}'.format(i))
            if i == 0:
                f['fx'].append(fx)
            else:
                f['fx'].append(1-fx)
    
    # define chl distribution and get chl conc
    chl = define_case2_chlDist(phyto_class_frxn, maxpft)
    
    # phyto classes
    classIOPs = {}
    classIOPs['TotChl'] = chl
    classIOPs = phyto_iops(phyto_class_frxn, phy_library, classIOPs)
    iops['Phyto'] = classIOPs    

####################### NON ALGAL PARTICLES ####################################
#%
    # MINERALS
    
    run_mins = random.choices(list(datamin.keys()), k=2) # allows repeats
    fx = np.random.choice(frxns)
    min_frxn = {run_mins[0]+'1': fx,
                run_mins[1]+'2': 1-fx}
        
    sigma,scale = lognorm_params(.5,500)
    napData = lognorm_random(sigma, scale, 20000)  
    nap = round(np.random.choice(napData), 3)
    sf = np.random.choice(np.linspace(.6, .95, 50))
    minl = nap * sf

    # mineral component
    minIOPs = {}
    minIOPs['Tot_conc'] = minl
    minIOPs = min_iops(min_frxn, datamin, minIOPs)    
    iops['Min'] = minIOPs 
    
    # DETRITUS
    
    detIOPs = {}
    cdet = nap - minl 
    detIOPs = det_iops(cdet, datadet, detIOPs)
    iops['Det'] = detIOPs

##################### CDOM ######################################################
#%
    sigma,scale = lognorm_params(.2,20)
    domData = lognorm_random(sigma, scale, 20000)  
    domData = domData[domData < 200]
    ag440 = round(np.random.choice(domData), 3)
    cdomIOPs = cdom_iops(ag440)
    iops['CDOM'] = cdomIOPs

################### DEPTH FUNCTION #############################################
#%
    depth = np.random.choice(np.arange(1,21,1)) * -1
    s1 = np.arange(.005, .1, .005)
    s2 = np.arange(.1,.6,.05)
    s3 = np.arange(.6,1,.1)
    s = np.concatenate([s1,s2,s3])
    slope = np.random.choice(s)
    c = 30 # hypothetical (doesnt matter for xfactor)
    d = np.arange(0,depth,-.5)
    yfactor = []
    xfactor = []
    for k in d:
        y = c * np.exp(-slope*-k)
        x = y/c
        yfactor.append(y)
        if k == 0:
            xfactor.append(0)
        else:
            xfactor.append(x)
    
    dprops = {'Depth':d,
              'xfactor':xfactor,
              'slope':slope,
              'Dmax':d.min()*-1}
    iops['Depth'] = dprops

##################### Benthic Reflectance #######################################
#%
    groups = ['Bleached coral','Blue coral','Brown coral','Brown/red algae',
              'CCA/turf/MPB/cyano','Green algae','Green coral','Mud','Sand/rock',
              'Seagrass/weed']
    benthic_frxn = dirichlet_benthic(alphas,groups,benthic_lib)
    iops['Benthic'] = benthic_frxn

##################### Adjacency Reflectance ####################################
#%
    groups = ['Ice','Manmade','Non-photo Veg','Soil','Vegetation']
    adj_frxn = dirichlet_adj(alphas,groups,adj_lib)
    iops['Adjacency'] = adj_frxn 

#################### ATMOSPHERE ################################################
#%  
    atm =  {'aero': aero_lib.sample(),
            'atm_prof': np.random.choice(['afglt','afglms','afglmw','afglss','afglsw',
                                         'afglus']),
            'VZA': np.random.choice(range(10,50)),
            'VAA': np.random.choice(range(60,120))}
    iops['Atm'] = atm
    
    return iops