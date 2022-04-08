#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:27:27 2022

@author: jakravit
"""
import numpy as np
import pandas as pd
from luts.luts import LUT, MLUT, Idx, merge, read_mlut
from smartg.tools.phase import integ_phase
from smartg.albedo import Albedo_cst
from smartg.water import IOP
from scipy.interpolate import interp1d
from smartg.tools.phase import integ_phase, calc_iphase

WAV = np.linspace(400, 900., num=201, dtype=np.float32)
WAV_CDOM = np.arange(240., 900., 2.5, dtype=np.float32)

class Albedo_speclib2(object):
    '''
    Albedo from speclib
    (http://speclib.jpl.nasa.gov/)
    '''
    def __init__(self, df):
        # data = np.genfromtxt(filename, skip_header=26)
        # convert X axis from micrometers to nm
        # convert Y axis from percent to dimensionless
        l = [df.index.values.astype(np.float32)]
        r = df.values
        self.data = LUT(r, axes=l, names=['wavelength'])

    def get(self, wl):
        return self.data[Idx(wl)]

class Albedo_spectrum2(object):
    '''
    Albedo R(lambda)

    R spectral albedo, lam in nm
    '''
    def __init__(self, df):
        l = [df.index.values.astype(np.float32)]
        r = df.values / 100
        self.data = LUT(r, axes=l, names=['wavelength'])

    def get(self, wl):
        return self.data[Idx(wl)]

def norm_vsf(vsf, angles):
    '''
    return the normalized VSF in the sense of SMART-G
    i.e. Sum(-1,1) P(mu) dmu = 2.
    output VSF has the same shape as the input VSF
    '''
    theta = np.deg2rad(angles)
    d1    = np.diff(theta)
    dtheta = np.append(d1,d1[0])
    norm   = np.sum(vsf * np.sin(theta) * dtheta, axis=2)
    
    return 2*vsf/norm[:,:,None]


def import_vsf(vsf_list, wav, Z, angles):
    '''
    format the list of input VSF's to SMART-G
    each VSF should have the shape (1, wav.size, angles.size)
    the different VSFs in the list correspond to the vertical starting 
    from the surface
    the max number of VSF's is Z.size, if it is lower, last VSF in the list is applied
    until the sea bottom
    
    Output:
        LUT object adapted to SMART-G, with 4 dimensions (WAV, Z, Stokes parameter (4), angles)
    '''
    Nlevel = len(vsf_list)
    Nwav   = vsf_list[0].shape[1]
    Nangle = vsf_list[0].shape[2]
    Nstk   = 4
    assert wav.shape[0]==Nwav
    assert angles.shape[0]==Nangle
    dat    = np.zeros((Nwav, Nlevel, Nstk, Nangle), dtype=np.float32)
    test   = np.concatenate([norm_vsf(vsf, angles) for vsf in vsf_list]).swapaxes(0,1)[:,:,:]
    dat[:,:,0,:] = test
    dat[:,:,1,:] = test
    
    return LUT(dat, names = ['wav_phase_oc', 'z_phase_oc', 'stk', 'theta_oc'],
                        axes = [wav, Z[:Nlevel], None, angles])


def truncate(pha, ang_trunc=10.):
    '''
    Phase matrix truncation
    
    Inputs:
        Phase Matix LUT and truncation angle
    Outputs:
        Truncated Phase Matrix LUT and truncation coefficient
    '''
    norm=integ_phase(np.radians(pha.axis('theta_oc')), (pha.data[:,:,0,:] + pha.data[:,:,1,:])/2.)
    NANG=pha.axis('theta_oc').size
    ang = np.linspace(0, np.pi, NANG, dtype='float64')
    itronc = int(NANG * ang_trunc/180.)
    #
    pha.data[:,:,0,:itronc] = pha.data[:,:,0,itronc][:,:,None]
    pha.data[:,:,1,:itronc] = pha.data[:,:,1,itronc][:,:,None]
    pha.data[:,:,2,:] = 0.
    pha.data[:,:,3,:] = 0.
    trunc = norm/integ_phase(np.radians(pha.axis('theta_oc')),
                             (pha.data[:,:,0,:] + pha.data[:,:,1,:])/2.) 
    
    return pha * trunc[:,:,None,None], trunc.flatten()


def import_iop(ap_list, bp_list, ac_list, vsf_list, wav, wav_vsf, Z, angles, 
               aw_list=None, bw_list=None, ang_trunc=None, ALB=Albedo_cst(0.)):
    '''
    format all IOP's to SMART-G inputs
    
    Inputs:
        ap_list : list of ap (particulate absorption in m-1) arrays of size wav.size, 
            the list should have Z.size elements ans represents the vertical distribution from surface to bottom. 
            the first ap arrays is ignored in the computation.
        bp_list : same for particulate scattering
        ac_list : same for CDOM absorption
        vsf_list: each VSF should have the shape (1, wav.size, angles.size)the different VSFs 
            in the list correspond to the vertical starting from the surface. The max number of VSF's is Z.size,
            if it is lower, last VSF in the list is applied until the sea bottom
        wav : wavelengths of absorption and scattering coefficients
        wav_vsf : wavlengths os VSF's
        Z   : depth grid in m (negative convention)
        angles : scateering angles in degree
    
    Keywords:
        aw_list: same as ap_list but for pure water (default: None, computed from SMART-G itself)
        bw_list: same as bp_list but for pure water (default: None, computed from SMART-G itself)
        ang_trunc: truncation angle in degree (default: None, no truncation)
        ALB : Albedo object from SMART-G (Albedo_spectrum, Albedo_speclib, Albedo_cst) 
            for seafloor albedo (default: black albedo, Albedo_cst(0.))
            
    Output : MLUT water optical properties computed for the grid (wav, Z), for input in the Smartg run() method,
             water keyword
        
    '''
    pha = import_vsf(vsf_list, wav_vsf, Z, angles)
    ap  = np.stack(ap_list, axis=1)
    ac  = np.stack(ac_list, axis=1)
    bp  = np.stack(bp_list, axis=1)
    aw  = np.stack(aw_list, axis=1) if aw_list is not None else None
    bw  = np.stack(bw_list, axis=1) if bw_list is not None else None
    if ang_trunc is not None:
        pha_trunc, trunc = truncate(pha, ang_trunc=ang_trunc)
        _,ipha_trunc = calc_iphase(pha_trunc, wav, Z)
        bp /= trunc[ipha_trunc]
        return IOP(phase=pha_trunc, ap=ap, bp=bp, aCDOM=ac, aw=aw, bw=bw, Z=Z, ALB=ALB).calc(wav)
    else:
        return IOP(phase=pha      , ap=ap, bp=bp, aCDOM=ac, aw=aw, bw=bw, Z=Z, ALB=ALB).calc(wav)


def mix(runData):
    '''    
    Compute the optical properties of the mixing
    particulate absorption and scattering coefficients
    CDOM absorption coefficient
    particulate VSF
    Inputs : 
        runData
    Outputs : 
        Absorption of particulate, CDOM, scattering of particulate, VSF and scattering angles of the
        mixing at the WAV grid
    '''
    ap  = np.zeros_like(WAV)
    ac  = np.zeros_like(WAV)
    bp  = np.zeros_like(WAV)
    vsf = np.zeros((1,WAV.size,1801), dtype=np.float32)
    angles = runData['VSF_angles']
    for key in runData.keys():
        comp = runData[key]
        if key in ['Phyto','Min','Det']:
            ap += comp['a_tot']
            bp += comp['b_tot']
            vsf+= comp['b_tot'][None,:,None] * comp['VSF_tot']
        elif key in ['CDOM']:
            ac += interp1d(WAV_CDOM, comp['a_tot'], kind='cubic', fill_value='extrapolate')(WAV)
        # if 'VSF_angles' in comp.keys():
        #     angles = comp['VSF_angles']
            
    return ap,ac,bp,vsf/bp[None,:,None],angles
