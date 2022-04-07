#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:06:20 2022

@author: jakravit
"""
import numpy as np
import pandas as pd

##
def lognorm_params(mode, stddev):
    import numpy as np
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale

##
def lognorm_random(sigma,scale,num):
    import numpy as np
    mu = np.log(scale)
    s = np.random.lognormal(mu, sigma, num)
    return s

##
def dirichlet_phyto (alphas):
    pftlist = ['Haptophytes','Diatoms','Dinoflagellates','Cryptophytes','Green_algae',
            'Cyano_blue','Heterokonts','Cyano_red','Rhodophytes']
    dist = np.random.choice(alphas)
    distributions = np.random.dirichlet(np.ones(9) / dist, size=1)[0]
    # dmax = distributions.max()
    dx = np.argmax(distributions)
    phyto_class_frxn = {   'Haptophytes': {'cfrxn': distributions[0], 'sps':[], 'fx':[]}, 
                            'Diatoms': {'cfrxn': distributions[1], 'sps': [], 'fx':[]},
                            'Dinoflagellates': {'cfrxn': distributions[2], 'sps':[], 'fx':[]},
                            'Cryptophytes': {'cfrxn': distributions[3], 'sps':[], 'fx':[]},
                            'Green_algae': {'cfrxn': distributions[4], 'sps':[], 'fx':[]},
                            'Cyano_blue': {'cfrxn': distributions[5], 'sps':[], 'fx':[]},
                            'Heterokonts': {'cfrxn': distributions[6], 'sps':[], 'fx':[]},
                            'Cyano_red': {'cfrxn': distributions[7], 'sps':[], 'fx':[]},
                            'Rhodophytes': {'cfrxn': distributions[8], 'sps':[], 'fx':[]}
                            }   
    maxpft = pftlist[dx]
    return phyto_class_frxn, maxpft

##
def sp_dist (c, lib, f):
    frxns = np.linspace(.05, .95, 50)
    if c in ['Ice','Manmade','Non-photo Veg','Soil','Vegetation']:
        specs = lib[lib.Group == c].sample(n=2,random_state=42)
    else:
        specs = lib[lib.Class == c].sample(n=2,random_state=42)
    specs = specs.iloc[:,3:]
    fx = np.random.choice(frxns)
    fxs = [fx, 1-fx]
    specs2 = specs.mul(fxs, axis=0)
    spec3 = specs2.sum() * f   
    return spec3
    

def dirichlet_benthic (alphas,groups,benthic_lib):
    l = np.arange(400, 902.5, 2.5) 
    specsTot = []
    specsAlgae = []
    specsSed = []
    specsSeagrass = []
    specsBleach = []
    specsCoral = []
    algaefx = []
    sedfx = []
    coralfx = []
    dist = np.random.choice(alphas)
    distributions = np.random.dirichlet(np.ones(len(groups)) / dist, size=1)[0]
    # dx = np.argmax(distributions)
    benthic_frxn = {'Algae':{},
                    'Sediment':{},
                    'Coral':{},
                    'Seagrass/weed':{},
                    'Bleached coral':{}}
    
    for i,c in enumerate(groups):
        f = distributions[i]
        if c in ['Brown/red algae','Green algae','CCA/turf/MPB/cyano']:
            spec = sp_dist(c, benthic_lib, f)
            benthic_frxn['Algae'][c] = {'cfx':f, 'spec':spec}
            specsTot.append(spec)
            specsAlgae.append(spec)
            algaefx.append(f)
        elif c in ['Sand/rock', 'Mud']:
            spec = sp_dist(c, benthic_lib, f)
            benthic_frxn['Sediment'][c] = {'cfx':f, 'spec':spec}
            specsTot.append(spec)
            specsSed.append(spec)
            sedfx.append(f)
        elif c in ['Brown coral','Blue coral', 'Green coral']:
            spec = sp_dist(c, benthic_lib, f)
            benthic_frxn['Coral'][c] = {'cfx':f, 'spec':spec}
            specsTot.append(spec)
            specsCoral.append(spec)
            coralfx.append(f)
        elif c in ['Seagrass/weed']:
            spec = sp_dist(c, benthic_lib, f)
            benthic_frxn[c] = {'gfx':f, 'spec':spec}
            specsTot.append(spec)
            specsSeagrass.append(spec)
        elif c in ['Bleached coral']:
            spec = sp_dist(c, benthic_lib, f)
            benthic_frxn[c] = {'gfx':f, 'spec':spec}
            specsTot.append(spec)
            specsBleach.append(spec)
    
    benthic_frxn['Algae']['tot'] = pd.DataFrame(specsAlgae).sum()
    benthic_frxn['Algae']['gfx'] = sum(algaefx)
    benthic_frxn['Sediment']['tot'] = pd.DataFrame(specsSed).sum()
    benthic_frxn['Sediment']['gfx'] = sum(sedfx)
    benthic_frxn['Coral']['tot'] = pd.DataFrame(specsCoral).sum()
    benthic_frxn['Coral']['gfx'] = sum(coralfx)
    benthic_frxn['Seagrass/weed']['tot'] = pd.DataFrame(specsSeagrass).sum()
    benthic_frxn['Bleached coral']['tot'] = pd.DataFrame(specsBleach).sum()
    benthic_frxn['Tot'] = pd.DataFrame(specsTot).sum()
    
    return benthic_frxn

##
def dirichlet_adj (alphas,groups,adj_lib):
    l = np.arange(400, 902.5, 2.5) 
    specsTot = []
    dist = np.random.choice(alphas)
    distributions = np.random.dirichlet(np.ones(len(groups)) / dist, size=1)[0]
    # dx = np.argmax(distributions)
    adj_frxn = {'Ice':{},
                'Manmade':{},
                'Non-photo Veg':{},
                'Soil':{},
                'Vegetation':{}}
    
    for i,c in enumerate(groups):
        f = distributions[i]
        spec = sp_dist(c, adj_lib, f)
        specsTot.append(spec)
        adj_frxn[c] = {'gfx':f, 'spec':spec}
        
    adj_frxn['Tot'] = pd.DataFrame(specsTot).sum()
    
    return adj_frxn

##
def define_case2_chlDist (phyto_class_frxn, maxpft):
    # if cyano bloom
    if maxpft == 'Cyano_blue' and 'M. aeruginosa' in phyto_class_frxn[maxpft]['sps']:
        
        # chl range for microcystis blooms
        if 0 < phyto_class_frxn[maxpft]['cfrxn'] < .3:
            sigma,scale = lognorm_params(3,5)
            chlDist = lognorm_random(sigma, scale, 20000)      
            # plt.hist(chlData, bins=500)
        
        elif .3 < phyto_class_frxn[maxpft]['cfrxn'] < .7:
            sigma,scale = lognorm_params(10,20)
            chlDist = lognorm_random(sigma, scale, 20000)
            # plt.hist(chlData, bins=500)
            
        else:
            sigma,scale = lognorm_params(50,300)
            chlDist = lognorm_random(sigma, scale, 20000)
            # plt.hist(chlData, bins=500)
    
    else:
        
        # chl ranges for everything else
        if 0 < phyto_class_frxn[maxpft]['cfrxn'] < .5:
            sigma,scale = lognorm_params(3,5)
            chlDist = lognorm_random(sigma, scale, 20000)  
            # plt.hist(chlData, bins=500)
            
        else:
            sigma,scale = lognorm_params(10,15)
            chlDist = lognorm_random(sigma, scale, 20000) 
    
    chl = round(np.random.choice(chlDist), 3)
    return chl
            
##
def phyto_iops (phyto_class_frxn, phy_library, classIOPs):            
    l = np.arange(400, 902.5, 2.5)  
    idx675 = np.where(l==675)[0]
    idx620 = np.where(l==620)[0]
    pc_tot = []
    for c, f in phyto_class_frxn.items():
        classIOPs[c] = {}
        
        # class chl contribution
        class_chl = f['cfrxn'] * classIOPs['TotChl']
        classIOPs[c]['class_chl'] = class_chl
        classIOPs[c]['class_frxn'] = f['cfrxn']
        
        for i, sp in enumerate(f['sps']):
            classIOPs[c][sp] = {}
            sims = phy_library[c][sp[:-2]]
            sim_idx = np.random.choice(list(sims), 1)[0] 
            info = sims[sim_idx]
            idx = np.random.choice(len(info['astar']), 1) 
            sfrxn = f['fx'][i] # species fraction
            classIOPs[c][sp]['sp_frxn'] = sfrxn
            deff = info['astar'].iloc[idx,:].index.values
            sp_chl = class_chl * sfrxn
            a = sp_chl * info['astar'].iloc[idx,:].values[0]
            classIOPs[c][sp]['a'] = a
            classIOPs[c][sp]['b'] = sp_chl * info['bstar'].iloc[idx,:].values[0]
            classIOPs[c][sp]['c'] = sp_chl * info['cstar'].iloc[idx,:].values[0]
            classIOPs[c][sp]['bb'] = sp_chl * info['bbstar'].iloc[idx,:].values[0]
            classIOPs[c][sp]['VSF'] = sp_chl * info['VSF'][idx,:,:]
            classIOPs[c][sp]['Deff'] = deff[0]
            classIOPs[c][sp]['sp_chl_conc'] = sp_chl
            for kk in ['ci','class','ncore','nshell','PFT1','PFT2','psdvol','size_class','Veff','Vs']:
                classIOPs[c][sp][kk] = info[kk]
            classIOPs[c][sp]['psdmax'] = info['psd'].max()           
            
            # phycocyanin model
            if c == 'Cyano_blue':
                # calculcate PC abs at 620 (remove chls)
                achla620 = a[idx675] * 0.179 # chla absorption at 620
                achlb620 = (a[idx620] - achla620) * 0.44 * 0.64 # chlb abs at 620
                achlc620 = (a[idx620] - achla620) * 0.059 * 1.14 # chlc abs at 620
                apc620 = a[idx620] - (achla620 + achlb620 + achlc620) # pc abs at 620
                
                # calculate a*pc at 620 based on admixture
                # scaled admixture eq (Kravitz et al. 2021)
                ad = f['cfrxn'] * sfrxn
                Sad = 0.0159*np.exp(6.2309*ad)
                astarpc620 = 0.0093*Sad**-0.717
                
                # Calculcate PC concentration per species
                sp_PC = apc620 / astarpc620
                classIOPs[c][sp]['sp_PC_conc'] = sp_PC[0]
                pc_tot.append(sp_PC)            
            
            
        # phyto class tot IOPs
        for p in ['a','b','c','bb','VSF']:
            class_tot_iop = 0
            for sp in f['sps']:
                class_tot_iop = class_tot_iop + classIOPs[c][sp][p]
            classIOPs[c]['{}_tot'.format(p)] = class_tot_iop
                  
        # phyto class size avg
        class_tot_sz = []
        for sp in f['sps']:
            class_tot_sz.append(classIOPs[c][sp]['Deff'])        
        cl_sz = np.mean(class_tot_sz)
        classIOPs[c]['class_Deff'] = cl_sz
        if cl_sz <= 2:
            cl_sz_cl = 'pico'
        elif 2 < cl_sz <= 11:
            cl_sz_cl = 'small_nano'
        elif 11 < cl_sz <= 20:
            cl_sz_cl = 'large_nano'
        elif cl_sz > 20:
            cl_sz_cl = 'micro'
        classIOPs[c]['class_sz_class'] = cl_sz_cl             
            
    # phyto component total iops
    for p in ['a','b','c','bb','VSF']:
        comp_tot_iop = 0
        for c in phyto_class_frxn.keys():
            comp_tot_iop = comp_tot_iop + classIOPs[c]['{}_tot'.format(p)]
        classIOPs['{}_tot'.format(p)] = comp_tot_iop           

    # tot PC
    classIOPs['TotPC'] = sum(pc_tot)[0]
    return classIOPs

##
def min_iops (min_frxn, datamin, minIOPs):
    l = np.arange(400, 902.5, 2.5)
    idx700 = int(np.where(l==700)[0])
    for c, f in min_frxn.items():
        minIOPs[c] = {}
        sims = datamin[c[:-1]]
        sim_idx = np.random.choice(list(sims.keys()))
        info = sims[sim_idx]
        cminl = minIOPs['Tot_conc'] * f
        minIOPs[c]['class_frxn'] = f
        minIOPs[c]['class_conc'] = cminl
        astar = info['astar'][0]
        minIOPs[c]['a'] = cminl * astar
        minIOPs[c]['b'] = cminl * info['bstar'][0]
        minIOPs[c]['bb'] = cminl * info['bbstar'][0]
        minIOPs[c]['VSF'] = cminl * info['VSF'][0]
        minIOPs[c]['class_slope'] = np.polyfit(l[:idx700], np.log(astar[:idx700]),1)[0]
        for kk in ['j','nreal','rho']:
            minIOPs[c][kk] = info[kk]
        minIOPs[c]['psdmax'] = info['dmax']

    # mineral component total iops
    for p in ['a','b','bb','VSF']:
        comp_tot_iop = 0 # iops
        for c in min_frxn.keys():
            comp_tot_iop = comp_tot_iop + minIOPs[c]['{}'.format(p)]
            minIOPs['{}_tot'.format(p)] = comp_tot_iop
            #comp_tot_C = comp_tot_C + minIOPs[c]['class_conc']
    
    minIOPs['Tot_slope'] = np.polyfit(l[:idx700], np.log(minIOPs['a_tot'][:idx700]),1)[0]    
    return minIOPs

##
def det_iops (cdet, datadet, detIOPs):
    l = np.arange(400, 902.5, 2.5)
    idx700 = int(np.where(l==700)[0])
    sim_idx = np.random.choice(list(datadet.keys()))
    info = datadet[sim_idx]
    astar = info['astar'][0]
    # cdet = nap - minl 
    detIOPs['Tot_conc'] = cdet
    detIOPs['a_tot'] = cdet * astar
    detIOPs['b_tot'] = cdet * info['bstar'][0]
    detIOPs['bb_tot'] = cdet * info['bbstar'][0]
    detIOPs['VSF_tot'] = cdet * info['VSF'][0]
    detIOPs['Tot_slope'] = np.polyfit(l[:idx700], np.log(astar[:idx700]),1)[0]       
    for kk in ['j','nreal','rho']:
        detIOPs[kk] = info[kk]
    detIOPs['psdmax'] = info['dmax']   
    return detIOPs

##
def cdom_iops (ag440):
    l = np.arange(240,900, 2.5)
    for k in range(10):       
        # random slope (240-900) from normal dist.
        slopes = np.random.normal(.03,.005,5000)
        slopes = slopes[slopes > 0]
        slope = np.random.choice(slopes)
        ag1 = ag440 * np.exp(-slope * (l-440))
        ag350 = ag1[np.where(l==350)]
        ag265 = ag1[np.where(l==265)]
        cdomIOPs = {}
        comp_data = {1: {'mu': np.random.normal(269,3,1000),
                         'std': np.random.normal(20,3,1000)}, 
                     2: {'mu': np.random.normal(299,5,1000),
                         'std': np.random.normal(20,5,1000)},
                     3: {'mu': np.random.normal(320,5,1000),
                         'std': np.random.normal(30,3,1000)},
                     4: {'mu': np.random.normal(345,5,1000),
                         'std': np.random.normal(55,5,1000)},
                     5: {'mu': np.random.normal(375,5,1000),
                         'std': np.random.normal(70,5,1000)},
                     6: {'mu': np.random.normal(407,5,1000),
                         'std': np.random.normal(90,10,1000)},
                     7: {'mu': np.random.normal(440,10,1000),
                         'std': np.random.normal(115,10,1000)},
                     8: {'mu': np.random.normal(500,10,1000),
                         'std': np.random.normal(130,10,1000)},
                     }
        gc = np.random.choice(range(0,9),1)[0]
        qx = np.random.choice(np.linspace(.1,.2,20))
        cx = ag265 * qx
        comps = {'phis':[],
                 'mus':[],
                 'stds':[],
                 'gs':[]}
        
        for c in range(gc):
            c += 1
            if c == 1:
                comps['phis'].append(cx)
                comps['mus'].append(np.random.choice(comp_data[c]['mu']))
                comps['stds'].append(np.random.choice(comp_data[c]['std']))
            else:
                cx = cx*.8
                comps['phis'].append(cx)
                comps['mus'].append(np.random.choice(comp_data[c]['mu']))
                comps['stds'].append(np.random.choice(comp_data[c]['std']))
               
        def ag_gauss(ag0,s,l,l0,comps):
            ag = ag0 * np.exp(-s * (l-l0))
            gcomps = 0
            for i in range(len(comps['phis'])):
                # print (i)
                gs = comps['phis'][i] * np.exp(-(l-comps['mus'][i])**2 / (2*comps['stds'][i]**2))
                comps['gs'].append(gs)
                gcomps += gs
            return ag, gcomps, comps
        
        ag, gcomps, comps = ag_gauss(ag350, slope, l, 350,comps)
        agtot = ag + gcomps
                        
        import scipy as sp
        import scipy.optimize
        
        def exp(l, a0, s):
            return a0 * np.exp(-s * (l-350))
        
        def fit_exp(l, y):
            opt_parms, parm_cov = sp.optimize.curve_fit(exp, l, y, p0=[.001, .001])
            a0, s = opt_parms
            return a0, s 
        
        # slopes
        # 275-295
        li0 = np.where(l == 275)[0][0]
        li1 = np.where(l == 295)[0][0]
        a, s275_295 = fit_exp(l[li0:li1],agtot[li0:li1])        
        # S240-700
        li0 = np.where(l == 240)[0][0]
        li1 = np.where(l == 700)[0][0]
        a, s240_700 = fit_exp(l[li0:li1],agtot[li0:li1])
        # 300-700
        li0 = np.where(l == 300)[0][0]
        li1 = np.where(l == 700)[0][0]
        a, s300_700 = fit_exp(l[li0:li1],agtot[li0:li1])
        # 350-400
        li0 = np.where(l == 350)[0][0]
        li1 = np.where(l == 400)[0][0]
        a, s350_400 = fit_exp(l[li0:li1],agtot[li0:li1])
        # 350-550
        li0 = np.where(l == 350)[0][0]
        li1 = np.where(l == 550)[0][0]
        a, s350_550 = fit_exp(l[li0:li1],agtot[li0:li1])
        # 400-450
        li0 = np.where(l == 400)[0][0]
        li1 = np.where(l == 450)[0][0]
        a, s400_450 = fit_exp(l[li0:li1],agtot[li0:li1])
        # 400-700
        li0 = np.where(l == 400)[0][0]
        li1 = np.where(l == 700)[0][0]
        a, s400_700 = fit_exp(l[li0:li1],agtot[li0:li1])
        # slope ratio (Helms 2008)
        sr = s275_295 / s350_400
        
        if sr > .7:
            break
    
    cdomIOPs['a_tot'] = agtot
    cdomIOPs['ag440'] = ag440
    cdomIOPs['ag_exp'] = ag
    cdomIOPs['gaus_comps'] = comps
    cdomIOPs['tot_gcomps'] = gcomps
    cdomIOPs['num_gcomps'] = gc
    cdomIOPs['S275_295'] = s275_295
    cdomIOPs['S240_700'] = s240_700
    cdomIOPs['S300_700'] = s300_700
    cdomIOPs['S350_400'] = s350_400
    cdomIOPs['S350_550'] = s350_550
    cdomIOPs['S400_450'] = s400_450
    cdomIOPs['S400_700'] = s400_700
    cdomIOPs['slope_ratio'] = sr   
    return cdomIOPs

##
def col_name_append(name):
    if 'cdom' in name.split('_'):
        l = np.arange(240, 900, 2.5)
    else:
        l = np.arange(400, 902.5, 2.5)  
    l2 = [str(x) for x in l]
    l3 = ['{}'.format(name) + lx for lx in l2]  
    return l3

##
def dict_to_df (iops, k):
    tots = ['TotChl','TotPC','class_chl','class_frxn','class_Deff','class_sz_class',
            'Tot_conc','Tot_slope','S240_700', 
            'S275_295', 'S300_700', 'S350_400', 'S350_550',
            'S400_450', 'S400_700','slope_ratio','a_tot', 'b_tot', 'bb_tot',]
    
    classes = ['Green_algae', 'Cryptophytes','Diatoms','Dinoflagellates','Heterokonts',
               'Haptophytes','Cyano_blue','Cyano_red','Rhodophytes']
    row = []
    col_names = []
    
    # phyto totals
    classIOPs = iops['Phyto']
    for p in tots:
        if p in ['TotChl','TotPC']:
            d = classIOPs[p]
            row.append(d)
            col_names.append(p)
        elif p in ['a_tot', 'b_tot', 'bb_tot']:
            d = classIOPs[p]
            row.append(d)
            col_names.append(col_name_append('ph_{}_'.format(p)))
        else:
            continue
    
    # pft data
    for i, p in classIOPs.items():
        if i in classes:
            for ii in tots:
            # for ii, pp in p.items():
                if ii in ['class_chl','class_frxn','class_Deff','class_sz_class']:
                    row.append(p[ii])
                    col_names.append('{}_{}'.format(i,ii))
                elif ii in ['a_tot', 'b_tot', 'bb_tot']:
                    row.append(p[ii])
                    col_names.append(col_name_append('{}_{}_'.format(i,ii)))
                else:
                    continue
      
    # minerals
    minIOPs = iops['Min']
    for p in tots:
        if p in ['Tot_conc','Tot_slope']:
            d = minIOPs[p]
            row.append(d)
            col_names.append('min_{}'.format(p))
        elif p in ['a_tot', 'b_tot', 'bb_tot']:
            d = minIOPs[p]
            row.append(d)
            col_names.append(col_name_append('min_{}_'.format(p)))              
        else:
            continue
    
    # detritus
    detIOPs = iops['Det']
    for p in tots:
        if p in ['Tot_conc','Tot_slope']:
            d = detIOPs[p]
            row.append(d)
            col_names.append('det_{}'.format(p))
        elif p in ['a_tot', 'b_tot', 'bb_tot']:
            d = minIOPs[p]
            row.append(d)
            col_names.append(col_name_append('det_{}_'.format(p)))              
        else:
            continue
    
    # cdom
    cdomIOPs = iops['CDOM']
    for p in tots:
        if p in ['S240_700', 'S275_295', 'S300_700', 'S350_400', 'S350_550',
                 'S400_450', 'S400_700','slope_ratio']:
            d = cdomIOPs[p]
            row.append(d)
            col_names.append('cdom_{}'.format(p))
        elif p == 'a_tot':
            d = cdomIOPs[p]
            row.append(d)
            col_names.append(col_name_append('cdom_{}_'.format(p)))
    
    # benthic
    benIOPs = iops['Benthic']
    for i, k in benIOPs.items():
        if i == 'Tot':
            d = k.values
            row.append(d)
            col_names.append(col_name_append('benthic_tot_'))
        
        elif i in ['Algae','Coral','Sediment']:
            a = k['gfx']
            b = k['tot'].values
            row.append(a)
            row.append(b)
            col_names.append('benthic_{}_gfx'.format(i))
            col_names.append(col_name_append('benthic_{}_tot_'.format(i)))
            
            for ii, kk in k.items():
                if ii not in ['gfx','tot']:
                    a = kk['cfx']
                    b = kk['spec'].values
                    row.append(a)
                    row.append(b)
                    col_names.append('benthic_{}_cfx'.format(ii))
                    col_names.append(col_name_append('benthic_{}_'.format(ii)))
        
        else:
            a = k['gfx']
            b = k['spec'].values
            row.append(a)
            row.append(b)
            col_names.append('benthic_{}_cfx'.format(i))
            col_names.append(col_name_append('benthic_{}_'.format(i)))
        
    # adjacency
    adjIOPs = iops['Adjacency']
    for i,k in adjIOPs.items():
        if i == 'Tot':
            d = k.values
            row.append(d)
            col_names.append(col_name_append('adj_tot_'))
        else:
            d = k['gfx']
            row.append(d)
            col_names.append('adj_{}_gfx'.format(i))

    # atmosphere
    atm = iops['Atm']
    row.append(atm['aero'].iloc[0,3:].values)
    col_names.append(atm['aero'].add_prefix('atm_').iloc[0,3:].index.values)
    row.append(atm['atm_prof'])
    row.append(atm['VZA'])
    row.append(atm['VAA'])
    col_names.append(['prof','VZA','VAA'])


    col_names_final = np.hstack(col_names)
    final = np.hstack((row))    
    
    df = pd.Series(final, index=col_names_final)
    
    return df




