import numpy as np
import os, sys
from scipy.optimize import curve_fit
from tqdm import tqdm
import glob

import nuSQUIDSpy as nsq
import nuSQUIDSTools

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../weighting/'))
from event_info import EventInfo
from configs.config import config

def translatePDGtoInfo(pdgList):
    flavList = (np.abs(pdgList) - 12) / 2
    nnbarList = (-1 * np.sign(pdgList) + 1) / 2
    return flavList, nnbarList

##open all nuSQuIDS cached states to create spline
def initPropFiles(flux_path, f_type, selection, earth):
    norm_list = config.normList
    print(f'Opening nuSQuIDS files for {f_type}, {selection} (n_norms = {len(norm_list)})')
    #this_path = os.path.dirname(os.path.abspath(__file__))
    #start_path = os.path.join(this_path, '../')
    #glob_str = os.path.join(start_path, f'*_{f_type}_{selection}.hdf')
    if earth == 'normal':
        #f_str = f'nuSQuIDS_flux_cache_*_{f_type}_{selection}.hdf'
        f_str = f'nuSQuIDS_flux_cache_*_toleranceUp_-2.37_{f_type}.hdf'
    elif earth == 'up' or earth == 'down':
        f_str = f'nuSQuIDS_flux_cache_*_{earth}_{f_type}_{selection}.hdf'    
    glob_str = os.path.join(flux_path, f_str)
    fileList = sorted(glob.glob(glob_str))
    if len(fileList) == 0:
        raise IOError(f'glob could not find any files at {glob_str}')
    splineList, normList = createAllNSQ(fileList, norm_list)
    return splineList, normList

def createNSQ(filename, norm_list=None):
    if os.path.exists(filename):
        bname = os.path.basename(filename)
        split = bname.split('_')
        norm = float(split[3])
        if norm_list != None:
            if norm not in norm_list:
                return None, norm
        ##if norm is in the list, or no list given
        print(f'Opened: {filename}')
        return nsq.nuSQUIDSAtm(filename), norm
    else:
        raise IOError(f'{filename} does not exist!')

def createAllNSQ(fileList, norm_list=[None]):
    nuSQList = []
    normList = []

    int_norm_list = []
    if norm_list[0] != None:
        for _n in norm_list:
            int_norm_list.append(float(_n))

    for filename in fileList:
        nuSQ, norm = createNSQ(filename, int_norm_list)
        if norm == 5:
            print(f'Skipping sigma_n = 5 for fitting!')
            continue
        if norm == 0.2:
            print(f'Skipping sigma_n = 0.2 for fitting!')
            continue
        if norm < 0:
            continue

        if normList != None:
            bname = os.path.basename(filename)
            split = bname.split('_')
            norm = float(split[3])
            if norm not in norm_list:
                continue

        nuSQList.append(nuSQ)
        normList.append(norm)

    args = np.argsort(normList)
    normList = np.sort(normList)
    nuSQList = np.array(nuSQList)[args]
   
    if len(nuSQList) == 0 or len(normList) == 0:
        raise IOError(f'Could not locate nuSQuIDS files at {fileList}')

    return nuSQList, normList

###fitting function for propWeight
def linear_func(x, a, b):
    return a * x + b

def exp_func(x, exponent, norm):
    val = norm * np.power(x, exponent)
    return val

def getFitFlag(e, cosz):
    fitFlags = ['exp'] * len(e)
    e_mask = e <= 2e13
    z_mask = cosz >= 0
    mask = np.logical_or(e_mask, z_mask)
    fitFlags = np.asarray(fitFlags)
    fitFlags[mask] = 'linear'
    return fitFlags

def runFit(normList, propW, fitFlag):
    if fitFlag == 'linear' or fitFlag == 'lin':
        p0 = [-1, np.mean(propW)]
        popt, pcov = curve_fit(linear_func, normList, propW, p0=p0)
    elif fitFlag == 'exp':
        p0 = [-10, np.mean(propW)]
        try:
            popt, pcov = curve_fit(exp_func, normList, propW, p0=p0)
        except:
            try:
                print('Trying to fit with steeper slope')
                p0[0] = -150
                popt, pcov = curve_fit(exp_func, normList, propW, p0=p0)
            except:
                try:
                    print('Trying other method')
                    p0[1] = propW[-1]
                    popt, pcov = curve_fit(exp_func, normList, propW, p0=p0)
                except:
                    print(f'Error in curve_fit! Debug fitting with {fitFlag}:')
                    print(f'norm list: {normList}')
                    print(f'prop weight: {propW}')
                    print(f'p0: {p0}')
                    exit(1)
    else:
        raise ValueError(f'No valid fitFlag {fitFlag}!')
    return popt

def propWeightFit(eventInfo, splineList, normList, f_type):

    ##energy is in GeV
    ##make sure e is in eV!
    eList    = np.array(eventInfo.nu_energy) * 1e9
    coszList = np.cos(eventInfo.nu_zenith)
    if len(eList) == 0:
        print('<fitting> Size of eventInfo is 0, skipping')
        return eventInfo

    ##check which events are GR
    ccnc  = eventInfo.ccnc
    gr_tags = np.array(ccnc) == 'GR'

    ##make the flavours and n/n-bar friendly to nuSQuIDS
    flavs, nnbars = translatePDGtoInfo(eventInfo.pdg)

    ##pre-determine which fit should be used based on e and cos(z)
    fitFlags = getFitFlag(eList, coszList)

    ##grab pre-calculated baseline flux from the eventInfo storage
    if f_type == 'astro':
        fluxes = np.asarray(eventInfo.flux_astro)
    if f_type == 'atmo':
        fluxes = np.asarray(eventInfo.flux_atmo)

    fit_x      = []
    fit_y      = []
    fit_params = []

    ##Loop over each event
    for e, cosz, fitFlag, flav, nnbar, flux, is_gr in zip(eList, coszList, fitFlags, 
            flavs, nnbars, fluxes, gr_tags):
        
        ##splineList and normList are the same length
        ##splineList is the group of nuSQuIDS flux splines
        ##normList is the cross section normalisation list
        propW = np.ones(len(splineList))
        
        ##Loop over each normalistaion
        for i, nuSQ in enumerate(splineList):
            
            ##new_flux is the flux at the detector assuming 
            ##some cross section normalisation
            new_flux = nuSQ.EvalFlavor(int(flav), cosz, e, int(nnbar))
            
            ##divide out nominal flux contribution (which is the flux at the detector)
            ##multiply in the new flux contribution (also at detector)
            ##scale interaction probability in the detector (linear)
            ##except for GR events (cross section at detetor is the same!)
            if is_gr == True:
                propW[i] = (1 / flux) * new_flux * 1.0
            else:
                propW[i] = (1/ flux) * new_flux * normList[i]
        
        ##Fit across the new scale factor for each normalisation
        popt = runFit(normList, propW, fitFlag) 

        fit_x.append(normList)
        fit_y.append(propW)
        fit_params.append(popt)

    setattr(eventInfo, f'fit_type_{f_type}', fitFlags)
    setattr(eventInfo, f'fit_x_{f_type}', fit_x)
    setattr(eventInfo, f'fit_y_{f_type}', fit_y)
    setattr(eventInfo, f'fit_params_{f_type}', fit_params)

    return eventInfo

##end
