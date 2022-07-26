import numpy as np
import os, sys
from scipy.optimize import curve_fit
from tqdm import tqdm
import glob

#SQuIDS
sys.path.append('/data/user/chill/SQuIDS/lib/')
##nuSQuIDS libraries
sys.path.append('/data/user/chill/nuSQuIDS/resources/python/bindings/')
import nuSQUIDSpy as nsq
import nuSQUIDSTools

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from event_info import EventInfo

def translatePDGtoInfo(pdgList):
    flavList = (np.abs(pdgList) - 12) / 2
    nnbarList = (-1 * np.sign(pdgList) + 1) / 2
    return flavList, nnbarList

##open all nuSQuIDS cached states to create spline
def initPropFiles(flux_path, norm_list, f_type, selection):
    print(f'Opening nuSQuIDS files for {f_type}, {selection} (n = {len(norm_list)})')
    this_path = os.path.dirname(os.path.abspath(__file__))
    start_path = os.path.join(this_path, '../')
    glob_str = os.path.join(start_path, f'{flux_path}*_{f_type}_{selection}.hdf')
    fileList = sorted(glob.glob(glob_str))
    if len(fileList) == 0:
        raise IOError(f'glob could not file any files at {glob_str}')
    splineList, normList = createAllNSQ(fileList, norm_list)
    return splineList, normList

def createNSQ(filename):
    if os.path.exists(filename):
        bname = os.path.basename(filename)
        split = bname.split('_')
        norm = float(split[3])
        return nsq.nuSQUIDSAtm(filename), norm
    else:
        raise IOError(f'{filename} does not exist!')

def createAllNSQ(fileList, norm_list=None):
    nuSQList = []
    normList = []
    for filename in fileList:
        nuSQ, norm = createNSQ(filename)
        if norm == 5:
            #print(f'Skipping sigma_n = 5 for fitting!')
            continue
        if norm == 0.2:
            #print(f'Skipping sigma_n = 0.2 for fitting!')
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
    e_mask = e <= 1e14
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
        p0 = [-1, np.mean(propW)]
        popt, pcov = curve_fit(exp_func, normList, propW, p0=p0)
    else:
        raise ValueError(f'No valid fitFlag {fitFlag}!')
    return popt

def propWeightFit(eventInfo, splineList, normList, sType='none'):

    ##energy is in GeV
    ##make sure e is in eV!
    eList    = np.array(eventInfo.nu_energy) * 1e9
    coszList = np.cos(eventInfo.nu_zenith)
    if len(eList) == 0:
        print('<fitting> Size of eventInfo is 0, skipping')
        return eventInfo

    flavs, nnbars = translatePDGtoInfo(eventInfo.pdg)
    fitFlags = getFitFlag(eList, coszList)

    if sType == 'astro':
        fluxes = np.asarray(eventInfo.flux_astro)

    if sType == 'atmo':
        fluxes = np.asarray(eventInfo.flux_atmo)

    if sType == 'none':
        fluxes = np.asarray(eventInfo.flux_astro) + np.asarray(eventInfo.flux_atmo)

    fit_x      = []
    fit_y      = []
    fit_params = []

    k = 0
    for e, cosz, fitFlag, flav, nnbar, flux in tqdm(zip(eList, coszList, fitFlags, flavs, nnbars, fluxes)):
        propW = np.ones(len(splineList))
        for i, nuSQ in enumerate(splineList):
            w = nuSQ.EvalFlavor(int(flav), cosz, e, int(nnbar))
            propW[i] = w / flux
        popt = runFit(normList, propW, fitFlag) 

        fit_x.append(normList)
        fit_y.append(propW)
        fit_params.append(popt)

        k += 1

    if sType == 'none':
        eventInfo.fit_type   = fitFlags
        eventInfo.fit_x      = fit_x
        eventInfo.fit_y      = fit_y
        eventInfo.fit_params = fit_params
    if sType == 'astro':
        eventInfo.fit_type_astro  = fitFlags
        eventInfo.fit_x_astro     = fit_x
        eventInfo.fit_y_astro     = fit_y
        eventInfo.fit_params_astro = fit_params
    if sType == 'atmo':
        eventInfo.fit_type_atmo  = fitFlags
        eventInfo.fit_x_atmo     = fit_x
        eventInfo.fit_y_atmo     = fit_y
        eventInfo.fit_params_atmo = fit_params

    return eventInfo

##end
