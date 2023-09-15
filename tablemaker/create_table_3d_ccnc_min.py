##try to create a 3D fit over (w, norm, gamma)

##open all nuSQuIDS files and evaluate the points, extract the weights
import pandas as pd
from glob import glob
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
import click

import nuSQUIDSpy as nsq
import nuSQUIDSTools

from fluxes import DiffFlux, AtmFlux, InitAtmFlux
from fitting.fitting import translatePDGtoInfo
from configs import config

def getnuSQGridFiles(df, cache):
    #nsq_filepath =os.path.join(config.inner, 'propagation_grid/output_ccnc'
    nsq_filepath =os.path.join(config.inner, 'propagation_grid/output_minXsec')
    fluxType = 'astro'
    glob_str = f'{nsq_filepath}/nuSQuIDS_flux_cache_*_{fluxType}.hdf'
    files = sorted(glob(glob_str))
    if len(files) == 0:
        raise IOError(f'No file found at {files}')
    else:
        print(f'Found {len(files)} files!')
    ##because of gamma weighting in all files, tracks & cascades are the same!
    ##(with the exception of the normalisation value, but this is stored in the dataframe)
    eList = df.nu_energy.values
    cosList = np.cos(df.nu_zenith.values)
    flavList, nnbarList = translatePDGtoInfo(df.pdg.values)
    fluxList = df.flux_astro.values
    pdgList = df.pdg.values
    selectionList = df.Selection.values

    weightsList = np.zeros([len(files), len(eList)])
    ncList = np.zeros(len(files)*3)
    ccList = np.zeros(len(files)*3)
  
    fweightsList = np.zeros([len(files)*3, 500])
    c1weightsList = np.zeros([len(files), 60])
    c2weightsList = np.zeros([len(files), 60])
    c3weightsList = np.zeros([len(files), 60])
    erange = np.logspace(2, 8, 500)

    gamma = -2.45 
    i = 0

    ccnc_flux_cache = 'flux_cache_ccnc/ccnc_flux_cache_min.npy'
    normOrder = []
    for f in tqdm(files):
        bname = os.path.basename(f)
        split = bname.split('_')

        norm_cc = float(split[3])
        ccList[(i*3)+0] = norm_cc
        ccList[(i*3)+1] = norm_cc
        ccList[(i*3)+2] = norm_cc
        norm_nc = float(split[4])
        ncList[(i*3)+0] = norm_nc
        ncList[(i*3)+1] = norm_nc
        ncList[(i*3)+2] = norm_nc

        if cache == False:
            #nuSQ = nsq.nuSQUIDSAtm(f)
            #getWeightsGrid(weightsList, i, gamma, nuSQ, eList, cosList, flavList, nnbarList, pdgList, fluxList, selectionList)
            pass
        if not os.path.exists(ccnc_flux_cache):
            nuSQ = nsq.nuSQUIDSAtm(f)
            getFluxPlotInfo(fweightsList, c1weightsList, c2weightsList, c3weightsList, i, nuSQ, erange)

        i += 1

    ##sorting
    inds = np.lexsort((ncList, ccList))
    ccList = ccList[inds]
    ncList = ncList[inds]
    #weightsList = weightsList[inds]   
    fweightsList = fweightsList[inds]
    if not os.path.exists(ccnc_flux_cache):
        np.save(ccnc_flux_cache, fweightsList) 
        np.save('flux_cache_ccnc/ccnc_c1_min_cache.npy', c1weightsList)
        np.save('flux_cache_ccnc/ccnc_c2_min_cache.npy', c2weightsList)
        np.save('flux_cache_ccnc/ccnc_c3_min_cache.npy', c3weightsList) 
    if os.path.exists(ccnc_flux_cache):
        fweightsList = np.load(ccnc_flux_cache)
    makeFluxPlots(fweightsList, c1weightsList, c2weightsList, c3weightsList, ccList, ncList)

    return weightsList, ccList, ncList

def makeFluxPlots(fweightsList, c1weightsList, c2weightsList, c3weightsList, ccList, ncList):
    erange = np.logspace(2, 8, 500)
    ##each slice has the flux over the energy bins
    ##for some CC/NC weight
    fig_nc, ax_nc = plt.subplots()
    fig_cc, ax_cc = plt.subplots()
    fig_ncz, ax_ncz = plt.subplots()
    fig_ccz, ax_ccz = plt.subplots()
    fig_ncc, ax_ncc = plt.subplots()
    fig_ccc, ax_ccc = plt.subplots()
    fig_nccz, ax_nccz = plt.subplots()
    fig_cccz, ax_cccz = plt.subplots()

    fig_cce, ax_cce = plt.subplots()
    fig_nce, ax_nce = plt.subplots()

    ##mix cc and nc to compare
    fig_m, ax_m = plt.subplots()
    fig_zero, ax_zero = plt.subplots()

    cValues = np.zeros(len(fweightsList[0]))
    for i, flux in enumerate(fweightsList):    
        ccVal = ccList[i]
        ncVal = ncList[i]
        if ncVal == 1.0 and ccVal == 1.0:
            cValues = flux
            ax_m.plot(erange, cValues/cValues, color='black', label='CC: 1.0, NC: 1.0')
            break
    for i, flux in enumerate(fweightsList):
        ccVal = ccList[i]
        ncVal = ncList[i]
        mask = erange > 1e5

        if ncVal == 1.0:
            if ccVal == 1.0:
                ax_cc.plot(erange, flux, '--', label=f'{ccVal}')
                ax_ccz.plot(erange[mask], flux[mask], '--', label=f'{ccVal}')
                ax_cce.plot(erange, flux*(erange**2.45), '--', label=f'{ccVal}')
                print(f'CC:{ccVal}, NC:{ncVal}: {np.sum(flux)}')
            if ccVal != 1.0:
                ax_cc.plot(erange, flux, label=f'{ccVal}')
                ax_ccz.plot(erange[mask], flux[mask], label=f'{ccVal}')
                ax_cce.plot(erange, flux*(erange**2.45), label=f'{ccVal}')
                print(f'CC:{ccVal}, NC:{ncVal}: {np.sum(flux)}')
                

            ax_ccc.plot(erange, flux/cValues, label=f'{ccVal}')
            ax_cccz.plot(erange[mask], flux[mask]/cValues[mask], label=f'{ccVal}')
            
            if ccVal in [0.5, 2.0]:
                ax_m.plot(erange, flux/cValues, label=f'{ccVal}, {ncVal}')


        if ccVal == 1.0:
            if ncVal == 1.0:
                ax_nc.plot(erange, flux, '--', label=f'{ncVal}')
                ax_ncz.plot(erange[mask], flux[mask], '--', label=f'{ncVal}')
                ax_nce.plot(erange, flux*(erange**2.45), '--', label=f'{ncVal}')
                print(f'CC:{ccVal}, NC:{ncVal}: {np.sum(flux)}')
            if ncVal != 1.0:
                ax_nc.plot(erange, flux, label=f'{ncVal}')
                ax_ncz.plot(erange[mask], flux[mask], label=f'{ncVal}')
                ax_nce.plot(erange, flux*(erange**2.45), label=f'{ncVal}')
                print(f'CC:{ccVal}, NC:{ncVal}: {np.sum(flux)}')
            
            ax_ncc.plot(erange, flux/cValues, label=f'{ncVal}')
            ax_nccz.plot(erange[mask], flux[mask]/cValues[mask], label=f'{ncVal}')
            
            if ncVal in [0.5, 2.0, 1e-6]:
                ax_m.plot(erange, flux/cValues, '--', label=f'{ccVal}, {ncVal}')

        if ccVal in [0.5, 2.0, 1e-6]:
            if ncVal in [0.5, 2.0, 1e-6]:
                ax_m.plot(erange, flux/cValues, '-.', label=f'{ccVal}, {ncVal}')
                print(f'CC:{ccVal}, NC:{ncVal}: {np.sum(flux)}')

                ax_zero.plot(erange, flux*(erange**2.45))
                #ax_zero.plot(erange, [1.0]*len(erange), '--')

    fig_zero.savefig('plots3d/prop_flux_zero_min.pdf')

    ax_nc.set_xlabel('Energy [GeV]')
    ax_nc.set_ylabel('Flux')
    ax_nc.set_xscale('log')
    ax_nc.set_yscale('log')
    ax_nc.legend()
    ax_nc.set_title(f'NC Range: [{np.min(ncList)}, {np.max(ncList)}]')
    fig_nc.tight_layout()
    fig_nc.savefig('plots3d/prop_flux_nc_min.pdf')    
    plt.close(fig_nc)

    ax_cc.set_xlabel('Energy [GeV]')
    ax_cc.set_ylabel('Flux')
    ax_cc.set_xscale('log')
    ax_cc.set_yscale('log')
    ax_cc.legend()
    ax_cc.set_title(f'CC Range: [{np.min(ccList)}, {np.max(ccList)}]')
    fig_cc.tight_layout()
    fig_cc.savefig('plots3d/prop_flux_cc_min.pdf')    
    plt.close(fig_cc)    
    
    ax_cce.set_xlabel('Energy [GeV]')
    ax_cce.set_ylabel(r'Flux x E^$\gamma$')
    ax_cce.set_xscale('log')
    ax_cce.set_yscale('log')
    ax_cce.legend()
    ax_cce.set_title(f'CC Range: [{np.min(ccList)}, {np.max(ccList)}]')
    fig_cce.tight_layout()
    fig_cce.savefig('plots3d/prop_flux_cc_e_min.pdf')
    ax_cce.set_ylim(1e-8, 1e-5)
    fig_cce.savefig('plots3d/prop_flux_cc_e_min_zoom.pdf')
    plt.close(fig_cce) 
    
    ax_nce.set_xlabel('Energy [GeV]')
    ax_nce.set_ylabel(r'Flux x E^$\gamma$')
    ax_nce.set_xscale('log')
    ax_nce.set_yscale('log')
    ax_nce.legend()
    ax_nce.set_title(f'NC Range: [{np.min(ncList)}, {np.max(ncList)}]')
    fig_nce.tight_layout()
    fig_nce.savefig('plots3d/prop_flux_nc_e_min.pdf')    
    ax_nce.set_ylim(1e-8, 1e-5)
    fig_nce.savefig('plots3d/prop_flux_nc_e_min_zoom.pdf')
    plt.close(fig_nce) 

    ax_ncz.set_xlabel('Energy [GeV]')
    ax_ncz.set_ylabel('Flux')
    ax_ncz.set_xscale('log')
    ax_ncz.set_yscale('log')
    ax_ncz.legend()
    ax_ncz.set_title(f'NC Range: [{np.min(ncList)}, {np.max(ncList)}]')
    fig_ncz.tight_layout()
    fig_ncz.savefig('plots3d/prop_flux_nc_min_zoom.pdf')    
    plt.close(fig_ncz)

    ax_ccz.set_xlabel('Energy [GeV]')
    ax_ccz.set_ylabel('Flux')
    ax_ccz.set_xscale('log')
    ax_ccz.set_yscale('log')
    ax_ccz.legend()
    ax_ccz.set_title(f'CC Range: [{np.min(ccList)}, {np.max(ccList)}]')
    fig_ccz.tight_layout()
    fig_ccz.savefig('plots3d/prop_flux_cc_min_zoom.pdf')    
    plt.close(fig_ccz)
    
    ax_ncc.set_xlabel('Energy [GeV]')
    ax_ncc.set_ylabel('Flux / Flux | CC=1.0')
    ax_ncc.set_xscale('log')
    ax_ncc.set_yscale('log')
    ax_ncc.legend(loc=2)
    ax_ncc.set_title(f'NC Range: [{np.min(ncList)}, {np.max(ncList)}]')
    fig_ncc.tight_layout()
    fig_ncc.savefig('plots3d/prop_norm_flux_nc_min.pdf')    
    plt.close(fig_ncc)

    ax_ccc.set_xlabel('Energy [GeV]')
    ax_ccc.set_ylabel('Flux / Flux | NC=1.0')
    ax_ccc.set_xscale('log')
    ax_ccc.set_yscale('log')
    ax_ccc.legend(loc=2)
    ax_ccc.set_title(f'CC Range: [{np.min(ccList)}, {np.max(ccList)}]')
    fig_ccc.tight_layout()
    fig_ccc.savefig('plots3d/prop_norm_flux_cc_min.pdf')    
    plt.close(fig_ccc)
    
    ax_nccz.set_xlabel('Energy [GeV]')
    ax_nccz.set_ylabel('Flux / Flux | CC=1.0')
    ax_nccz.set_xscale('log')
    ax_nccz.set_yscale('log')
    ax_nccz.legend(loc=2)
    ax_nccz.set_title(f'NC Range: [{np.min(ncList)}, {np.max(ncList)}]')
    fig_nccz.tight_layout()
    fig_nccz.savefig('plots3d/prop_norm_flux_nc_min_zoom.pdf')    
    plt.close(fig_nccz)

    ax_cccz.set_xlabel('Energy [GeV]')
    ax_cccz.set_ylabel('Flux / Flux | NC=1.0')
    ax_cccz.set_xscale('log')
    ax_cccz.set_yscale('log')
    ax_cccz.legend(loc=2)
    ax_cccz.set_title(f'CC Range: [{np.min(ccList)}, {np.max(ccList)}]')
    fig_cccz.tight_layout()
    fig_cccz.savefig('plots3d/prop_norm_flux_cc_min_zoom.pdf')    
    plt.close(fig_cccz)

    ax_m.set_xlabel('Energy [GeV]')
    ax_m.set_ylabel('Flux / Flux | CC=1.0, NC=1.0')
    ax_m.set_xscale('log')
    ax_m.set_yscale('log')
    ax_m.legend(loc=2)
    ax_m.set_title(f'CC & NC Variations')
    fig_m.tight_layout()
    fig_m.savefig('plots3d/prop_norm_flux_mix_min.pdf')    
    plt.close(fig_m)
    
    for i, c in enumerate(np.linspace(-1, 1, 60)):
        w = c1weightsList[i]

    print("plotting then exiting early")
    exit(1)

def getFluxPlotInfo(fweightsList, c1weightsList, c2weightsList, c3weightsList, i, nuSQ, erange):

    ##evaluate the spline for a given E range
    ##fix the flav, cos, and nnbar
    ##REMEMBER NUSQUIDS GOES FROM 0, 1, 2 for FLAVOUR
    flav = 0
    nnbar = 0
    cos = -0.99
    j = 0
    #for e in erange:
    #    prop_f = nuSQ.EvalFlavor(int(flav), cos, e*1e9, int(nnbar))
    #    fweightsList[i][j] = prop_f
    #    j += 1
    
    for e in erange:
        for k in [0, 1, 2]:
            prop_f = nuSQ.EvalFlavor(int(k), cos, e*1e9, int(nnbar))
            fweightsList[(i*3)+(k)][j] = prop_f
        j += 1

    j = 0
    for c in np.linspace(-1, 1, 60):
        prop_f1 = nuSQ.EvalFlavor(int(flav), c, 1e3*1e9, int(nnbar))
        prop_f2 = nuSQ.EvalFlavor(int(flav), c, 1e5*1e9, int(nnbar))
        prop_f3 = nuSQ.EvalFlavor(int(flav), c, 1e7*1e9, int(nnbar))
        c1weightsList[i][j] = prop_f1
        c2weightsList[i][j] = prop_f2
        c3weightsList[i][j] = prop_f3
        j += 1

##for now just doing astro
def getWeightsGrid(weightsList, i, gamma, nuSQ, eList, cosList, flavList, nnbarList, pdgList, fluxList, selectionList):
    j = 0
    #wList = np.zeros(len(eList))
    for e, cos, flav, nnbar, pdg, flux, selection in zip(eList, cosList, flavList, nnbarList, pdgList, fluxList, selectionList):
        prop_w = nuSQ.EvalFlavor(int(flav), cos, e*1e9, int(nnbar))
        
        flux = DiffFlux(e, selection, gamma)
        ##need to fix the normalisation based on selection
        ##in condor jobs norm is 1.36e-18
        if selection == 'track':
            flux = (1.36 / 1.44) * flux
        elif selection == 'cascade':
            flux = (1.36 / 1.66) * flux 
        else:
            raise ValueError(f'Selection {selection} is not valid!')
        
        w = prop_w / flux
        weightsList[i][j] = w
        j += 1

def plot2DSpace(ccList, ncList):
    fig, ax = plt.subplots()
    ax.plot(ccList, ncList, 'o', color='royalblue', label='Simulated Points')
    ax.legend()
    ax.set_xlabel('CC Cross Section Norm')
    ax.set_ylabel('NC Cross Section Norm')
    fig.savefig('plots3d/sample_area_ccnc.pdf')
    plt.close(fig)

def make3DMesh(df, ccList, ncList, weightCollection):
    eList = df.nu_energy.values
    cosList = np.cos(df.nu_zenith.values)
    pdg = df.pdg.values
    i = 0

    inds = np.where(weightCollection == 0)
    weightCollection[inds] = 1

    print('ccList, ncList:')
    print(np.unique(ccList), np.unique(ncList))
    for j in range(weightCollection.shape[1]):
        
        if i >= 30 and eList[j] <= 8e6:
            continue

        ##get all weights for a given event
        ##across all norm/gamma pairs!
        weights = weightCollection[:, j]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        ax.scatter(ccList, ncList, weights)
        ax.set_xlabel('CC Cross Section Norm.')
        ax.set_ylabel('NC Cross Section Norm.')
        ax.set_zlabel('Prop. W')
        ax.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f} (PDG={pdg[j]})')
        #ax.view_init(20, 180)
        fig.tight_layout()
        fig.savefig(f'plots3d/map_ccnc_min_{i}_astro.pdf')

        weights = np.reshape(weights, (len(np.unique(ccList)), len(np.unique(ncList))))
        weights_t = weights.T

        spline = RectBivariateSpline(np.unique(ncList), np.unique(ccList), weights_t)
        evalNC = np.linspace(np.min(ncList), np.max(ncList), 100)
        evalCC  = np.linspace(np.min(ccList), np.max(ccList), 100)
        splineVals = spline(evalNC, evalCC)
        meshN, meshG = np.meshgrid(evalCC, evalNC)
        ax.plot_wireframe(meshN, meshG, splineVals, color='goldenrod')
        fig.tight_layout()
        fig.savefig(f'plots3d/map_ccnc_min_spline_{i}_astro.pdf')
        plt.close(fig)

        fig1, ax1 = plt.subplots()
        plot = ax1.pcolormesh(np.unique(ccList), np.unique(ncList), weights_t, cmap='viridis', shading='auto')
        cbar = fig.colorbar(plot, ax=ax1)
        cbar.set_label(r'Propagation Weight', labelpad=18, rotation=270)
        ax1.set_xlabel('CC Cross Section Norm.')
        ax1.set_ylabel('NC Cross Section Norm.')
        ax1.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f}')
        fig1.tight_layout()
        fig1.savefig(f'plots3d/map_ccnc_min_mesh_{i}_astro.pdf')
        plt.close(fig1)

        def_cycler = (
            cycler(marker=['o', 'v', '^', '<', '>']) *
            cycler(color=mcolors.to_rgba_array(
                       ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']))
        )

        ##marginalise over gamma to get projection
        ##i.e. w vs norm for different gammas
        fig2, ax2 = plt.subplots()
        for k, g in enumerate(np.unique(ncList)):
            w = weights[:, k]
            ax2.plot(np.unique(ccList), w, 'o', label=g)

        ax2.set_prop_cycle(def_cycler)
        ax2.legend(title='NC')
        ax2.set_xlabel('CC Cross Section Norm.')
        ax2.set_ylabel('Prop. W')
        ax2.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f} (PDG={pdg[j]})')
        ax2.grid()
        fig2.tight_layout()
        fig2.savefig(f'plots3d/weight_vs_cc_nc_min_{i}_astro.pdf')
        plt.close(fig2)
        
        ##marginalise over norm to get projection
        ##i.e. w vs gamma for different norms
        fig3, ax3 = plt.subplots()
        for k, n in enumerate(np.unique(ccList)):
            w = weights[k]
            ax3.plot(np.unique(ncList), w, 'o', label=n)

        ax3.set_prop_cycle(def_cycler)
        ax3.legend(title='CC')
        ax3.set_xlabel('NC Cross Section Norm.')
        ax3.set_ylabel('Prop. W')
        ax3.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f}')
        ax3.grid()
        fig3.tight_layout()
        fig3.savefig(f'plots3d/weight_vs_nc_cc_min_{i}_astro.pdf')
        plt.close(fig3)

        i += 1    

def main():
    fpath = config.dataframes_dir
    cache_file = 'cached2D_astro_ccnc_min.npy'

    cache = False
    if os.path.exists(cache_file):
        cache = True
    df = pd.read_hdf(os.path.join(fpath, 'li_total.hdf5'))
    
    ##use new cached files
    weightCollection, ccList, ncList  = getnuSQGridFiles(df, cache)
    if os.path.exists(cache_file) == False:
        np.save(cache_file, weightCollection)  
    else:
        print('Loading cache!')
        weightCollection = np.load(cache_file)

    plot2DSpace(ccList, ncList)
    make3DMesh(df, ccList, ncList, weightCollection)

if __name__ == "__main__":
    main()
##end
