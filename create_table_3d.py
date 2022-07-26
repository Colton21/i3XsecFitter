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

#SQuIDS
sys.path.append('/data/user/chill/SQuIDS/lib/')
##nuSQuIDS libraries
sys.path.append('/data/user/chill/nuSQuIDS/resources/python/bindings/')
import nuSQUIDSpy as nsq
import nuSQUIDSTools

from fluxes import DiffFlux, AtmFlux, InitAtmFlux
from fitting.fitting import translatePDGtoInfo


def getnuSQGridFiles(df, cache):
    nsq_filepath ='/data/user/chill/icetray_LWCompatible/propagation_grid/output'
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
    gammaList = np.zeros(len(files))
    normList = np.zeros(len(files))
   
    i = 0
    m = 0 
    ebins = 20
    check_weights = np.zeros((6, ebins, 76))
    normOrder = []
    for f in tqdm(files):
        bname = os.path.basename(f)
        split = bname.split('_')

        norm = float(split[3])
        gamma = float(split[4])
        gammaList[i] = gamma
        normList[i]  = norm

        if cache == False:
            nuSQ = nsq.nuSQUIDSAtm(f)
            getWeightsGrid(weightsList, i, gamma, nuSQ, eList, cosList, flavList, nnbarList, pdgList, fluxList, selectionList)

        if gamma == -2.6:
            nuSQ = nsq.nuSQUIDSAtm(f)
            specificCheck(check_weights, nuSQ, m, ebins)
            normOrder.append(norm)
            m += 1
        i += 1


    specificPlots(normOrder, check_weights, ebins) 

    inds = np.lexsort((gammaList, normList))
    normList  = normList[inds]
    gammaList = gammaList[inds]
    weightsList = weightsList[inds]

    print(weightsList)
    print(np.sum(weightsList, axis=1))

    return weightsList, normList, gammaList

def specificPlots(normOrder, check_weights, ebins):
    e_range = np.linspace(5e15, 1e17, ebins)
    tag = ['nue', 'nuebar', 'numu', 'numubar', 'nutau', 'nutaubar']
    for flav in range(6):
        fig, ax = plt.subplots()
        for i in range(10):
            e = e_range[i]/1e9
            p_w = check_weights[flav][i]/np.max(check_weights[flav][i])
            ax.plot(normOrder, p_w, 'o', label=f'{e:.0f}')

        ax.set_xlabel('Cross Section Norm.')
        ax.set_ylabel('Prop. W')
        ax.grid()
        ax.legend(title=r'$E_{\nu}$ [GeV]')
        fig.tight_layout()
        fig.savefig(f'plots3d/energy_check_{tag[flav]}.pdf')
    exit(1)

def specificCheck(weights, nuSQ, m, ebins):
    for i, flav in enumerate([0, 1, 2]):
        for j, nnbar in enumerate([0, 1]):
            for k, e in enumerate(np.linspace(1e16, 1e17, ebins)):
                prop_w = nuSQ.EvalFlavor(flav, -0.3, e, nnbar)
                weights[2*i+j][k][m] = prop_w

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

def getnuSQFiles(cache):
    nsq_filepath ='/data/user/chill/icetray_LWCompatible/nuSQuIDS_propagation_files'
    gammaList = []
    normList = []
    nuSQList = []
    selectionList = []
    fluxType = 'astro'
    for selection in ['cascade', 'track']:
        glob_str = f'{nsq_filepath}/nuSQuIDS_flux_cache_*_{fluxType}_{selection}.hdf'
        files = sorted(glob(glob_str))
        if len(files) == 0:
            raise IOError(f'No file found at {files}')
        for f in files:
            bname = os.path.basename(f)
            split = bname.split('_')

            if float(split[3]) > 0:
                norm = float(split[3])
                if norm == 0.2 or norm == 5.0:
                    continue
                if len(split) == 7:
                    gamma = float(split[4])
                elif len(split) == 6:
                    if selection == 'cascade':
                        gamma = -2.53
                    if selection == 'track':
                        gamma == -2.28

            if float(split[3]) < 0:
                gamma = float(split[3])
                norm = 1.0

            if cache == False:
                nuSQ = nsq.nuSQUIDSAtm(f)
            else:
                nuSQ = None

            gammaList.append(gamma)
            normList.append(norm)
            nuSQList.append(nuSQ)
            selectionList.append(selection)

    return nuSQList, normList, gammaList, selectionList


def getWeights(df, nuSQList, gammaList):
    eList = df.nu_energy.values
    cosList = np.cos(df.nu_zenith.values)
    flavList, nnbarList = translatePDGtoInfo(df.pdg.values)
    selectionList = df.Selection.values
    weightCollection = []
    fluxList = df.flux_astro.values
    for e, cos, selection, flav, nnbar, pdg, flux in zip(eList, cosList, selectionList, flavList, nnbarList, df.pdg.values, fluxList):
        wList = np.zeros(len(nuSQList))
        j = 0
        ##need to fix the normalisation based on selection
        ##in condor jobs norm is 1.36e-18
        if selection == 'track':
            flux = (1.36 / 1.44) * flux
        elif selection == 'cascade':
            flux = (1.36 / 1.66) * flux 
        else:
            raise ValueError(f'Selection {selection} is not valid!')
        for nuSQ, gamma in zip(nuSQList, gammaList):
            prop_w = nuSQ.EvalFlavor(int(flav), cos, e*1e9, int(nnbar))
            #fluxVal = DiffFlux(e, selection, gamma)
            w = prop_w / flux
            wList[j] = w
            j += 1
        weightCollection.append(wList)
    return weightCollection

def plot2DSpace(normList, gammaList):
    fig, ax = plt.subplots()
    ax.plot(normList, gammaList, 'o', color='royalblue', label='Simulated Points')
    ax.legend()
    ax.set_xlabel('Cross Section Norm')
    ax.set_ylabel(r'$\gamma_{Astro}$')
    fig.savefig('plots3d/sample_area.pdf')
    plt.close(fig)

def make3DMesh(df, normList, gammaList, weightCollection):
    eList = df.nu_energy.values
    cosList = np.cos(df.nu_zenith.values)
    pdg = df.pdg.values
    i = 0

    inds = np.where(weightCollection == 0)
    weightCollection[inds] = 1

    print(np.unique(normList), np.unique(gammaList))
    for j in range(weightCollection.shape[1]):
        
        if i >= 30 and eList[j] <= 5e6:
            continue

        ##get all weights for a given event
        ##across all norm/gamma pairs!
        weights = weightCollection[:, j]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        ax.scatter(normList, gammaList, weights)
        ax.set_xlabel('Cross Section Norm.')
        ax.set_ylabel(r'$\gamma_{Astro}$')
        ax.set_zlabel('Prop. W')
        ax.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f} (PDG={pdg[j]})')
        #ax.view_init(20, 180)
        fig.tight_layout()
        fig.savefig(f'plots3d/map_{i}_astro.pdf')

        weights = np.reshape(weights, (len(np.unique(normList)), len(np.unique(gammaList))))
        weights_t = weights.T

        spline = RectBivariateSpline(np.unique(gammaList), np.unique(normList), weights_t)
        evalGamma = np.linspace(np.min(gammaList), np.max(gammaList), 100)
        evalNorm  = np.linspace(np.min(normList), np.max(normList), 100)
        splineVals = spline(evalGamma, evalNorm)
        meshN, meshG = np.meshgrid(evalNorm, evalGamma)
        ax.plot_wireframe(meshN, meshG, splineVals, color='goldenrod')
        fig.tight_layout()
        fig.savefig(f'plots3d/map_spline_{i}_astro.pdf')
        plt.close(fig)


        print(weights.shape, np.sum(weights))

        fig1, ax1 = plt.subplots()
        plot = ax1.pcolormesh(np.unique(normList), np.unique(gammaList), weights_t, cmap='viridis', shading='auto')
        cbar = fig.colorbar(plot, ax=ax1)
        cbar.set_label(r'Propagation Weight', labelpad=18, rotation=270)
        ax1.set_xlabel('Cross Section Norm.')
        ax1.set_ylabel(r'$\gamma_{Astro}$')
        ax1.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f}')
        fig1.tight_layout()
        fig1.savefig(f'plots3d/map_mesh_{i}_astro.pdf')
        plt.close(fig1)

        def_cycler = (
            cycler(marker=['o', 'v', '^', '<', '>']) *
            cycler(color=mcolors.to_rgba_array(
                       ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']))
        )

        ##marginalise over gamma to get projection
        ##i.e. w vs norm for different gammas
        fig2, ax2 = plt.subplots()
        for k, g in enumerate(np.unique(gammaList)):
            w = weights[:, k]
            ax2.plot(np.unique(normList), w, 'o', label=g)

        ax2.set_prop_cycle(def_cycler)
        ax2.legend(title=r'$\gamma_{Astro}$')
        ax2.set_xlabel('Cross Section Norm.')
        ax2.set_ylabel('Prop. W')
        ax2.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f} (PDG={pdg[j]})')
        ax2.grid()
        fig2.tight_layout()
        fig2.savefig(f'plots3d/weight_vs_norm_gammas_{i}_astro.pdf')
        plt.close(fig2)
        
        ##marginalise over norm to get projection
        ##i.e. w vs gamma for different norms
        fig3, ax3 = plt.subplots()
        for k, n in enumerate(np.unique(normList)):
            w = weights[k]
            ax3.plot(np.unique(gammaList), w, 'o', label=n)

        ax3.set_prop_cycle(def_cycler)
        ax3.legend(title='Cross Section Norm.')
        ax3.set_xlabel(r'$\gamma_{Astro}$')
        ax3.set_ylabel('Prop. W')
        ax3.set_title(f'E={eList[j]:.0f} GeV, '+ r'cos($\theta$' + f')={cosList[j]:.2f}')
        ax3.grid()
        fig3.tight_layout()
        fig3.savefig(f'plots3d/weight_vs_gamma_norms_{i}_astro.pdf')
        plt.close(fig3)

        i += 1    

def main():
    fpath = '/data/user/chill/icetray_LWCompatible/dataframes/'
    cache_file = 'cached2D_astro.npy'

    cache = False
    if os.path.exists(cache_file):
        cache = True
    df = pd.read_hdf(os.path.join(fpath, 'li_total.hdf5'))
    
    ##memory runs out using old method and new files...
    ##use old cached files
    #nuSQList, normList, gammaList, selectionList = getnuSQFiles(cache)
    #if os.path.exists(cache_file) == False:
    #    weightCollection = getWeights(df, nuSQList, gammaList)
    #    np.save(cache_file, weightCollection)
    #else:
    #    print('Loading cache!')
    #    weightCollection = np.load(cache_file)
    
    ##use new cached files
    weightCollection, normList, gammaList  = getnuSQGridFiles(df, cache)
    if os.path.exists(cache_file) == False:
        np.save(cache_file, weightCollection)  
    else:
        print('Loading cache!')
        weightCollection = np.load(cache_file)

    plot2DSpace(normList, gammaList)
    make3DMesh(df, normList, gammaList, weightCollection)

if __name__ == "__main__":
    main()
##end
