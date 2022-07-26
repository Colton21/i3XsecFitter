import sys, os
import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, wait

##local libraries
from fluxes import InitAtmFlux, AtmFlux, DiffFlux

#SQuIDS
sys.path.append('/data/user/chill/SQuIDS/lib/')
##nuSQuIDS libraries
sys.path.append('/data/user/chill/nuSQuIDS/resources/python/bindings/')
import nuSQUIDSpy as nsq
import nuSQUIDSTools

def set_energy():
    units = nsq.Const()
    E_min = 100.0*units.GeV
    E_max = 100.0*units.PeV
    E_nodes = 500
    energy_nodes = nsq.logspace(E_min,E_max,E_nodes)
    return energy_nodes


def set_angle():
    cth_min = -1.0
    cth_max = 1.0
    cth_nodes = 160
    c_zen_nodes = nsq.linspace(cth_min,cth_max,cth_nodes)
    return c_zen_nodes


def ConfigFlux(neutrino_flavours, c_zen_nodes, energy_nodes, flux_type, mode, gamma_factor):
    #setup neutrino flux
    InitialFlux = np.zeros((len(c_zen_nodes), len(energy_nodes), 2, neutrino_flavours))
    if flux_type == 'all' or flux_type == 'atmo':
        atm_flux = InitAtmFlux(mode='modern')
    index = 0
    for i in range(len(c_zen_nodes)):
        for j in range(len(energy_nodes)):
            if flux_type == 'all' or flux_type == 'atmo':
                nue_val       = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i],  12)
                nue_bar_val   = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i], -12)
                numu_val      = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i],  14)
                numu_bar_val  = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i], -14)
                nutau_val     = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i],  16)
                nutau_bar_val = AtmFlux(atm_flux, energy_nodes[j]*1e-9, c_zen_nodes[i], -16)
            if flux_type == 'all' or flux_type == 'astro':
                Astroval = DiffFlux(energy_nodes[j]*1e-9, mode, spectrum=gamma_factor)
            if flux_type == 'atmo':
                InitialFlux[i][j][0][0] = nue_val
                InitialFlux[i][j][1][0] = nue_bar_val
                InitialFlux[i][j][0][1] = numu_val
                InitialFlux[i][j][1][1] = numu_bar_val
                InitialFlux[i][j][0][2] = nutau_val
                InitialFlux[i][j][1][2] = nutau_bar_val
            if flux_type == 'astro':
                InitialFlux[i][j][0][0] = Astroval
                InitialFlux[i][j][1][0] = Astroval
                InitialFlux[i][j][0][1] = Astroval
                InitialFlux[i][j][1][1] = Astroval
                InitialFlux[i][j][0][2] = Astroval
                InitialFlux[i][j][1][2] = Astroval
            if flux_type == 'all':
                InitialFlux[i][j][0][0] = nue_val + Astroval
                InitialFlux[i][j][1][0] = nue_bar_val + Astroval
                InitialFlux[i][j][0][1] = numu_val + Astroval
                InitialFlux[i][j][1][1] = numu_bar_val + Astroval
                InitialFlux[i][j][0][2] = nutau_val + Astroval
                InitialFlux[i][j][1][2] = nutau_bar_val + Astroval

    return InitialFlux


def propagate_events(xsec_norm, flux_type, mode, cache, auto=False, gamma_factor='default'):
    neutrino_flavours = 3
    energy_nodes = set_energy()
    c_zen_nodes = set_angle()
    InitialFlux = ConfigFlux(neutrino_flavours, c_zen_nodes, energy_nodes, flux_type, mode, gamma_factor)

    pref = "/data/user/chill/snowstorm_nugen_ana/xsec/data/"
    if xsec_norm == '1.0':
        xsec_file = pref + "csms.h5"
    else:
        xsec_file = pref + f"csms_{xsec_norm}.h5"
    if os.path.exists(xsec_file) is False:
        raise IOError(f"Please check input params for xsec - file does not exist: {xsec_file}")

    #### nuSQuIDS stuff ####
    print("- Preparing nuSQuIDS -")
    units = nsq.Const()
    interactions = True
    nuSQ = nsq.nuSQUIDSAtm(c_zen_nodes, energy_nodes, neutrino_flavours, nsq.NeutrinoType.both, interactions)
    xs_obj = nsq.NeutrinoDISCrossSectionsFromTables(xsec_file)
    nuSQ.SetNeutrinoCrossSections(xs_obj)
    nuSQ.Set_MixingParametersToDefault()
    nuSQ.Set_initial_state(np.array(InitialFlux),  nsq.Basis.flavor)
    ##Sally set this to 1e-35! But she's not setting h_max -- not working
    ##I had it working with 1e-21, but this wasn't enough
    nuSQ.Set_rel_error(1.0e-27)##needed high precision to avoid floating point errs
    nuSQ.Set_abs_error(1.0e-27)##some example uses -17, others -7
    nuSQ.Set_h_max(500.0*units.km)
    nuSQ.Set_TauRegeneration(True)
    nuSQ.Set_IncludeOscillations(True)
    nuSQ.Set_ProgressBar(True)
    nuSQ.Set_GlashowResonance(True)
    nuSQ.EvolveState()
    ####################

    if cache == True:
        if gamma_factor == 'default':
            outfile_name = f'../nuSQuIDS_propagation_files/nuSQuIDS_flux_cache_{xsec_norm}_{flux_type}_{mode}.hdf'
        else:            
            outfile_name = f'../nuSQuIDS_propagation_files/nuSQuIDS_flux_cache_{xsec_norm}_{gamma_factor}_{flux_type}_{mode}.hdf'
        nuSQ.WriteStateHDF5(os.path.join(os.path.dirname(os.path.abspath(__file__)), outfile_name), True)
        print(f'Created outfile_name')
    
    ## issue with nuSQ being picklable when 
    ## running with parallel processing
    if auto == False:
        return nuSQ
    else:
        return 0

@click.command()
@click.option('--xsec_norm', '-n', default='1.0')
@click.option('--flux_type', '-f', type=click.Choice(['atmo', 'astro', 'all']))
@click.option('--mode', '-m', type=click.Choice(['track', 'cascade']))
@click.option('--gamma', '-g', default='default')
@click.option('--cache', is_flag=True)
@click.option('--auto', is_flag=True)
def main(xsec_norm, flux_type, mode, cache, auto, gamma):
    
    gamma_factor = gamma
    if gamma_factor != 'default':
        gamma_factor = float(gamma_factor)

    norm_list = [0.985, 0.99, 0.995, 1.005, 1.01, 1.015]

    if auto:
        print("Automatically Handling Separated Flux Propagation For Tracks and Cascades for all listed normalisations")
        modes = ['track', 'cascade']
        types = ['atmo', 'astro']

        ##start multi-threading here
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            ##loop over all datasets, CCNC
            for m in modes:
                for t in types:
                    for n in norm_list:
                        futures.append(executor.submit(
                            propagate_events, n, t, m, cache=True, auto=auto, gamma_factor=gamma_factor))
        results = wait(futures)
        for result in results.done:
            print(result.result())

    if not auto:
        if not cache:
            print("=== WARNING - nuSQ State will not be saved! To save, use --cache ===")
        nuSQ = propagate_events(xsec_norm, flux_type, mode, cache, gamma_factor=gamma_factor)

    print("Done")

if __name__ == "__main__":
    main()
##end
