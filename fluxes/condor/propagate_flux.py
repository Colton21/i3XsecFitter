#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, wait

##local libraries
sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter/fluxes')
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


def ConfigFlux(neutrino_flavours, c_zen_nodes, energy_nodes, flux_type, gamma_factor):
    #setup neutrino flux
    InitialFlux = np.zeros((len(c_zen_nodes), len(energy_nodes), 2, neutrino_flavours))
    if flux_type == 'all' or flux_type == 'atmo':
        atm_flux = InitAtmFlux(mode='modern')
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
                Astroval = DiffFlux(energy_nodes[j]*1e-9, mode='classic', spectrum=gamma_factor)
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


def propagate_events(xsec_norm, flux_type, auto=False, gamma_factor='default',
                     earth='default'):
    neutrino_flavours = 3
    energy_nodes = set_energy()
    c_zen_nodes = set_angle()
    InitialFlux = ConfigFlux(neutrino_flavours, c_zen_nodes, energy_nodes, flux_type, gamma_factor)

    pref = "/data/user/chill/snowstorm_nugen_ana/xsec/data/"
    xsec_file = pref + f"csms_{xsec_norm}.h5"
    if os.path.exists(xsec_file) is False:
        raise IOError(f"Please check input params for xsec - file does not exist: {xsec_file}")

    #### nuSQuIDS stuff ####
    print("- Preparing nuSQuIDS -")
    units = nsq.Const()
    interactions = True
    nuSQ = nsq.nuSQUIDSAtm(c_zen_nodes, energy_nodes, neutrino_flavours, nsq.NeutrinoType.both, interactions)
    xs_obj = nsq.NeutrinoDISCrossSectionsFromTables(xsec_file)
    
    if earth != 'default':
        earth_name = f'EARTH_MODEL_PREM_{earth}.dat'
        earth_path = '/data/user/chill/icetray_LWCompatible/i3XsecFitter/fluxes/'
        earth = os.path.join(earth_path, f'EARTH_MODEL_PREM_{earth}.dat')
        if not os.path.exists(earth):
            raise FileNotFoundError(f'{earth} not found!')
        earth_atm = nsq.EarthAtm(earth)
        nuSQ.Set_EarthModel(earth_atm)
        earth_setting_a = earth_name.split('_')[3]
        earth_setting_b = earth_name.split('_')[4]
        earth_setting_b = earth_setting_b.split('.')[0]
        earth_setting = earth_setting_a + earth_setting_b
        earth_setting = earth_setting.lower()

    nuSQ.SetNeutrinoCrossSections(xs_obj)
    nuSQ.Set_MixingParametersToDefault()
    nuSQ.Set_initial_state(np.array(InitialFlux),  nsq.Basis.flavor)
    ##Sally set this to 1e-35! But she's not setting h_max -- not working
    ##I had it working with 1e-21, but this wasn't enough
    ##Eventually got 1e-27 working, but at 1e-31 on NPX output was 0 everywhere
    nuSQ.Set_rel_error(1.0e-30)##needed high precision to avoid floating point errs
    nuSQ.Set_abs_error(1.0e-30)##some example uses -17, others -7
    nuSQ.Set_h_max(500.0*units.km)
    nuSQ.Set_TauRegeneration(True)
    nuSQ.Set_IncludeOscillations(True)
    nuSQ.Set_ProgressBar(True)
    nuSQ.Set_GlashowResonance(True)
    nuSQ.EvolveState()
    ####################

    outfile_name = f'/data/user/chill/icetray_LWCompatible/propagation_grid/output/nuSQuIDS_flux_cache_{xsec_norm}_{earth_setting}_{gamma_factor}_{flux_type}.hdf'
    nuSQ.WriteStateHDF5(outfile_name, True)
    print(f'Created {outfile_name}')
    return 0

@click.command()
@click.option('--xsec_norm', '-n', default='100')
@click.option('--flux_type', '-f', type=click.Choice(['atmo', 'astro', 'all']))
@click.option('--gamma', '-g', default='default')
@click.option('--earth', '-e', default='default')
def main(xsec_norm, flux_type, gamma, earth):

    ####THIS IS THE CONDOR VERSION - BE CAREFUL OF CHANGES!###
    ##Simulates 1 file at a time
    ##input handling changed
    ##track and cascade treated together
    ##cache always true


    xsec_norm = float(xsec_norm) / 100
    if xsec_norm > 2.1 or xsec_norm < 0.2:
        raise ValueError(f'xsec_norm should be between 0.2 and 2.1! Not {xsec_norm}!')    

    if gamma == 'default':
        gamma = 237
    gamma_factor = gamma
    gamma_factor = -1 * float(gamma_factor) / 100
    if gamma_factor > 0 or gamma_factor < -3.0:
        raise ValueError(f'gamma_factor should be less than 0 and greater than -3.0! Not {gamma_factor}')
    
    print(f'Running with {xsec_norm}, {gamma_factor}')

    nuSQ = propagate_events(xsec_norm, flux_type, gamma_factor=gamma_factor,
                            earth=earth)

    print("Done")

if __name__ == "__main__":
    main()
##end
