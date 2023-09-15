import sys, os
import numpy as np
import pandas as pd
import click
import matplotlib
import matplotlib.pyplot as plt

#local libs
from fluxes import InitAtmFlux, AtmFlux, DiffFlux
from propagate_flux import set_energy, set_angle
from propagate_flux import ConfigFlux
from configs.config import config
import nuSQUIDSpy as nsq
import nuSQUIDSTools

def transmissionPlots(cc, nc, nom_nuSQList, mod_nuSQList_atmo, mod_nuSQList_astro, pp, cosz=-1):
    fig_nu,     ax_nu     = plt.subplots()
    fig_nubar,  ax_nubar  = plt.subplots()
    fig_nuebar, ax_nuebar = plt.subplots()
    fig_flux,   ax_flux   = plt.subplots()

    e_range = np.logspace(2.5, 8, 200)
    #nu_type_dict = {'nue': 0, 'numu': 1, 'nutau': 2}
    nu_type_dict = {12: 0, 14: 1, 16: 2, -12: 0, -14: 1, -16: 2}
    color_list={12:'royalblue', 14:'goldenrod', 16:'firebrick',
                -12: 'royalblue', -14: 'goldenrod', -16: 'firebrick'}
    pdg_code = 12

    nom = np.zeros(len(e_range))
    for index, e_ in enumerate(e_range):
        for _nuSQ in nom_nuSQList:
            nom[index] += _nuSQ.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 0)
    
    nlen = len(cc)
    for i, _cc in enumerate(cc):
        for j, _nc in enumerate(nc):
            _nuSQ_atmo  = mod_nuSQList_atmo[(i*nlen)+j]
            _nuSQ_astro = mod_nuSQList_astro[(i*nlen)+j]
            mod = np.zeros(len(e_range))
            for index, e_ in enumerate(e_range):
                ##sum the astro & atmo components
                mod[index] = _nuSQ_atmo.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 0)
                mod[index] += _nuSQ_astro.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 0)
            transmissionP = mod / nom
            ax_nu.plot(e_range, transmissionP, label=f'{_cc}CC {_nc}NC')
            ax_flux.plot(e_range, transmissionP, label=f'{_cc}CC {_nc}NC')

            #print(f'--- For Norms of {_cc}CC & {_nc}NC at {cosz} deg---')
            #print(f'--- Summed Flux for Nominal : {np.sum(nom)} ---')
            #print(f'--- Summed Flux for Modified: {np.sum(mod)} ---')
            #print(f'--- Ratio: {np.sum(mod)/np.sum(nom):.5f}')
            #print('-'*20)

    ax_flux.legend(loc=0)
    ax_flux.set_xscale('log')
    ax_flux.set_yscale('log')
    ax_flux.set_xlabel('Energy [GeV]')
    ax_flux.set_ylabel('Flux')
    ax_flux.set_title(r'Neutrinos - $cos(\theta_{z})$ =' + f' {cosz}')
    fig_flux.savefig(f'{pp}/flux_cos{cosz}.pdf')
    plt.close(fig_flux) 

    ax_nu.hlines(1, 1e2, 1e8, color='black', linestyle='--', label = '1:1')
    ax_nu.legend(loc=0)
    ax_nu.set_xscale('log')
    ax_nu.set_xlabel('Energy [GeV]')
    ax_nu.set_ylabel(r'$\nu$ Ratios' + f' Mod / Nom')
    ax_nu.set_title(r'Neutrinos - $cos(\theta_{z})$ =' + f' {cosz}')
    fig_nu.savefig(f'{pp}/nu_cos{cosz}_ratio.pdf')
    ax_nu.set_yscale('log')
    fig_nu.savefig(f'{pp}/nu_log_cos{cosz}_ratio.pdf')
    plt.close(fig_nu)

def getPropFlux(cc, nc, selection):
    fpath = config.fluxPath
    mod_nuSQList_atmo = []
    mod_nuSQList_astro = []
    nom_nuSQList = []
    print('Getting nominal nuSQuIDS Files')
    for _flux in ['atmo', 'astro']:
        in_file = os.path.join(fpath, 
                    f'nuSQuIDS_flux_cache_1.0_{_flux}_{selection}.hdf')
        nuSQ = nsq.nuSQUIDSAtm(in_file)
        nom_nuSQList.append(nuSQ)

    print('Getting Modified Files')    
    for _cc in cc:
        print(f'Grabbing CC = {_cc}')
        for _nc in nc:
            _fpath = os.path.join(fpath, 'ccnc')
            in_file = os.path.join(_fpath, 
                f'nuSQuIDS_flux_cache_{_cc}CC_{_nc}NC_atmo_{selection}.hdf')
            nuSQ = nsq.nuSQUIDSAtm(in_file)
            mod_nuSQList_atmo.append(nuSQ)
            in_file = os.path.join(_fpath, 
                f'nuSQuIDS_flux_cache_{_cc}CC_{_nc}NC_astro_{selection}.hdf')
            nuSQ = nsq.nuSQUIDSAtm(in_file)
            mod_nuSQList_astro.append(nuSQ)
    return nom_nuSQList, mod_nuSQList_atmo, mod_nuSQList_astro

def propagationPlots(cc, nc, selection, nom_nuSQList, mod_nuSQList, pp):
    #print('Setting up Input Flux')
    #inFluxList = []
    #for _flux in ['atmo', 'astro']:
    #    inputFlux = ConfigFlux(3, set_angle(), set_energy(), _flux, selection, 'default')
    #    inFluxList.append(inputFlux)

    xx, yy = np.meshgrid(set_angle(), set_energy())
    print('-'*20)
    print('Evaluating and Plotting')
    for flav in [0, 1, 2]:
        print(f'Flavour: {flav}')
        for nnbar in [0, 1]:
            print(f'n/nbar: {nnbar}')
            #in_weights  = np.zeros([len(set_energy()), len(set_angle())])
            out_nom_weights = np.zeros([len(set_energy()), len(set_angle())])
            out_mod_weights = np.zeros([len(set_energy()), len(set_angle())])
            for i, e in enumerate(set_energy()):
                for j, csz in enumerate(set_angle()):
                    ##e should be in eV!
                    for _nuSQ in nom_nuSQList:
                        prop_w = _nuSQ.EvalFlavor(flav, csz, e, nnbar)
                        if prop_w < 0:
                            prop_w = 1e-60
                        out_nom_weights[i][j] += prop_w
                    for _nuSQ in mod_nuSQList:
                        prop_w = _nuSQ.EvalFlavor(flav, csz, e, nnbar)
                        if prop_w < 0:
                            prop_w = 1e-60
                        out_mod_weights[i][j] += prop_w
                        #for inputFlux in inFluxList:
                        #    flux_w = inputFlux[j][i][nnbar][flav] 
                        #    if flux_w < 0:
                        #        flux_w = 1e-60
                        #    in_weights[i][j] += flux_w

            ##allow full range of z (color)
            fig1, ax1   = plt.subplots()
            plot = ax1.pcolormesh(xx, yy/1e9, out_mod_weights/out_nom_weights,
                                cmap='viridis', shading='auto',
                                norm=matplotlib.colors.LogNorm())
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$E_\nu$ [GeV]')
            ax1.set_xlabel(r'$\cos \theta_\nu$')
            plt.title(f'{cc}CC, {nc}NC ' + r'$\times$ Standard Model cross-section')
            cbar = fig1.colorbar(plot, ax=ax1)
            cbar.set_label(r'$\Phi_{mod}$/$\Phi_{nom}$', 
                           labelpad=18, rotation=270)
            fig1.savefig(f"{pp}/prop_flux_ratio_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            ax1.set_xlim(-1, 0.2)
            fig1.savefig(f"{pp}/prop_flux_ratio_zoom_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            plt.close(fig1)
            
            fig1, ax1   = plt.subplots()
            plot = ax1.pcolormesh(xx, yy/1e9, out_mod_weights/out_nom_weights,
                                cmap='viridis', shading='auto')
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$E_\nu$ [GeV]')
            ax1.set_xlabel(r'$\cos \theta_\nu$')
            plt.title(f'{cc}CC, {nc}NC ' + r'$\times$ Standard Model cross-section')
            cbar = fig1.colorbar(plot, ax=ax1)
            cbar.set_label(r'$\Phi_{mod}$/$\Phi_{nom}$', 
                           labelpad=18, rotation=270)
            fig1.savefig(f"{pp}/prop_flux_ratio_nolog_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            ax1.set_xlim(-1, 0.2)
            fig1.savefig(f"{pp}/prop_flux_ratio_zoom_nolog_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            plt.close(fig1)
   
            if flav == 0 and nnbar == 0: 
                print(f'--- For Norms of {cc}CC & {nc}NC at all angles ---')
                print(f'--- Summed Flux for Nominal : {np.sum(out_nom_weights)} ---')
                print(f'--- Summed Flux for Modified: {np.sum(out_mod_weights)} ---')
                print(f'--- Ratio: {np.sum(out_mod_weights)/np.sum(out_nom_weights):.5f}')
                print('-'*20)

@click.command()
@click.option('--selection', '-s')
def main(selection):
    pp = 'propagation_plots_select'
    #cc = ['1e-09', '0.5', '0.9', '1.0', '1.1', '2.0']
    #nc = ['1e-09', '0.5', '0.9', '1.0', '1.1', '2.0']
    cc = ['0.5', '1.0', '2.0']
    nc = ['0.5', '1.0', '2.0']
    nom_List, mod_List_atmo, mod_List_astro = getPropFlux(cc, nc, selection='track')
    #propagationPlots(cc, nc, selection, nom_List, mod_List, pp)
    for cosz in [-1, -0.8, -0.2, 0.0, 0.2]:
        transmissionPlots(cc, nc, nom_List, mod_List_atmo, mod_List_astro, pp, cosz=cosz)

if __name__ == "__main__":
    main()

##end
