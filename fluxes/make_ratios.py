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
sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter')
from configs.config import config

#SQuIDS
sys.path.append('/data/user/chill/SQuIDS/lib/')
##nuSQuIDS libraries
sys.path.append('/data/user/chill/nuSQuIDS/resources/python/bindings/')
import nuSQUIDSpy as nsq
import nuSQUIDSTools

def transmissionPlots(cc, nc, nom_nuSQList, mod_nuSQList, pp, cosz=-1):
    fig_nu,     ax_nu     = plt.subplots()
    fig_nubar,  ax_nubar  = plt.subplots()
    fig_nuebar, ax_nuebar = plt.subplots()
    fig_flux,   ax_flux   = plt.subplots()

    e_range = np.logspace(2.5, 8, 200)
    #nu_type_dict = {'nue': 0, 'numu': 1, 'nutau': 2}
    nu_type_dict = {12: 0, 14: 1, 16: 2, -12: 0, -14: 1, -16: 2}
    color_list={12:'royalblue', 14:'goldenrod', 16:'firebrick',
                -12: 'royalblue', -14: 'goldenrod', -16: 'firebrick'}

    for pdg_code in [12, 14, 16, -12, -14, -16]:
        nom = np.zeros(len(e_range))
        mod = np.zeros(len(e_range))
        for index, e_ in enumerate(e_range):
            if pdg_code < 0:
                ##sum the astro & atmo components
                for _nuSQ in nom_nuSQList:
                    nom[index] += _nuSQ.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 1)
                for _nuSQ in mod_nuSQList:
                    mod[index] += _nuSQ.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 1)
            if pdg_code > 0:
                ##sum the astro & atmo components
                for _nuSQ in nom_nuSQList:
                    nom[index] += _nuSQ.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 0)
                for _nuSQ in mod_nuSQList:
                    mod[index] += _nuSQ.EvalFlavor(nu_type_dict[pdg_code], cosz, e_*1e9, 0)
        #propagation weight
        transmissionP = mod / nom
        if pdg_code > 0:
            ax_nu.plot(e_range, transmissionP, label=f'{pdg_code}',
                       color=color_list[pdg_code])
        if pdg_code == 12:
            ax_flux.plot(e_range, nom, linestyle='--', label=f'Nominal',
                   color=color_list[pdg_code])
            ax_flux.plot(e_range, mod, label=f'Mod: {cc}CC, {nc}NC',
                   color=color_list[pdg_code])
        if pdg_code < 0:
            ax_nubar.plot(e_range, transmissionP, label=f'{pdg_code}',
                          color=color_list[pdg_code])
        if pdg_code == -12:
            ax_nuebar.plot(e_range, transmissionP, label=f'{pdg_code}',
                          color=color_list[pdg_code])
            ax_nuebar.legend(loc=0)
            ax_nuebar.set_xscale('log')
            ax_nuebar.set_yscale('log')
            ax_nuebar.set_xlabel('Energy [GeV]')
            ax_nuebar.set_ylabel(r'$\bar{\nu}$ Ratio')
            ax_nuebar.set_title(r'- $cos(\theta_{z})$ =' + f' {cosz}')
            fig_nuebar.tight_layout()
            fig_nuebar.savefig(f'{pp}/nuebar_cos{cosz}_{cc}cc_{nc}nc_ratio.pdf')
            plt.close(fig_nuebar)

    print(f'--- For Norms of {cc}CC & {nc}NC at {cosz} deg---')
    print(f'--- Summed Flux for Nominal : {np.sum(nom)} ---')
    print(f'--- Summed Flux for Modified: {np.sum(mod)} ---')
    print(f'--- Ratio: {np.sum(mod)/np.sum(nom):.5f}')
    print('-'*20)

    ax_flux.legend(loc=0)
    ax_flux.set_xscale('log')
    ax_flux.set_yscale('log')
    ax_flux.set_xlabel('Energy [GeV]')
    ax_flux.set_ylabel('Flux')
    ax_flux.set_title(r'Neutrinos - $cos(\theta_{z})$ =' + f' {cosz}')
    fig_flux.tight_layout()
    fig_flux.savefig(f'{pp}/flux_cos{cosz}_{cc}cc_{nc}nc.pdf')
    plt.close(fig_flux) 

    ax_nu.hlines(1, 1e2, 1e8, color='black', linestyle='--', label = '1:1')
    ax_nu.legend(loc=0)
    ax_nu.set_xscale('log')
    ax_nu.set_xlabel('Energy [GeV]')
    ax_nu.set_ylabel(r'$\nu$ Ratio' + f' {cc}CC & {nc}NC / Nom')
    ax_nu.set_title(r'Neutrinos - $cos(\theta_{z})$ =' + f' {cosz}')
    fig_nu.tight_layout()
    fig_nu.savefig(f'{pp}/nu_cos{cosz}_{cc}cc_{nc}nc_ratio.pdf')
    ax_nu.set_yscale('log')
    fig_nu.tight_layout()
    fig_nu.savefig(f'{pp}/nu_log_cos{cosz}_{cc}cc_{nc}nc_ratio.pdf')
    plt.close(fig_nu)
    ax_nubar.hlines(1, 1e2, 1e8, color='black', linestyle='--', label = '1:1')
    ax_nubar.legend(loc=0)
    ax_nubar.set_xscale('log')
    ax_nubar.set_xlabel('Energy [GeV]')
    ax_nubar.set_ylabel(r'$\bar{\nu}$ Ratio' + f' {cc}CC & {nc}NC / Nom')
    ax_nubar.set_title(r'Anti-Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nubar.tight_layout()
    fig_nubar.savefig(f'{pp}/nubar_cos{cosz}_{cc}cc_{nc}nc_ratio.pdf')
    ax_nubar.set_yscale('log')
    fig_nubar.tight_layout()
    fig_nubar.savefig(f'{pp}/nubar_log_cos{cosz}_{cc}cc_{nc}nc_ratio.pdf')
    plt.close(fig_nubar)    

def getPropFlux(cc, nc, selection):
    fpath = config.fluxPath
    mod_nuSQList = []
    nom_nuSQList = []
    print('Getting nuSQuIDS File')
    for _flux in ['atmo', 'astro']:
        in_file = os.path.join(fpath, 
                    f'nuSQuIDS_flux_cache_1.0_{_flux}_{selection}.hdf')
        nuSQ = nsq.nuSQUIDSAtm(in_file)
        nom_nuSQList.append(nuSQ)
    for _flux in ['atmo', 'astro']:
        _fpath = os.path.join(fpath, 'ccnc')
        in_file = os.path.join(_fpath, 
                    f'nuSQuIDS_flux_cache_{cc}CC_{nc}NC_{_flux}_{selection}.hdf')
        nuSQ = nsq.nuSQUIDSAtm(in_file)
        mod_nuSQList.append(nuSQ)
    return nom_nuSQList, mod_nuSQList

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
@click.option('--cc', required=True)
@click.option('--nc', required=True)
@click.option('--selection', '-s', required=True)
def main(cc, nc, selection):
        if selection not in ['cascade', 'track']:
            raise NotImplementedError(f'{selection} should be cascade or track!')
        
        pp = 'propagation_plots_ratio'
        nom_List, mod_List = getPropFlux(cc, nc, selection)
        propagationPlots(cc, nc, selection, nom_List, mod_List, pp)
        for cosz in [-1, -0.8, 0.0, 0.2]:
            transmissionPlots(cc, nc, nom_List, mod_List, pp, cosz=cosz)

if __name__ == "__main__":
    main()

##end
