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

def transmissionPlots(infile, cc, nc):

    fig_nu, ax_nu       = plt.subplots()
    fig_nubar, ax_nubar = plt.subplots()
    fig_nue,       ax_nue       = plt.subplots()
    fig_nuebar,    ax_nuebar    = plt.subplots()
    fig_numu,      ax_numu      = plt.subplots()
    fig_numubar,   ax_numubar   = plt.subplots()
    fig_nutau,     ax_nutau     = plt.subplots()
    fig_nutaubar,  ax_nutaubar  = plt.subplots()

    e_range = np.logspace(3, 8, 200)
    atm_flux = InitAtmFlux() 
    #nu_type_dict = {'nue': 0, 'numu': 1, 'nutau': 2}
    nu_type_dict = {12: 0, 14: 1, 16: 2, -12: 0, -14: 1, -16: 2}
    color_list={12:'royalblue', 14:'goldenrod', 16:'firebrick',
                -12: 'royalblue', -14: 'goldenrod', -16: 'firebrick'}
    lstyle={norm_list[0]:'dotted', norm_list[1]:'solid', norm_list[2]:'dashdot'}

    fig_nu_r ,      ax_nu_r       = plt.subplots()
    fig_nubar_r,    ax_nubar_r    = plt.subplots()
    fig_nue_r,      ax_nue_r      = plt.subplots()
    fig_nuebar_r,   ax_nuebar_r   = plt.subplots()
    fig_numu_r,     ax_numu_r     = plt.subplots()
    fig_numubar_r,  ax_numubar_r  = plt.subplots()
    fig_nutau_r,    ax_nutau_r    = plt.subplots()
    fig_nutaubar_r, ax_nutaubar_r = plt.subplots()

    for pdg_code in [12, 14, 16, -12, -14, -16]:
        flux = np.zeros(len(e_range))
        fi_nsqd = np.zeros(len(e_range))
        for index, e_ in enumerate(e_range):
            f_atm = AtmFlux(atm_flux, e_, -1, pdg_code)
            f_diff = DiffFlux(e_)
            flux[index] = f_atm + f_diff
            if pdg_code < 0:
                fi_nsqd[index] = nuSQ.EvalFlavor(nu_type_dict[pdg_code], -1, e_*1e9, 1)
            if pdg_code > 0:
                fi_nsqd[index] = nuSQ.EvalFlavor(nu_type_dict[pdg_code], -1, e_*1e9, 0)
        #propagation weight
        transmissionP = fi_nsqd / flux
        if pdg_code > 0:
            ax_nu.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                       color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nu_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                       color=color_list[pdg_code])
            ax_nu_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                       color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code < 0:
            ax_nubar.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nubar_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                       color=color_list[pdg_code])
            ax_nubar_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                       color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == 12:
            ax_nue.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nue_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_nue_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == -12:
            ax_nuebar.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nuebar_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_nuebar_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == 14:
            ax_numu.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_numu_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_numu_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == -14:
            ax_numubar.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_numubar_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_numubar_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == 16:
            ax_nutau.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nutau_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_nutau_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')
        if pdg_code == -16:
            ax_nutaubar.plot(e_range, transmissionP, label=f'{pdg_code}: {norm}',
                          color=color_list[pdg_code], linestyle=lstyle[norm])
            ax_nutaubar_r.plot(e_range, flux, label=f'{pdg_code}: {norm}' + r' $F_{0}$',
                                 color=color_list[pdg_code])
            ax_nutaubar_r.plot(e_range, fi_nsqd, label=f'{pdg_code}: {norm} F',
                                 color=color_list[pdg_code], linestyle='dashdot')


        ax_nu_r.legend(loc=0)
        ax_nu_r.set_xscale('log')
        ax_nu_r.set_yscale('log')
        ax_nu_r.set_xlabel('Energy [GeV]')
        ax_nu_r.set_ylabel(r'$\nu$ Rate')
        ax_nu_r.set_title(f'Neutrinos {norm}' + r' - $cos(\theta_{z})$ = -1')
        fig_nu_r.tight_layout()
        fig_nu_r.savefig(f'plots/transmission_probability_nu_{norm}_rate.pdf')
        plt.close(fig_nu_r)    
        ax_nubar_r.legend(loc=0)
        ax_nubar_r.set_xscale('log')
        ax_nubar_r.set_yscale('log')
        ax_nubar_r.set_xlabel('Energy [GeV]')
        ax_nubar_r.set_ylabel(r'$\bar{\nu}$ Rate')
        ax_nubar_r.set_title(f'Anti-Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_nubar_r.tight_layout()
        fig_nubar_r.savefig(f'plots/transmission_probability_nubar_{norm}_rate.pdf')
        plt.close(fig_nubar_r)        

        ax_nue_r.legend(loc=0)
        ax_nue_r.set_xscale('log')
        ax_nue_r.set_yscale('log')
        ax_nue_r.set_xlabel('Energy [GeV]')
        ax_nue_r.set_ylabel(r'$\nu$ Rate')
        ax_nue_r.set_title(f'Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_nue_r.tight_layout()
        fig_nue_r.savefig(f'plots/transmission_probability_nue_{norm}_rate.pdf')
        plt.close(fig_nue_r)
        ax_nuebar_r.legend(loc=0)
        ax_nuebar_r.set_xscale('log')
        ax_nuebar_r.set_yscale('log')
        ax_nuebar_r.set_xlabel('Energy [GeV]')
        ax_nuebar_r.set_ylabel(r'$\bar{\nu}$ Rate')
        ax_nuebar_r.set_title(f'Anti-Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_nuebar_r.tight_layout()
        fig_nuebar_r.savefig(f'plots/transmission_probability_nuebar_{norm}_rate.pdf')
        plt.close(fig_nuebar_r)

        ax_numu_r.legend(loc=0)
        ax_numu_r.set_xscale('log')
        ax_numu_r.set_yscale('log')
        ax_numu_r.set_xlabel('Energy [GeV]')
        ax_numu_r.set_ylabel(r'$\nu$ Rate')
        ax_numu_r.set_title(f'Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_numu_r.tight_layout()
        fig_numu_r.savefig(f'plots/transmission_probability_numu_{norm}_rate.pdf')
        plt.close(fig_numu_r)
        ax_numubar_r.legend(loc=0)
        ax_numubar_r.set_xscale('log')
        ax_numubar_r.set_yscale('log')
        ax_numubar_r.set_xlabel('Energy [GeV]')
        ax_numubar_r.set_ylabel(r'$\bar{\nu}$ Rate')
        ax_numubar_r.set_title(f'Anti-Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_numubar_r.tight_layout()
        fig_numubar_r.savefig(f'plots/transmission_probability_numubar_{norm}_rate.pdf')
        plt.close(fig_numubar_r)        

        ax_nutau_r.legend(loc=0)
        ax_nutau_r.set_xscale('log')
        ax_nutau_r.set_yscale('log')
        ax_nutau_r.set_xlabel('Energy [GeV]')
        ax_nutau_r.set_ylabel(r'$\nu$ Rate')
        ax_nutau_r.set_title(f'Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_nutau_r.tight_layout()
        fig_nutau_r.savefig(f'plots/transmission_probability_nutau_{norm}_rate.pdf')
        plt.close(fig_nutau_r)
        ax_nutaubar_r.legend(loc=0)
        ax_nutaubar_r.set_xscale('log')
        ax_nutaubar_r.set_yscale('log')
        ax_nutaubar_r.set_xlabel('Energy [GeV]')
        ax_nutaubar_r.set_ylabel(r'$\bar{\nu}$ Rate')
        ax_nutaubar_r.set_title(f'Anti-Neutrinos {norm}' + r'- $cos(\theta_{z})$ = -1')
        fig_nutaubar_r.tight_layout()
        fig_nutaubar_r.savefig(f'plots/transmission_probability_nutaubar_{norm}_rate.pdf')
        plt.close(fig_nutaubar_r)

    ax_nu.legend(loc=0)
    ax_nu.set_xscale('log')
    ax_nu.set_xlabel('Energy [GeV]')
    ax_nu.set_ylabel(r'$\nu$ Transmission Probability')
    ax_nu.set_title(r'Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nu.tight_layout()
    fig_nu.savefig('plots/transmission_probability_nu.pdf')
    plt.close(fig_nu)
    ax_nubar.legend(loc=0)
    ax_nubar.set_xscale('log')
    ax_nubar.set_xlabel('Energy [GeV]')
    ax_nubar.set_ylabel(r'$\bar{\nu}$ Transmission Probability')
    ax_nubar.set_title(r'Anti-Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nubar.tight_layout()
    fig_nubar.savefig('plots/transmission_probability_nubar.pdf')
    plt.close(fig_nubar)    

    ax_nue.legend(loc=0)
    ax_nue.set_xscale('log')
    ax_nue.set_xlabel('Energy [GeV]')
    ax_nue.set_ylabel(r'$\nu$ Transmission Probability')
    ax_nue.set_title(r'Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nue.tight_layout()
    fig_nue.savefig('plots/transmission_probability_nue.pdf')
    plt.close(fig_nue)
    ax_nuebar.legend(loc=0)
    ax_nuebar.set_xscale('log')
    ax_nuebar.set_xlabel('Energy [GeV]')
    ax_nuebar.set_ylabel(r'$\bar{\nu}$ Transmission Probability')
    ax_nuebar.set_title(r'Anti-Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nuebar.tight_layout()
    fig_nuebar.savefig('plots/transmission_probability_nuebar.pdf')
    plt.close(fig_nuebar)

    ax_numu.legend(loc=0)
    ax_numu.set_xscale('log')
    ax_numu.set_xlabel('Energy [GeV]')
    ax_numu.set_ylabel(r'$\nu$ Transmission Probability')
    ax_numu.set_title(r'Neutrinos - $cos(\theta_{z})$ = -1')
    fig_numu.tight_layout()
    fig_numu.savefig('plots/transmission_probability_numu.pdf')
    plt.close(fig_numu)
    ax_numubar.legend(loc=0)
    ax_numubar.set_xscale('log')
    ax_numubar.set_xlabel('Energy [GeV]')
    ax_numubar.set_ylabel(r'$\bar{\nu}$ Transmission Probability')
    ax_numubar.set_title(r'Anti-Neutrinos - $cos(\theta_{z})$ = -1')
    fig_numubar.tight_layout()
    fig_numubar.savefig('plots/transmission_probability_numubar.pdf')
    plt.close(fig_numubar)

    ax_nutau.legend(loc=0)
    ax_nutau.set_xscale('log')
    ax_nutau.set_xlabel('Energy [GeV]')
    ax_nutau.set_ylabel(r'$\nu$ Transmission Probability')
    ax_nutau.set_title(r'Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nutau.tight_layout()
    fig_nutau.savefig('plots/transmission_probability_nutau.pdf')
    plt.close(fig_nutau)
    ax_nutaubar.legend(loc=0)
    ax_nutaubar.set_xscale('log')
    ax_nutaubar.set_xlabel('Energy [GeV]')
    ax_nutaubar.set_ylabel(r'$\bar{\nu}$ Transmission Probability')
    ax_nutaubar.set_title(r'Anti-Neutrinos - $cos(\theta_{z})$ = -1')
    fig_nutaubar.tight_layout()
    fig_nutaubar.savefig('plots/transmission_probability_nutaubar.pdf')
    plt.close(fig_nutaubar)

def propagationPlots(in_file, cc, nc, influx, selection):
    fpath = config.fluxPath
    
    nuSQList = []
    if in_file != None:    
        nuSQ = nsq.nuSQUIDSAtm(in_file)
    
    elif in_file == None:    
        print('Getting nuSQuIDS File')
        if influx == 'atmo' or influx == 'astro':
            if cc == '1.0' and nc == '1.0':
                in_file = os.path.join(fpath, 
                            f'nuSQuIDS_flux_cache_1.0_{influx}_{selection}.hdf')
            else:
                _fpath = os.path.join(fpath, 'ccnc')
                in_file = os.path.join(_fpath, 
                            f'nuSQuIDS_flux_cache_{cc}CC_{nc}NC_{influx}_{selection}.hdf')
            nuSQ = nsq.nuSQUIDSAtm(in_file)
        if influx == 'all':
            for _flux in ['atmo', 'astro']:
                if cc == '1.0' and nc == '1.0':
                    in_file = os.path.join(fpath, 
                                f'nuSQuIDS_flux_cache_1.0_{_flux}_{selection}.hdf')
                else:
                    _fpath = os.path.join(fpath, 'ccnc')
                    in_file = os.path.join(_fpath, 
                                f'nuSQuIDS_flux_cache_{cc}CC_{nc}NC_{_flux}_{selection}.hdf')
                nuSQ = nsq.nuSQUIDSAtm(in_file)
                nuSQList.append(nuSQ)
   
    if cc == '1.0' and nc == '1.0': 
        pp = 'propagation_plots'
    else: 
        pp = 'propagation_plots_ccnc'

    print('Setting up Input Flux')
    inFluxList = []
    if influx in ['atmo', 'astro']:
        inputFlux = ConfigFlux(3, set_angle(), set_energy(), influx, selection, 'default')
    else:
        for _flux in ['atmo', 'astro']:
            inputFlux = ConfigFlux(3, set_angle(), set_energy(), _flux, selection, 'default')
            inFluxList.append(inputFlux)

    xx, yy = np.meshgrid(set_angle(), set_energy())
    print('-'*20)
    print('Evaluating and Plotting')
    for flav in [0, 1, 2]:
        print(f'Flavour: {flav}')
        for nnbar in [0, 1]:
            print(f'n/nbar: {nnbar}')
            in_weights  = np.zeros([len(set_energy()), len(set_angle())])
            out_weights = np.zeros([len(set_energy()), len(set_angle())])
            for i, e in enumerate(set_energy()):
                for j, csz in enumerate(set_angle()):
                    ##e should be in eV!
                    if influx in ['atmo', 'astro']:
                        prop_w = nuSQ.EvalFlavor(flav, csz, e, nnbar)
                        if prop_w < 0:
                            prop_w = 1e-60
                        out_weights[i][j] = prop_w
                        flux_w = inputFlux[j][i][nnbar][flav] 
                        if flux_w < 0:
                            flux_w = 1e-60
                        in_weights[i][j] = flux_w
                    if influx == 'all':
                        for _nuSQ in nuSQList:
                            prop_w = _nuSQ.EvalFlavor(flav, csz, e, nnbar)
                            if prop_w < 0:
                                prop_w = 1e-60
                            out_weights[i][j] += prop_w
                        for inputFlux in inFluxList:
                            flux_w = inputFlux[j][i][nnbar][flav] 
                            if flux_w < 0:
                                flux_w = 1e-60
                            in_weights[i][j] += flux_w

            ##allow full range of z (color)
            fig1, ax1   = plt.subplots()

            plot = ax1.pcolormesh(xx, yy/1e9, out_weights/in_weights,
                                cmap='viridis', shading='auto',
                                norm=matplotlib.colors.LogNorm())
            ax1.set_yscale('log')
            #ax1.set_ylabel(r'$log_{10}$($E_\nu$) [GeV]')
            ax1.set_ylabel(r'$E_\nu$ [GeV]')
            ax1.set_xlabel(r'$\cos \theta_\nu$')
            plt.title(f'{cc}CC, {nc}NC ' + r'$\times$ Standard Model cross-section')
            cbar = fig1.colorbar(plot, ax=ax1)
            cbar.set_label(r'$\Phi$/$\Phi_{0}$ [GeV cm$^2$ Str s]$^{-1}$', 
                           labelpad=18, rotation=270)
            fig1.savefig(f"{pp}/prop_flux_ratio_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            ax1.set_xlim(-1, 0.2)
            fig1.savefig(f"{pp}/prop_flux_ratio_zoom_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            plt.close(fig1)
            
            fig1, ax1   = plt.subplots()
            plot = ax1.pcolormesh(xx, yy/1e9, out_weights/in_weights,
                                cmap='viridis', shading='auto')
            ax1.set_yscale('log')
            #ax1.set_ylabel(r'$log_{10}$($E_\nu$) [GeV]')
            ax1.set_ylabel(r'$E_\nu$ [GeV]')
            ax1.set_xlabel(r'$\cos \theta_\nu$')
            plt.title(f'{cc}CC, {nc}NC ' + r'$\times$ Standard Model cross-section')
            cbar = fig1.colorbar(plot, ax=ax1)
            cbar.set_label(r'$\Phi$/$\Phi_{0}$', 
                           labelpad=18, rotation=270)
            fig1.savefig(f"{pp}/prop_flux_ratio_nolog_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            ax1.set_xlim(-1, 0.2)
            fig1.savefig(f"{pp}/prop_flux_ratio_zoom_nolog_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                         dpi=320)
            plt.close(fig1)

            fig2d, ax2d = plt.subplots()
            plot = ax2d.pcolormesh(xx, yy/1e9, out_weights,
                                cmap='viridis', shading='auto',
                                norm=matplotlib.colors.LogNorm())
            ax2d.set_yscale('log')
            #ax2d.set_ylabel(r'$log_{10}$($E_\nu$) [GeV]')
            ax2d.set_ylabel(r'$E_\nu$ [GeV]')
            ax2d.set_xlabel(r'$\cos \theta_\nu$')
            plt.title(f'{cc}CC, {nc}NC ' + r'$\times$ Standard Model cross-section')
            cbar = fig2d.colorbar(plot, ax=ax2d)
            cbar.set_label(r'Flux at Detector [GeV cm$^2$ Str s]$^{-1}$', 
                           labelpad=18, rotation=270)
            fig2d.savefig(f"{pp}/prop_flux_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", dpi=320)
            ax2d.set_xlim(-1, 0.2)
            fig2d.savefig(f"{pp}/prop_flux_zoom_{influx}_{flav}_{nnbar}_{cc}cc_{nc}nc.png", 
                          dpi=320)
            plt.close(fig2d)

@click.command()
@click.option('--in_file', default=None)
@click.option('--cc', required=True)
@click.option('--nc', required=True)
@click.option('--flux', '-f', required=True)
@click.option('--selection', '-s', required=True)
def main(in_file, cc, nc, flux, selection):

        if in_file != None:
            print('--- Input file was specified ---')
            propagationPlots(in_file, cc, nc, flux, selection)
            return


        if flux not in ['astro', 'atmo', 'all']:
            raise NotImplementedError(f'{influx} should be astro or atmo or all!')
        if selection not in ['cascade', 'track']:
            raise NotImplementedError(f'{selection} should be cascade or track!')

        propagationPlots(in_file, cc, nc, flux, selection)

        #transmissionPlots(in_file, cc, nc)

if __name__ == "__main__":
    main()

##end
