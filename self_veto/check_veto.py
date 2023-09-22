import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os, sys
import click
import pandas as pd
from glob import glob

sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter')
from helper.reco_helper import remove_bad_reco
#from weighting.create_weight_df_data import calc_live_time
from configs.config import config
        
def saveHelper(fig, figname, year, subbin='', norm=False, dpi=None, ignore_close=False):
    figpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if subbin != '':
        figname = f'{figname}_{subbin}'
    if norm == True:
        figname = f'{figname}_norm'
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        print(f'Created plotting directory {figpath}')
    fig.tight_layout()
    if year == 'total':
        if dpi != None:        
            fig.savefig(os.path.join(figpath, f'{figname}.png'), dpi=dpi)
        else:
            fig.savefig(os.path.join(figpath, f'{figname}.pdf'))
    elif year == 'truth':
        tfigpath = os.path.join(figpath, f'{year}')
        if not os.path.exists(tfigpath):
            os.mkdir(tfigpath)
            print(f'Created plotting directory {tfigpath}')
        if dpi != None:        
            fig.savefig(os.path.join(tfigpath, f'{figname}.png'), dpi=dpi)
        else:
            fig.savefig(os.path.join(tfigpath, f'{figname}.pdf'))
    else:
        yfigpath = os.path.join(figpath, f'{year}')
        if not os.path.exists(yfigpath):
            os.mkdir(yfigpath)
            print(f'Created plotting directory {yfigpath}')
        if dpi != None:        
            fig.savefig(os.path.join(yfigpath, f'{figname}.png'), dpi=dpi)
        else:
            fig.savefig(os.path.join(yfigpath, f'{figname}_{year}.pdf'))
    if ignore_close == False:
        plt.close(fig)
    
    if dpi != None:        
        print(f'Saved {figname}.png')
    else:
        print(f'Saved {figname}.pdf')

def simple_truth(df_li, opt='track', f_type='all', do_norms=False, best_fit=True):
    atmo_norm = 1
    if opt == 'track':
        label = 'Track'
        if best_fit == True:
            atmo_norm = config.track_atmo_norm
    if opt == 'cascade':
        label = 'Cascade'
        if best_fit == True:
            atmo_norm = config.cascade_atmo_norm

    yr_const = 60 * 60 * 24 * 3600

    e_li = df_li.nu_energy.values
    z_li = np.cos(df_li.nu_zenith.values)
    w_li_atmo  = (df_li['weight1.0_atmo'].values * (yr_const / df_li.LiveTime.values[0]) * 
                  atmo_norm)
    w_li_atmo_veto  = (df_li['weight1.0_atmo'].values * (yr_const / df_li.LiveTime.values[0]) * 
                  atmo_norm * df_li['veto_pf'].values)
    w_li_astro = df_li['weight1.0_astro'].values * (yr_const / df_li.LiveTime.values[0])
    w_li = w_li_atmo + w_li_astro
    w_li_v = w_li_atmo_veto + w_li_astro

    print(f'-- Truth Plots {opt} --')
    print(f'Lepton Injector No Veto: {np.sum(w_li)}')
    print(f'Lepton Injector w/ Veto: {np.sum(w_li_v)}')
    print(f'veto / no veto : {np.sum(w_li_v)/np.sum(w_li)}')

    binning  = np.logspace(2, 8, 35)
    zbinning = np.linspace(-1, 1, 24)
    bin_mids = (binning[1:] + binning[:-1])/2
    zbin_mids = (zbinning[1:] + zbinning[:-1])/2

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('True Neutrino Energy [GeV]')
    ax1.hist(e_li, bins=binning, weights=w_li_atmo,   histtype='step', 
             label=f'Atmo No Veto',  color='royalblue')
    ax1.hist(e_li, bins=binning, weights=w_li_atmo_veto, histtype='step', 
             label=f'Atmo w/ Veto',  color='goldenrod')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(title=f'{label}')
    ax1.set_ylabel('Events per year')
    saveHelper(fig1, f'veto_energy_atmo_only_{opt}', 'truth')
    
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('True Neutrino Energy [GeV]')
    ax1.hist(e_li, bins=binning, weights=w_li,   histtype='step', 
             label=f'No Veto',  color='royalblue')
    ax1.hist(e_li, bins=binning, weights=w_li_v, histtype='step', 
             label=f'With Veto',  color='goldenrod')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(title=f'{label}')
    ax1.set_ylabel('Events per year')
    saveHelper(fig1, f'veto_energy_{opt}', 'truth')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r'True Neutrino cos($\theta_{z}$)')
    ax3.hist(z_li, bins=zbinning, weights=w_li_atmo, 
             histtype='step', label=f'Atmo No Veto',  color='royalblue')
    ax3.hist(z_li, bins=zbinning, weights=w_li_atmo_veto, 
             histtype='step', label=f'Atmo w/ Veto',  color='goldenrod')
    ax3.set_yscale('log')
    ax3.legend(title=f'{label}')
    ax3.set_ylabel('Events per Year')
    saveHelper(fig3, f'veto_zenith_atmo_only_{opt}', 'truth')
    
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r'True Neutrino cos($\theta_{z}$)')
    ax3.hist(z_li, bins=zbinning, weights=w_li, 
             histtype='step', label=f'No Veto',  color='royalblue')
    ax3.hist(z_li, bins=zbinning, weights=w_li_v, 
             histtype='step', label=f'With Veto',  color='goldenrod')
    ax3.set_yscale('log')
    ax3.legend(title=f'{label}')
    ax3.set_ylabel('Events per Year')
    saveHelper(fig3, f'veto_zenith_{opt}', 'truth')

def get_dfs(opt='lepton_weighter'):
    df_dir = '/data/user/chill/icetray_LWCompatible/dataframes/'
    if opt.lower() == 'lepton_weighter':
        print("Getting Lepton Injector Monte Carlo")
        df = pd.read_hdf(os.path.join(df_dir, 'li_total.hdf5'))
        return df
    elif opt.lower() == 'data':
        print("Getting burn sample data")
        df = pd.read_hdf(os.path.join(df_dir, 'data_total.hdf5'))
        return df
    else:
        raise NotImplementedError(f'Option {opt} to open dfs not valid')

@click.command()
@click.option('--norm', '-n', is_flag=True)
@click.option('--ignore_nugen', '-i', is_flag=True)
@click.option('--best_fit', '-b', is_flag=True)
@click.option('--fast', is_flag=True)
def main(norm, ignore_nugen, best_fit, fast):

    if best_fit == True:
        print("=== Modifying MC Weight Normalisation to best fit values ===")

    df_data = get_dfs('data')
    df_li = get_dfs('lepton_weighter')
 
    df_li_c = df_li[df_li.Selection == 'cascade']
    df_li_t = df_li[df_li.Selection == 'track']

    ##also cut away the GR events
    df_li_c = df_li_c[(df_li_c.IntType == 'CC') | (df_li_c.IntType == 'NC')]
    df_li_t = df_li_t[(df_li_t.IntType == 'CC') | (df_li_t.IntType == 'NC')]
    
    ## check if weights are separate or merged
    f_type = 'separate'
    #for key in df_li_c.keys():
    #    if key == 'weight1.0':
    #        f_type='all'
    print(f'=== MC Weights were calculated f_type: {f_type} ===')

    ## do_norms will plot for select xsec normalisations
    simple_truth(df_li_c, opt='cascade', f_type=f_type, do_norms=False, best_fit=best_fit)

    print('Done')

if __name__ == "__main__":
    main()

##end
