import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helper.reco_helper import remove_bad_reco
from configs.config import config

YR_CONST = 60 * 60 * 24 * 3600

def remove_gr(df):
    _df = df[(df.IntType == 'CC') | (df.IntType == 'NC')]
    return _df

def get_weights(df, opt, normList, gammaOpt=False):

    lt_const = YR_CONST / df.LiveTime.values[0]

    if opt == 'cascade':
        atmo_norm = config.cascade_atmo_norm
    if opt == 'track':
        atmo_norm = config.track_atmo_norm

    w_atmo  = []
    w_astro = []
    w_total = []
    for norm in normList:
        if gammaOpt != False:

        else:
            _w_atmo  = df[f'weight{norm}_atmo'].values  * lt_const * atmo_norm
            _w_astro = df[f'weight{norm}_astro'].values * lt_const

        w_atmo.append(_w_atmo)
        w_astro.append(_w_astro)
        w_total.append(_w_atmo + _w_astro)

    return w_atmo, w_astro, w_total

def handle_weights(df, opt, normList, earth, ax_n_t, ax_n_c, gammaOpt=False):            
    _df = df[df.Selection == opt]
    w_atmo, w_astro, w_total = get_weights(_df, opt, normList, gammaOpt='')
    if opt == 'track':
        ax_n_t.plot(normList, np.sum(w_total, axis=1), 'o', 
                    linewidth=0, label=f'{earth}')
    if opt == 'cascade':
        ax_n_c.plot(normList, np.sum(w_total, axis=1), 'o', 
                    linewidth=0, label=f'{earth}')

    return w_atmo, w_astro, w_total

def getNormList(d_norm, d_file):
    df = pd.read_hdf(d_file)
    keys = df.keys()
    kList = []
    for k in keys:
        if 'weight' in k:
            _str = k.split('_')[0]
            _str = _str[6:]
            kList.append(float(_str))

    mod_unique = np.unique(kList)
    
    df = pd.read_hdf(d_norm)
    keys = df.keys()
    kList = []
    for k in keys:
        if 'weight' in k:
            _str = k.split('_')[0]
            _str = _str[6:]
            kList.append(float(_str))

    norm_unique = np.unique(kList)
    return np.intersect1d(norm_unique, mod_unique)
    
def frac_plot(opt, normList, w_total_nu, w_total_nd, plot_dir):
    fig_dt, ax_dt = plt.subplots()
    ax_dt.plot(normList, 100*np.sum(w_total_nu, axis=1)/len(w_total_nu[0]), 
               'o', linewidth=0, label='Up', color='royalblue')
    ax_dt.plot(normList, 100*np.sum(w_total_nd, axis=1)/len(w_total_nd[0]), 
               'o', linewidth=0, label='Down', color='goldenrod')
    ax_dt.hlines(0, np.min(normList), np.max(normList), linestyle='--', color='black')
    ax_dt.legend()
    ax_dt.set_xlabel('Cross Section Normalisation')
    ax_dt.set_ylabel('Frac. Diff. [%] Num. Events / 1 yr')
    fig_dt.savefig(os.path.join(plot_dir, f'event_rate_diff_{opt}.pdf'))
    plt.close(fig_dt)

def true_gamma_plot(opt, normList, dfList, wList, plot_dir):
    true_plot(opt, normList, dfList, wList, plot_dir, gamma=True)

def true_plot(opt, normList, dfList, wList, plot_dir, gamma=False):
    
    df_n  = dfList[0][dfList[0].Selection == opt]
    df_u  = dfList[1][dfList[1].Selection == opt]
    df_d  = dfList[2][dfList[2].Selection == opt]
    df_uu = dfList[3][dfList[3].Selection == opt]
    df_dd = dfList[4][dfList[4].Selection == opt]
    df_list = [df_n, df_u, df_d, df_uu, df_dd]
    if gamma == True:
        df_gu = dfList[5][dfList[5].Selection == opt]
        df_gd = dfList[6][dfList[6].Selection == opt]
        df_list.append(df_gu)
        df_list.append(df_gd)

    eList = []
    zList = []
    w_list = []
    for df in df_list:
        e.append(df.nu_energy.values)
        z.append(np.cos(df_d.nu_zenith.values))

    ##pick a normalisation
    ##normally do 1.0
    for i, norm in enumerate(normList):
        if norm == 1.0:
            for w_total in wList:
                w_list.append(w_total[i])
            break

    binning  = np.logspace(2, 8, 14)
    fig_e, ax_e = plt.subplots()
    vals_n, _, _  = ax_e.hist(eList[0], binning, weights=w_list[0], color='royalblue', 
              histtype='step', label=f'Norm:{np.sum(w_list[0])}')
    vals_u, _, _  = ax_e.hist(eList[1], binning, weights=w_list[1], color='goldenrod',
              histtype='step', label=f'Core Up:{np.sum(w_list[1])}')
    vals_d, _, _  = ax_e.hist(eList[2], binning, weights=w_list[2], color='firebrick',
              histtype='step', label=f'Core Down:{np.sum(w_list[2])}')
    vals_uu, _, _ = ax_e.hist(eList[3], binning, weights=w_list[3], color='olive',
              histtype='step', label=f'All Up:{np.sum(w_list[3])}')
    vals_dd, _, _ = ax_e.hist(eList[4], binning, weights=w_list[4], color='salmon',
              histtype='step', label=f'All Down:{np.sum(w_list[4])}')
    
    if gamma == True:
        vals_gu, _, _  = ax_e.hist(eList[5], binning, weights=w_list[5], color='indigo',
                  histtype='step', label=f'Gamma Up:{np.sum(w_list[5])}')
        vals_gd, _, _  = ax_e.hist(eList[6], binning, weights=w_list[6], color='black',
                  histtype='step', label=f'Gamma Down:{np.sum(w_list[6])}')


    ax_e.set_xlabel('True Neutrino Energy [GeV]')
    ax_e.set_ylabel('Events per bin / 1 yr')
    ax_e.set_title(f'{opt}')
    ax_e.set_xscale('log')
    ax_e.set_yscale('log')
    ax_e.legend()
    fig_e.tight_layout()
    fig_e.savefig(os.path.join(plot_dir, f'true_e_{opt}.pdf'))
    plt.close(fig_e)

    fig_d, ax_d = plt.subplots()
    ax_d.plot(binning[:-1], 100*((vals_u/vals_n) - 1), color='goldenrod', 
              label=f'Core Up')
    ax_d.plot(binning[:-1], 100*((vals_d/vals_n) - 1), color='firebrick',
              label=f'Core Down')
    ax_d.plot(binning[:-1], 100*((vals_uu/vals_n) - 1), color='olive', 
              label=f'All Up')
    ax_d.plot(binning[:-1], 100*((vals_ddd/vals_n) - 1), color='salmon',
              label=f'All Down')
    
    if gamma == True:
        ax_d.plot(binning[:-1], 100*((vals_gu/vals_n) - 1), color='indigo', 
                  label=f'Gamma Up')
        ax_d.plot(binning[:-1], 100*((vals_gd/vals_n) - 1), color='black',
                  label=f'Gamma Down')


    ax_d.set_xlabel('True Neutrino Energy [GeV]')
    ax_d.set_ylabel(r'Rel. $\Delta$ Events per bin [%]')
    ax_d.set_title(f'{opt}')
    ax_d.set_xscale('log')
    ax_d.legend()
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(plot_dir, f'true_e_diff_{opt}.pdf'))
    plt.close(fig_d)

    mask_n    = z_n <= -0.9
    mask_nn   = (z_n > -0.9) & (z_n <= -0.6)
    mask_nnn  = (z_n > -0.6) & (z_n <= 0.0)
    mask_nnnn = (z_n > 0.0)
    nlabels = ['-1.0, -0.9', '-0.9, -0.6', '-0.6, 0.0', '0.0, 1.0']

    i = 0
    for mask, label in zip([mask_n, mask_nn, mask_nnn, mask_nnnn], nlabels):
        fig_z, ax_z = plt.subplots()
        vals_n, _, _  = ax_z.hist(eList[0][mask], binning, weights=w_list[0][mask], 
                color='royalblue', histtype='step', label=f'Norm:{np.sum(w_list[0][mask])}')
        vals_u, _, _  = ax_z.hist(eList[1][mask], binning, weights=w_list[1][mask], 
                color='goldenrod', histtype='step', label=f'Core Up:{np.sum(w_list[1][mask])}')
        vals_d, _, _  = ax_z.hist(eList[2][mask], binning, weights=w_list[2][mask], 
                color='firebrick', histtype='step', label=f'Core Down:{np.sum(w_list[2][mask])}')
        vals_uu, _, _ = ax_z.hist(eList[3][mask], binning, weights=w_list[3][mask], 
                color='olive', histtype='step', label=f'All Up:{np.sum(w_list[3][mask])}')
        vals_dd, _, _ = ax_z.hist(eList[4][mask], binning, weights=w_list[4][mask], 
                color='salmon', histtype='step', label=f'All Down:{np.sum(w_list[4][mask])}')
        if gamma == True:
            vals_gu, _, _  = ax_z.hist(eList[5][mask], binning, weights=w_list[5][mask], 
                color='indigo', histtype='step', label=f'Gamma Up:{np.sum(w_list[5][mask])}')
            vals_gd, _, _  = ax_z.hist(eList[6][mask], binning, weights=w_list[6][mask], 
                color='black', histtype='step', label=f'Gamma Down:{np.sum(w_list[6][mask])}')
        ax_z.set_xlabel('True Neutrino Energy [GeV]')
        ax_z.set_ylabel('Events per bin / 1 yr')
        ax_z.set_title(f'{opt}, {label}')
        ax_z.set_xscale('log')
        ax_z.set_yscale('log')
        ax_z.legend()
        fig_z.tight_layout()
        fig_z.savefig(os.path.join(plot_dir, f'true_e_{i}_{opt}.pdf'))
        plt.close(fig_z)
        
        fig_d, ax_d = plt.subplots()
        ax_d.plot(binning[:-1], 100*((vals_u/vals_n) - 1), color='royalblue', 
                  label=f'Core Up')
        ax_d.plot(binning[:-1], 100*((vals_d/vals_n) - 1), color='goldenrod',
                  label=f'Core Down')
        ax_d.plot(binning[:-1], 100*((vals_uu/vals_n) - 1), color='olive', 
              label=f'All Up')
        ax_d.plot(binning[:-1], 100*((vals_dd/vals_n) - 1), color='salmon',
              label=f'All Down')
        if gamma == True:
            ax_d.plot(binning[:-1], 100*((vals_gu/vals_n) - 1), color='indigo', 
                      label=f'Gamma Up')
            ax_d.plot(binning[:-1], 100*((vals_gd/vals_n) - 1), color='black',
                      label=f'Gamma Down')
        ax_d.set_xlabel('True Neutrino Energy [GeV]')
        ax_d.set_ylabel(r'Rel. $\Delta$ Events per bin [%]')
        ax_d.set_title(f'{opt}, {label}')
        ax_d.set_xscale('log')
        ax_d.legend()
        fig_d.tight_layout()
        fig_d.savefig(os.path.join(plot_dir, f'true_e_diff_{i}_{opt}.pdf'))
        plt.close(fig_d)
        
        i += 1

def main():
    gamma = True
    ##open central value, core down, all down, core up, and all up

    plot_dir = os.path.join(config.install, 'earth_ana/plots')
    df_path  = os.path.join(config.inner, 'dataframes')

    ##determine the norm list from the perturbed samples, not the default
    ##but check it's present in the default for comparison
    normList = getNormList(os.path.join(df_path, 'li_total.hdf5'),
                           os.path.join(df_path, 'li_core_up_total.hdf5'))
    print(f'Testing for {normList}')

    fig_n_t, ax_n_t = plt.subplots()
    fig_n_c, ax_n_c = plt.subplots()


    for i, earth in enumerate(['core_up', 'all_up', 'core_down', 'all_down', 'normal']):
        if earth == 'normal':
            print('-- Opening nominal Earth model case --')
            df = pd.read_hdf(os.path.join(df_path, 'li_total.hdf5'))
        else:
            print(f'-- Opening {earth} Earth model case --')
            df = pd.read_hdf(os.path.join(df_path, f'li_{earth}_total.hdf5'))
        df = remove_bad_reco(df)
        df = remove_gr(df)
        if earth == 'normal':
            df_n  = df
        elif earth == 'core_up':
            df_u  = df
        elif earth == 'core_down':
            df_d  = df
        elif earth == 'all_up':
            df_uu = df
        elif earth == 'all_down':
            df_dd = df
        else:
            raise NameError(f'string for {earth} is wrong!')


        for opt in ['track', 'cascade']:
            w_atmo, w_astro, w_total = handle_weights(df, opt, normList, earth, ax_n_t, ax_n_c)
            if opt == 'track':
                if earth == 'normal':
                    w_atmo_t_n  = w_atmo
                    w_astro_t_n = w_astro
                    w_total_t_n = w_total
                if earth == 'core_up':
                    w_atmo_t_u  = w_atmo
                    w_astro_t_u = w_astro
                    w_total_t_u = w_total
                if earth == 'core_down':
                    w_atmo_t_d  = w_atmo
                    w_astro_t_d = w_astro
                    w_total_t_d = w_total
                if earth == 'all_up':
                    w_atmo_t_uu  = w_atmo
                    w_astro_t_uu = w_astro
                    w_total_t_uu = w_total
                if earth == 'all_down':
                    w_atmo_t_dd  = w_atmo
                    w_astro_t_dd = w_astro
                    w_total_t_dd = w_total
            if opt == 'cascade':
                if earth == 'normal':
                    w_atmo_c_n  = w_atmo
                    w_astro_c_n = w_astro
                    w_total_c_n = w_total
                if earth == 'core_up':
                    w_atmo_c_u  = w_atmo
                    w_astro_c_u = w_astro
                    w_total_c_u = w_total
                if earth == 'core_down':
                    w_atmo_c_d  = w_atmo
                    w_astro_c_d = w_astro
                    w_total_c_d = w_total
                if earth == 'all_up':
                    w_atmo_c_uu  = w_atmo
                    w_astro_c_uu = w_astro
                    w_total_c_uu = w_total
                if earth == 'all_down':
                    w_atmo_c_dd  = w_atmo
                    w_astro_c_dd = w_astro
                    w_total_c_dd = w_total

    if gamma == True:
        df = pd.read_hdf(os.path.join(df_path, 'li_total_modGamma_mod.hdf5'))
        df = remove_bad_reco(df)
        df = remove_gr(df)
        w_atmo_t_gu, w_astro_t_gu, w_total_t_gu = handle_weights(df, 'track', 
                normList, 'Gamma Up', ax_n_t, ax_n_c)
        w_atmo_c_gu, w_astro_c_gu, w_total_c_gu = handle_weights(df, 'cascade', 
                normList, 'Gamma Down', ax_n_t, ax_n_c)

    ax_n_t.set_xlabel('Cross Section Normalisation')
    ax_n_t.set_ylabel('Num. Events / 1 yr')
    ax_n_t.legend()
    fig_n_t.tight_layout()
    fig_n_t.savefig(os.path.join(plot_dir, 'event_rates_track.pdf'))
    plt.close(fig_n_t)
    ax_n_c.set_xlabel('Cross Section Normalisation')
    ax_n_c.set_ylabel('Num. Events / 1 yr')
    ax_n_c.legend()
    fig_n_c.tight_layout()
    fig_n_c.savefig(os.path.join(plot_dir, 'event_rates_cascade.pdf'))
    plt.close(fig_n_c)

    ##show the difference from the model shift
    w_total_t_nu  = (np.array(w_total_t_u)  - np.array(w_total_t_n)) / np.array(w_total_t_n)
    w_total_t_nd  = (np.array(w_total_t_d)  - np.array(w_total_t_n)) / np.array(w_total_t_n)
    w_total_c_nu  = (np.array(w_total_c_u)  - np.array(w_total_c_n)) / np.array(w_total_c_n)
    w_total_c_nd  = (np.array(w_total_c_d)  - np.array(w_total_c_n)) / np.array(w_total_c_n)
    w_total_t_nuu = (np.array(w_total_t_uu) - np.array(w_total_t_n)) / np.array(w_total_t_n)
    w_total_t_ndd = (np.array(w_total_t_dd) - np.array(w_total_t_n)) / np.array(w_total_t_n)
    w_total_c_nuu = (np.array(w_total_c_uu) - np.array(w_total_c_n)) / np.array(w_total_c_n)
    w_total_c_ndd = (np.array(w_total_c_dd) - np.array(w_total_c_n)) / np.array(w_total_c_n)

    weightsList_t = [w_total_n, w_total_t_nu, w_total_t_nd, w_total_t_uu, w_total_t_dd]
    weightsList_c = [w_total_n, w_total_c_nu, w_total_c_nd, w_total_c_uu, w_total_c_dd]
    dfList = [df_n, df_u, df_d, df_uu, df_dd]
    

    frac_plot('track',   normList, weightsList_t, plot_dir)
    frac_plot('cascade', normList, weightsList_c, plot_dir)
    true_plot('track',   normList, dfList, weightsList_t, plot_dir)
    true_plot('cascade', normList, dfList, weightsList_c, plot_dir)

    ##for the gamma comparison case
    if gamma == True
        weightsList_t.append(w_total_t_gu)
        weightsList_t.append(w_total_t_gd)
        weightsList_t.append(w_total_c_gu)
        weightsList_t.append(w_total_c_gd)
        dfList.append(df_gu)
        dfList.append(df_gd)

        true_gamma_plot('track',   normList, dfList, weightsList_t, plot_dir)
        true_gamma_plot('cascade', normList, dfList, weightsList_c, plot_dir)

    print('Done')

if __name__ == "__main__":
    main()
##end
