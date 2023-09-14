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
from weighting.create_weight_df_data import calc_live_time
from configs.config import config

##compare monte carlo to data (or MC to MC)
def ratio_info(li, w_li, ng, w_ng, data, binning):
    binned_li = np.histogram(li, bins=binning, weights=w_li)[0]
    binned_ng = np.histogram(ng, bins=binning, weights=w_ng)[0]
    r_li = binned_li / data
    r_li_err = r_li * np.sqrt(np.sqrt(binned_li)**2 / binned_li**2 + np.sqrt(data)**2 / data**2)
    r_ng = binned_ng / data
    r_ng_err = r_ng * np.sqrt(np.sqrt(binned_ng)**2 / binned_ng**2 + np.sqrt(data)**2 / data**2)

    print(r_li)
    return r_li, r_li_err, r_ng, r_ng_err
        
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

def compare_reco(df_data_c, df_data_t, df_li_c, df_li_t, norm=False, f_type='all', 
                 ignore_nugen=True, best_fit=True): 

    ##return either yearly live time, or just total
    ##figure it out from the dataframe
    d_live_time_t, years_t = calc_live_time(df_data_t)
    d_live_time_c, years_c = calc_live_time(df_data_c)

    if years_t != years_c:
        raise ValueError(f'Years for tracks and cascades are different! \
                           You need to be careful when plotting the year-by-year!')

    ##if d_live_time is length 1 - only makes one plot (all years)
    ##if d_live_time is > 1 - makes plots for all years

    ##TODO - create a shared canvas to compare year by year

    for this_year, lt_t, lt_c in zip(years_t, d_live_time_t, d_live_time_c):
        w_info = get_weights_info(df_data_c, df_data_t, df_li_c, df_li_t,
                        norm=norm, f_type=f_type, year=this_year,
                        live_time_c=lt_c, live_time_t=lt_t,
                        best_fit=best_fit)
        y_livetime_c = lt_c / (3600 * 24 * 365)
        y_livetime_t = lt_t / (3600 * 24 * 365)
        ##append to the tuple
        w_info = w_info + (y_livetime_c, y_livetime_t)

        print(f'-- Plotting {this_year} --')
        prepare_plotting(df_data_c, df_data_t, df_li_c, df_li_t, 
                     norm=norm, f_type=f_type, w_info=w_info,
                     ignore_nugen=ignore_nugen, best_fit=best_fit,
                     year=this_year)

    ##if more than 1 year, also make the total plot
    if len(d_live_time_t) > 1 and len(d_live_time_c) > 1:
        lt_c = np.sum(d_live_time_c)
        lt_t = np.sum(d_live_time_t)
        w_info = get_weights_info(df_data_c, df_data_t, df_li_c, df_li_t,
                        norm=norm, f_type=f_type, year='total',
                        live_time_c=lt_c, live_time_t=lt_t,
                        best_fit=best_fit)
        y_livetime_c = lt_c / (3600 * 24 * 365)
        y_livetime_t = lt_t / (3600 * 24 * 365)
        w_info = w_info + (y_livetime_c, y_livetime_t)
        prepare_plotting(df_data_c, df_data_t, df_li_c, df_li_t, 
                     norm=norm, f_type=f_type, w_info=w_info,
                     ignore_nugen=ignore_nugen, best_fit=best_fit,
                     year='total')

        year_by_year_plot(df_data_c, df_data_t, df_li_c, df_li_t,
                          w_info=w_info, best_fit=best_fit)

    
def get_weights_info(df_data_c, df_data_t, df_li_c, df_li_t, norm=False,
                     f_type='all', year='total', 
                     live_time_c=0, live_time_t=0, best_fit=True, ignore_veto=False):

    if year != 'total':
        year = int(year)
        df_data_t = df_data_t[df_data_t.year.values == year]
        df_data_c = df_data_c[df_data_c.year.values == year]

    if best_fit == True:
        track_atmo_norm = config.track_atmo_norm
        cascade_atmo_norm = config.cascade_atmo_norm
    else:
        track_atmo_norm = 1
        cascade_atmo_norm = 1

    n_data_c = len(df_data_c.reco_energy.values)
    n_data_t = len(df_data_t.reco_energy.values)

    print(f'- Live Time in Data [s] -: Tracks   = {live_time_t}')
    print(f'- Live Time in Data [s] -: Cascades = {live_time_c}')
    print('=== Data Num Cascades ===')
    print(f'{n_data_c} events, {n_data_c/live_time_t} event / live time')
    print('=== Data Num Tracks ===')
    print(f'{n_data_t} events, {n_data_t/live_time_c} event / live time')
    print('='*10)

    ##scale lepton injector to data
    trackLiveTimeRatioLI   = live_time_t / df_li_t.LiveTime.values[0]
    cascadeLiveTimeRatioLI = live_time_c / df_li_c.LiveTime.values[0]

    ##if atmo and astro weights are together
    if f_type == 'all':
        w_li_c = df_li_c['weight1.0'].values * cascadeLiveTimeRatioLI
        w_li_t = df_li_t['weight1.0'].values * trackLiveTimeRatioLI
        n_li_c = np.sum(w_li_c) #* cascadeLiveTimeRatioLI
        n_li_t = np.sum(w_li_t) #* trackLiveTimeRatioLI
    
        w_li_c_atmo = 0
        w_li_t_atmo = 0
        w_li_c_astro = 0
        w_li_t_astro = 0


    ##if atmo and astro weights are separate
    if f_type == 'separate':
        if ignore_veto == False:
            w_li_c_atmo  = (df_li_c['weight1.0_atmo'].values * 
                        cascadeLiveTimeRatioLI * cascade_atmo_norm * df_li_c['veto_pf'].values)
        if ignore_veto == True:
            w_li_c_atmo  = (df_li_c['weight1.0_atmo'].values * 
                        cascadeLiveTimeRatioLI * cascade_atmo_norm)
        w_li_t_atmo  = (df_li_t['weight1.0_atmo'].values * 
                        trackLiveTimeRatioLI * track_atmo_norm)
        w_li_c_astro = df_li_c['weight1.0_astro'].values * cascadeLiveTimeRatioLI
        w_li_t_astro = df_li_t['weight1.0_astro'].values * trackLiveTimeRatioLI
        w_li_c = w_li_c_atmo + w_li_c_astro
        w_li_t = w_li_t_atmo + w_li_t_astro

        n_li_c_atmo  = np.sum(w_li_c_atmo)  #* cascadeLiveTimeRatioLI * cascade_atmo_norm 
        n_li_t_atmo  = np.sum(w_li_t_atmo)  #* trackLiveTimeRatioLI   * track_atmo_norm
        n_li_c_astro = np.sum(w_li_c_astro) #* cascadeLiveTimeRatioLI
        n_li_t_astro = np.sum(w_li_t_astro) #* trackLiveTimeRatioLI
        n_li_c = n_li_c_atmo + n_li_c_astro
        n_li_t = n_li_t_atmo + n_li_t_astro

   
    if norm == True:
        w_li_c / n_li_c
        w_li_t / n_li_t
    
    ##total number of events in each MC sample
    ##compared to total number in data
    print(f'Monte Carlo Cascades: {n_li_c}, Data Cascades: {n_data_c}')
    print(f'Ratio MC/Data = {(n_li_c / n_data_c):.2f}')
    print(f'Monte Carlo Tracks  : {n_li_t}, Data Cascades: {n_data_t}')
    print(f'Ratio MC/Data = {(n_li_t / n_data_t):.2f}')

    ##with 10 yr burn sample, rough estimate of mis-match
    #estimated
    #fake_track_scale = 1.3
    #fake_cascade_scale = 1.65
    ##from ratio of events
    fake_track_scale = n_li_t / n_data_t
    fake_cascade_scale = n_li_c / n_data_c
    
    w_li_c       = w_li_c       * (1/fake_cascade_scale)
    w_li_c_atmo  = w_li_c_atmo  * (1/fake_cascade_scale)
    w_li_c_astro = w_li_c_astro * (1/fake_cascade_scale)
    w_li_t       = w_li_t       * (1/fake_track_scale)
    w_li_t_atmo  = w_li_t_atmo  * (1/fake_track_scale)
    w_li_t_astro = w_li_t_astro * (1/fake_track_scale)

    cascadeLiveTimeRatioLI = cascadeLiveTimeRatioLI * (1/fake_cascade_scale)
    trackLiveTimeRatioLI   = trackLiveTimeRatioLI   * (1/fake_track_scale)

    print('='*20)
    print('WARNING - FAKE SCALING IS APPLIED!!!')
    print(f'Tracks: {fake_track_scale}')
    print(f'Cascade: {fake_cascade_scale}')
    print('='*20)

    return w_li_c, w_li_c_atmo, w_li_c_astro, cascadeLiveTimeRatioLI, cascade_atmo_norm, \
           w_li_t, w_li_t_atmo, w_li_t_astro, trackLiveTimeRatioLI, track_atmo_norm
        

def year_by_year_plot(df_data_c, df_data_t, df_li_c, df_li_t, w_info, best_fit=False):
    ##unpack weight info
    w_li_c, w_li_c_atmo, w_li_c_astro, cascadeLiveTimeRatioLI, cascade_atmo_norm, \
        w_li_t, w_li_t_atmo, w_li_t_astro, trackLiveTimeRatioLI, track_atmo_norm, \
            y_livetime_c, y_livetime_t = w_info

    fig1t, ax1t = plt.subplots()
    fig2t, ax2t = plt.subplots()
    fig1c, ax1c = plt.subplots()
    fig2c, ax2c = plt.subplots()
    binning  = np.logspace(2, 8, 28)
    bin_mids = (binning[1:] + binning[:-1])/2
    zbinning = np.linspace(-1, 1, 15)
    zbin_mids = (zbinning[1:] + zbinning[:-1])/2

    e_li_c = df_li_c.reco_energy.values
    e_li_t = df_li_t.reco_energy.values
    z_li_c = np.cos(df_li_c.reco_zenith.values)
    z_li_t = np.cos(df_li_t.reco_zenith.values)
    #ax2t.hist(e_li_t, bins=binning, weights=w_li_t, histtype='step', density=True,
    #          label=f'LepIn Track', color='royalblue')
    #ax2c.hist(e_li_c, bins=binning, weights=w_li_c, histtype='step', density=True,
    #          label=f'LepIn Cascade', color='royalblue')
    
    years = df_data_c.year.unique()
    for year in years:
        year = int(year)
        _df_data_t = df_data_t[df_data_t.year.values == year]
        _df_data_c = df_data_c[df_data_c.year.values == year]
        e_data_c = _df_data_c.reco_energy.values
        e_data_t = _df_data_t.reco_energy.values
        z_data_c = np.cos(_df_data_c.reco_zenith.values)
        z_data_t = np.cos(_df_data_t.reco_zenith.values)
        d_info_c = np.histogram(e_data_c, bins=binning)
        d_vals_c = d_info_c[0]
        d_errs_c = np.sqrt(d_vals_c)
        d_info_t = np.histogram(e_data_t, bins=binning)
        d_vals_t = d_info_t[0]
        d_errs_t = np.sqrt(d_vals_t)
        
        d_zinfo_c = np.histogram(z_data_c, bins=zbinning)
        d_zvals_c = d_zinfo_c[0]
        d_zerrs_c = np.sqrt(d_zvals_c)
        d_zinfo_t = np.histogram(z_data_t, bins=zbinning)
        d_zvals_t = d_zinfo_t[0]
        d_zerrs_t = np.sqrt(d_zvals_t)
       
        ##no norm 
        ax1t.errorbar(bin_mids, d_vals_t, yerr=d_errs_t, 
                      xerr=np.diff(binning)/2, linewidth=0, alpha=0.8,
                      marker='o', elinewidth=2, capsize=2, label=f'{year}: {np.sum(d_vals_t)}')
        ax1c.errorbar(bin_mids, d_vals_c, yerr=d_errs_c, 
                      xerr=np.diff(binning)/2, linewidth=0, alpha=0.8,
                      marker='o', elinewidth=2, capsize=2, label=f'{year}: {np.sum(d_vals_c)}')
        ##norm to check shape
        _d_vals_t = d_vals_t / _df_data_t.live_time.values[0]
        _d_vals_c = d_vals_c / _df_data_c.live_time.values[0]
        ax2t.errorbar(bin_mids, _d_vals_t, 
                      #linewidth=0, 
                      alpha=0.8,
                      marker='o', elinewidth=2, capsize=2,
                      label=f'{year}: {1000*np.sum(d_vals_t)/_df_data_t.live_time.values[0]:.2f}')
        ax2c.errorbar(bin_mids, _d_vals_c, 
                      #linewidth=0, 
                      alpha=0.8,
                      marker='o', elinewidth=2, capsize=2,
                      label=f'{year}: {1e6*np.sum(d_vals_c)/_df_data_c.live_time.values[0]:.2f}')

    ax1t.legend(title='N Events')
    ax1t.set_xscale('log')
    ax1t.set_xlabel('(Deposited) Reco Energy [GeV]')
    ax1t.set_ylabel('Entries')
    fig1t.tight_layout()
    saveHelper(fig1t, 'year_by_year_track', 'total', 
                subbin='', norm=False, dpi=None, ignore_close=True)
    ax1t.set_xlim(100, 1e6)
    saveHelper(fig1t, 'year_by_year_track_zoom', 'total', 
                subbin='', norm=False, dpi=None, ignore_close=False)

    ax2t.legend(title='Rate [mHz]')
    ax2t.set_xscale('log')
    ax2t.set_xlabel('(Deposited) Reco Energy [GeV]')
    ax2t.set_ylabel('Rate [Hz]')
    fig2t.tight_layout()
    saveHelper(fig2t, 'year_by_year_track', 'total', 
                subbin='', norm=True, dpi=None, ignore_close=True)
    ax2t.set_xlim(100, 1e6)
    saveHelper(fig2t, 'year_by_year_track_zoom', 'total', 
                subbin='', norm=True, dpi=None, ignore_close=False)
    
    ax1c.legend(title='N Events')
    ax1c.set_xscale('log')
    ax1c.set_xlabel('(Deposited) Reco Energy [GeV]')
    ax1c.set_ylabel('Entries')
    fig1c.tight_layout()
    saveHelper(fig1c, 'year_by_year_cascade', 'total', 
                subbin='', norm=False, dpi=None, ignore_close=True)
    ax1c.set_xlim(100, 1e6)
    saveHelper(fig1c, 'year_by_year_cascade_zoom', 'total', 
                subbin='', norm=False, dpi=None, ignore_close=False)

    ax2c.legend(title=r'Rate [$\mu$Hz]')
    ax2c.set_xscale('log')
    ax2c.set_xlabel('(Deposited) Reco Energy [GeV]')
    ax2c.set_ylabel('Rate [Hz]')
    fig2c.tight_layout()
    saveHelper(fig2c, 'year_by_year_cascade', 'total', 
                subbin='', norm=True, dpi=None, ignore_close=True)
    ax2c.set_xlim(100, 1e6)
    saveHelper(fig2c, 'year_by_year_cascade_zoom', 'total', 
                subbin='', norm=True, dpi=None, ignore_close=False)


def prepare_plotting(df_data_c, df_data_t, df_li_c, df_li_t, 
                     norm=False, f_type='all', w_info=None,
                     ignore_nugen=True, best_fit=True, year='total'):
    if w_info == None:
        raise NotImplementedError('w_info must be prepared, use get_weights_info first!')

    w_li_c, w_li_c_atmo, w_li_c_astro, cascadeLiveTimeRatioLI, cascade_atmo_norm, \
        w_li_t, w_li_t_atmo, w_li_t_astro, trackLiveTimeRatioLI, track_atmo_norm, \
            y_livetime_c, y_livetime_t = w_info

    if year != 'total':
        year = int(year)
        df_data_t = df_data_t[df_data_t.year.values == year]
        df_data_c = df_data_c[df_data_c.year.values == year]

    ##get reco energy and zenith info
    e_li_c = df_li_c.reco_energy.values
    e_li_t = df_li_t.reco_energy.values
    e_data_c = df_data_c.reco_energy.values
    e_data_t = df_data_t.reco_energy.values
    
    z_li_c = np.cos(df_li_c.reco_zenith.values)
    z_li_t = np.cos(df_li_t.reco_zenith.values)
    z_data_c = np.cos(df_data_c.reco_zenith.values)
    z_data_t = np.cos(df_data_t.reco_zenith.values)

    ##start plotting
    binning  = np.logspace(2, 8, 28)
    ##original settings
    zbinning_t = np.linspace(-1, 1, 15)
    ##settings matching LLH
    zbinning_c = np.linspace(-1, 1, 15)

    ##start binning data
    bin_mids = (binning[1:] + binning[:-1])/2
    d_info_c = np.histogram(e_data_c, bins=binning)
    d_vals_c = d_info_c[0]
    d_errs_c = np.sqrt(d_vals_c)
    d_info_t = np.histogram(e_data_t, bins=binning)
    d_vals_t = d_info_t[0]
    d_errs_t = np.sqrt(d_vals_t)
    
    zbin_mids_c = (zbinning_c[1:] + zbinning_c[:-1])/2
    d_zinfo_c = np.histogram(z_data_c, bins=zbinning_c)
    d_zvals_c = d_zinfo_c[0]
    d_zerrs_c = np.sqrt(d_zvals_c)
    zbin_mids_t = (zbinning_t[1:] + zbinning_t[:-1])/2
    d_zinfo_t = np.histogram(z_data_t, bins=zbinning_t)
    d_zvals_t = d_zinfo_t[0]
    d_zerrs_t = np.sqrt(d_zvals_t)

    reco_plots(e_li_c, z_li_c, w_li_c, d_vals_c, d_errs_c, d_zvals_c, d_zerrs_c,
               binning, zbinning_c, bin_mids, zbin_mids_c,
               y_livetime_c, opt='cascade', norm=norm, ignore_nugen=ignore_nugen,
               year=year)
    reco_plots(e_li_t, z_li_t, w_li_t, d_vals_t, d_errs_t, d_zvals_t, d_zerrs_t,
               binning, zbinning_t, bin_mids, zbin_mids_t,
               y_livetime_t, opt='track', norm=norm, ignore_nugen=ignore_nugen,
               year=year)
    
    ##checking mc vs data for depth
    for df_data, df_li, label, _w in zip([df_data_c, df_data_t], 
                                     [df_li_c, df_li_t], 
                                     ['cascade', 'track'],
                                     [w_li_c, w_li_t]):
        
        fig0, ax0 = plt.subplots()
        if label == 'track':
            _dbinning = np.linspace(-600, 500, 100)
            ax0.set_ylabel(f'Events per {y_livetime_t:.2f} yrs (IC86)')
        if label == 'cascade':        
            _dbinning = np.linspace(-450, 400, 60)
            ax0.set_ylabel(f'Events per {y_livetime_c:.2f} yrs (IC86)')
        _bin_mids = (_dbinning[1:] + _dbinning[:-1])/2
        _vals = df_data.reco_z.values
        dz_info_c = np.histogram(_vals, bins=_dbinning)
        dz_vals = dz_info_c[0]
        dz_errs = np.sqrt(dz_vals)
        ax0.hist(df_li.reco_z.values, _dbinning, weights=_w, histtype='step', color='royalblue')
        ax0.errorbar(_bin_mids, dz_vals, yerr=dz_errs, xerr=np.diff(_dbinning)/2, 
                 linewidth=0, color='black',
                 marker='o', elinewidth=2, capsize=2, label=f'Data {label}')

        ax0.set_xlabel('Reco Z Position [m]')
        ax0.set_title(label)
        saveHelper(fig0, f'reco_z_data_mc_{label}', year)

        fig0a, ax0a = plt.subplots()
        ax0a.plot(df_data.reco_z.values, np.cos(df_data.reco_zenith.values), 'o', linewidth=0)
        ax0a.set_xlabel('Reco Z Position [m]')
        ax0a.set_ylabel(r'Reco cos($\theta_z$)')
        ax0a.set_title(label)
        saveHelper(fig0a, f'reco_z_zenith_data_{label}', year)

    ##if flux weight information is available for atmo and astro   
    if f_type == 'separate': 
        separate_reco_plots(e_li_c, z_li_c, w_li_c_atmo, w_li_c_astro, 
               d_vals_c, d_errs_c, d_zvals_c, d_zerrs_c,
               binning, zbinning_c, bin_mids, zbin_mids_c,
               y_livetime_c, opt='cascade', norm=norm, year=year, skip_data=True)
        separate_reco_plots(e_li_t, z_li_t, w_li_t_atmo, w_li_t_astro, 
               d_vals_t, d_errs_t, d_zvals_t, d_zerrs_t,
               binning, zbinning_t, bin_mids, zbin_mids_t,
               y_livetime_t, opt='track', norm=norm, year=year, skip_data=True)
        separate_reco_plots(e_li_c, z_li_c, w_li_c_atmo, w_li_c_astro, 
               d_vals_c, d_errs_c, d_zvals_c, d_zerrs_c,
               binning, zbinning_c, bin_mids, zbin_mids_c,
               y_livetime_c, opt='cascade', norm=norm, year=year)
        separate_reco_plots(e_li_t, z_li_t, w_li_t_atmo, w_li_t_astro, 
               d_vals_t, d_errs_t, d_zvals_t, d_zerrs_t,
               binning, zbinning_t, bin_mids, zbin_mids_t,
               y_livetime_t, opt='track', norm=norm, year=year)

        ##try to find where there is excess in data over MC
        #e_range_list = [1e2, 1e3, 5e3, 1e4, 5e8]
        e_range_list = [1e2, 5e3, 1e9]
        for i in range(len(e_range_list)-1):
            #_df_li_c = df_li_c[(e_range_list[i] < np.log10(df_li_c.reco_energy)) & (np.log10(df_li_c.reco_energy)  <= e_range_list[i+1])]
            _df_li_c = df_li_c[(e_range_list[i] < df_li_c.reco_energy) & (df_li_c.reco_energy  <= e_range_list[i+1])]
            e_li_c = _df_li_c.reco_energy.values
            z_li_c = np.cos(_df_li_c.reco_zenith.values)
            w_li_c_atmo  = (_df_li_c['weight1.0_atmo'].values * 
                            cascadeLiveTimeRatioLI * cascade_atmo_norm)
            w_li_c_astro = _df_li_c['weight1.0_astro'].values * cascadeLiveTimeRatioLI
            
            _df_data_c = df_data_c[(e_range_list[i] < df_data_c.reco_energy) & 
                                   (df_data_c.reco_energy <= e_range_list[i+1])]
            e_data_c = _df_data_c.reco_energy.values
            z_data_c = np.cos(_df_data_c.reco_zenith.values)
            d_info_c = np.histogram(e_data_c, bins=binning)
            d_vals_c = d_info_c[0]
            d_errs_c = np.sqrt(d_vals_c)
            d_zinfo_c = np.histogram(z_data_c, bins=zbinning_c)
            d_zvals_c = d_zinfo_c[0]
            d_zerrs_c = np.sqrt(d_zvals_c)
            subbin_str = f'bin{i}'
            separate_reco_plots(e_li_c, z_li_c, w_li_c_atmo, w_li_c_astro, 
                                d_vals_c, d_errs_c, d_zvals_c, d_zerrs_c,
                                binning, zbinning_c, bin_mids, zbin_mids_c, y_livetime_c,
                                opt='cascade', norm=norm, subbin=subbin_str, year=year)


def separate_reco_plots(e_li, z_li, w_li_atmo, w_li_astro, d_vals, d_errs, d_zvals, d_zerrs,
                        binning, zbinning, bin_mids, zbin_mids, 
                        y_livetime, opt, norm, subbin='', year='total', skip_data=False):
    if opt == 'track':
        label = 'Track'
    if opt == 'cascade':
        label = 'Cascade'

    w_li = w_li_atmo + w_li_astro

    print(f'-- Reco Plots {opt} --')
    if subbin != '':
        print(f'Slice: {subbin}')
    print(f'Atmo  {np.sum(w_li_atmo)}')
    print(f'Astro {np.sum(w_li_astro)}')
    print(f'Total {np.sum(w_li)}')
    print(f'Data  {np.sum(d_vals)}')

    #### energy plots ####

    ##plot ratios if not normalised
    if not norm and not skip_data:
        fig1 = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[4, 1]) 
        ax1 = plt.subplot(gs[0, :]) 
        ax1_r = plt.subplot(gs[1, :], sharex=ax1) 
        ax1.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')
    else:
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('(Deposited) Reco Energy [GeV]')
        if norm == True:
            ax1.set_ylabel('Events (Normalised)')
        else:
            ax1.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')

    ### top plot ###
    ax1.hist(e_li, bins=binning, weights=w_li,       histtype='step', label=f'Total', color='royalblue')
    ax1.hist(e_li, bins=binning, weights=w_li_atmo,  histtype='step', label=f'Atmo',  color='goldenrod')
    ax1.hist(e_li, bins=binning, weights=w_li_astro, histtype='step', label=f'Astro', color='firebrick')
    if skip_data == False:
        ax1.errorbar(bin_mids, d_vals, yerr=d_errs, xerr=np.diff(binning)/2, linewidth=0, color='black',
                 marker='o', elinewidth=2, capsize=2, label=f'Data {label}')
    ax1.legend(title=f'{label}')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ### bottom plot ###
    if not norm and not skip_data:
        r_li, r_li_err, _, _ = ratio_info(e_li, w_li, 0 , 0, d_vals, binning)
        ax1_r.set_xscale('log')
        ax1_r.grid()
        ax1_r.errorbar(bin_mids, r_li, yerr=r_li_err, xerr=np.diff(binning)/2, fmt='o',
                       label=r'$R_{LI/DATA}$', color='royalblue')
        ax1_r.hlines(1.0, bin_mids[0], bin_mids[-1], linestyle='--', color='goldenrod')
        ax1_r.legend()
        ax1_r.set_xlabel('(Deposited) Reco Energy [GeV]')
        ax1_r.set_ylabel('Ratios')
    if skip_data == True:   
        saveHelper(fig1, f'simple_reco_energy_no_data_{opt}', year, subbin, norm)
    else:   
        saveHelper(fig1, f'simple_reco_energy_{opt}', year, subbin, norm)

    #### cos(z) plots ####
    
    ##plot ratios if not normalised
    if not norm and not skip_data:
        fig1z = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[4, 1]) 
        ax1z = plt.subplot(gs[0, :]) 
        ax1z_r = plt.subplot(gs[1, :], sharex=ax1z) 
        ax1z_r.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')
        ax1z_r.set_xlabel(r'Reco cos($\theta_{z}$)')
    else:
        fig1z, ax1z = plt.subplots()
        ax1z.set_xlabel(r'Reco cos($\theta_{z}$)')
        if norm == True:
            ax1z.set_ylabel('Events (Normalised)')
        else:
            ax1z.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')

    ### top plot ###
    ax1z.hist(z_li, bins=zbinning, weights=w_li,       histtype='step', label=f'Total', color='royalblue')
    ax1z.hist(z_li, bins=zbinning, weights=w_li_atmo,  histtype='step', label=f'Atmo',  color='goldenrod')
    ax1z.hist(z_li, bins=zbinning, weights=w_li_astro, histtype='step', label=f'Astro', color='firebrick')
    if skip_data == False:
        ax1z.errorbar(zbin_mids, d_zvals, yerr=d_zerrs, xerr=np.diff(zbinning)/2, 
                      linewidth=0, color='black',
                      marker='o', elinewidth=2, capsize=2, label=f'Data {label}')
    ax1z.legend(title=f'{label}')
    ax1z.set_yscale('log')
    ### bottom plot ###
    if not norm:
        r_li, r_li_err, _, _ = ratio_info(z_li, w_li, 0, 0, d_zvals, zbinning)
        if not skip_data:
            ax1z_r.grid()
            ax1z_r.errorbar(zbin_mids, r_li, yerr=r_li_err, xerr=np.diff(zbinning)/2, 
                            fmt='o', label=r'$R_{LI/DATA}$', color='royalblue')
            ax1z_r.hlines(1.0, zbin_mids[0], zbin_mids[-1], linestyle='--', color='goldenrod')
            ax1z_r.legend()
            ax1z_r.set_ylabel('Ratios')
        if subbin == '':
            if skip_data == True:
                saveHelper(fig1z, f'simple_reco_zenith_log_no_data_{opt}', year, subbin, 
                            norm, ignore_close=True)
            else:
                saveHelper(fig1z, f'simple_reco_zenith_log_{opt}', year, subbin, 
                            norm, ignore_close=True)
            ax1z.set_yscale('linear')

    if skip_data == True:
        saveHelper(fig1z, f'simple_reco_zenith_no_data_{opt}', year, subbin, norm)
    else:
        saveHelper(fig1z, f'simple_reco_zenith_{opt}', year, subbin, norm)

def reco_plots(e_li, z_li, w_li, d_vals, d_errs, d_zvals, d_zerrs,
               binning, zbinning, bin_mids, zbin_mids, 
               y_livetime, opt='track', norm=False, ignore_nugen=True,
               year='total'):
    if opt == 'track':
        label = 'Track'
    if opt == 'cascade':
        label = 'Cascade'

    print(f'-- Reco Plots {opt} --')
    print(f'Lepton Injector Sum Weights: {np.sum(w_li)}')

    #### energy plots ####
    ##plot ratios if not normalised
    if not norm:
        fig1 = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[4, 1]) 
        ax1 = plt.subplot(gs[0, :]) 
        ax1_r = plt.subplot(gs[1, :], sharex=ax1) 
        ax1.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')
    else:
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('(Deposited) Reco Energy [GeV]')
        ax1.set_ylabel('Events (Normalised)')

    ### top plot ###
    ax1.hist(e_li, bins=binning, weights=w_li, histtype='step', label=f'LepIn {label}', color='royalblue')
    ax1.errorbar(bin_mids, d_vals, yerr=d_errs, xerr=np.diff(binning)/2, linewidth=0, color='black',
                 marker='o', elinewidth=2, capsize=2, label=f'Data {label}')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ### bottom plot ###
    if not norm:
        r_li, r_li_err, _, _ = ratio_info(e_li, w_li, 0, 0, d_vals, binning)
        ax1_r.set_xscale('log')
        ax1_r.grid()
        ax1_r.errorbar(bin_mids, r_li, yerr=r_li_err, xerr=np.diff(binning)/2, fmt='o', label=r'$R_{LI/DATA}$', color='royalblue')
        ax1_r.hlines(1.0, bin_mids[0], bin_mids[-1], linestyle='--', color='goldenrod')
        ax1_r.legend()
        ax1_r.set_xlabel('(Deposited) Reco Energy [GeV]')
        ax1_r.set_ylabel('Ratios')
    saveHelper(fig1, f'compare_reco_energy_{opt}', year, norm=norm)

    #### cos(z) plots ####
    
    ##plot ratios if not normalised
    if not norm:
        fig1z = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[4, 1]) 
        ax1z = plt.subplot(gs[0, :]) 
        ax1z_r = plt.subplot(gs[1, :], sharex=ax1z) 
        ax1z.set_ylabel(f'Events per {y_livetime:.2f} yrs (IC86)')
        ax1z_r.set_xlabel(r'Reco cos($\theta_{z}$)')
    else:
        fig1z, ax1z = plt.subplots()
        ax1z.set_xlabel(r'Reco cos($\theta_{z}$)')
        ax1z.set_ylabel('Events (Normalised)')

    ### top plot ###
    ax1z.hist(z_li, bins=zbinning, weights=w_li, histtype='step', label=f'LepIn {label}', color='royalblue')
    ax1z.errorbar(zbin_mids, d_zvals, yerr=d_zerrs, xerr=np.diff(zbinning)/2, linewidth=0, color='black',
                 marker='o', elinewidth=2, capsize=2, label=f'Data {label}')
    ax1z.legend()
    ax1z.set_yscale('log')
    ### bottom plot ###
    if not norm:
        r_li, r_li_err, _, _ = ratio_info(z_li, w_li, 0, 0, d_zvals, zbinning)
        ax1z_r.grid()
        ax1z_r.errorbar(zbin_mids, r_li, yerr=r_li_err, xerr=np.diff(zbinning)/2, fmt='o', label=r'$R_{LI/DATA}$', color='royalblue')
        ax1z_r.hlines(1.0, zbin_mids[0], zbin_mids[-1], linestyle='--', color='goldenrod')
        ax1z_r.legend()
        ax1z_r.set_ylabel('Ratios')
    saveHelper(fig1z, f'compare_reco_zenith_{opt}', year, norm=norm)

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
    if f_type == 'all':
        w_li = df_li['weight1.0'].values * (yr_const / df_li.LiveTime.values[0])
    if f_type == 'separate':
        w_li_atmo  = df_li['weight1.0_atmo'].values * (yr_const / df_li.LiveTime.values[0]) * atmo_norm
        w_li_astro = df_li['weight1.0_astro'].values * (yr_const / df_li.LiveTime.values[0])
        w_li = w_li_atmo + w_li_astro

    if do_norms == True:
        wd_li_atmo  = df_li['weight0.95_atmo'].values * (yr_const / df_li.LiveTime.values[0]) * atmo_norm
        wd_li_astro = df_li['weight0.95_astro'].values * (yr_const / df_li.LiveTime.values[0])
        wd_li = wd_li_atmo + wd_li_astro
        wu_li_atmo  = df_li['weight1.05_atmo'].values * (yr_const / df_li.LiveTime.values[0]) * atmo_norm
        wu_li_astro = df_li['weight1.05_astro'].values * (yr_const / df_li.LiveTime.values[0])
        wu_li = wu_li_atmo + wu_li_astro

    print(f'-- Truth Plots {opt} --')
    print(f'Lepton Injector : {np.sum(w_li)}')

    binning  = np.logspace(2, 8, 14)
    zbinning = np.linspace(-1, 1, 10)
    bin_mids = (binning[1:] + binning[:-1])/2
    zbin_mids = (zbinning[1:] + zbinning[:-1])/2

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('True Neutrino Energy [GeV]')
    ax1.hist(e_li, bins=binning, weights=w_li,        histtype='step', label=f'Total', color='royalblue')
    ax1.hist(e_li, bins=binning, weights=w_li_atmo,   histtype='step', label=f'Atmo',  color='goldenrod')
    ax1.hist(e_li, bins=binning, weights=w_li_astro,  histtype='step', label=f'Astro', color='firebrick')
    if do_norms == True:
        ax1.hist(e_li, bins=binning, weights=wd_li_atmo,  histtype='step', label=r'Atmo 0.95$\sigma$',  linestyle='dashed', color='goldenrod')
        ax1.hist(e_li, bins=binning, weights=wd_li_astro, histtype='step', label=r'Astro 0.95$\sigma$', linestyle='dashed', color='firebrick')
        ax1.hist(e_li, bins=binning, weights=wu_li_atmo,  histtype='step', label=r'Atmo 1.05$\sigma$',  linestyle='dotted', color='goldenrod')
        ax1.hist(e_li, bins=binning, weights=wu_li_astro, histtype='step', label=r'Astro 1.05$\sigma$', linestyle='dotted', color='firebrick')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(title=f'{label}')
    ax1.set_ylabel('Events (Normalised)')
    saveHelper(fig1, f'simple_energy_{opt}', 'truth')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r'True Neutrino cos($\theta_{z}$)')
    ax3.hist(z_li, bins=zbinning, weights=w_li,        histtype='step', label=f'Total', color='royalblue')
    ax3.hist(z_li, bins=zbinning, weights=w_li_atmo,   histtype='step', label=f'Atmo',  color='goldenrod')
    ax3.hist(z_li, bins=zbinning, weights=w_li_astro,  histtype='step', label=f'Astro', color='firebrick')
    if do_norms == True:
        ax3.hist(z_li, bins=zbinning, weights=wd_li_atmo,  histtype='step', label=r'Atmo 0.95$\sigma$',  linestyle='dashed', color='goldenrod')
        ax3.hist(z_li, bins=zbinning, weights=wd_li_astro, histtype='step', label=r'Astro 0.95$\sigma$', linestyle='dashed', color='firebrick')
        ax3.hist(z_li, bins=zbinning, weights=wu_li_atmo,  histtype='step', label=r'Atmo 1.05$\sigma$',  linestyle='dotted', color='goldenrod')
        ax3.hist(z_li, bins=zbinning, weights=wu_li_astro, histtype='step', label=r'Astro 1.05$\sigma$', linestyle='dotted', color='firebrick')
    ax3.set_yscale('log')
    ax3.legend(title=f'{label}')
    ax3.set_ylabel('Events (Normalised)')
    saveHelper(fig3, f'simple_zenith_{opt}', 'truth')

def get_dfs(opt='lepton_weighter'):
    df_dir = '/data/user/chill/icetray_LWCompatible/dataframes/'
    if opt.lower() == 'nugen':
        print("Getting nuGen Monte Carlo")
        #files_c = glob('../../snowstorm_nugen_ana/monte_carlo/merged_dfs/data_cache_neo*_cascade*')
        files_c = glob('../../snowstorm_nugen_ana/data_cache_neo*_cascade_weighted*')
        files_t = glob('../../snowstorm_nugen_ana/monte_carlo/merged_dfs/data_cache*_track*')
        df_l_c = []
        df_l_t = []
        for c, t in zip(files_c, files_t):
            _df_c = pd.read_hdf(c)
            _df_t = pd.read_hdf(t)
            df_l_c.append(_df_c)
            df_l_t.append(_df_t)
        df_c = pd.concat(df_l_c)
        df_t = pd.concat(df_l_t)
        return df_c, df_t
    elif opt.lower() == 'lepton_weighter':
        print("Getting Lepton Injector Monte Carlo")
        df = pd.read_hdf(os.path.join(df_dir, 'li_total.hdf5'))
        return df
    elif opt.lower() == 'data':
        print("Getting burn sample data")
        df = pd.read_hdf(os.path.join(df_dir, 'data_total.hdf5'))
        return df
    else:
        raise NotImplementedError(f'Option {opt} to open dfs not valid')

def compare_reco_true(df, opt):
    fig0, ax0 = plt.subplots()
    binning2d = [np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)]
    ax0.hist2d(np.cos(df.nu_zenith.values), np.cos(df.reco_zenith.values), binning2d, norm=matplotlib.colors.LogNorm())
    ax0.set_xlabel(r'True cos($\theta_z$)')
    ax0.set_ylabel(r'Reco cos($\theta_z$)')
    saveHelper(fig0, f'reco_true_zenith_{opt}', 'truth')
    

    fig1, ax1 = plt.subplots()
    binning = np.linspace(-1, 1, 20)
    ax1.hist(np.cos(df.nu_zenith.values),   binning, histtype='step', color='royalblue', label='Truth')
    ax1.hist(np.cos(df.reco_zenith.values), binning, histtype='step', color='goldenrod', label='Reco')
    ax1.set_xlabel(r'cos($\theta_z$)')
    ax1.set_ylabel('Entries')
    ax1.legend()
    saveHelper(fig1, f'reco_and_true_zenith_{opt}', 'truth')

    fig1b, ax1b = plt.subplots()
    reco_z = np.cos(df.reco_zenith.values)
    true_z = np.cos(df.nu_zenith.values)
    diff = reco_z - true_z
    true_e = df.nu_energy.values
    if opt == 'cascade':
        ebinning2d = [np.logspace(2, 8, 61), np.linspace(-2, 2, 40)]
    if opt == 'track':
        ebinning2d = [np.logspace(2, 8, 61), np.linspace(-0.3, 0.3, 40)]
    ax1b.hist2d(true_e, diff, ebinning2d, norm=matplotlib.colors.LogNorm())
    ax1b.set_xlabel('True Neutrino Energy [GeV]')
    ax1b.set_xscale('log')
    ax1b.set_ylabel(r'Reco - True cos($\theta_{z}$)')
    saveHelper(fig1b, f'reco_minus_true_zenith_energy_{opt}', 'truth')
    
    fig1b2, ax1b2 = plt.subplots()
    ax1b2.hist2d(df.nu_energy.values, df.reco_energy.values, ebinning2d, 
                norm=matplotlib.colors.LogNorm())
    ax1b2.set_xscale('log')
    ax1b2.set_yscale('log')
    ax1b2.set_xlabel(r'True Neutrino Energy [GeV]')
    ax1b2.set_ylabel(r'Reco Neutrino Energy [GeV]')
    saveHelper(fig1b2, f'reco_true_energy_{opt}', 'truth')
    
    fig1b3, ax1b3 = plt.subplots()
    ax1b3.plot(df.nu_energy.values, 
               (df.reco_energy.values - df.nu_energy.values)/df.nu_energy.values, 
               'o', color='royalblue', alpha=0.8, markersize=0.9)
    ax1b3.axhline(0, np.min(df.nu_energy.values), np.max(df.nu_energy.values))
    ax1b3.set_xscale('log')
    ax1b3.set_xlabel(r'True Neutrino Energy [GeV]')
    ax1b3.set_ylabel(r'(Reco - True) / True Neutrino Energy')
    saveHelper(fig1b3, f'reco_true_energy_resolution_{opt}', 'truth')
    
    mask = df.nu_energy.values >= 1e5
    fig1b4, ax1b4 = plt.subplots()
    ax1b4.plot(df.nu_energy.values[mask], 
            (df.reco_energy.values[mask] - df.nu_energy.values[mask])/df.nu_energy.values[mask], 
            'o', color='royalblue', alpha=0.8, markersize=0.9)
    ax1b4.axhline(0, np.min(df.nu_energy.values[mask]), np.max(df.nu_energy.values[mask]))
    ax1b4.axhline(np.median(
        (df.reco_energy.values[mask] - df.nu_energy.values[mask])/df.nu_energy.values[mask]
        ),
        np.min(df.nu_energy.values[mask]), np.max(df.nu_energy.values[mask]),
        color='goldenrod')
    ax1b4.set_xscale('log')
    ax1b4.set_xlabel(r'True Neutrino Energy [GeV]')
    ax1b4.set_ylabel(r'(Reco - True) / True Neutrino Energy')
    saveHelper(fig1b4, f'reco_true_energy_resolution_highE_{opt}', 'truth')

    print(opt)
    z_mask_up         = reco_z < -0.9
    print(f'z_mask_up: {np.sum(z_mask_up)/len(reco_z)}')    
    z_mask_horizontal = (reco_z > -0.15) & (reco_z < 0.15)
    print(f'z_mask_horizontal: {np.sum(z_mask_horizontal)/len(reco_z)}')    
    z_mask_down       = reco_z > 0.9
    print(f'z_mask_down: {np.sum(z_mask_down)/len(reco_z)}')    
    maskList = [z_mask_up, z_mask_horizontal, z_mask_down]
    labelList = ['up', 'horizontal', 'down']

    fig1d, ax1d = plt.subplots()
    fig1e, ax1e = plt.subplots()
    if opt == 'cascade':
        diff_binning1d    = np.linspace(-2, 2, 40)
        diff_absbinning1d = np.linspace(0, 2, 40)
    if opt == 'track':
        diff_binning1d    = np.linspace(-0.3, 0.3, 40)
        diff_absbinning1d = np.linspace(0, 0.3, 40)
    colors = ['royalblue', 'goldenrod', 'salmon']
    ind = 0
    for mask, label in zip(maskList, labelList):
        _reco_z = reco_z[mask]
        _true_z = true_z[mask]
        _true_e = true_e[mask]
        _diff = _reco_z - _true_z
        fig1c, ax1c = plt.subplots()
        ax1c.hist2d(_true_e, _diff, ebinning2d, norm=matplotlib.colors.LogNorm())
        ax1c.set_xlabel('True Neutrino Energy [GeV]')
        ax1c.set_xscale('log')
        ax1c.set_ylabel(r'Reco - True cos($\theta_{z}$)')
        ax1c.set_title(label)
        saveHelper(fig1c, f'reco_minus_true_zenith_energy_{opt}_{label}', 'truth')

        ax1d.hist(_diff, diff_binning1d, histtype='step', label=label, color=colors[ind])
        ax1e.hist(abs(_diff), diff_absbinning1d, histtype='step', label=label, color=colors[ind])
        ind += 1
    
    ax1d.set_ylabel('Entries')
    ax1d.set_xlabel(r'Reco - True cos($\theta_{z}$)')
    ax1d.legend()
    saveHelper(fig1d, f'reco_minus_true_zenith_{opt}', 'truth')
    
    ax1e.set_ylabel('Entries')
    ax1e.set_xlabel(r'|Reco - True| cos($\theta_{z}$)')
    ax1e.legend()
    saveHelper(fig1e, f'reco_minus_true_zenith_abs_{opt}', 'truth')

    _df = df[df.reco_energy.values < 5e3]
    fig2, ax2 = plt.subplots()
    ax2.hist(np.cos(_df.nu_zenith.values),   binning, histtype='step', color='royalblue', label='Truth')
    ax2.hist(np.cos(_df.reco_zenith.values), binning, histtype='step', color='goldenrod', label='Reco')
    ax2.set_xlabel(r'cos($\theta_z$)')
    ax2.set_ylabel('Entries')
    ax2.legend()
    saveHelper(fig2, f'reco_and_true_zenith_lowE_{opt}', 'truth')

    fig3, ax3 = plt.subplots()
    ax3.hist2d(np.cos(_df.nu_zenith.values), np.cos(_df.reco_zenith.values), binning2d, norm=matplotlib.colors.LogNorm())
    ax3.set_xlabel(r'True cos($\theta_z$)')
    ax3.set_ylabel(r'Reco cos($\theta_z$)')
    saveHelper(fig3, f'reco_true_zenith_lowE_{opt}', 'truth')

    min_z = np.min(df.reco_z.values)
    max_z = np.max(df.reco_z.values)

    fig4, ax4 = plt.subplots()
    ax4.hist(df.reco_z.values, np.linspace(min_z, max_z, 100), histtype='step', color='royalblue')
    ax4.set_xlabel('Reco Z Position [m]')
    ax4.set_ylabel('Entries')
    saveHelper(fig4, f'reco_z_{opt}', 'truth')

    fig4a, ax4a = plt.subplots()
    ax4a.hist(df.reco_z.values, np.linspace(min_z, max_z, 100), 
              histtype='step', color='royalblue', label='All', density=True)
    ax4a.hist(df.reco_z.values[z_mask_horizontal], np.linspace(min_z, max_z, 100), 
              histtype='step', color='goldenrod', label='Horizontal', density=True)
    ax4a.set_xlabel('Reco Z Position [m]')
    ax4a.set_ylabel('Entries (Normalised)')
    ax4a.legend()
    saveHelper(fig4a, f'reco_z_slice_{opt}', 'truth')
    
    if opt == 'cascade':
        zbinning2d = [np.linspace(min_z, max_z, 100), np.linspace(-2, 2, 40)]
    if opt == 'track':
        zbinning2d = [np.linspace(min_z, max_z, 100), np.linspace(-0.3, 0.3, 40)]
    fig4b, ax4b = plt.subplots()
    ax4b.hist2d(df.reco_z.values, diff, zbinning2d, norm=matplotlib.colors.LogNorm())
    ax4b.set_xlabel('Reco Z Position [m]')
    ax4b.set_ylabel(r'Reco - True cos($\theta_{z}$)')
    saveHelper(fig4b, f'reco_minus_true_zenith_depth_{opt}', 'truth')
    
    fig4c, ax4c = plt.subplots()
    ax4c.hist2d(df.reco_z.values[z_mask_horizontal], diff[z_mask_horizontal], 
                zbinning2d, norm=matplotlib.colors.LogNorm())
    ax4c.set_xlabel('Reco Z Position [m]')
    ax4c.set_ylabel(r'Reco - True cos($\theta_{z}$)')
    saveHelper(fig4c, f'reco_minus_true_zenith_depth_slice_{opt}', 'truth')
    
    zbinning2d = [np.linspace(-1, 1, 20), np.linspace(min_z, max_z, 100)]
    fig5, ax5 = plt.subplots()
    ax5.hist2d(reco_z, df.reco_z.values, zbinning2d, norm=matplotlib.colors.LogNorm())
    ax5.set_xlabel(r'Reco cos($\theta_z$)')
    ax5.set_ylabel('Reco Z Position')
    saveHelper(fig5, f'reco_zenith_z_{opt}', 'truth', dpi=300)


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
    for key in df_li_c.keys():
        if key == 'weight1.0':
            f_type='all'

    print(f'=== MC Weights were calculated {f_type} ===')

    ## do_norms will plot for select xsec normalisations
    simple_truth(df_li_c, opt='cascade', f_type=f_type, do_norms=False, best_fit=best_fit)
    simple_truth(df_li_t, opt='track', f_type=f_type, do_norms=False, best_fit=best_fit)
    
    ##remove badly reconstructed events
    print("==== Removing Bad Reco ====")
    print("--- Data ---")
    df_data_c = df_data[df_data.Selection == 'cascade']
    df_data_t = df_data[df_data.Selection == 'track']
    print("-- Data Tracks   --")
    df_data_t = remove_bad_reco(df_data_t)
    print("-- Data Cascades --")
    df_data_c = remove_bad_reco(df_data_c)
    print("--- LeptonInjector ---")
    print("-- LeptonInjector - Cascade --")
    df_li_c = remove_bad_reco(df_li_c)
    print("-- LeptonInjecotr - Track --")
    df_li_t = remove_bad_reco(df_li_t)
    print('='*20)


    compare_reco_true(df_li_c, opt='cascade')
    compare_reco_true(df_li_t, opt='track')
    if fast == False:
        compare_reco(df_data_c, df_data_t, df_li_c, df_li_t, norm, f_type, ignore_nugen, best_fit)
    else:
        print('Skipping time consuming step of data/MC comparisons')

    print('Done')

if __name__ == "__main__":
    main()

##end
