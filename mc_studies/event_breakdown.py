from overlap_control import check_overlap_no_index
import matplotlib.pyplot as plt
import click
import numpy as np
import pandas as pd
import os, sys
import tables

def doConcatDF(df_list):
    df = pd.concat([df_list[0], df_list[1], df_list[2]])
    return df

def energy_breakdown(df_tracks, df_cascades, outdir):
    
    yr_const = 1 #weights already in units of years

    df_track_nue   = df_tracks[(df_tracks.pdg == 12) | (df_tracks.pdg == -12)]
    df_track_nue   = df_track_nue[df_track_nue.IntType != 'GR']
    df_track_numu  = df_tracks[(df_tracks.pdg == 14) | (df_tracks.pdg == -14)]
    df_track_numu  = df_track_numu[df_track_numu.IntType != 'GR']
    df_track_nutau = df_tracks[(df_tracks.pdg == 16) | (df_tracks.pdg == -16)]
    df_track_nutau = df_track_nutau[df_track_nutau.IntType != 'GR']
    df_track_nue_cc,   df_track_nue_nc   = slice_by_interaction(df_track_nue)
    df_track_numu_cc,  df_track_numu_nc  = slice_by_interaction(df_track_numu)
    df_track_nutau_cc, df_track_nutau_nc = slice_by_interaction(df_track_nutau)
    df_track_nc   = doConcatDF([df_track_nue_nc,   df_track_numu_nc,  df_track_nutau_nc])
    df_track_gr = df_tracks[df_tracks.IntType == 'GR']
    
    df_cascade_nue   = df_cascades[(df_cascades.pdg == 12) | (df_cascades.pdg == -12)]
    df_cascade_nue   = df_cascade_nue[df_cascade_nue.IntType != 'GR']
    df_cascade_numu  = df_cascades[(df_cascades.pdg == 14) | (df_cascades.pdg == -14)]
    df_cascade_numu  = df_cascade_numu[df_cascade_numu.IntType != 'GR']
    df_cascade_nutau = df_cascades[(df_cascades.pdg == 16) | (df_cascades.pdg == -16)]
    df_cascade_nutau = df_cascade_nutau[df_cascade_nutau.IntType != 'GR']
    df_cascade_nue_cc,   df_cascade_nue_nc   = slice_by_interaction(df_cascade_nue)
    df_cascade_numu_cc,  df_cascade_numu_nc  = slice_by_interaction(df_cascade_numu)
    df_cascade_nutau_cc, df_cascade_nutau_nc = slice_by_interaction(df_cascade_nutau)
    df_cascade_nc   = doConcatDF([df_cascade_nue_nc,   df_cascade_numu_nc,  df_cascade_nutau_nc])
    df_cascade_gr = df_cascades[df_cascades.IntType == 'GR']
    
    df_tracks_no_gr   = df_tracks[  (df_tracks.IntType == 'CC')   | (df_tracks.IntType == 'NC')]
    df_cascades_no_gr = df_cascades[(df_cascades.IntType == 'CC') | (df_cascades.IntType == 'NC')]

    df_track_cc = doConcatDF([df_track_nue_cc, df_track_numu_cc, df_track_nutau_cc])
    df_cascade_cc = doConcatDF([df_cascade_nue_cc, df_cascade_numu_cc, df_cascade_nutau_cc])

    hist_bins = np.logspace(2, 8, 16)
    zhist_bins = np.linspace(-1.0, 1.0, 12)

    ratehist1d(df_tracks, df_cascades, hist_bins, title='Rate vs True Energy',
                outdir=outdir, outfile='selection_compare_energy.pdf')
    ratehist1d(df_tracks_no_gr, df_cascades_no_gr, hist_bins, title='Rate vs True Energy (No GR)',
                outdir=outdir, outfile='selection_no_gr_compare_energy.pdf')
    ratehist1d(df_track_nue_cc, df_cascade_nue_cc, hist_bins, title='Nue CC - Rate vs True Energy',
                outdir=outdir, outfile='ccnc_true_energy_rate_nue_cc.pdf')
    ratehist1d(df_track_numu_cc, df_cascade_numu_cc, hist_bins, 
                title='Numu CC - Rate vs True Energy',
                outdir=outdir, outfile='ccnc_true_energy_rate_numu_cc.pdf')
    ratehist1d(df_track_nutau_cc, df_cascade_nutau_cc, hist_bins, 
                title='Nutau CC - Rate vs True Energy',
                outdir=outdir, outfile='ccnc_true_energy_rate_nutau_cc.pdf')
    ratehist1d(df_track_nc, df_cascade_nc, hist_bins, title='NC - Rate vs True Energy',
                outdir=outdir, outfile='ccnc_true_energy_rate_nc.pdf')
    ratehist1d(df_track_gr, df_cascade_gr, hist_bins, title='GR - Rate vs True Energy',
                outdir=outdir, outfile='ccnc_true_energy_rate_gr.pdf')
   
    ratehist1d(df_track_cc, df_track_nc, hist_bins, title='Track Selection - CC/NC',
                labels=['CC', 'NC'], outdir=outdir,
                outfile='ccnc_true_energy_rate_track.pdf')
    ratehist1d(df_cascade_cc, df_cascade_nc, hist_bins, title='Cascade Selection - CC/NC',
                labels=['CC', 'NC'], outdir=outdir,
                outfile='ccnc_true_energy_rate_cascade.pdf', ratio=True)
    ratehist1dzen(df_track_cc, df_track_nc, zhist_bins, title='Track Selection - CC/NC',
                labels=['CC', 'NC'], outdir=outdir,
                outfile='ccnc_true_zenith_rate_track.pdf')
    ratehist1dzen(df_cascade_cc, df_cascade_nc, zhist_bins, title='Cascade Selection - CC/NC',
                labels=['CC', 'NC'], outdir=outdir,
                outfile='ccnc_true_zenith_rate_cascade.pdf')
 
    ratehist1dfull([[df_track_nue_cc,  df_cascade_nue_cc],
                   [df_track_numu_cc,  df_cascade_numu_cc],
                   [df_track_nutau_cc, df_cascade_nutau_cc],
                   [df_track_nc,       df_cascade_nc],
                   [df_track_gr,       df_cascade_gr]],
                   hist_bins, title='Selected Flavour Rates vs True Energy',
                   outdir=outdir,
                   outfile='ccnc_true_energy_rate_all.pdf')

    scale_pts = [0.9, 1.0, 1.1]
    #scale_pts = [0.2, 1.0, 5.0]
    ratehist1dWeights([[df_track_nue_cc,   df_cascade_nue_cc],
                       [df_track_numu_cc,  df_cascade_numu_cc],
                       [df_track_nutau_cc, df_cascade_nutau_cc]],
                       hist_bins, zhist_bins, scale_pts, 
                       title='Selected Flavour Rates - Scaled',
                       outdir=outdir,
                       outfile='ccnc_true_energy_rate_all_scaling.pdf',
                       zoutfile='ccnc_true_zenith_rate_all_scaling.pdf',
                       zzoutfile='ccnc_true_energy_rate_cosz-1_scaling.pdf',
                       coutfile='ccnc_true_zenith_rate_cascade_scaling.pdf')

    ratehist1dnnbar([[df_track_nue_cc,   df_cascade_nue_cc],
                     [df_track_numu_cc,  df_cascade_numu_cc],
                     [df_track_nutau_cc, df_cascade_nutau_cc],
                     [df_track_nc,       df_cascade_nc],
                     [df_track_gr,       df_cascade_gr]],
                     hist_bins, title=r'Selected Rates $\nu$/$\bar{\nu}$',
                     outdir=outdir, 
                     outfile='ccnc_true_energy_rate_all_nnbar')

def ratehist1dnnbar(dfs, binning, title, outdir, outfile, use_reco=False):
    yr_const = 1
    #reco_eng_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.energy'
    #reco_eng_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.energy'
    #reco_eng_str3 = 'L3_MonopodFit4_AmptFit.energy'
    #reco_zen_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.zenith'
    #reco_zen_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.zenith'
    #reco_zen_str3 = 'L3_MonopodFit4_AmptFit.zenith'
    colours = ['royalblue', 'goldenrod', 'salmon', 'seagreen', 'firebrick']    
    labels_n    = ['Nue CC', 'Numu CC', 'Nutau CC', 'NC', 'GR']
    labels_nbar = ['Nuebar CC', 'Numubar CC', 'Nutaubar CC', 'NC', 'GR']
    plt_name = ['nue', 'numu', 'nutau', 'nc', 'gr']

    fig, ax = plt.subplots()

    for index, df in enumerate(dfs):
        df_t = df[0]
        df_c = df[1]

        df_t_n    = df_t.loc[df_t.pdg > 0] 
        df_t_nbar = df_t.loc[df_t.pdg < 0] 
        df_c_n    = df_c.loc[df_c.pdg > 0]
        df_c_nbar = df_c.loc[df_c.pdg < 0] 
            
        df_w_n    = np.append(df_t_n['weight1.0'],    df_c_n['weight1.0']) * yr_const
        df_w_nbar = np.append(df_t_nbar['weight1.0'], df_c_nbar['weight1.0']) * yr_const
        
        if use_reco != True:
            df_e_n    = np.append(df_t_n.nu_energy, df_c_n.nu_energy)
            df_e_nbar = np.append(df_t_nbar.nu_energy, df_c_nbar.nu_energy)
        if use_reco == True:
            df_e_n    = np.append(df_t_n[reco_eng_str1], df_c_n[reco_eng_str3])
            df_e_nbar = np.append(df_t_nbar[reco_eng_str1], df_c_nbar[reco_eng_str3])

        fig1d, ax1d = plt.subplots()
        #for GR could be 0 - only nu-bar!
        if len(df_e_n) != 0:
            ax1d.hist(df_e_n,    binning, color=colours[index], weights=df_w_n, 
                      histtype='step', label=labels_n[index])
        else:
            if labels_n[index].lower() != 'gr':
                print(f'Nu - No entries for Index: {index} ({labels_n[index]})')
        if len(df_e_nbar) != 0:
            ax1d.hist(df_e_nbar, binning, color=colours[index], weights=df_w_nbar,
                      histtype='step', label=labels_nbar[index], linestyle='dashdot')
        else:
            print(f'Nubar - No entries for Index: {index} ({labels_nbar[index]})')
        if index == 0 or index == 4:
            ax.hist(df_e_nbar, binning, color=colours[index], weights=df_w_nbar,
                    histtype='step', label=labels_nbar[index])
            if index == 0:
                e_bar, e_bar_bins = np.histogram(df_e_nbar, binning, weights=df_w_nbar)
            if index == 4:
                gr, gr_bins = np.histogram(df_e_nbar, binning, weights=df_w_nbar)

        ax1d.set_yscale('log')
        ax1d.set_xscale('log')
        ax1d.set_ylabel('Events per Year')
        ax1d.set_xlabel('Neutrino Energy [GeV]')
        ax1d.set_title(title)
        ax1d.legend()
        fig1d.savefig(os.path.join(outdir, outfile + '_' + plt_name[index] + '.pdf'))
        print(f"Created: {outfile}_{plt_name[index]}.pdf")
        plt.close(fig1d)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Events per Year')
    ax.set_xlabel('Neutrino Energy [GeV]')
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(outdir, outfile + '_nue_ccgr.pdf'))
    print(f"Created: {outfile}_nue_ccgr.pdf")
    plt.close(fig)

    with np.errstate(divide='ignore',invalid='ignore'):
        gr_e_bar_ratio = gr / e_bar
        gr_e_bar_ratio = np.nan_to_num(gr_e_bar_ratio)
        print(f'Ratio between GR and nue-bar Rate per energy bin: {gr_e_bar_ratio}')

    fig_r, ax_r = plt.subplots()
    ax_r.hist(np.logspace(np.log10(binning[0]), np.log10(binning[-2]), len(binning)-1), binning, 
              weights=gr_e_bar_ratio, histtype='step')
    ax_r.set_yscale('log')
    ax_r.set_ylim([1e-4, 100])
    ax_r.set_xscale('log')
    ax_r.set_ylabel(r'$\bar{\nu_{e}}$ GR / CC per Year')
    ax_r.set_xlabel('Neutrino Energy [GeV]')
    fig_r.savefig(os.path.join(outdir, outfile + '_nue_ccgr_ratio.pdf'))
    plt.close(fig_r)

def ratehist1dWeights(dfs, binning, zbinning, scale_pts, title, outdir, 
                      outfile, zoutfile, zzoutfile, 
                      coutfile, use_reco=False):
    yr_const = 1
    reco_eng_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.energy'
    reco_eng_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.energy'
    reco_eng_str3 = 'L3_MonopodFit4_AmptFit.energy'
    reco_zen_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.zenith'
    reco_zen_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.zenith'
    reco_zen_str3 = 'L3_MonopodFit4_AmptFit.zenith'
    colours = ['royalblue', 'goldenrod', 'firebrick', 'seagreen', 'salmon']    
    #colours = ['royalblue', 'goldenrod', 'firebrick']    

    #energy
    fig1d, ax1d = plt.subplots()
    fig1c, ax1c = plt.subplots()
    #also do cosine zenith
    fig1z, ax1z = plt.subplots()
    fig1zt, ax1zt = plt.subplots()
    fig1zc, ax1zc = plt.subplots()
    #for cos(z) = -1
    fig1dz, ax1dz = plt.subplots()    

    index = 0
    for scale in scale_pts:
        for df in dfs:
            ##get the respective track and cascade dfs
            ##nue cc, numu cc, nutau cc, nc, gr
            df_t = df[0]
            df_c = df[1]

            df_t_z = df_t.loc[np.cos(df_t.nu_zenith) <= -0.9]
            df_c_z = df_c.loc[np.cos(df_c.nu_zenith) <= -0.9]

            try:
                key = 'weight' + str(scale)
            except KeyError:
                key = 'weight'
            df_w = np.append(df_t[key], df_c[key]) * yr_const
            df_w_z = np.append(df_t_z[key], df_c_z[key]) * yr_const
            if use_reco != True:
                df_e = np.append(df[0].nu_energy, df[1].nu_energy)
                df_z = np.append(df[0].nu_zenith, df[1].nu_zenith)
                df_e_z = np.append(df_t_z.nu_energy, df_c_z.nu_energy)
        ax1d.hist(df_e, bins=binning, color=colours[index], weights=df_w, histtype='step', label=str(scale)) 
        ax1z.hist(np.cos(df_z), bins=zbinning, color=colours[index], 
                   weights=df_w, histtype='step', label=str(scale)) 
        ax1zt.hist(np.cos(df_t.nu_zenith), bins=zbinning, color=colours[index], 
                   weights=df_t[key], histtype='step', label=str(scale)) 
        ax1zc.hist(np.cos(df_c.nu_zenith), bins=zbinning, color=colours[index], 
                   weights=df_c[key], histtype='step', label=str(scale)) 
        ax1dz.hist(df_e_z, bins=binning, color=colours[index], weights=df_w_z, histtype='step',
                   label=str(scale))
        ax1c.hist(np.cos(df_c.loc[df_c.nu_energy >= 6e4, 'nu_zenith']), 
                   bins=zbinning, color=colours[index],
                   weights=df_c.loc[df_c.nu_energy >= 6e4, key], 
                   histtype='step', label=str(scale))
        index += 1

    #both selection
    ax1d.set_ylabel('Total CC Events per Year')
    ax1d.set_xlabel('Neutrino Energy [GeV]')
    ax1d.set_xscale('log')
    ax1d.set_yscale('log')
    ax1d.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1d.set_title(title)
    fig1d.tight_layout()
    fig1d.savefig(os.path.join(outdir, outfile))
    print(f"--- Created {outfile} ---")
    plt.close(fig1d)
    
    #cascade only zenith
    ax1c.set_ylabel('Total CC Cascades per Year | E >= 60 TeV')
    ax1c.set_xlabel(r'$cos(\theta_{z})$')
    ax1c.set_yscale('log')
    ax1c.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1c.set_title(title)
    fig1c.tight_layout()
    fig1c.savefig(os.path.join(outdir, coutfile))
    print(f"--- Created {coutfile} ---")
    plt.close(fig1c)

    #only for cos(z) <= 0.9
    ax1dz.set_ylabel(r'Total CC Events per Year | $cos(\theta_{z})$ <= -0.9')
    ax1dz.set_xlabel('Neutrino Energy [GeV]')
    ax1dz.set_xscale('log')
    ax1dz.set_yscale('log')
    ax1dz.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1dz.set_title(title)
    fig1dz.tight_layout()
    fig1dz.savefig(os.path.join(outdir, zzoutfile))
    print(f"--- Created {zzoutfile} ---")
    plt.close(fig1dz)
    
    ax1z.set_ylabel('Total CC Events per Year')
    ax1z.set_xlabel(r'$cos(\theta_{z})$')
    ax1z.set_yscale('log')
    ax1z.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1z.set_title(title)
    fig1z.tight_layout()
    fig1z.savefig(os.path.join(outdir, zoutfile))
    print(f"--- Created {zoutfile} ---")
    plt.close(fig1z)
    
    ax1zt.set_ylabel('Track Selected CC Events per Year')
    ax1zt.set_xlabel(r'$cos(\theta_{z})$')
    ax1zt.set_yscale('log')
    ax1zt.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1zt.set_title(title)
    fig1zt.tight_layout()
    fig1zt.savefig(os.path.join(outdir, 'ccnc_true_zenith_rate_track_cc_scaling.pdf'))
    plt.close(fig1zt)
    ax1zc.set_ylabel('Cascade Selected CC Events per Year')
    ax1zc.set_xlabel(r'$cos(\theta_{z})$')
    ax1zc.set_yscale('log')
    ax1zc.legend(loc=0, prop={'size': 7}, title=r'$\sigma$ CSMS')
    ax1zc.set_title(title)
    fig1zc.tight_layout()
    fig1zc.savefig(os.path.join(outdir, 'ccnc_true_zenith_rate_cascade_cc_scaling.pdf'))
    plt.close(fig1zc)


def ratehist1dfull(dfs, binning, title, outdir, outfile, use_reco=False):
    yr_const = 1
    reco_eng_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.energy'
    reco_eng_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.energy'
    reco_eng_str3 = 'L3_MonopodFit4_AmptFit.energy'
    colours = ['royalblue', 'goldenrod', 'salmon', 'seagreen', 'firebrick']    
    labels = ['Nue CC', 'Numu CC', 'Nutau CC', 'NC', 'GR']

    fig1d, ax1d = plt.subplots()
    index = 0
    for df in dfs:
        df_w = np.append(df[0]['weight1.0'], df[1]['weight1.0']) * yr_const
        if use_reco != True:
            df_e = np.append(df[0].nu_energy, df[1].nu_energy)
        if use_reco == True:
            df_e = np.append(df[0][reco_eng_str1], df[1][reco_eng_str3])
        ax1d.hist(df_e, bins=binning, color=colours[index], weights=df_w, histtype='step', label=labels[index]) 
        index += 1

    ax1d.set_ylabel('Events per Year')
    ax1d.set_xlabel('Neutrino Energy [GeV]')
    ax1d.set_xscale('log')
    ax1d.set_yscale('log')
    ax1d.legend(loc=0, prop={'size': 7})
    ax1d.set_title(title)
    fig1d.savefig(os.path.join(outdir, outfile))
    print(f"--- Created {outfile} ---")
    plt.close(fig1d)

def ratehist1d(df_t, df_c, binning, colours=['royalblue', 'goldenrod'],
               title='Event Rates vs True Energy', 
               labels=['Track', 'Cascade'], 
               outdir=None, outfile='ccnc_true_energy_rate.pdf', 
               use_reco=False,
               ratio=False):
    if outdir == None:
        raise NotImplementedError('You must give an outdir for ratehist1d!')
    yr_const = 1
    reco_eng_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.energy'
    reco_eng_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.energy'
    reco_eng_str3 = 'L3_MonopodFit4_AmptFit.energy'
    
    fig1d, ax1d = plt.subplots()
    if use_reco != True:
        ax1d.hist(df_t.nu_energy, bins=binning, facecolor=colours[0], weights=df_t['weight1.0'] * yr_const,
                histtype='step', label=labels[0])
        ax1d.hist(df_c.nu_energy, bins=binning, facecolor=colours[1],
                 weights=df_c['weight1.0'] * yr_const,
                histtype='step', label=labels[1])
        ax1d.set_xlabel('True Neutrino Energy [GeV]')
    if use_reco == True:
        ax1d.hist(df_t[reco_eng_str1], bins=binning, facecolor=colours[0], weights=df_t['weight1.0'] * yr_const,
                histtype='step', label=labels[0])
        ax1d.hist(df_c[reco_eng_str3], bins=binning, facecolor=colours[1],
                 weights=df_c['weight1.0'] * yr_const,
                histtype='step', label=labels[1])
        ax1d.set_xlabel('Reco Neutrino Energy [GeV]')

    ax1d.set_ylabel('Events per Year')
    ax1d.set_xscale('log')
    ax1d.set_yscale('log')
    ax1d.legend()
    ax1d.set_title(title)
    fig1d.savefig(os.path.join(outdir, outfile))
    print(f"--- Created {outfile} ---")
    plt.close(fig1d)

    if ratio == True:

        ##separate more finely
        fig2d, ax2d = plt.subplots()
        df_nue_cc = df_t[(df_t.pdg == 12) | (df_t.pdg == -12)]
        df_numu_cc = df_t[(df_t.pdg == 14) | (df_t.pdg == -14)]
        df_nutau_cc = df_t[(df_t.pdg == 16) | (df_t.pdg == -16)]
        ax2d.hist(df_nue_cc.nu_energy, bins=binning, facecolor='royalblue', weights=df_nue_cc['weight1.0'] * yr_const,
                    histtype='step', label='Nue CC')
        ax2d.hist(df_numu_cc.nu_energy, bins=binning, facecolor='goldenrod', weights=df_numu_cc['weight1.0'] * yr_const,
                    histtype='step', label='Numu CC')
        ax2d.hist(df_nutau_cc.nu_energy, bins=binning, facecolor='firebrick', weights=df_nutau_cc['weight1.0'] * yr_const,
                    histtype='step', label='Nutau CC')
        ax2d.hist(df_c.nu_energy, bins=binning, facecolor='black', weights=df_c['weight1.0'] * yr_const,
                    histtype='step', label=r'NC (e,$\mu$,$\tau$)')
    
        ax2d.set_ylabel('Events per Year')
        ax2d.set_xlabel('True Neutrino Energy [GeV]')
        ax2d.set_xscale('log')
        ax2d.set_yscale('log')
        ax2d.legend()
        ax2d.set_title(title)
        s2 = outfile.split('.')
        outfile2 = s2[0] + '_factorised.' + s2[1]
        fig2d.savefig(os.path.join(outdir, outfile2))
        print(f"--- Created {outfile2} ---")
        plt.close(fig2d)


        ##grab default xsec for comparison
        default_path="/data/user/chill/snowstorm_nugen_ana/xsec/data/csms.h5"
        t = tables.open_file(default_path)
        e_range = get_node(t, 'energies')
        node_list = ['s_CC_nu', 's_CC_nubar', 's_NC_nu', 's_NC_nubar']
        nu_cc = []
        nubar_cc = []
        nu_nc = []
        nubar_nc = []
        vals_list = [nu_cc, nubar_cc, nu_nc, nubar_nc]
        for i, n_name in enumerate(node_list):
            vals_list[i] = get_node(t, n_name)
            
        xsec_cc = 10**vals_list[0] + 10**vals_list[1]
        xsec_nc = 10**vals_list[2] + 10**vals_list[3]
        xsec_ratio = xsec_cc/xsec_nc

        df_t = df_t[(df_t.pdg == 12) | (df_t.pdg == -12)]
        df_c = df_c[(df_c.pdg == 12) | (df_c.pdg == -12)]

        fig1r, ax1r = plt.subplots()
        t = np.histogram(df_t.nu_energy, bins=binning, weights=df_t['weight1.0']*yr_const)
        c = np.histogram(df_c.nu_energy, bins=binning, weights=df_c['weight1.0']*yr_const)
        v_ratio = [1] * len(t[0])
        i = 0
        for v1, v2 in zip(t[0], c[0]):
            if v1 == 0 and v2 == 0:
                i += 1
                continue
            v_ratio[i] = v1/v2
            i += 1

        ##find good slice for xsec
        start = 0
        stop = -1
        for j, e in enumerate(e_range-9):
            if e == 2:
                start = j
            if e == 8:
                stop = j
                break

        ax1r.plot(binning[:-1], v_ratio, color='firebrick', linewidth=0, marker='o', label=f'Nue {labels[0]} / Nue {labels[1]}')
        ax1r.plot(10**(e_range-9)[start:stop], xsec_ratio[start:stop], color='black', linewidth=2, label='CC/NC Cross Section')

        ax1r.set_xlabel('True Neutrino Energy [GeV]')
        ax1r.set_ylabel(f'Ratio {labels[0]}/{labels[1]}')
        ax1r.set_xscale('log')
        #ax1r.set_yscale('log')
        ax1r.legend()
        ax1r.set_title(title)
        s = outfile.split('.')
        outfile_r = s[0] + '_ratio.' + s[1]
        fig1r.savefig(os.path.join(outdir, outfile_r))
        print(f"--- Created {outfile_r} ---")
        plt.close(fig1r)

def get_node(t, n_name):
    node = t.get_node("/" + n_name)
    vals = node.read()
    return vals


def ratehist1dzen(df_t, df_c, binning, colours=['royalblue', 'goldenrod'],
                  title='Event Rates vs cos(zenith)', labels=['Track', 'Cascade'],
                  outdir=None, outfile='ccnc_true_zenith_rate.pdf', use_reco=False):
    if outdir == None:
        raise NotImplementedError('You must give an outdir for ratehist1d!')
    yr_const = 1
    reco_eng_str1 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Neutrino.energy'
    reco_eng_str2 = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon.energy'
    reco_eng_str3 = 'L3_MonopodFit4_AmptFit.energy'
    
    fig1d, ax1d = plt.subplots()
    if use_reco != True:
        ax1d.hist(np.cos(df_t.nu_zenith), bins=binning, facecolor=colours[0], weights=df_t['weight1.0'] * yr_const,
                histtype='step', label=labels[0])
        ax1d.hist(np.cos(df_c.nu_zenith), bins=binning, facecolor=colours[1],
                 weights=df_c['weight1.0'] * yr_const,
                histtype='step', label=labels[1])
        ax1d.set_xlabel(r'True Neutrino cos($\theta_{zen}$) [GeV]')
    if use_reco == True:
        raise NotImplementedError('reco not yet supported')
    #    ax1d.hist(df_t[reco_eng_str1], bins=binning, facecolor=colours[0], weights=df_t['weight1.0'] * yr_const,
    #            histtype='step', label=labels[0])
    #    ax1d.hist(df_c[reco_eng_str3], bins=binning, facecolor=colours[1],
    #             weights=df_c['weight1.0'] * yr_const,
    #            histtype='step', label=labels[1])
    #    ax1d.set_xlabel('Reco Neutrino Energy [GeV]')

    ax1d.set_ylabel('Events per Year')
    #ax1d.set_xscale('log')
    ax1d.set_yscale('log')
    ax1d.legend()
    ax1d.set_title(title)
    fig1d.savefig(os.path.join(outdir, outfile))
    print(f"--- Created {outfile} ---")
    plt.close(fig1d)

def ratio_breakdown(df_track_nue_cc_total, df_cascade_nue_cc_total,
                    df_track_numu_cc_total, df_cascade_numu_cc_total,
                    df_track_nutau_cc_total, df_cascade_nutau_cc_total,
                    df_track_nc_total, df_cascade_nc_total,
                    df_track_gr_total, df_cascade_gr_total,
                    outdir, outfile):
    
    #yr_const = 60 * 60 * 24 * 365
    yr_const = 1   

    track_nue_cc   = np.zeros(3)
    track_numu_cc  = np.zeros(3)
    track_nutau_cc = np.zeros(3)
    track_nc       = np.zeros(3)
    track_gr       = np.zeros(3)

    cascade_nue_cc   = np.zeros(3)
    cascade_numu_cc  = np.zeros(3)
    cascade_nutau_cc = np.zeros(3)
    cascade_nc       = np.zeros(3)
    cascade_gr       = np.zeros(3)

    overlap_nue_cc   = np.zeros(3)
    overlap_numu_cc  = np.zeros(3)
    overlap_nutau_cc = np.zeros(3)
    overlap_nc       = np.zeros(3)
    overlap_gr       = np.zeros(3)
    
    # 3 samples (1e2-1e4, 1e4-1e6, 1e6-1e8), so size is 3 
    for i in range(3):
        o_nue_cc   = check_overlap(df_track_nue_cc_total[i],   df_cascade_nue_cc_total[i])
        o_numu_cc  = check_overlap(df_track_numu_cc_total[i],  df_cascade_numu_cc_total[i])
        o_nutau_cc = check_overlap(df_track_nutau_cc_total[i], df_cascade_nutau_cc_total[i])
        o_nc       = check_overlap(df_track_nc_total[i],       df_cascade_nc_total[i])
        o_gr       = check_overlap(df_track_gr_total[i],       df_cascade_gr_total[i])

        df_track_nue_cc   = df_track_nue_cc_total[i]
        df_track_numu_cc  = df_track_numu_cc_total[i]
        df_track_nutau_cc = df_track_nutau_cc_total[i]
        df_track_nc       = df_track_nc_total[i]
        df_track_gr       = df_track_gr_total[i]
        
        df_cascade_nue_cc   = df_cascade_nue_cc_total[i]
        df_cascade_numu_cc  = df_cascade_numu_cc_total[i]
        df_cascade_nutau_cc = df_cascade_nutau_cc_total[i]
        df_cascade_nc       = df_cascade_nc_total[i]
        df_cascade_gr       = df_cascade_gr_total[i]

        df_o_nue_cc   = df_track_nue_cc[df_track_nue_cc.index.isin(o_nue_cc)]
        df_o_numu_cc  = df_track_numu_cc[df_track_numu_cc.index.isin(o_numu_cc)]
        df_o_nutau_cc = df_track_nutau_cc[df_track_nutau_cc.index.isin(o_nutau_cc)]
        df_o_nc       = df_track_nc[df_track_nc.index.isin(o_nc)]
        df_o_gr       = df_track_gr[df_track_gr.index.isin(o_gr)]        

        track_nue_cc[i]     = np.sum(df_track_nue_cc['weight1.0'])     * yr_const
        track_numu_cc[i]    = np.sum(df_track_numu_cc['weight1.0'])    * yr_const
        track_nutau_cc[i]   = np.sum(df_track_nutau_cc['weight1.0'])   * yr_const
        track_nc[i]         = np.sum(df_track_nc['weight1.0'])         * yr_const
        track_gr[i]         = np.sum(df_track_gr['weight1.0'])         * yr_const

        cascade_nue_cc[i]   = np.sum(df_cascade_nue_cc['weight1.0'])   * yr_const
        cascade_numu_cc[i]  = np.sum(df_cascade_numu_cc['weight1.0'])  * yr_const
        cascade_nutau_cc[i] = np.sum(df_cascade_nutau_cc['weight1.0']) * yr_const
        cascade_nc[i]       = np.sum(df_cascade_nc['weight1.0'])       * yr_const
        cascade_gr[i]       = np.sum(df_cascade_gr['weight1.0'])       * yr_const        

        overlap_nue_cc[i]   = np.sum(df_o_nue_cc['weight1.0'])   * yr_const
        overlap_numu_cc[i]  = np.sum(df_o_numu_cc['weight1.0'])  * yr_const
        overlap_nutau_cc[i] = np.sum(df_o_nutau_cc['weight1.0']) * yr_const
        overlap_nc[i]       = np.sum(df_o_nc['weight1.0'])       * yr_const
        overlap_gr[i]       = np.sum(df_o_gr['weight1.0'])       * yr_const

    n_nue_cc   = np.zeros(3)
    n_numu_cc  = np.zeros(3)
    n_nutau_cc = np.zeros(3)
    n_nc       = np.zeros(3)
    n_gr       = np.zeros(3)
    n_total    = np.zeros(3)

    for i in range(3): 
        n_nue_cc[i]   = track_nue_cc[i]   + cascade_nue_cc[i]
        n_numu_cc[i]  = track_numu_cc[i]  + cascade_numu_cc[i]
        n_nutau_cc[i] = track_nutau_cc[i] + cascade_nutau_cc[i]
        n_nc[i]       = track_nc[i]       + cascade_nc[i]
        n_gr[i]       = track_gr[i]       + cascade_gr[i]
        n_total[i]    = n_nue_cc[i] + n_numu_cc[i] + n_nutau_cc[i] + n_nc[i] + n_gr[i]

    labels = ['1e2 - 1e4', '1e4 - 1e6', '1e6 - 1e8']
    print("=== CC / NC Breakdown Info ===")
    for i in range(3):
        print(f"Energy Range: {labels[i]} GeV - Total {n_total[i]}")
        print(f"Nue CC   : {n_nue_cc[i]} ({n_nue_cc[i]     / n_total[i] * 100} [%])")
        print(f"Numu CC  : {n_numu_cc[i]} ({n_numu_cc[i]   / n_total[i] * 100} [%])")
        print(f"Nutau CC : {n_nutau_cc[i]} ({n_nutau_cc[i] / n_total[i] * 100} [%])")
        print(f"NC       : {n_nc[i]} ({n_nc[i] / n_total[i] * 100} [%]")
        print(f"GR       : {n_gr[i]} ({n_gr[i] / n_total[i] * 100} [%]")
        print(f"CC / NC    : {(n_nue_cc[i] + n_numu_cc[i] + n_nutau_cc[i]) / n_nc[i] * 100} [%]")
        print(f"CC / Total : {(n_nue_cc[i] + n_numu_cc[i] + n_nutau_cc[i]) / n_total[i] * 100} [%]")
 
    x = np.arange(len(labels))  # the label locations
    x[1] += 4
    x[2] += 8
    width = 3.0  # the width of the bars
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, n_nue_cc,   width/4, label=r'$\nu_{e}$',    color='royalblue')
    bar2 = ax.bar(x - width/4, n_numu_cc,  width/4, label=r'$\nu_{\mu}$',  color='goldenrod')
    bar3 = ax.bar(x,           n_nutau_cc, width/4, label=r'$\nu_{\tau}$', color='salmon')
    bar4 = ax.bar(x + width/4, n_nc,       width/4, label=r'NC',           color='seagreen')
    bar5 = ax.bar(x + width/2, n_gr,       width/4, label='GR',            color='firebrick')
 
    ax.set_title('CC / NC / GR Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylabel('Selected Events per Year')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, outfile))
    ax.set_yscale('log')
    split = outfile.split(".")
    outfile_log = split[0] + '_log.' + split[1]
    fig.savefig(os.path.join(outdir, outfile_log))

def get_df(f):
    df = pd.read_hdf(f)
    df = df.drop_duplicates()
    return df

def conditions(df, i):
    if i == 0:
        df_c = df[df.nu_energy <= 1e4]
    elif i == 1:
        df_c = df[(df.nu_energy > 1e4) & (df.nu_energy <= 1e6)]
    elif i == 2:
        df_c = df[df.nu_energy < 1e8]
    else:
        raise NotImplementedError('Outside simulation binning range!')
    return df_c
    
def plot_summary(n_tracks, n_cascades, n_overlap, n_total, neutrino_type=None, 
                 ccnc=None, use_weights=False, outdir=None, out_file=None, verbose=False):
    if outdir == None:
        raise NotImplementedError('You must give an outdir for plot_summary!')
    labels = ['1e2 - 1e4', '1e4 - 1e6', '1e6 - 1e8']
    x = np.arange(len(labels))  # the label locations
    x[1] += 4
    x[2] += 8
    width = 3.0  # the width of the bars
    #yr_const = 60 * 60 * 24 * 365
    yr_const = 1.0 ##all weights already scaled by 1 year
    
    if neutrino_type == None:
        neutrino_type = 'All Flavours'
    if ccnc == None:
        ccnc = 'CC+NC'
    if verbose:
        if use_weights == False:
            print(f"--- Unweighted {neutrino_type} : {ccnc} ---")
        if use_weights == True:
            print(f"--- Weighted {neutrino_type} : {ccnc} ---")
        print(f"Energy [GeV],     Tracks%,     Cascades%,     Overlap%")
        print(f"1e2-1e4 GeV: {(n_tracks[0] / n_total[0]) * 100}, {(n_cascades[0] / n_total[0]) * 100}, {(n_overlap[0] / n_total[0]) * 100}")
        print(f"1e4-1e6 GeV: {(n_tracks[1] / n_total[1]) * 100}, {(n_cascades[1] / n_total[1]) * 100}, {(n_overlap[1] / n_total[1]) * 100}")
        print(f"1e6-1e8 GeV: {(n_tracks[2] / n_total[2]) * 100}, {(n_cascades[2] / n_total[2]) * 100}, {(n_overlap[2] / n_total[2]) * 100}")
    
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, n_tracks, width/4, label='Tracks', color='royalblue')
    bar2 = ax.bar(x - width/4, n_cascades, width/4, label='Cascades', color='goldenrod')
    bar3 = ax.bar(x + width/4, n_overlap, width/4, label='Overlap', color='salmon')
    bar4 = ax.bar(x + width/2, n_total, width/4, label='Unique Total', color='seagreen')

    ax.set_title(f'Selection Breakdown: {neutrino_type} : {ccnc}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    if use_weights is True:
        ax.set_ylabel('Selected Events per Year')
    if use_weights is False:
        ax.set_ylabel('Selected Events (Unweighted)')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    if use_weights is False:
        autolabel(bar1)
        autolabel(bar2)
        autolabel(bar3)
        autolabel(bar4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, out_file))
    ax.set_yscale('log')
    split = out_file.split(".")
    out_file_log = split[0] + '_log.' + split[1]
    fig.savefig(os.path.join(outdir, out_file_log))
    plt.close(fig)

def bar_breakdown(df_tracks, df_cascades, nu_type=None, out_file=None, 
                  ccnc=None, verbose=False, outdir=None):
    if outdir == None:
        raise NotImplementedError('You must give an outdir for bar_breakdown!')
    labels = ['1e2 - 1e4', '1e4 - 1e6', '1e6 - 1e8']

    n_tracks   = [0] * len(labels)
    n_cascades = [0] * len(labels)
    n_overlap  = [0] * len(labels)
    n_total    = [0] * len(labels)
    
    n_tracks_w   = [0] * len(labels)
    n_cascades_w = [0] * len(labels)
    n_overlap_w  = [0] * len(labels)
    n_total_w    = [0] * len(labels)

    ##returns dataframes with overlap
    ##df_t_o and df_c_o have the same events
    ##so only need to count once
    ##but could have different reco applied

    df_t_o, df_c_o = check_overlap_no_index(df_tracks, df_cascades)

    for i in range(len(labels)):
        df_t = conditions(df_tracks, i)
        df_c = conditions(df_cascades, i)
        n_tracks_w[i]   = np.sum(df_t['weight1.0'])
        n_cascades_w[i] = np.sum(df_c['weight1.0'])
        n_overlap_w[i]  = np.sum(df_t_o['weight1.0'])
        n_tracks[i]     = len(df_t.index.values)
        n_cascades[i]   = len(df_c.index.values)
        n_overlap[i]    = len(df_t_o.index.values)

        n_total[i]   = n_tracks[i]   + n_cascades[i]   - n_overlap[i]
        n_total_w[i] = n_tracks_w[i] + n_cascades_w[i] - n_overlap_w[i]
   
    if nu_type != None:
        out_file  = os.path.join(outdir, f'bar_breakdown_{nu_type}_unweighted.pdf')
        out_file_w= os.path.join(outdir, f'bar_breakdown_{nu_type}_weighted.pdf')
    if nu_type == None:
        s1 = out_file.split('.')[0]
        out_file_w = f'{s1}_weighted.pdf'

    plot_summary(n_tracks,   n_cascades,   n_overlap,   n_total,   nu_type, ccnc, 
                 use_weights=False, outdir=outdir, out_file=out_file, verbose=verbose) 
    plot_summary(n_tracks_w, n_cascades_w, n_overlap_w, n_total_w, nu_type, ccnc, 
                 use_weights=True,  outdir=outdir, out_file=out_file_w, verbose=verbose) 

    #return overlapping dfs to avoid computing again
    return df_t_o, df_c_o

def slice_by_interaction(df):
    df_cc = df[df.IntType == 'CC']
    df_nc = df[df.IntType == 'NC']
    return df_cc, df_nc

def get_files(nu_type):
    loc = '/data/user/chill/snowstorm_nugen_ana/monte_carlo/'
    pref = 'data_cache_' + nu_type
    if nu_type == 'numu':
        df_track1   = loc + pref + '_21430_track.hdf'
        df_cascade1 = loc + pref + '_21430_cascade.hdf'        
        df_track2   = loc + pref + '_21431_track.hdf'
        df_cascade2 = loc + pref + '_21431_cascade.hdf'        
        df_track3   = loc + pref + '_21432_track.hdf'
        df_cascade3 = loc + pref + '_21432_cascade.hdf'        
    if nu_type == 'nue':
        df_track1   = loc + pref + '_21468_track.hdf'
        df_cascade1 = loc + pref + '_21468_cascade.hdf'        
        df_track2   = loc + pref + '_21469_track.hdf'
        df_cascade2 = loc + pref + '_21469_cascade.hdf'        
        df_track3   = loc + pref + '_21470_track.hdf'
        df_cascade3 = loc + pref + '_21470_cascade.hdf'        
    if nu_type == 'nutau':
        df_track1   = loc + pref + '_21471_track.hdf'
        df_cascade1 = loc + pref + '_21471_cascade.hdf'        
        df_track2   = loc + pref + '_21472_track.hdf'
        df_cascade2 = loc + pref + '_21472_cascade.hdf'        
        df_track3   = loc + pref + '_21473_track.hdf'
        df_cascade3 = loc + pref + '_21473_cascade.hdf'        
    if nu_type not in ['numu', 'nue', 'nutau']:
        raise ValueError("Neutrino Type not valid")

    df_track_list = [df_track1, df_track2, df_track3]
    df_cascade_list = [df_cascade1, df_cascade2, df_cascade3]
    return df_track_list, df_cascade_list

def format_string(nu_type):
    if nu_type == 'nue':
        fmt_nu_type = r'$\nu_e$'
    if nu_type == 'numu':
        fmt_nu_type = r'$\nu_{\mu}$'
    if nu_type == 'nutau':
        fmt_nu_type = r'$\nu_{\tau}$'
    return fmt_nu_type

def setup_weights(df):

    df['weight1.0'] = df['weight1.0_atmo'] + df['weight1.0_astro']
    df['weight1.1'] = df['weight1.1_atmo'] + df['weight1.1_astro']
    df['weight0.9'] = df['weight0.9_atmo'] + df['weight0.9_astro']
    return df

@click.command()
@click.argument('df_file')
@click.option('--verbose', '-v', is_flag=True)
def main(df_file, verbose):
    df_total = pd.read_hdf(df_file)
    df_total = setup_weights(df_total)
    df_tracks = df_total[df_total.Selection == 'track']
    df_cascades = df_total[df_total.Selection == 'cascade']

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print(f'Created {outdir}')

    nu_dict = {'nue': 12, 'numu': 14, 'nutau': 16, 'gr': -12}
    nu_str = ['nue', 'numu', 'nutau', 'gr']

    df_tracks_overlap, df_cascades_overlap = bar_breakdown(df_tracks, df_cascades, nu_type=None,
                                 out_file='bar_breakdown_all_flavour.pdf', 
                                 verbose=verbose, outdir=outdir)
    for nu in nu_str:
        nu_id = nu_dict[nu]
        if nu != 'gr':
            _df_tracks   = df_tracks[(df_tracks.pdg == nu_id)     | (df_tracks.pdg == -1*nu_id)]
            _df_tracks   = _df_tracks[_df_tracks.IntType != 'GR']
            _df_cascades = df_cascades[(df_cascades.pdg == nu_id) | (df_cascades.pdg == -1*nu_id)]
            _df_cascades = _df_cascades[_df_cascades.IntType != 'GR']

        if nu == 'gr':
            _df_tracks = df_tracks[(df_tracks.pdg == nu_id) & (df_tracks.IntType == 'GR')]
            _df_cascades = df_cascades[(df_cascades.pdg == nu_id) & (df_cascades.IntType == 'GR')]
            
        
        bar_breakdown(_df_tracks, _df_cascades, nu_type=nu, 
                        out_file=None, verbose=verbose, outdir=outdir)
        if nu != 'gr':
            ##now slice based on CC or NC
            ##input dfs are either nue, numu, nutau, or GR
            ##so no need to slice GR
            df_tracks_cc, df_tracks_nc = slice_by_interaction(_df_tracks)
            df_cascades_cc, df_cascades_nc = slice_by_interaction(_df_cascades)
            bar_breakdown(df_tracks_cc, df_cascades_cc, nu_type=nu, 
                            out_file=f'bar_breakdown_{nu}_cc.pdf', 
                            ccnc='CC', verbose=verbose, outdir=outdir)
            bar_breakdown(df_tracks_nc, df_cascades_nc, nu_type=nu, 
                            out_file=f'bar_breakdown_{nu}_nc.pdf', 
                            ccnc='NC', verbose=verbose, outdir=outdir)

    energy_breakdown(df_tracks, df_cascades, outdir)


if __name__ == "__main__":
    main()

##end
