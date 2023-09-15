##this script needs icetray, will open i3 files
import sys, os
import numpy as np
import h5py
from glob import glob
import click
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from I3Tray import I3Tray
from icecube import icetray, dataio, dataclasses

from weighting.event_info import EventInfo
from configs import config

def get_info(fList):

    eventInfo = EventInfo()
    for f in tqdm(fList):
        data_file = dataio.I3File(f, 'r')
        # scan over the frames
        while data_file.more():
            try:
                frame = data_file.pop_daq()
            ## if no physics frames are in the file - skip
            except RuntimeError:
                continue
            if "MCPrimary1" in frame:
                MCPrimary = frame["MCPrimary1"]
            else:
                i = 0
                is_neutrino = False
                while is_neutrino == False:
                    MCPrimary = frame["I3MCTree"].primaries[i]
                    if MCPrimary.is_neutrino == True:
                        is_neutrino = True
                    else:
                        i += 1
                if i != 0:
                    print(f'Neutrino primary was {i}!')
            frame["MCPrimary1"]         = MCPrimary
            MCPrimary1 = frame['MCPrimary1']
            eventInfo.nu_energy.append(MCPrimary1.energy)
            eventInfo.nu_azimuth.append(MCPrimary1.dir.azimuth)
            eventInfo.nu_zenith.append(np.cos(MCPrimary1.dir.zenith))
            eventInfo.nu_x.append(MCPrimary1.pos.x)
            eventInfo.nu_y.append(MCPrimary1.pos.y)
            eventInfo.nu_z.append(MCPrimary1.pos.z)
        data_file.close()

    return eventInfo

def make_plots(info, global_tag):

    energy = info.nu_energy
    azimuth = info.nu_azimuth
    zenith = info.nu_zenith
    pos_x = info.nu_x
    pos_y = info.nu_y
    pos_z = info.nu_z    

    varList = [energy, azimuth, zenith, pos_x, pos_y, pos_z]
    tagList = ['Energy [GeV]', 'Azimuth', 'cos(zenith)', 'Vertex x [m]',
                'Vertex y [m]', 'Vertex z [m]']
    plt_tagList = ['energy', 'azimuth', 'cosz', 'vertex_x',
                'vertex_y', 'vertex_z']

    for var, tag, ptag in zip(varList, tagList, plt_tagList):
        fig, ax = plt.subplots()
        if tag == 'Energy [GeV]':
            ax.set_xscale('log')
            binning = np.logspace(np.log10(np.min(var)), np.log10(np.max(var)), 150)
        else:
            binning = np.linspace(np.min(var), np.max(var), 150)

        ax.hist(var, binning, histtype='step', color='royalblue')

        ax.set_xlabel(tag)
        ax.set_ylabel('Entries (Unweighted)')
        fig.savefig(f'plots_unweighted/{global_tag}_{ptag}.pdf')
        plt.close(fig)

def make_selected_plots(df, df_pair, global_tag):
    varList = ['nu_energy', 'nu_azimuth', 'nu_zenith', 'nu_x', 'nu_y', 'nu_z', 'mmc_energy']
    tagList = ['Neutrino Energy [GeV]', r'Azimuth$_{\nu}$', r'$cos(z_{\nu})$', 'Vertex x [m]',
                'Vertex y [m]', 'Vertex z [m]', 'MMCTrackList Ei_max [GeV]']
    plt_tagList = ['energy', 'azimuth', 'cosz', 'vertex_x',
                'vertex_y', 'vertex_z', 'mmc_energy']

    event_w = df['weight1.0_atmo'].values + df['weight1.0_astro'].values
    p_event_w = df_pair['weight1.0_atmo'].values + df_pair['weight1.0_astro'].values
    for _w in event_w:
        if _w <= 0:
            print(f'Negative/Zero Weight!: {_w}')
    for _w in p_event_w:
        if _w <= 0:
            print(f'Negative/Zero Weight!: {_w} in paired df')

    mask1 = event_w > 0
    mask2 = p_event_w > 0
    #mask = mask1 * mask2

    event_w = event_w[mask1]
    p_event_w = p_event_w[mask2]

    for var, tag, ptag in zip(varList, tagList, plt_tagList):
        if var == 'mmc_energy':
            if global_tag != 'GR':
                continue
        #val = df[f'{var}'].values
        #valp = df_pair[f'{var}'].values
        val = df[f'{var}'].values[mask1]
        valp = df_pair[f'{var}'].values[mask2]
        if len(valp) == 0:
            print(var)
            print('Old MC dataframe had 0 entry')
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        figp, axp = plt.subplots()
        fig1p, ax1p = plt.subplots()
        if var == 'nu_energy' or var == 'mmc_energy':
            ax.set_xscale('log')
            ax1.set_xscale('log')
            axp.set_xscale('log')
            ax1p.set_xscale('log')
            if global_tag == 'GR':
                binning = np.logspace(np.log10(np.min(val)), np.log10(np.max(val)), 20)
            else:
                binning = np.logspace(np.log10(np.min(val)), np.log10(np.max(val)), 60)
        
        elif var == 'nu_zenith':
            val = np.cos(val)
            valp = np.cos(valp)
            binning = np.linspace(-1, 1, 150)
        else:
            binning = np.linspace(np.min(val), np.max(val), 150)

        ax.hist(val, binning, histtype='step', color='royalblue')
        ax.set_xlabel(tag)
        ax.set_ylabel('Entries (Unweighted)')
        fig.tight_layout()
        fig.savefig(f'plots_unweighted/selected_{global_tag}_{ptag}.pdf')
        plt.close(fig)
        
        ax1.hist(val, binning, weights=event_w, histtype='step', color='royalblue')
        ax1.set_xlabel(tag)
        ax1.set_ylabel('Events / yr')
        fig1.tight_layout()
        fig1.savefig(f'plots_unweighted/selected_weighted_{global_tag}_{ptag}.pdf')
        plt.close(fig1)
        
        axp.hist(val,  binning, density=True, histtype='step', color='royalblue', label='New')
        axp.hist(valp, binning, density=True, histtype='step', color='goldenrod', label='Current')
        axp.set_xlabel(tag)
        axp.legend()
        axp.set_ylabel('Entries (Unweighted)')
        figp.tight_layout()
        figp.savefig(f'plots_unweighted/selected_{global_tag}_compare_{ptag}.pdf')
        plt.close(figp)
        
        ax1p.hist(val,  binning, weights=event_w,   #density=True, 
                  histtype='step', color='royalblue', label='New')
        ax1p.hist(valp, binning, weights=p_event_w, #density=True, 
                  histtype='step', color='goldenrod', label='Current')
        ax1p.set_xlabel(tag)
        ax1p.legend()
        ax1p.set_ylabel('Events / yr')
        fig1p.tight_layout()
        fig1p.savefig(f'plots_unweighted/selected_weighted_{global_tag}_compare_{ptag}.pdf')
        plt.close(fig1p)

@click.command()
@click.option('--energy', '-e', required=True)
@click.option('--weight_only', '-w', is_flag=True)
@click.option('--compare', '-c', is_flag=True)
def main(energy, weight_only, compare):
    ##NOTE: this script is for verification of MC ONLY
    ##only for simple verification
    ##grab the files manually in the generated folder
   
    print("THIS SCRIPT IS DEPRICATED SINCE 2023-09 -- FUTURE USE REQUIRES MODIFICATIONS")

    ##high E sample
    if energy == 'high':
        fPath = '/data/sim/IceCube/2020/generated/lepton-injector/22369/0000000-0000999/'
        tag = 'highE'
        df = pd.read_hdf('../../weights/weight_df_022369_track_CC.hdf')
        df_pair = pd.read_hdf('../../weights/weight_df_021397_track_CC.hdf')
    ##low E sample
    elif energy == 'low':
        fPath = '/data/sim/IceCube/2020/generated/lepton-injector/22375/0000000-0000999/'
        tag = 'lowE'
        df = pd.read_hdf('../../weights/weight_df_022375_track_CC.hdf')
        df_pair = pd.read_hdf('../../weights/weight_df_021395_track_CC.hdf')
    elif energy == 'gr':
        fPath = '/data/user/chill/icetray_LWCompatible/temp_gr/'
        tag = 'GR'
        df = pd.read_hdf('../../weights/weight_df_022398_track_GR.hdf')
        df_pair = pd.read_hdf('../../weights/weight_df_021408_track_GR.hdf')


    ##make the unweighted plots
    if weight_only == False:
        if tag != 'GR':
            fList = glob(os.path.join(fPath, '*.i3.zst'))
        else:
            fList = glob(os.path.join(fPath, 'LeptonInjector*.i3.zst'))
        if len(fList) == 0:
            raise FileNotFoundError(f'{fList} is of size 0!')
        eventInfo = get_info(fList)
        make_plots(eventInfo, tag)
    make_selected_plots(df, df_pair, tag)

if __name__ == "__main__":
    main()
##end
