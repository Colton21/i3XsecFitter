##get the unweighted MC, plot in various binning configurations
##sample until bin has ~10 events, or until hitting a detector/physics boundary
import os, sys
import numpy as np
import pandas as pd

def main():
    df_dir = '/data/user/chill/icetray_LWCompatible/dataframes/'
    print("Getting Lepton Injector Monte Carlo")
    df_li = pd.read_hdf(os.path.join(df_dir, 'li_total.hdf5'))
    df_li_c = df_li[df_li.Selection == 'cascade']
    df_li_t = df_li[df_li.Selection == 'track']
    ##also cut away the GR events
    df_li_c = df_li_c[(df_li_c.IntType == 'CC') | (df_li_c.IntType == 'NC')]
    df_li_t = df_li_t[(df_li_t.IntType == 'CC') | (df_li_t.IntType == 'NC')]

    ##cascade reco limit ~10 degrees
    z_lim_c = np.cos(10/180*3.1415)
    zmin = -1
    zmax = 1
    #zrange_c = int((zmax - zmin)/z_lim_c)
    #print(f'Maximum number of zenith bins for cascades are: {zrange_c}')

    e_lim_c = 1
    emin = 2.6
    emax = 8
    
    e = df_li_c.reco_energy.values
    z = np.cos(df_li_c.reco_zenith.values)
    print(f'Min Energy {np.min(e)}')

    #for enbins_c in range(20, 100):
    #    for znbins_c in range(5, 18):
    for enbins_c in range(12, 25):
        for znbins_c in range(3, 16):
            ebins_c = np.logspace(emin, emax, enbins_c)
            zbins_c = np.linspace(zmin, zmax, znbins_c)
            vals, xedges, yedges = np.histogram2d(e, z, bins=(ebins_c, zbins_c))
            i_min = np.argmin(vals, axis=0)
            j_min = np.argmin(vals[i_min[0]])
            if vals[i_min[0]][j_min] < 1:
                continue
            print('---')
            print(f'nEng={enbins_c}, nZen={znbins_c}')
            print(f'e={xedges[i_min[0]]}, z={yedges[j_min]}')
            print(f'Minimum Bin Val: {vals[i_min[0]][j_min]}')
            #j_min = np.argmin(vals[i_min])
            #print(f'Minimum Bin: {i_min},{j_min} - E/Z {edges[i_min][j_min]}')

if __name__ == "__main__":
    main()
##end
