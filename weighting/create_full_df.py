import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
from glob import glob
import time

def gather_files(dataset, selection, CCNC, modGamma=False):
    cc_nc = ['CC', 'NC', 'GR']
    selections = ['track', 'cascade']
    datasets = [21395, 21396, 21397, 21398, 21399, 21400, 21401, 21402, 21403, 21408]
    mc_dir = '/data/user/chill/icetray_LWCompatible/weights'
    if dataset not in datasets or selection not in selections or CCNC not in cc_nc:
        raise NotImplementedError('Bad input to find file')
    if modGamma == False:
        df_file = glob(f'{mc_dir}/weight_df_0{dataset}_{selection}_{CCNC}.hdf')
    if modGamma == True:
        df_file = glob(f'{mc_dir}/weight_df_0{dataset}_{selection}_{CCNC}_modGamma.hdf')

    print(df_file)
    if len(df_file) != 1:
        raise IOError(f'Num Files Found With Glob: {len(df_file)} != 1 !')
    df = pd.read_hdf(df_file[0])
    return df

def df_wrapper(import_all, cache, modGamma=False):
    if not import_all:
        raise NotImplementedError('For now load all files please using "-ia" ')
    cc_nc = ['CC', 'NC']
    selections = ['track', 'cascade']
    datasets = [21395, 21396, 21397, 21398, 21399, 21400, 21401, 21402, 21403, 21408]
    df_list = []
    for dataset in datasets:
        for selection in selections:
            for CCNC in cc_nc:
                if dataset == 21408 and CCNC == 'CC':
                    CCNC = 'GR'
                if dataset == 21408 and CCNC == 'NC':
                    continue
                df = gather_files(dataset, selection, CCNC, modGamma)
                df_list.append(df)    

    if len(df_list) != 38:
        raise IOError(f'Number of data frames to be combined is wrong! Found Length: {len(df_list)}')

    df_total = pd.concat(df_list, sort=False)
    if cache:
        if not modGamma:
            df_total.to_hdf(f'/data/user/chill/icetray_LWCompatible/dataframes/li_total.hdf5', key='df', mode='w')
        if modGamma:
            df_total.to_hdf(f'/data/user/chill/icetray_LWCompatible/dataframes/li_total_modGamma.hdf5', key='df', mode='w')

def df_data_wrapper(cache):
    df_c = pd.read_hdf('/data/user/chill/icetray_LWCompatible/weights/weight_df_data_cascade.hdf')
    df_t = pd.read_hdf('/data/user/chill/icetray_LWCompatible/weights/weight_df_data_track.hdf')

    df_total = pd.concat([df_t, df_c], sort=False)
    if cache:
        df_total.to_hdf('/data/user/chill/icetray_LWCompatible/dataframes/data_total.hdf5', key='df', mode='w')

@click.command()
@click.option('--import_all', '-ia', is_flag=True)
@click.option('--data', is_flag=True)
@click.option('--cache', is_flag=True)
@click.option('--mod_gamma', '-g', is_flag=True)
def main(import_all, data, cache, mod_gamma):
    if cache == False:
        print('============ RUNNING IN TEST MODE! ===============')
        print('====== YOU ARE NOT CACHING THIS DATAFRAME! =======')
        time.sleep(0.5)
    if data == True:
        print('Combining data')
        df_data_wrapper(cache)
        return

    df_wrapper(import_all, cache, mod_gamma)

    print("Done")

if __name__ == "__main__":
    main()

##end
