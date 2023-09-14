import os
import pandas as pd
import matplotlib.pyplot as plt
import click

def check_sys_dist(df):
    keys_to_check = ['ice_absorption', 'ice_scattering', 'dom_efficiency',
       'ice_anisotropy_scale', 'hole_ice_forward_p0', 'hole_ice_forward_p1',]
    for key in keys_to_check:
        vals = df[f'{key}'].values 

        fig, ax = plt.subplots()
        ax.hist(vals, histtype='step', color='royalblue')
        ax.set_xlabel(key)
        ax.set_ylabel('Entries')
        fig.savefig(f'sys_plots/{key}_hist.pdf')
        plt.close(fig)

@click.command()
@click.option('--infile', '-i', default=None)
def main(infile):
    if infile == None:
        infile = '../../dataframes/li_total.hdf5'
    if infile != None:
        if not os.path.exists(infile):
            raise FileNotFoundError(f'{infile} not found!')
    print(f'Checking {infile}')
    df = pd.read_hdf(infile)
    check_sys_dist(df)

    print('Done')

if __name__ == "__main__":
    main()
##end
