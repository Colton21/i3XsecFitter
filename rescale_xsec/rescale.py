import tables
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import click

sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter/')
from configs.config import config

def open_default(path="/data/user/chill/snowstorm_nugen_ana/xsec/data/csms.h5"):
    if os.path.exists(path):
        t = tables.open_file(path)
    else:
        raise IOError(f"File at path {path} does not exist!")
    return t

def get_node(t, n_name):
    node = t.get_node("/" + n_name)
    vals = node.read()
    return vals

def plot_vals_default(e_range, vals_list):
    fig1, ax1 = plt.subplots()
    ax1.plot(e_range-9, 10**vals_list[0], label=r'$\nu$ CC')
    ax1.plot(e_range-9, 10**vals_list[1], label=r'$\bar{\nu}$ CC')
    ax1.plot(e_range-9, 10**vals_list[2], label=r'$\nu$ NC')
    ax1.plot(e_range-9, 10**vals_list[3], label=r'$\bar{\nu}$ NC')
    ax1.set_xlabel('log(E) [GeV]')
    ax1.set_ylabel(r'$\sigma_{CSMS}$ [cm$^2$]')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('CSMS Cross Section')
    fig1.savefig('/data/user/chill/snowstorm_nugen_ana/xsec/plots/csms_nominal.pdf')
    plt.close(fig1)

def plot_vals_scaled(e_range, vals_list, scaling, ccnc):
    if scaling > 5.0 or scaling < 0.0:
        raise ValueError(f"Value of {scaling} not between 0.0 and 5.0")
    up = scaling
    down = 1. / up
    
    fig1, ax1 = plt.subplots()
    if ccnc == 'cc':
        ax1.plot(e_range-9, 10**vals_list[0], color='royalblue', label=r'$\nu$ CC')
        ax1.plot(e_range-9, 10**vals_list[1], color='goldenrod', label=r'$\bar{\nu}$ CC')
        ax1.plot(e_range-9, 10**vals_list[0] * up, linestyle='dashdot', color='royalblue', 
                label=r'$\nu$ CC +/-' + str(scaling) + 'x')
        ax1.plot(e_range-9, 10**vals_list[1] * up, linestyle='dashdot', color='goldenrod', 
                label=r'$\bar{\nu}$ CC +/-' + str(scaling) + 'x')
        ax1.plot(e_range-9, 10**vals_list[0] * down, linestyle='dashdot', color='royalblue')
        ax1.plot(e_range-9, 10**vals_list[1] * down, linestyle='dashdot', color='goldenrod') 
    if ccnc == 'nc':
        ax1.plot(e_range-9, 10**vals_list[2], color='indigo',    label=r'$\nu$ NC')
        ax1.plot(e_range-9, 10**vals_list[3], color='firebrick', label=r'$\bar{\nu}$ NC')
        ax1.plot(e_range-9, 10**vals_list[2] * up, linestyle='dashdot', color='indigo',
                label=r'$\nu$ NC +/-' + str(scaling) + 'x')
        ax1.plot(e_range-9, 10**vals_list[3] * up, linestyle='dashdot', color='firebrick',
                label=r'$\bar{\nu}$ NC +/-' + str(scaling) + 'x')
        ax1.plot(e_range-9, 10**vals_list[2] * down, linestyle='dashdot', color='indigo')
        ax1.plot(e_range-9, 10**vals_list[3] * down, linestyle='dashdot', color='firebrick')
    
    ax1.set_xlabel('log(E) [GeV]')
    ax1.set_ylabel(r'$\sigma_{CSMS}$ [cm$^2$]')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('CSMS Cross Section')
    fig1.savefig(f'/data/user/chill/snowstorm_nugen_ana/xsec/plots/csms_{ccnc}_{scaling}.pdf')
    plt.close(fig1)

##scaling for CC & NC are the same
def scale_table(vals_list, name_list, scaling):
    scaled_vals_list = []
    for name, vals in zip(name_list, vals_list):
        if name not in ['energies', 'zs']:
            scaled_vals_list.append(vals + np.log10(scaling))
        if name in ['energies', 'zs']:
            scaled_vals_list.append(vals)
    return scaled_vals_list

##scaling for CC & NC differently
def scale_table_CCNC(vals_list, name_list, scalingCC, scalingNC):
    scaled_vals_list = []
    for name, vals in zip(name_list, vals_list):
        if name not in ['energies', 'zs']:
            if name in ['dsdy_CC_nu', 'dsdy_CC_nubar', 's_CC_nu', 's_CC_nubar']:
                scaled_vals_list.append(vals + np.log10(scalingCC))
            if name in ['dsdy_NC_nu', 'dsdy_NC_nubar', 's_NC_nu', 's_NC_nubar']:
                scaled_vals_list.append(vals + np.log10(scalingNC))
        if name in ['energies', 'zs']:
            scaled_vals_list.append(vals)
    return scaled_vals_list

def save_scaled(scaled_vals_list, name_list, scaling):
    filename = f'/data/user/chill/snowstorm_nugen_ana/xsec/data/csms_{scaling}.h5'
    with tables.open_file(filename, 'w') as open_file:
            for name, vals in zip(name_list, scaled_vals_list):
                open_file.create_array('/', str(name), vals, '')
    print(f"Finished creating table for scale: {scaling}")

def save_scaled_CCNC(scaled_vals_list, name_list, scalingCC, scalingNC):
    filename = f'/data/user/chill/snowstorm_nugen_ana/xsec/data/csms_{scalingCC}CC_{scalingNC}NC.h5'
    with tables.open_file(filename, 'w') as open_file:
            for name, vals in zip(name_list, scaled_vals_list):
                open_file.create_array('/', str(name), vals, '')
    print(f"Finished creating table for scale: {scalingCC} CC & {scalingNC} NC")

@click.command()
@click.option("--in_file", "-i")
@click.option("--norm", "-n")
@click.option('--plot', '-p', is_flag=True)
def main(in_file, norm, plot):
    if norm is not None:
        try:
            norm = float(norm)
        except:
            raise ValueError("Please give the csms scaling as a value")
        scaling_list = [norm]
    else:
        raise ValueError("Please specify the normalisation using '-n' ")

    if in_file is None:
        t = open_default()
    else:
        t = open_default(in_file)

    ######### used for plotting ###########################
    #in units of log(E) eV, starting at 10 GeV
    e_range = get_node(t, 'energies') 
    nu_cc = []
    nubar_cc = []
    nu_nc = []
    nubar_nc = []
    node_list = ['s_CC_nu', 's_CC_nubar', 's_NC_nu', 's_NC_nubar']
    vals_list = [nu_cc, nubar_cc, nu_nc, nubar_nc]
    for i, n_name in enumerate(node_list):
        vals_list[i] = get_node(t, n_name)
    if plot == True:
        plot_vals_default(e_range, vals_list)
        plot_vals_scaled(e_range, vals_list, scaling_list[0], 'cc')
        plot_vals_scaled(e_range, vals_list, scaling_list[0], 'nc')
    #######################################################

    for_scaling_list = []
    name_list = []
    node_list = t.list_nodes('/')
    for node in node_list:
        name = node.name
        for_scaling_list.append(node.read())
        name_list.append(name)


    print("Reminder, doing CC and NC scales independently!")
    scaling_listCC = config.normList
    scaling_listNC = scaling_listCC

    for scaleCC in scaling_listCC:
        for scaleNC in scaling_listNC:
            scaleCC = float(scaleCC)
            scaleNC = float(scaleNC)

            scaled_vals_list = scale_table_CCNC(for_scaling_list, name_list, scaleCC, scaleNC)
            save_scaled_CCNC(scaled_vals_list, name_list, scaleCC, scaleNC)

if __name__ == "__main__":
    main()

##end
