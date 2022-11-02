##This file loads and plots the default PREM model
##Also provides functionality to perterb the PREM model
import numpy as np
import click
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371 ##km
CORE_RADIUS  = 0.545##aprox

def get_prem():
    ##col1 = radius in units of 6371 km
    ##col2 = earth density in units of g/cm3
    ##col3 = electron fraction
    data = np.loadtxt('EARTH_MODEL_PREM.dat')
    rad = [0] * len(data)
    density = [0] * len(data)
    ye = [0] * len(data)
    for i, d in enumerate(data):
        rad[i] = d[0]
        density[i] = d[1]
        ye[i] = d[2]

    icore = np.argmax(abs(np.diff(density)))
    core_start = rad[icore]
    print(f'Core Start: {core_start}')

    return rad, density, ye, core_start

def reweight(radius, density, core_start=CORE_RADIUS, vals=None):

    print(f'Reweighting {vals}')

    if vals == 'up':
        weight = 1.1
    if vals == 'down':
        weight = 0.9

    radius  = np.array(radius)
    density = np.array(density)

    mask = radius <= core_start
    nmask = radius > core_start

    new_density_weight = mask * weight
    weighting = nmask + new_density_weight

    new_density = density * weighting

    return new_density

def plot_prem(models=[], labels=[], tags=[]):

    if len(models) > 1:
        fig1, ax1 = plt.subplots()

    ##loop over list of values
    for model, label, tag in zip(models, labels, tags):
        radius, density = model
        fig, ax = plt.subplots()
        ax.plot(radius, density, '-', color='royalblue')
        ax.set_xlabel('Radius / 6371 km')
        ax.set_ylabel(r'Density / g/cm$^{3}$')
        ax.set_title(label)
        fig.savefig(f'model_{tag}.pdf')
        plt.close(fig)

        if len(models) > 1:
            ax1.plot(radius, density, '--', label=label)
    
    if len(models) > 1:
        ax1.set_xlabel('Radius / 6371 km')
        ax1.set_ylabel(r'Density / g/cm$^{3}$')
        ax1.legend()
        fig1.savefig(f'model_updown.pdf')
        plt.close(fig1)

##save file in exected format
def save_file(radius, density, ye, filename):
    with open(filename, mode='w') as _file:
        for r, d, y in zip(radius, density, ye):
            _str = f'{r} {d} {y}\n'
            _file.write(_str)

@click.command()
@click.option('--up', is_flag=True)
@click.option('--down', is_flag=True)
def main(up, down):
    ##get default PREM
    radius, density, ye, core_start = get_prem()

    model_list = []
    label_list = []
    tag_list   = []
    model_list.append((radius, density))
    label_list.append('Nominal PREM')
    tag_list.append('prem')
    if up == False and down == False:
        print('Performing Default Behaviour - plotting PREM')
    else:
        if up == True:
            new_density_up   = reweight(radius, density, core_start, 'up')
            model_list.append((radius, new_density_up))
            label_list.append(r'$\rho$ Core +10%')
            tag_list.append('up')
            save_file(radius, new_density_up, ye, filename='EARTH_MODEL_PREM_UP.dat')
            #np.savetxt('EARTH_MODEL_PREM_UP.dat', (radius, new_density_up, ye))
        if down == True:
            new_density_down = reweight(radius, density, core_start, 'down')
            model_list.append((radius, new_density_down))
            label_list.append(r'$\rho$ Core -10%')
            tag_list.append('down')
            #np.savetxt('EARTH_MODEL_PREM_DOWN.dat', (radius, new_density_down, ye))
            save_file(radius, new_density_down, ye, filename='EARTH_MODEL_PREM_DOWN.dat')

    plot_prem(model_list, label_list, tag_list)
    print('Finished')

if __name__ == "__main__":
    main()
##end
