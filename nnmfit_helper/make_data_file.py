from datetime import datetime
import pandas as pd
import click
import sys, os
import numpy as np

def gauss(x, norm, peak, width):
    val = norm * np.exp(-(x-peak)**2/(2 * width**2))
    return val

@click.command()
@click.argument('input_file')
@click.option('--selection', '-s', required=True)
@click.option('--alt_weight', is_flag=True)
def main(input_file, selection, alt_weight):
    if os.path.exists(input_file):
        df = pd.read_hdf(input_file)
    else:
        raise IOError(f'{input_file} does not exist!')

    if selection not in ['track', 'cascade']:
        raise IOError(f'Option {selection} is not valid! Must be track or cascade')
    
    _df = df
    ##nnmfit needs tracks and cascades separated
    _df = _df[_df.Selection == selection]
   
    nnm_w_atmo  = _df['weight1.0_atmo'].values / _df.LiveTime.values
    nnm_w_astro = _df['weight1.0_astro'].values / _df.LiveTime.values

    ##apply weighting
    ##then re-fit, in this energy range always linear I think
    if alt_weight == True:
        alt_pulls = [1] * len(nnm_w_astro)
        print('Running with alt weights for 5 Tev to 30 TeV')
        emin = 1e3
        emax = 120e3
        peak = 30e3
        width = 8e3
        norm = 0.4

        mask_alt = (_df['nu_energy'].values > emin) & (_df['nu_energy'].values < emax)
        print(f'Num to be modified: {np.sum(mask_alt)}')
        _pull = 1 + (gauss(_df['nu_energy'].values, norm, peak, width) * mask_alt)
        print('---')
        print(f'Weights in region (astro, total): {np.sum(nnm_w_astro)}, {np.sum(nnm_w_astro+nnm_w_atmo)}')
        nnm_w_astro = nnm_w_astro * _pull
        nnm_w_atmo = nnm_w_atmo * _pull
        alt_pulls = _pull
        print(f'After (astro, total): {np.sum(nnm_w_astro)}, {np.sum(nnm_w_astro+nnm_w_atmo)}')
        print('---')

        __pulls = np.array(_pull)[_pull != 1.0]
        #print(__pulls)

        #for i, _w in enumerate(nnm_w_astro):
        #    if _df['nu_energy'].values[i] > 1e3 and _df['nu_energy'].values[i] < 5e3:
        #        _pull = 1 + gauss(_df['nu_energy'].values[i], norm, peak, width)
        #        nnm_w_atmo[i] = _pull * _w
        #        alt_pulls[i] = _pull
        

    ##fit_type_atmo/fit_type_astro are not strings!
    atmo_fits = [0] * len(_df.fit_type_atmo)
    for i, ft in enumerate(_df.fit_type_atmo):
        if ft == 'lin' or ft == 'linear':
           atmo_fits[i] = 0
        elif ft == 'exp' or ft == 'exponential':
           atmo_fits[i] = 1
        else:
            raise ValueError(f'Value of {ft} does not match lin or exp!')
    astro_fits = [0] * len(_df.fit_type_astro)
    for i, ft in enumerate(_df.fit_type_astro):
        if ft == 'lin' or ft == 'linear':
           astro_fits[i] = 0
        elif ft == 'exp' or ft == 'exponential':
           astro_fits[i] = 1
        else:
            raise ValueError(f'Value of {ft} does not match lin or exp!')

    _df['fit_type_atmo'] = atmo_fits
    _df['fit_type_astro'] = astro_fits

    ##Theano needs the columns to be simple - fit_params_astro --> fit_paramA_astro, fitparamB_astro
    paramA_atmo = [0] * len(_df.fit_params_atmo)
    paramB_atmo = [0] * len(_df.fit_params_atmo)
    for i, (a, b) in enumerate(_df.fit_params_atmo):
        paramA_atmo[i] = a
        paramB_atmo[i] = b
    

    paramA_astro = [0] * len(_df.fit_params_astro)
    paramB_astro = [0] * len(_df.fit_params_astro)
    for i, (a, b) in enumerate(_df.fit_params_astro):
        paramA_astro[i] = a
        paramB_astro[i] = b
        
        if alt_weight == True:
            paramB_astro[i] = alt_pulls[i] * b

    ##get rid of un-needed keys/columns
    keyList = ['nu_energy', 'nu_zenith', 'nu_azimuth', 'pdg', 'fit_type_atmo', 
               'fit_type_astro', 'reco_energy', 'reco_zenith', 'reco_azimuth']

    ##also add the systematics to the list
    ##but format the names for NNMFit
    keySysList = ['ice_absorption', 'ice_scattering', 'dom_efficiency',
                  'ice_anisotropy_scale', 'hole_ice_forward_p0', 
                  'hole_ice_forward_p1']
    keyList.extend(keySysList)
    sysNameMap = {
        'ice_absorption': 'IceAbsorption',
        'ice_scattering': 'IceScattering',
        'dom_efficiency': 'DOMEfficiency',
        'ice_anisotropy_scale': 'IceAnisotropyScale',
        'hole_ice_forward_p0': 'HoleIceForward_p0',
        'hole_ice_forward_p1': 'HoleIceForward_p1'
    }

    dropList = []
    for key in _df.keys():
        if key not in keyList:
            dropList.append(key)
    _df = _df.drop(columns=dropList)
    _df.rename(columns=sysNameMap, inplace=True)

    _df['fit_paramA_atmo'] = paramA_atmo
    _df['fit_paramB_atmo'] = paramB_atmo
    _df['fit_paramA_astro'] = paramA_astro
    _df['fit_paramB_astro'] = paramB_astro

    ##needs "_exists" columns
    e_exists = _df.reco_energy > 0
    d_exists = _df.reco_zenith > -180
    _df['reco_energy_exists'] = e_exists
    _df['reco_dir_exists'] = d_exists
    _df['reco_energy_fit_status'] = [0] * len(e_exists)
    _df['reco_dir_fit_status'] = [0] * len(d_exists)

    date = datetime.now().strftime('%Y-%m-%d')
    ##nnm fit needs the name of the weight to be something specific?
    ##looks like it should be the flux name
    _df['mceq_conv_H3a_SIBYLL23c'] = nnm_w_atmo
    _df['mceq_conv_H4a_SIBYLL23c'] = nnm_w_atmo
    _df['powerlaw'] = nnm_w_astro
    if alt_weight == True:
        _df['alt_pulls'] = alt_pulls
        _df.to_hdf(f'data/{selection}_alt_weight_{date}.hdf', key='nnmfit', mode='w')
    else:
        _df.to_hdf(f'data/{selection}_{date}.hdf', key='nnmfit', mode='w')

    print('-'*20)
    print(f'The final keys are: {_df.keys()}')
    if alt_weight == True:
        print(f'Created output file data/{selection}_alt_weight_{date}.hdf')
    else:
        print(f'Created output file data/{selection}_{date}.hdf')

if __name__ == "__main__":
    main()
##end
