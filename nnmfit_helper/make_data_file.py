import pandas as pd
import click
import sys, os

@click.command()
@click.argument('input_file')
@click.argument('selection')
def main(input_file, selection):
    if os.path.exists(input_file):
        df = pd.read_hdf(input_file)
    else:
        raise IOError(f'{input_file} does not exist!')

    if selection not in ['track', 'cascade']:
        raise IOError(f'Option {selection} is not valid! Must be track or cascade')
    
    _df = df
    ##nnmfit needs tracks and cascades separated
    _df = _df[_df.Selection == selection]
    
    ##nnmfit needs the livetime #### not the case I think ...###and flux divided out
    ##(maybe will be modified in the future of my scripts)
    #nnm_w = ((_df['weight1.0_atmo'].values / _df.flux_atmo.values) + (_df['weight1.0_astro'].values / _df.flux_astro.values)) / _df.LiveTime.values
    nnm_w_atmo  = _df['weight1.0_atmo'].values / _df.LiveTime.values
    nnm_w_astro = _df['weight1.0_astro'].values / _df.LiveTime.values

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

    ##get rid of un-needed keys/columns
    keyList = ['nu_energy', 'nu_zenith', 'pdg', 'fit_type_atmo', 
               'fit_type_astro', 'reco_energy', 'reco_zenith']

    dropList = []
    for key in _df.keys():
        if key not in keyList:
            dropList.append(key)
    _df = _df.drop(columns=dropList)

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


    ##nnm fit needs the name of the weight to be something specific?
    ##looks like it should be the flux name
    _df['mceq_conv_H3a_SIBYLL23c'] = nnm_w_atmo
    _df['powerlaw'] = nnm_w_astro
    _df.to_hdf('test.hdf', key='nnmfit', mode='w')

if __name__ == "__main__":
    main()
##end
