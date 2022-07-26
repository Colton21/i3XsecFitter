###flux functions to be fed into nuSQuIDS

##icecube specific
import nuflux
import numpy as np
import matplotlib.pyplot as plt

##spectrum only used if mode set to simple
def InitAtmFlux(mode='modern'):
    if mode == 'classic':
        model_option = 'honda2006'
    elif mode == 'modern':
        model_option = 'H3a_SIBYLL23C'
    elif mode == 'experimental':
        model_option = 'H3a_SIBYLL21'
    else:
        raise NotImplementedError(f'{mode} flux option not implemented for this module!')
    print(f"Using {model_option} as Atm. flux")
    atm_flux = nuflux.makeFlux(model_option)
    return atm_flux 

def AtmFlux(atm_flux, e, z, flavour, flag=None, spectrum=-2.4):
    if flag == None:
        if atm_flux is None:
            raise ValueError("Initialise flux first, then pass into AtmFlux")
        f_atm = atm_flux.getFlux(flavour, e, z)
        
    if flag == 'simple':
        ## -3.5 +/- 0.1, 0.2
        eprior = 100e3
        norm = 2.e-18        
        f_atm = norm * np.power(e/eprior, spectrum)
    
    return f_atm

def DiffFlux(e, mode, spectrum='default'):
    eprior = 100e3

    if spectrum != 'default':
        if spectrum >= 0:
            raise ValueError(f'Spectral Index should be less than 0 not {spectrum}')
        e = np.asarray(e)
        mode = np.asarray(mode)
        norm = np.zeros(e.shape)
        strList = ['classic', 'cascade', 'track']
        for s in strList:
            mask = mode == s
            _e = e[mask]
            if s == 'classic':
                norm[mask] = 1.36e-18
            if s == 'track':
                norm[mask] = 1.44e-18
            if s == 'cascade':
                norm[mask] = 1.66e-18
        return norm * np.power(e/eprior, spectrum)

    '''
    ##gamma provided from script
    if spectrum != 'default':
        if mode == 'track':
            norm = 1.44e-18
        elif mode == 'cascade':
            norm = 1.66e-18
        elif mode == 'classic':
            norm = 1.36e-18
        else:
            raise NotImplementedError(f'{mode} not supported for this module!')
        flux_ast = norm * np.power(e/eprior, spectrum)
        return flux_ast
    '''
    ##gamma not fixed by script
    if mode == 'classic':
        norm = 1.36e-18
        spectrum = -2.37

    elif mode == 'track':
        ## from 10 yr diffuse track
        ## https://arxiv.org/pdf/1908.09551.pdf
        norm = 1.44e-18
        spectrum = -2.28

    elif mode == 'cascade':
        ## 6 yr diffuse cascade paper
        ## https://arxiv.org/pdf/2001.09520.pdf
        norm = 1.66e-18
        spectrum = -2.53
    else:
        raise NotImplementedError(f'{mode} not supported for this module!')
    flux_ast = norm * np.power(e/eprior, spectrum)
    return flux_ast

def TranslateToPDG(flav, typ):
    pdg_list = [12, 14, 16, -12, -14, -16]
    if flav < 0 or flav >= 3:
        raise ValueError(f'flav should be 0, 1, or 2 not {flav}')
    if typ not in [0, 1]:
        raise ValueError(f'typ (nu/nubar) should be 0 or 1 {typ}')
    nu_ind = flav + (typ * 3)
    return pdg_list[nu_ind]

def plot_fluxes():

    atm_flux = InitAtmFlux('modern')
    atm_flux_classic = InitAtmFlux('classic')
    e_range = np.logspace(2, 8, 200) #already in GeV
    c_zen_range = np.linspace(-1, 1, 100)    
    pdg_list = [12, 14, 16, -12, -14, -16]

    atm_nue      = np.zeros(len(e_range))
    atm_numu     = np.zeros(len(e_range))
    atm_nutau    = np.zeros(len(e_range))
    atm_nuebar   = np.zeros(len(e_range))
    atm_numubar  = np.zeros(len(e_range))
    atm_nutaubar = np.zeros(len(e_range))
    atm_list = [atm_nue, atm_numu, atm_nutau, atm_nuebar, atm_numubar, atm_nutaubar] 
    atm_nue_classic      = np.zeros(len(e_range))
    atm_numu_classic     = np.zeros(len(e_range))
    atm_nutau_classic    = np.zeros(len(e_range))
    atm_nuebar_classic   = np.zeros(len(e_range))
    atm_numubar_classic  = np.zeros(len(e_range))
    atm_nutaubar_classic = np.zeros(len(e_range))
    atm_list_classic = [atm_nue_classic, atm_numu_classic, atm_nutau_classic, atm_nuebar_classic, atm_numubar_classic, atm_nutaubar_classic] 

    diff_nue_track      = np.zeros(len(e_range))
    diff_numu_track     = np.zeros(len(e_range))
    diff_nutau_track    = np.zeros(len(e_range))
    diff_nuebar_track   = np.zeros(len(e_range))
    diff_numubar_track  = np.zeros(len(e_range))
    diff_nutaubar_track = np.zeros(len(e_range))
    diff_track_list = [diff_nue_track, diff_numu_track, diff_nutau_track, diff_nuebar_track, diff_numu_track, diff_nutaubar_track]
    diff_nue_cascade      = np.zeros(len(e_range))
    diff_numu_cascade     = np.zeros(len(e_range))
    diff_nutau_cascade    = np.zeros(len(e_range))
    diff_nuebar_cascade   = np.zeros(len(e_range))
    diff_numubar_cascade  = np.zeros(len(e_range))
    diff_nutaubar_cascade = np.zeros(len(e_range))
    diff_cascade_list = [diff_nue_cascade, diff_numu_cascade, diff_nutau_cascade, diff_nuebar_cascade, diff_numu_cascade, diff_nutaubar_cascade]

    for atm, atm_classic, diff_t, diff_c, pdg in zip(atm_list, atm_list_classic, diff_track_list, diff_cascade_list, pdg_list):
        for ind, e in enumerate(e_range):
            f_atm = []
            f_atm_classic = []
            f_diff_t = []
            f_diff_c = []
            for c_zen in c_zen_range:
                f_atm.append(AtmFlux(atm_flux, e, c_zen, pdg))
                if pdg != 16 and pdg != -16:
                    f_atm_classic.append(AtmFlux(atm_flux_classic, e, c_zen, pdg))
                f_diff_t.append(DiffFlux(e, 'track'))
                f_diff_c.append(DiffFlux(e, 'cascade'))
            atm[ind] = np.mean(f_atm)
            atm_classic[ind] = np.mean(f_atm_classic)
            diff_t[ind] = np.mean(f_diff_t)
            diff_c[ind] = np.mean(f_diff_c)

    atm_nue_ave = (atm_nue + atm_nuebar) / 2
    atm_nue_ave_classic = (atm_nue_classic + atm_nuebar_classic) / 2
    atm_numu_ave = (atm_numu + atm_numubar) / 2

    fig1, ax1 = plt.subplots()
    ax1.plot(e_range, atm_nue_ave,   color='royalblue', label=r'Ave. Atm $\nu_{e}$ H3a_SIBYLL23c')
    ax1.plot(e_range, atm_nue_ave_classic, color='goldenrod', label=r'Ave. Atm $\nu_{e}$ Honda2006')
    #ax1.plot(e_range, atm_numu_ave,  color='goldenrod', label=r'Ave. Atm $\nu_{\mu}$')
    #ax1.plot(e_range, atm_nutau, color='salmon',    label=r'Atm $\nu_{\tau}$')
    #ax1.plot(e_range, atm_nuebar,   linestyle='dashdot', color='royalblue', label=r'Atm $\bar{\nu_{e}}$')
    #ax1.plot(e_range, atm_numubar,  linestyle='dashdot', color='goldenrod', label=r'Atm $\bar{\nu_{\mu}}$')
    #ax1.plot(e_range, atm_nutaubar, linestyle='dashdot', color='salmon',    label=r'Atm $\bar{\nu_{\tau}}$')
    ax1.plot(e_range, diff_nue_track,    color='green', label=r'Diff $\nu_{\chi}$ Track Params')
    ax1.plot(e_range, diff_nue_cascade,  color='salmon', label=r'Diff $\nu_{\chi}$ Cascade Params')
    #ax1.plot(e_range, diff_nuebar_track, color='coral', linestyle='dashdot',
    #            label=r'Diff $\bar{\nu_{\chi}}$') 
    ax1.set_xlabel('Neutrino Energy [GeV]')
    ax1.set_ylabel(r'Zen. Averaged Flux [GeV cm$^2$ Str s]$^{-1}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Input Flux')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('plots/input_flux.pdf')
    print("Created flux plot")

if __name__ == "__main__":
    plot_fluxes()

##end
