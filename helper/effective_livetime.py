import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_ef(binning, xlabel, tag, xscale='linear'):
    df = pd.read_hdf('../dataframes/li_total.hdf5')
    ##remove glashow events
    df_gr = df[df.IntType == 'GR']
    df = df[(df.IntType == 'CC') | (df.IntType == 'NC')]

    df_c = df[df.Selection == 'cascade']
    df_t = df[df.Selection == 'track']

    bin_centers = (binning[1:] + binning[:-1])/2

    eff_l_c = []
    eff_l_t = []
    eff_l_gr = []
    mc_c  = []
    mc_t  = []
    mc_gr = []

    year_scaling = 60 * 60 * 24 * 365
   
    scales = [0.9, 1.0, 1.1]
    weightListC = []
    weightListT = []
    for s in scales:
        emptyListC = []
        emptyListT = []
        weightListC.append(emptyListC)
        weightListT.append(emptyListT)

    for i in range(len(binning)):
        if i == 0:
            continue
        else:
            if tag == 'energy':
                if i == len(binning)-1:
                    slice_c  = df_c[(df_c.nu_energy <= binning[i])   & (df_c.nu_energy >= binning[i-1])]
                    slice_t  = df_t[(df_t.nu_energy <= binning[i])   & (df_t.nu_energy >= binning[i-1])]
                    slice_gr = df_gr[(df_gr.nu_energy <= binning[i]) & (df_gr.nu_energy >= binning[i-1])]
                else:
                    slice_c  = df_c[(df_c.nu_energy < binning[i])   & (df_c.nu_energy >= binning[i-1])]
                    slice_t  = df_t[(df_t.nu_energy < binning[i])   & (df_t.nu_energy >= binning[i-1])]
                    slice_gr = df_gr[(df_gr.nu_energy < binning[i]) & (df_gr.nu_energy >= binning[i-1])]
                
            elif tag == 'zenith':
                if i == len(binning)-1:
                    slice_c  = df_c[(np.cos(df_c.nu_zenith) <= binning[i])   & (np.cos(df_c.nu_zenith) >= binning[i-1])]
                    slice_t  = df_t[(np.cos(df_t.nu_zenith) <= binning[i])   & (np.cos(df_t.nu_zenith) >= binning[i-1])]
                    slice_gr = df_gr[(np.cos(df_gr.nu_zenith) <= binning[i]) & (np.cos(df_gr.nu_zenith) >= binning[i-1])]
                else:
                    slice_c  = df_c[(np.cos(df_c.nu_zenith) < binning[i])   & (np.cos(df_c.nu_zenith) >= binning[i-1])]
                    slice_t  = df_t[(np.cos(df_t.nu_zenith) < binning[i])   & (np.cos(df_t.nu_zenith) >= binning[i-1])]
                    slice_gr = df_gr[(np.cos(df_gr.nu_zenith) < binning[i]) & (np.cos(df_gr.nu_zenith) >= binning[i-1])]

        if len(slice_c.index.values) == 0:
            eff_l_c.append(0)
        else:
            w_c  = slice_c['weight1.0'].values / slice_c.LiveTime.values[0]
            eff_l_c.append(np.sum(w_c)/np.sum(w_c*w_c))
        if len(slice_t.index.values) == 0:
            eff_l_t.append(0)
        else:
            w_t  = slice_t['weight1.0'].values / slice_t.LiveTime.values[0]
            eff_l_t.append(np.sum(w_t)/np.sum(w_t*w_t))
        
        mc_t.append(len(slice_t.index.values))
        mc_c.append(len(slice_c.index.values))

        j = 0
        for scale in scales:
            key = f'weight{scale}'
            
            if len(slice_c.index.values) == 0:
                val_c = 0
            else:
                w_cs = slice_c[key].values / slice_c.LiveTime.values[0]
                val_c = np.sum(w_cs)/np.sum(w_cs*w_cs)
                val_c = val_c/year_scaling
            
            if len(slice_t.index.values) == 0:
                val_t = 0
            else:
                w_ts = slice_t[key].values / slice_t.LiveTime.values[0]
                val_t = np.sum(w_ts)/np.sum(w_ts*w_ts)
                val_t = val_t/year_scaling
            
            weightListC[j].append(val_c)
            weightListT[j].append(val_t)
            j += 1

        if len(slice_gr.index.values) != 0:
            w_gr = slice_gr['weight1.0'].values / slice_gr.LiveTime.values[0]
            eff_l_gr.append(np.sum(w_gr)/np.sum(w_gr*w_gr))
        else:
            eff_l_gr.append(0)
        mc_gr.append(len(slice_gr.index.values))

    eff_l_t = np.array(eff_l_t)
    eff_l_c = np.array(eff_l_c)
    eff_l_gr = np.array(eff_l_gr)

    fig1, ax1 = plt.subplots()
    ax1.plot(bin_centers, eff_l_t/year_scaling,  label='Track', color='royalblue', marker='o')
    ax1.plot(bin_centers, eff_l_c/year_scaling,  label='Cascade', color='goldenrod', marker='o')
    ax1.plot(bin_centers, eff_l_gr/year_scaling, label='GR Events', color='firebrick', marker='o', linewidth=0)
    ax1.set_xscale(xscale)
    ax1.set_yscale('log')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Effective Livetime [yr]')
    ax1.legend(loc=0, title='Selection')
    ax1.set_title('Effective Livetime per Selection')
    fig1.tight_layout()
    fig1.savefig(f'plots/effective_livetime_{tag}_both_with_gr.pdf')

    fig2, ax2 = plt.subplots()
    ax2.plot(bin_centers, eff_l_t/year_scaling,  label='Track', color='royalblue', marker='o')
    ax2.plot(bin_centers, eff_l_c/year_scaling,  label='Cascade', color='goldenrod', marker='o')
    ax2.set_xscale(xscale)
    ax2.set_yscale('log')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Effective Livetime [yr]')
    ax2.legend(loc=0, title='Selection')
    ax2.set_title('Effective Livetime per Selection')
    fig2.tight_layout()
    fig2.savefig(f'plots/effective_livetime_{tag}_both.pdf')

    fig2t, ax2t = plt.subplots()
    for l, scale in zip(weightListT, scales):
        ax2t.plot(bin_centers, l,  label=f'{scale}', marker='o')
    ax2t.set_xscale(xscale)
    ax2t.set_yscale('log')
    ax2t.set_xlabel(xlabel)
    ax2t.set_ylabel('Effective Livetime [yr]')
    ax2t.legend(loc=0, title='Scaling')
    ax2t.set_title('Track Effective Livetime per Scale')
    fig2t.tight_layout()
    fig2t.savefig(f'plots/effective_livetime_{tag}_track_scale.pdf')

    fig2c, ax2c = plt.subplots()
    for l, scale in zip(weightListC, scales):
        ax2c.plot(bin_centers, l,  label=f'{scale}', marker='o')
    ax2c.set_xscale(xscale)
    ax2c.set_yscale('log')
    ax2c.set_xlabel(xlabel)
    ax2c.set_ylabel('Effective Livetime [yr]')
    ax2c.legend(loc=0, title='Scaling')
    ax2c.set_title('Cascade Effective Livetime per Scale')
    fig2c.tight_layout()
    fig2c.savefig(f'plots/effective_livetime_{tag}_cascade_scale.pdf')

    fig3, ax3 = plt.subplots()
    ax3.plot(bin_centers, mc_t,  label='Track', color='royalblue', marker='o')
    ax3.plot(bin_centers, mc_c,  label='Cascade', color='goldenrod', marker='o')
    ax3.set_xscale(xscale)
    ax3.set_yscale('log')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Number of Events (unweighted)')
    ax3.legend(loc=0, title='Selection')
    ax3.set_title('Events per Selection')
    fig3.tight_layout()
    fig3.savefig(f'plots/events_unweighted_{tag}.pdf')

def main():
    binning = np.logspace(2, 8, 14)
    calc_ef(binning, 'True Neutrino Energy [GeV]', tag='energy', xscale='log')

    cosz_binning = np.linspace(-1, 1, 10)
    calc_ef(cosz_binning, r'True Neutrino cos($\theta_{zen}$)', tag='zenith')

if __name__ == "__main__":
    main()

##end
