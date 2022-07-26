# i3XsecFitter
Analysis using LeptonInjector and nuSQuIDS to evaluate the neutrino cross section in IceCube

This project has dependencies: LeptonWeighter, nuSQuIDS, SQuIDS, nuflux.
Currently handled with specific env variables in the IceTray environment.

Usage:

============
Propagation
============

Propagating a new flux with nuSQuIDS and creating an output file for a given xsec normalisation.

```python3 propagate_fluxes.py -n {norm} -f {flux} -m {selection} --cache```

The flags are: norm is a float, typically between 0.2 and 5, flux is either 'atmo' or 'astro', and selection is 'cascade' or 'track'.
After calculating you want the state to be saved, so run with --cache unless you're just testing.

For ease of use, also use '--auto', which will run over both selections, both fluxes, and a hard-coded range of normalisations. 

=============
Reweighting
=============

To reweight to all of the listed normalisations and fluxes for a given configuration for all datasets (1e2-1e4, 1e4-1e6, 1e6-1e8) all flavours:

```python3 create_weight_df.py --do_all -s {selection}```

