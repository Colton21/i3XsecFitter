# i3XsecFitter
Analysis using LeptonInjector and nuSQuIDS to evaluate the neutrino cross section in IceCube

This project has dependencies: LeptonWeighter, nuSQuIDS, SQuIDS, nuflux.
Currently handled with specific env variables in the IceTray environment.

Usage:
------------

Propagation
============

Propagating a new flux with nuSQuIDS and creating an output file for a given xsec normalisation.

```python3 propagate_fluxes.py -n {norm} -f {flux} -m {selection} --cache```

The flags are: norm is a float, typically between 0.2 and 5, flux is either 'atmo' or 'astro', and selection is 'cascade' or 'track'.
After calculating you want the state to be saved, so run with --cache unless you're just testing.

For ease of use, also use '--auto', which will run over both selections, both fluxes, and a hard-coded range of normalisations. 

The Earth Model PREM can also be modified up or down.
This is configured just in the core region, as this is the region where we have the least amount of information.

```python3 earth_model.py --up --down```

The new earth model files can be fed into the propagation code with:

```--earth prem_file```


Reweighting
=============

To reweight to all of the listed normalisations and fluxes for a given configuration for all datasets (1e2-1e4, 1e4-1e6, 1e6-1e8) all flavours:

```python3 create_weight_df.py --do_all -s {selection}```

Combining
===========

Once the files are reweighted and updated, you need to combine them into a single file using

```create_full_df.py -ia --cache```

This will make a single easy to use pandas-readable file.

