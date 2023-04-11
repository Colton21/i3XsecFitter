class config:
    #track_reco = 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Neutrino'
    #track_reco = 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon'
    track_reco = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon'
    cascade_reco = 'cscdSBU_MonopodFit4_noDC'

    path_track = '/data/ana/Diffuse/AachenUpgoingTracks/exp/Pass2'
    track_year_list = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn/final_cascade'
    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn/final_cascade_hansbdt'
    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn' #PATH IS DEPRICATED!
    #path_cascade  = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2016/burn/'
    #path_cascade  = '/data/user/zzhang1/pass2_GlobalFit/data'
    path_cascade  = '/data/ana/analyses/diffuse/cascades/pass2/data/'
    cascade_year_list = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]


    good_run_year_list = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    
    ## best fit values info
    ## from Joeran's thesis
    track_atmo_norm = 1.21

    ## from cascade paper
    ### https://arxiv.org/pdf/2001.09520.pdf
    cascade_atmo_norm = 1.07

    fluxPath = '/data/user/chill/icetray_LWCompatible/nuSQuIDS_propagation_files'
    xsecPath = '/data/user/chill/snowstorm_nugen_ana/xsec/data/'

    normList = [1e-9, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 
                1.8, 1.9, 2.0, 3.0]

##end
