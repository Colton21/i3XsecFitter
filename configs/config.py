class config:
    #track_reco = 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Neutrino'
    #track_reco = 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon'
    track_reco = 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon'
    cascade_reco = 'cscdSBU_MonopodFit4_noDC'

    path_track = '/data/ana/Diffuse/AachenUpgoingTracks/exp/Pass2/IC2013'

    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn/final_cascade'
    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn/final_cascade_hansbdt'
    #path_cascade = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2013/burn' #PATH IS DEPRICATED!
    path_cascade  = '/data/user/zzhang1/pass2_GlobalFit/data/IC86_2016/burn/'


    ## best fit values info
    ## from Joeran's thesis
    track_atmo_norm = 1.21

    ## from cascade paper
    ### https://arxiv.org/pdf/2001.09520.pdf
    cascade_atmo_norm = 1.07

    fluxPath = '/data/user/chill/icetray_LWCompatible/nuSQuIDS_propagation_files'

##end
