##simple container class for info
##from LeptonWeighter events or i3 frames
class EventInfo(object):
    def __init__(self):

        self.dataset      = []
        self.run          = []
        self.event_id     = []
        self.sub_event_id = []
        #self.weight       = []

        self.pdg         = []
        self.is_neutrino = []
        self.nu_energy   = []
        self.nu_azimuth  = []
        self.nu_zenith   = []
        self.nu_x        = []
        self.nu_y        = []
        self.nu_z        = []

        self.li_energy   = []
        self.li_azimuth  = []
        self.li_zenith   = []

        #self.fit_x      = []
        #self.fit_y      = []
        #self.fit_type   = []
        #self.fit_params = []

        self.flux_atmo  = []
        self.flux_astro = []

##cascade specific info to be set
##for both data and MC
class CascadeInfo(object):
    def __init__(self, reco_choice):
        self.reco = reco_choice
        self.reco_energy = []
        self.reco_zenith = []
        self.reco_azimuth = []
        self.reco_x = []
        self.reco_y = [] 
        self.reco_z = []
        #self.reco_pos_x = []
        #self.reco_pos_y = []
        #self.reco_pos_z = [] 

    @property
    def reco(self):
        return self.__reco
    @reco.setter
    def reco(self, reco_choice):
        self.validateReco(reco_choice)
        self.__reco = reco_choice

    def validateReco(self, reco_choice):
        valid_recos= ['cscdSBU_MonopodFit4_noDC', 'L3_MonopodFit4_AmptFit']
        if reco_choice not in valid_recos:
            raise NotImplementedError(f'{reco_choice} not a valid reco options for cascades!')

class TrackInfo(object):
    def __init__(self, reco_choice):
        self.reco = reco_choice
        self.reco_energy = []
        self.reco_zenith = []
        self.reco_azimuth = []
        self.reco_x = []
        self.reco_y = [] 
        self.reco_z = []
        #self.reco_pos_x = []
        #self.reco_pos_y = []
        #self.reco_pos_z = [] 

    @property
    def reco(self):
        return self.__reco
    @reco.setter
    def reco(self, reco_choice):
        self.validateReco(reco_choice)
        self.__reco = reco_choice

    def validateReco(self, reco_choice):
        valid_recos= ['SplineMPEICTruncatedEnergy_SPICEMie_AllDOMS_Neutrino', 'SplineMPEICTruncatedEnergySPICEMie_AllDOMS_Muon']
        if reco_choice not in valid_recos:
            raise NotImplementedError(f'{reco_choice} not a valid reco options for tracks!')

class DataEventInfo(object):
    def __init__(self):
        self.live_time = []
        self.year = []
        self.run = []
        self.sub_run = []
        self.event = []
        self.sub_event = []
        self.Selection = []


##end
