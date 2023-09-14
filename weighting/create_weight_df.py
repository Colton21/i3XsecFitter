# used for parsing inputs
from optparse import OptionParser
# used for verifying files exist
import sys, os
import numpy as np
import h5py
from glob import glob
import click
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import time
from concurrent.futures import ProcessPoolExecutor, wait

#####
from I3Tray import I3Tray
from icecube import icetray, dataio, dataclasses, simclasses
from icecube.weighting import weighting, get_weighted_primary
from icecube.icetray import I3Units
from icecube import LeptonInjector
import LeptonWeighter as LW
#####

##local package
sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter/')
from event_info import EventInfo
from event_info import CascadeInfo, TrackInfo
from configs.config import config
from fitting.fitting import initPropFiles, propWeightFit
from fluxes import AtmFlux, DiffFlux, InitAtmFlux
from self_veto.apply_veto import apply_veto
##

def extract_event_info(eventInfo, recoInfo, frame, lwEvent, dataset, run, selection, atm_flux):
    if selection == 'track':
        MCPrimary1 = frame['MCPrimary1']
    if selection == 'cascade':    
        MCPrimary1 = frame['MCPrimary1']
    EventHeader = frame['I3EventHeader']
    try:
        event_id = EventHeader.event_id
        sub_event_id = EventHeader.sub_event_id
    except:
        print(f'!!! {dataset}, {run}, {selection} !!!')
        exit(1)


    eventInfo.dataset.append(dataset)
    eventInfo.run.append(run)
    eventInfo.event_id.append(event_id)
    eventInfo.sub_event_id.append(sub_event_id)

    eventInfo.pdg.append(MCPrimary1.pdg_encoding)
    eventInfo.is_neutrino.append(MCPrimary1.is_neutrino)
    eventInfo.nu_energy.append(MCPrimary1.energy)
    eventInfo.nu_azimuth.append(MCPrimary1.dir.azimuth)
    eventInfo.nu_zenith.append(MCPrimary1.dir.zenith)
    eventInfo.nu_x.append(MCPrimary1.pos.x)
    eventInfo.nu_y.append(MCPrimary1.pos.y)
    eventInfo.nu_z.append(MCPrimary1.pos.z)

    eventInfo.i3pos.append(MCPrimary1.pos)
    eventInfo.i3dir.append(MCPrimary1.dir)

    eventInfo.li_energy.append(lwEvent.energy)
    eventInfo.li_azimuth.append(lwEvent.azimuth)
    eventInfo.li_zenith.append(lwEvent.zenith)

    ##try to add MMCTrack info - energy entering
    mmc_ei = [-1]
    mmctrack_list = frame['MMCTrackList'] 
    for _mmc in mmctrack_list:
        _e = _mmc.Ei
        mmc_ei.append(_e)
    eventInfo.mmc_energy.append(np.max(mmc_ei))

    ##add flux info
    eventInfo.flux_atmo.append(AtmFlux(atm_flux, MCPrimary1.energy, 
                               np.cos(MCPrimary1.dir.zenith),MCPrimary1.pdg_encoding))
    #eventInfo.flux_astro.append(DiffFlux(MCPrimary1.energy, mode=selection))
    ##NOTE
    #print('Running with classic Diff Flux Setting - matching generated nuSQuIDS files!')
    eventInfo.flux_astro.append(DiffFlux(MCPrimary1.energy, mode='classic'))
    
    ##construct separate reco_info if no reco has been applied yet
    try:
        reco_info = frame[recoInfo.reco]
        recoInfo.reco_energy.append(reco_info.energy)
        recoInfo.reco_zenith.append(reco_info.dir.zenith)
        recoInfo.reco_azimuth.append(reco_info.dir.azimuth)
        recoInfo.reco_x.append(reco_info.pos.x)
        recoInfo.reco_y.append(reco_info.pos.y)
        recoInfo.reco_z.append(reco_info.pos.z)
    except KeyError:
        default = -9999
        #print(f"Unable to find {recoInfo.reco}, EventID:{event_id}, SubEventID:{sub_event_id}!!!")
        recoInfo.reco_energy.append(default)
        recoInfo.reco_zenith.append(default)
        recoInfo.reco_azimuth.append(default)
        recoInfo.reco_x.append(default)
        recoInfo.reco_y.append(default)
        recoInfo.reco_z.append(default)

    ##systematics / SnowStorm info
    ##based on frame['SnowstormParametrizations'], assign them
    if len(frame['SnowstormParameters']) != 6 and len(frame['SnowstormParametrizations']) != 5:
        #print(frame['SnowstormParameters'], frame['SnowstormParametrizations'])
        ##check if Anisotropy Scale is there
        if (len(frame['SnowstormParametrizations']) == 4 and 
            'Anisotropy' not in frame['SnowstormParametrizations']):
            eventInfo.ice_scattering.append(frame['SnowstormParameters'][0])
            eventInfo.ice_absorption.append(frame['SnowstormParameters'][1])
            eventInfo.dom_efficiency.append(frame['SnowstormParameters'][2])
            eventInfo.hole_ice_forward_p0.append(frame['SnowstormParameters'][3])
            eventInfo.hole_ice_forward_p1.append(frame['SnowstormParameters'][4])
            eventInfo.ice_anisotropy_scale.append(-999)
        else:
            raise KeyError(f'Unexpected sizes for the Snowstorm Systematics!')
    else:        
        eventInfo.ice_scattering.append(frame['SnowstormParameters'][0])
        eventInfo.ice_absorption.append(frame['SnowstormParameters'][1])
        eventInfo.ice_anisotropy_scale.append(frame['SnowstormParameters'][2])
        eventInfo.dom_efficiency.append(frame['SnowstormParameters'][3])
        eventInfo.hole_ice_forward_p0.append(frame['SnowstormParameters'][4])
        eventInfo.hole_ice_forward_p1.append(frame['SnowstormParameters'][5])

    ##eventInfo & recoInfo are ultimately combined later
    return eventInfo, recoInfo

def construct_lw_event(frame):
    LWevent = LW.Event()
    EventProperties                 = frame['EventProperties']
    LeptonInjectorProperties        = frame['LeptonInjectorProperties']
    LWevent.primary_type            = LW.ParticleType(EventProperties.initialType)
    LWevent.final_state_particle_0  = LW.ParticleType(EventProperties.finalType1)
    LWevent.final_state_particle_1  = LW.ParticleType(EventProperties.finalType2)
    LWevent.zenith                  = EventProperties.zenith
    LWevent.energy                  = EventProperties.totalEnergy
    LWevent.azimuth                 = EventProperties.azimuth
    LWevent.interaction_x           = EventProperties.finalStateX
    LWevent.interaction_y           = EventProperties.finalStateY
    LWevent.total_column_depth      = EventProperties.totalColumnDepth
    #volume events are nue CC & NC interactions - injection is different
    if isinstance(EventProperties, LeptonInjector.VolumeEventProperties):
        LWevent.radius              = EventProperties.radius
    else:
        LWevent.radius              = EventProperties.impactParameter

    ##use MCPrimary to get verticies
    ##from Ben Smither's implementation
    if "MCPrimary1" in frame:
        MCPrimary                   = frame["MCPrimary1"]
    else:
        #print("MCPrimary1 Not in Frame - trying to add to frame")
        i = 0
        is_neutrino = False
        while is_neutrino == False:
            MCPrimary = frame["I3MCTree"].primaries[i]
            if MCPrimary.is_neutrino == True:
                is_neutrino = True
            else:
                i += 1
        if i != 0:
            print(f'Neutrino primary was {i}!')
        frame["MCPrimary1"]         = MCPrimary

    LWevent.x                       = MCPrimary.pos.x 
    LWevent.y                       = MCPrimary.pos.y
    LWevent.z                       = MCPrimary.pos.z
    return LWevent

## LeptonWeighter weight_event depends on input flux - select for track/cascade & atmo/astro fluxes
def construct_weight_events(licfiles, flux_path, selection, CCNC, norm_list, f_type, 
                            earth, strange_flux):
    net_generation = []
    for lic in licfiles:
        net_generation += LW.MakeGeneratorsFromLICFile( lic )
    print("Finished building the generators")
    
    if CCNC.lower() in ['cc', 'nc']:
        xs = LW.CrossSectionFromSpline(
        "/data/user/bsmithers/cross_sections/dsdxdy_nu_CC_iso.fits",
        "/data/user/bsmithers/cross_sections/dsdxdy_nubar_CC_iso.fits",
        "/data/user/bsmithers/cross_sections/dsdxdy_nu_NC_iso.fits",
        "/data/user/bsmithers/cross_sections/dsdxdy_nubar_NC_iso.fits")
    elif CCNC.lower() == 'gr':
        xs = LW.GlashowResonanceCrossSection()
    else:
        raise IOError(f'CCNC type {CCNC} is not valid! Must be CC, NC, or GR')

    print("Load Pre-calculated nuSQuIDS Flux Files into LeptonWeighter")
    weight_event_list = []
    
    for norm in norm_list:
        if earth == 'normal':
            #f_str = f'nuSQuIDS_flux_cache_{norm}_{f_type}_{selection}.hdf'
            f_str = f'nuSQuIDS_flux_cache_{norm}_toleranceUp_-2.37_{f_type}.hdf'
        else:
            f_str = f'nuSQuIDS_flux_cache_{norm}_{earth}_{f_type}_{selection}.hdf'        

        flux = LW.nuSQUIDSAtmFlux(os.path.join(flux_path, f_str))
        weighter = LW.Weighter(flux, xs, net_generation)
        weight_event_list.append(weighter)

    return weight_event_list

def calculate_weights(i3files, licfiles, liveTime, flux_path, selection, CCNC, 
                      norm_list, f_type, splineList, earth, strange_flux=False):
    
    ## create empty storage object to manage information
    eventInfo = EventInfo()
    atm_flux = InitAtmFlux()
    
    ## weights depend on input flux
    ## either use merged flux or separate
    if f_type == 'all':
        weight_event_list = construct_weight_events(licfiles, flux_path, selection, 
                                                    CCNC, norm_list, f_type, earth)
        for norm in norm_list:
            setattr(eventInfo, f'weight{norm}', [])
        splineList = splineList[0]
    else:
        weight_event_list_atmo  = construct_weight_events(licfiles, flux_path, 
                                                          selection, CCNC, norm_list, 
                                                          earth=earth, f_type='atmo',
                                                          strange_flux=strange_flux)
        weight_event_list_astro = construct_weight_events(licfiles, flux_path, 
                                                          selection, CCNC, norm_list, 
                                                          earth=earth, f_type='astro',
                                                          strange_flux=strange_flux)
        #for norm in norm_list:
        #    setattr(eventInfo, f'weight{norm}_atmo',  [])
        #    setattr(eventInfo, f'weight{norm}_astro', [])
        setattr(eventInfo, f'weight1.0_astro', [])
        setattr(eventInfo, f'weight1.0_atmo',  [])
        splineListAtmo  = splineList[0]
        splineListAstro = splineList[1]

    ## determine which reco algo to search for in the i3 file
    if selection == 'track':
        recoInfo = TrackInfo(config.track_reco)
    if selection == 'cascade':
        recoInfo = CascadeInfo(config.cascade_reco)

    print(f'Start iterating over the i3 files for {selection}')
    print(f'Pre-evaluated norms are only 1.0 to save computation time')
    print(f'Instead use fitting later')
    ## loop over all i3 - lic file pairs to weight 
    for f, lic in tqdm(zip(i3files, licfiles)):
        dataset, run = get_filename_info(f, lic)
        data_file = dataio.I3File(f, 'r')
        # scan over the frames
        while data_file.more():
            try:
                frame = data_file.pop_physics()
            ## if no physics frames are in the file - skip
            except RuntimeError:
                continue

            ## construct the LeptonWeighter event needed for weighting
            lwEvent = construct_lw_event(frame)

            ## perform the weighting for each event for each normalisation
            for k, norm in enumerate(norm_list):
                if norm != 1.0:
                    continue

                ## if fluxes are together
                if f_type == 'all':
                    weight_event = weight_event_list[k]
                    ## scaling for interaction probability
                    weight = weight_event.weight(lwEvent, norm) * liveTime
                    if np.isfinite(weight) == False:
                        from IPython import embed
                        embed()
                
                    ## add the weight to the eventInfo container
                    _l = getattr(eventInfo, f'weight{norm}')
                    _l.append(weight)

                ## if fluxes are separated
                else:
                    weight_event_atmo = weight_event_list_atmo[k]
                    weight_event_astro = weight_event_list_astro[k]
                    ## if GR, we do not want to modify the 
                    ## at-detector interaction probability
                    if CCNC == 'GR':
                        weight_atmo = weight_event_atmo.weight(lwEvent, 1.0) * liveTime
                        weight_astro = weight_event_astro.weight(lwEvent, 1.0) * liveTime
                    else:
                        ## scaling for interaction probability
                        weight_atmo = weight_event_atmo.weight(lwEvent, norm) * liveTime
                        weight_astro = weight_event_astro.weight(lwEvent, norm) * liveTime
                    if np.isfinite(weight_atmo) == False:
                        from IPython import embed
                        embed()
                    if np.isfinite(weight_astro) == False:
                        from IPython import embed
                        embed()

                    ## add the weight to the eventInfo container
                    _l_atmo = getattr(eventInfo,  f'weight{norm}_atmo')
                    _l_atmo.append(weight_atmo)

                    _l_astro = getattr(eventInfo, f'weight{norm}_astro')
                    _l_astro.append(weight_astro)

            ## collect other information from the event
            eventInfo, recoInfo = extract_event_info(eventInfo, recoInfo, frame, 
                                        lwEvent, dataset, run, selection, atm_flux)
        ##end of the file
        data_file.close()   
    ##finish looping i3, lic file pairs

    ##indicates running as test
    if len(norm_list) == 1:
        return eventInfo, recoInfo

    ##open the nuSQuIDS propagation files - hold them in memory, then fit
    if f_type == 'all':
        eventInfo = propWeightFit(eventInfo, splineList, norm_list)
    else:
        eventInfo = propWeightFit(eventInfo, splineListAtmo,  norm_list, sType='atmo')
        eventInfo = propWeightFit(eventInfo, splineListAstro, norm_list, sType='astro')
   
    ##apply self-veto, only needed for cascades
    eventInfo = apply_veto(eventInfo, selection) 

    return eventInfo, recoInfo

def get_filename_info(file1, file2):
    f1 = os.path.basename(file1)
    f2 = os.path.basename(file2)
    f1 = f1.split(".")
    f2 = f2.split(".")

    dataset1 = f1[1]
    dataset2 = f2[1]
    runNumber1 = f1[2]
    runNumber2 = f2[2]
        
    if dataset1 != dataset2:
        raise IOError(f'Dataset numbers for {f1} and {f2} are not matching!')
    if runNumber1 != runNumber2:
        raise IOError(f'Run numbers for {f1} and {f2} are not matching!')
    return dataset1, runNumber1


#make sure dataset and run numbers are the same
#def check_nums(fileList1, fileList2):
def check_nums(fileList1, fileDir, fileType):
    ##force them to be same length - usually it's fine!
    #_fileList2 = fileList2[:len(fileList1)]
    #fileList2 = sorted(glob(filePath))
    datasetList = [''] * len(fileList1)
    runList = [''] * len(fileList1)
    matchedList = [''] * len(fileList1)
    i = 0
    
    #for file1, file2 in zip(fileList1, _fileList2):
    ##loop through all i3 files
    for i, file1, in enumerate(fileList1):
        f1 = os.path.basename(file1)
        f1 = f1.split(".")
        dataset1 = f1[1]
        runNumber1 = f1[2]

        ##check for corresponding lic file
        f2 = fileType.split('.')
        if len(f2) == 3:
            f2 = f2[0] + f'.{dataset1}.{runNumber1}.' + f2[2]
        if len(f2) == 2:
            f2 = f2[0] + f'.{dataset1}.{runNumber1}.' + f2[1]
        _filePath = os.path.join(fileDir, f2)
        ##OLD - But works
        file2 = glob(_filePath)
        if len(file2) != 1:
            raise FileNotFoundError(f'Could not find matching file for {file1} at {_filePath}')
        matchedList[i] = file2[0]
        ##NEW - cuts glob in loop, but don't know actual path (dir structure)...
        #if not os.path.exists(_filePath):
        #    raise FileNotFoundError(f'Could not find matching file for {file1} at {_filePath}')
        #matchedList[i] = _filePath

        ##loop through all lic files
        #for file2 in fileList2:
        #    f2 = os.path.basename(file2)
        #    f2 = f2.split(".")
        #    dataset2 = f2[1]
        #    runNumber2 = f2[2]

            ##find the matching lic file
        #    if dataset1 == dataset2 and runNumber1 == runNumber2:
        #        matchedList[i] = file2
        #        break

        #if dataset1 != dataset2:
        #    raise IOError(f'Dataset numbers for {f1} and {f2} are not matching!')
        #if runNumber1 != runNumber2:
        #    print(f'Run numbers for {f1} and {f2} are not matching! - trying to correct')
        #    ##try to find correct number - set search tolerance to 5
        #    ##if there are lots of mis-ordered files, this becomes expensive
        #    for iprime in range(5):
        #        _f2 = os.path.basename(fileList2[i+iprime])
        #        _f2 = _f2.split(".")
        #        if runNumber1 == _f2[2]:
        #            print(f'Found match: {f1} and {_f2}')
        #            _fileList2[i] = fileList2[i+iprime]
        #            break
        #        if iprime == 4:
        #            raise IOError(f'Run numbers for {f1} and {f2} are not matching!')
        #
        datasetList[i] = dataset1
        runList[i] = runNumber1
        #i += 1        
    #from IPython import embed
    #embed()
    #if len(fileList1) == 0:
    #    raise IOError(f'i3 files not found in path!')
    #if len(_fileList2) == 0:
    #    raise IOError(f'lic files not found in path!')
    #if len(fileList1) != len(_fileList2):
    if len(fileList1) != len(matchedList):
        raise IOError(f'Mismatch in files! {len(fileList1)} vs {len(matchedList)}')

    return fileList1, matchedList, datasetList, runList

def get_files(i3file_dir, licfile_dir, num_files, selection, CCNC='CC'):
    #if not os.path.isdir(i3file_dir):
    #    raise IOError(f'{i3file_dir} is not a valid directory!')
    #if not os.path.isdir(licfile_dir):
    #    raise IOError(f'{licfile_dir} is not a valid directory!')
    if i3file_dir[-1] == '/':
        i3file_dir = i3file_dir[:-1]
    if licfile_dir[-1] == '/':
        licfile_dir = licfile_dir[:-1]
    ##track and cascade folder structure are different - catch both cases
    if CCNC == 'GR':
        if selection == 'cascade':
            i3files = sorted(glob(f'{i3file_dir}/*_All_GR*.i3.bz2'))
        elif selection == 'track':
            i3files = sorted(glob(f'{i3file_dir}/*_All_GR*.i3.zst'))
        if len(i3files) == 0:
            raise FileNotFoundError(f'No files found at {i3file_dir} for GR')
        lic_str = f'All_GR*.lic'
    #elif (int(licfile_dir.split('/')[-2]) == int(config.gr_id) or 
    #    int(licfile_dir.split('/')[-1]) == int(config.gr_id)):
    #    if selection == 'cascade':
    #        i3files  = sorted(glob(f'{i3file_dir}/*_All_GR_cascade*.i3.bz2'))
    #    if selection == 'track':
    #        i3files  = sorted(glob(f'{i3file_dir}/*_All_GR*.i3.zst'))
    #    #_licfiles = sorted(glob(f'{licfile_dir}/All_GR*.lic'))
    #    lic_str = f'All_GR*.lic'
    elif selection == 'cascade':
        i3files  = sorted(glob(f'{i3file_dir}/*_{CCNC}_cascade.*.i3.bz2'))
        lic_str = f'*_{CCNC}.*.lic'
    elif selection == 'track':
        i3files  = sorted(glob(f'{i3file_dir}/*_{CCNC}.*.i3.zst'))
        lic_str = f'*_{CCNC}.*.lic'
    else:
        raise NotImplementedError(f'option {selection} is not valid!')

    print(f'--- Found {len(i3files)} i3files for {selection}, {CCNC} ---')
    print('--- Looking for matching lic files ---')
    i3files, licfiles, datasetList, runList = check_nums(i3files, licfile_dir, lic_str)  

    return i3files, licfiles, datasetList, runList

def valid_dir(dataset=None, selection='track', do_all=False, legacy=False):
    valid_datasets = config.mc_ids
    if dataset == None and do_all == False:
        raise NotImplementedError(f'Use specific dataset ({valid_datasets}) or set do_all = True!')

    if do_all == False:
        if int(dataset) not in valid_datasets:
            raise ValueError(f'Dataset number {dataset} not valid! Use {valid_datasets}!')
        i3_file_dir, lic_file_dir = build_path(int(dataset), selection, legacy)
        i3_file_dirs = [i3_file_dir]
        lic_file_dirs = [lic_file_dir]
    if do_all == True:
        i3_file_dirs = [''] * len(valid_datasets)
        lic_file_dirs = [''] * len(valid_datasets)        
        for i, dataset in enumerate(valid_datasets):
            i3, lic = build_path(int(dataset), selection, legacy)
            i3_file_dirs[i] = i3
            lic_file_dirs[i] = lic

    return i3_file_dirs, lic_file_dirs

def build_path(dataset, selection='track', legacy=False):
    for _c in config.legacy_ids:
        if dataset == int(_c):
            legacy = True
            break
    valid = False
    for _c in config.mc_ids:
        if dataset == int(_c):
            valid = True
            break

    if legacy == True:
        lic_file_base_path = config.legacy_lic_file_base_path    
    else:
        lic_file_base_path = config.lic_file_base_path    

    if selection == 'track':
        if legacy == True:
            i3_file_base_path = config.legacy_i3_track_base_path
        elif valid == True:
            i3_file_base_path = config.i3_track_base_path
        else:
            raise ValueError(f'No {dataset} for {config.mc_ids} or {config.legacy_ids}')
    elif selection == 'cascade':
        if legacy == True:
            i3_file_base_path = config.legacy_i3_cascade_base_path    
        elif valid == True:
            i3_file_base_path = config.i3_cascade_base_path    
        else:
            raise ValueError(f'No {dataset} for {config.mc_ids} or {config.legacy_ids}')
    else:
        raise NotImplementedError(f'selection {selection} not valid! should be track or cascade')

    i3_dpath = os.path.join(i3_file_base_path, f'{dataset}')
    extension_path = '000*/'
    #extension_path = '0000000-0000999/'
    #print('='*20)
    #print('='*20)
    #print(f'WARNING - only {extension_path} is checked! If you have more files, please fix!')
    #print('='*20)
    #print('='*20)

    if selection == 'track':
        i3_file_path = os.path.join(i3_dpath, extension_path)
    if selection == 'cascade':
        i3_file_path = i3_dpath

    lic_dpath = os.path.join(lic_file_base_path, f'{dataset}')
    lic_file_path = os.path.join(lic_dpath, extension_path)
    return i3_file_path, lic_file_path


def process_files(i3file_dir, licfile_dir, flux_path, selection, num_files, 
                  liveTime, w_dir, norm_list, f_type, earth, strange_flux):
    
    if f_type == 'all':
        splineList, norm_list = initPropFiles(flux_path, norm_list, 
                                              f_type='all', selection=selection,
                                              earth=earth)
        splineList = [splineList]
    else:
        splineListAtmo,  norm_list = initPropFiles(flux_path, norm_list, f_type='atmo',  
                                                   selection=selection, earth=earth)
        print(f'Atmo: {norm_list}')
        splineListAstro, norm_list = initPropFiles(flux_path, norm_list, f_type='astro',
                                                   selection=selection, earth=earth)
        print(f'Astro: {norm_list}')
        splineList = [splineListAtmo, splineListAstro]
    
    ##make an exception for how GR files are handled
    if i3file_dir.split('/')[-1] == '' and selection == 'track':
        dataset_num = i3file_dir.split('/')[-3]
    elif selection == 'track':
        dataset_num = i3file_dir.split('/')[-2]
    elif i3file_dir.split('/')[-1] == '' and selection == 'cascade':
        dataset_num = icfile_dir.split('/')[-2]
    else:
        dataset_num = i3file_dir.split('/')[-1]

    valid_datasets = config.mc_ids
    if int(dataset_num) not in valid_datasets:
        raise IOError(f'Could not correctly determine dataset number based on path {dataset_num}!')

    dataset_num = int(dataset_num)
    for CCNC in ['CC', 'NC']:
        if CCNC == 'NC' and dataset_num == int(config.gr_id):
            continue
        if CCNC == 'CC' and dataset_num == int(config.gr_id):
            CCNC = 'GR'
        print(f'Dataset: {dataset_num}, CC/NC: {CCNC}')
        i3files, licfiles, datasetList, runList = get_files(i3file_dir, licfile_dir, 
                                                            num_files, selection, CCNC)
        if len(i3files) == 0 and len(licfiles) == 0:
            print(f'No files found for this combination of {selection} and {CCNC}')
            print('No file will be created')
            return
        eventInfo, recoInfo = calculate_weights(i3files, licfiles, liveTime, flux_path, 
                                                selection, CCNC, norm_list, f_type, 
                                                splineList, earth, strange_flux)
        ccncList = [CCNC] * len(eventInfo.nu_energy) ##just pick anything
        liveTimeL = [liveTime] * len(eventInfo.nu_energy) ##just 1 year, arbitrarily
        selectionList = [selection] * len(eventInfo.nu_energy)

        ##unpack eventInfoList - items to save in the dataframe
        data = eventInfo.__dict__
        data.update({'Selection': selectionList, 'IntType': ccncList, 'LiveTime': liveTimeL})
        data.update(recoInfo.__dict__)

        ##removing some mess from the dict
        clean_track = False
        clean_cascade = False
        for k in data.keys():
            if k == '_TrackInfo__reco':
                clean_track = True
            if k == '_CascadeInfo__reco':
                clean_cascade = True
        if clean_track == True:
            del data['_TrackInfo__reco']
        if clean_cascade == True:
            del data['_CascadeInfo__reco']

        try:
            df = pd.DataFrame(data=data)
        except:
            print('Problem creating the dataframe!')
            print(data.keys())
            for k in data.keys():
                print(k, len(data[k]))
            from IPython import embed
            embed()

        if strange_flux == True:
            f_str = f'weight_df_{datasetList[0]}_strangeFlux_{selection}_{CCNC}.hdf'
        elif earth == 'normal':
            f_str = f'weight_df_{datasetList[0]}_{selection}_{CCNC}.hdf'
        else:
            f_str = f'weight_df_{datasetList[0]}_{earth}_{selection}_{CCNC}.hdf'
        df.to_hdf(os.path.join(w_dir, f_str), key='df', mode='w')
        print(f'Created: {f_str}')
        print(len(df.index.values))

def analysis_wrapper(dataset, selection, do_all, flux_path, num_files, 
                    f_type, earth, test=False, strange_flux=False, legacy=False):
    pi = np.pi
    proton_mass = 0.93827231 #GeV
    liveTime = 3.1536e7 #365 days in seconds
    w_dir = '/data/user/chill/icetray_LWCompatible/weights'
    
    if test == True:
        norm_list = [1.0]
    ##norm list is used for finding the nuSQuIDS propagation files
    ##use default PREM
    elif earth == 'normal' and test == False:
        norm_list = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0]
    ##use modified PREM
    elif earth in ['core_up', 'core_down', 'all_up', 'all_down']:
        #norm_list = [0.7, 0.8, 0.9, 0.98, 0.985, 0.99, 0.995,
        #             1.005, 1.01, 1.015, 1.02, 1.1, 1.2, 1.3]        
        norm_list = [1.0]        
    else:
        raise ValueError(f'Option for earth {earth} not valid! Use up or down')
    print(f'Running with norms = {norm_list}')
    time.sleep(2)

    if flux_path == None:
        flux_path = config.fluxPath

    ##if dataset is None and do_all is True - grabs all files for 1 selection
    i3file_dirs, licfile_dirs = valid_dir(dataset, selection, do_all, legacy)
   
    #i3file_dirs  = ['/data/user/chill/icetray_LWCompatible/temp_gr']
    #licfile_dirs = ['/data/user/chill/icetray_LWCompatible/temp_gr']
    #print('WARNING - i3file_dirs and licfile_dirs are FIXED FOR TESTING')
 
    ##test has norm_list configured to only 1.0
    if test == True:
        process_files(i3file_dirs[0], licfile_dirs[0], 
                      flux_path, selection, num_files, liveTime, 
                      w_dir, norm_list, f_type, earth, strange_flux)
        return        
    
    ##if do_all is False, size of dirs list is 1
    if do_all == False:
        process_files(i3file_dirs[0], licfile_dirs[0], flux_path, selection, 
                      num_files, liveTime, w_dir, norm_list, f_type, earth, strange_flux)
        return


    ##start multi-threading here
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        ##loop over all datasets, CCNC
        for i3file_dir, licfile_dir in zip(i3file_dirs, licfile_dirs):
            futures.append(executor.submit(process_files, i3file_dir, licfile_dir, 
                                           flux_path, selection, num_files, liveTime, 
                                           w_dir, norm_list, f_type, earth, strange_flux))
    results = wait(futures)
    for result in results.done:
        print(result.result())

@click.command()
@click.option('--dataset', '-d', default=None)
@click.option('--selection', '-s', default='track', type=click.Choice(['track', 'cascade']))
@click.option('--do_all', is_flag=True)
@click.option('--flux_path', '-f', default=None)
@click.option('--num_files', '-n', default=-1)
@click.option('--f_type', '-f', default='separate', type=click.Choice(['separate', 'all']))
@click.option('--earth', '-e', default='normal')
@click.option('--test', is_flag=True)
@click.option('--strange_flux', is_flag=True)
@click.option('--legacy', is_flag=True)
def main(dataset, selection, do_all, flux_path, num_files, f_type, earth, test, strange_flux, legacy):
    analysis_wrapper(dataset=dataset,
                     selection=selection, 
                     do_all=do_all,
                     flux_path=flux_path,
                     num_files=num_files,
                     f_type=f_type,
                     earth=earth,
                     test=bool(test),
                     strange_flux=strange_flux,
                     legacy=legacy)
    print("Done")

if __name__ == "__main__":
    main()

##end
