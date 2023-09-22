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
#from icecube.weighting import weighting, get_weighted_primary
#from icecube.icetray import I3Units
from icecube import LeptonInjector
import LeptonWeighter as LW
#####

##local package
from event_info import EventInfo
from event_info import CascadeInfo, TrackInfo
from configs.config import config
from fitting.fitting import initPropFiles, propWeightFit
from fluxes import AtmFlux, DiffFlux, InitAtmFlux
from self_veto.apply_veto import apply_veto
from debug import debug
##

def extract_event_info(eventInfo, recoInfo, frame, lwEvent, dataset, run, selection, atm_flux, CCNC):
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
    eventInfo.ccnc.append(CCNC)
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

##TODO - the fluxes are unified now? so selection is not needed?
def construct_weight_events(licfiles, flux_path, selection, CCNC, earth):
    net_generation = []
    for lic in licfiles:
        net_generation += LW.MakeGeneratorsFromLICFile( lic )
    print("Finished building the generators")
    
    if CCNC.lower() in ['cc', 'nc']:
        xs = LW.CrossSectionFromSpline(
        os.path.join(config.inner, "cross_sections/dsdxdy_nu_CC_iso.fits"),
        os.path.join(config.inner, "cross_sections/dsdxdy_nubar_CC_iso.fits"),
        os.path.join(config.inner, "cross_sections/dsdxdy_nu_NC_iso.fits"),
        os.path.join(config.inner, "cross_sections/dsdxdy_nubar_NC_iso.fits"))
    elif CCNC.lower() == 'gr':
        xs = LW.GlashowResonanceCrossSection()
    else:
        raise IOError(f'CCNC type {CCNC} is not valid! Must be CC, NC, or GR')

    
    if earth == 'normal':
        f_str_astro = f'nuSQuIDS_flux_cache_1.0_toleranceUp_-2.37_astro.hdf'
        f_str_atmo  = f'nuSQuIDS_flux_cache_1.0_toleranceUp_-2.37_atmo.hdf'
    else:
        f_str_astro = f'nuSQuIDS_flux_cache_1.0_{earth}_astro_{selection}.hdf'        
        f_str_atmo  = f'nuSQuIDS_flux_cache_1.0_{earth}_atmo_{selection}.hdf'        

    print("Load Pre-calculated nuSQuIDS Flux Files into LeptonWeighter")
    flux_astro = LW.nuSQUIDSAtmFlux(os.path.join(flux_path, f_str_astro))
    flux_atmo  = LW.nuSQUIDSAtmFlux(os.path.join(flux_path, f_str_atmo))

    weighter_astro = LW.Weighter(flux_astro, xs, net_generation)
    weighter_atmo  = LW.Weighter(flux_atmo,  xs, net_generation)
    
    return weighter_atmo, weighter_astro

def append_weights(eventInfo, lwEvent, lw_atmo, lw_astro): 
    for weighter, f_type in zip([lw_atmo, lw_astro], ['atmo', 'astro']):
        weight = lw_weighter.weight(lwEvent, 1.0) * config.liveTime
        if np.isfinite(weight) == False or weight == 0:
            debug()
        ## add the weight to the eventInfo container
        _l = getattr(eventInfo,  f'weight1.0_{f_type}')
        _l.append(weight)

def calculate_weights(i3file, licfile, flux_path, selection, CCNC, earth):

    ## create empty storage object to manage information
    eventInfo = EventInfo()
    ## determine which reco algo to search for in the i3 file
    if selection == 'track':
        recoInfo = TrackInfo(config.track_reco)
    if selection == 'cascade':
        recoInfo = CascadeInfo(config.cascade_reco)

    ##initialise once for nuFlux
    atm_flux = InitAtmFlux()
    
    ##construct lepton weighter objects
    lw_atmo, lw_astro  = construct_weight_events(licfile, flux_path, selection, CCNC, earth)

    dataset, run = get_filename_info(i3file, licfile)
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
        ## append weights to eventInfo for atmo & astro
        append_weights(eventInfo, lwEvent, lw_atmo, lw_astro)

        ## collect other information from the event
        eventInfo, recoInfo = extract_event_info(eventInfo, recoInfo, frame, 
                                    lwEvent, dataset, run, selection, atm_flux, CCNC)
    ##end of the file
    data_file.close()   
    ##finish looping i3, lic file pairs

    for f_type in ['atmo', 'astro']:
        splineList, norm_list = initPropFiles(flux_path, f_type=f_type,
                                              selection=selection, earth=earth)
        eventInfo = propWeightFit(eventInfo, splineList, norm_list, f_type=f_type)
   
    ##apply self-veto, only needed for cascades
    eventInfo = apply_veto(eventInfo, selection) 
    
    ##build and save pandas dataframe
    build_dataframe(eventInfo, recoInfo, selection, earth)

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
def check_nums(fileList1, fileDir, fileType):
    matchedList = [''] * len(fileList1)
    i = 0
    
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
        file2 = glob(_filePath)
        if len(file2) != 1:
            raise FileNotFoundError(f'Could not find matching file for {file1} at {_filePath}')
        matchedList[i] = file2[0]
    if len(fileList1) != len(matchedList):
        raise IOError(f'Mismatch in files! {len(fileList1)} vs {len(matchedList)}')

    return fileList1, matchedList

def get_files(dataset, num_files, selection, CCNC):
    
    ##validate directory, find path for a given dataset
    i3file_dir, licfile_dir = valid_dir(dataset, selection)
    
    ##track and cascade folder structure are different - catch both cases
    if CCNC == 'GR':
        i3files = os.path.join(f'{i3file_dir}', '*_All_GR*.i3.bz2')
        lic_str = f'All_GR*.lic'
    elif selection == 'cascade':
        i3files  = os.path.join(f'{i3file_dir}', '*_{CCNC}_cascade.*.i3.bz2')
        lic_str = f'*_{CCNC}.*.lic'
    elif selection == 'track':
        i3files  = os.path.join(f'{i3file_dir}', '*_{CCNC}.*.i3.zst')
        lic_str = f'*_{CCNC}.*.lic'
    else:
        raise NotImplementedError(f'option {selection} is not valid!')

    i3files = sorted(glob(i3files_path))
    if len(i3files) == 0:
        raise FileNotFoundError(f'No files found at {i3file_dir} for {CCNC}')

    print(f'--- Found {len(i3files)} i3files for {selection}, {CCNC} ---')
    print('--- Looking for matching lic files ---')
    i3files, licfiles  = check_nums(i3files, licfile_dir, lic_str)  
    
    if len(i3files) == 0 and len(licfiles) == 0:
        print(f'No files found for this combination of {selection} and {CCNC}')
        print('No file will be created')
        raise FileNotFoundError(f'Check your paths! LIC: {licfile_dir}, i3: {i3file_dir}')

    return i3files[num_files], licfiles[num_files]


def valid_dir(dataset, selection):
    if int(dataset) not in config.mc_ids:
        raise ValueError(f'Dataset number {dataset} not valid! Use {valid_datasets}!')
        
    if config.lic_file_base_path.split('/')[0] in ['disk19', 'disk20']:
        chiba = True
    else:
        chiba = False

    lic_file_base_path = config.lic_file_base_path
    i3_file_base_path = config.i3_cascade_base_path    

    if chiba == True:
        i3_dpath = os.path.join(i3_file_base_path, f'l5_{selection}_lepton_injector_{dataset}')
        lic_dpath = os.path.join(lic_file_base_path, f'lic_{dataset}')
    if chiba == False:
        if selection == 'track':
            extension_path = '000*/'
            i3_dpath = os.path.join(i3_file_base_path, f'{dataset}')
            i3_file_path = os.path.join(i3_dpath, extension_path)
        if selection == 'cascade':
            i3_file_path = i3_dpath
        lic_dpath = os.path.join(lic_file_base_path, f'{dataset}')
    
    if not os.path.isdir(i3_file_path):
        raise IOError(f'{i3_file_path} is not a valid directory for i3_file_path!')
    if not os.path.isdir(lic_file_path):
        raise IOError(f'{lic_file_path} is not a valid directory for lic_file_path!')
    
    return i3_file_path, lic_file_path



def build_dataframe(eventInfo, recoInfo, selection, earth):
    liveTimeL = [config.liveTime] * len(eventInfo.nu_energy) ##just 1 year, arbitrarily
    selectionList = [selection] * len(eventInfo.nu_energy)

    ##unpack eventInfoList - items to save in the dataframe
    data = eventInfo.__dict__
    data.update({'Selection': selectionList, 'IntType': eventInfo.ccnc, 'LiveTime': liveTimeL})
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
        debug()

    if earth == 'normal':
        f_str = f'weight_df_{eventInfo.dataset[0]}_{selection}_{eventInfo.ccnc[0]}.hdf'
    else:
        f_str = f'weight_df_{eventInfo.dataset[0]}_{earth}_{selection}_{eventInfo.ccnc[0]}.hdf'
    df.to_hdf(os.path.join(config.weights_dir, f_str), key='df', mode='w')
    print(f'Created: {f_str}')

def analysis_wrapper(dataset, selection, ccnc, flux_path, num_files, earth, test=False):
    
    print('='*20)
    print(f'Dataset: {dataset_num}, CC/NC: {ccnc}')
    print('='*20)
    
    if flux_path == None:
        flux_path = config.fluxPath

    if test == True:
        num_files = 2
    else:
        num_files = -1

    i3files, licfiles  = get_files(dataset, num_files, selection, ccnc)

    if test == True:
        for i3file, licfile in zip(i3files, licfiles):
            calculate_weights(i3file, licfile, flux_path, selection, ccnc, earth)
        return
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for i3file, licfile in zip(i3files, licfiles):
            futures.append(executor.submit(calculate_weights, 
                                           i3file=i3file,
                                           licfile=licfile,
                                           flux_path=flux_path,
                                           selection=selection,
                                           ccnc=ccnc,
                                           earth=earth))
        results = wait(futures)
        for result in results.done:
            print(result.result())

@click.command()
@click.option('--dataset', '-d', required=True)
@click.option('--selection', '-s', required=True, type=click.Choice(['track', 'cascade']))
@click.option('--ccnc', required=True, type=click.Choice(['cc','nc','gr']))
@click.option('--flux_path', '-f', default=None)
@click.option('--num_files', '-n', default=-1)
@click.option('--earth', '-e', default='normal')
@click.option('--test', is_flag=True)
def main(dataset, selection, ccnc, flux_path, num_files, earth, test):
    analysis_wrapper(dataset=dataset,
                     selection=selection, 
                     ccnc=ccnc,
                     flux_path=flux_path,
                     num_files=num_files,
                     earth=earth,
                     test=bool(test))
    print("Done")

if __name__ == "__main__":
    main()

##end
