# used for parsing inputs
from optparse import OptionParser
# used for verifying files exist
import os
import numpy as np
import h5py
from glob import glob
import click
from tqdm import tqdm
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, wait

#####
from I3Tray import I3Tray
from icecube import icetray, dataio, dataclasses
from icecube.weighting import weighting, get_weighted_primary
from icecube.icetray import I3Units
from icecube import LeptonInjector
import LeptonWeighter as LW
#####

##
from event_info import EventInfo
from event_info import CascadeInfo, TrackInfo
from config import config
from fitting.fitting import initPropFiles, propWeightFit
from fluxes import AtmFlux, DiffFlux, InitAtmFlux
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

    eventInfo.li_energy.append(lwEvent.energy)
    eventInfo.li_azimuth.append(lwEvent.azimuth)
    eventInfo.li_zenith.append(lwEvent.zenith)

    ##add flux info
    eventInfo.flux_atmo.append(AtmFlux(atm_flux, MCPrimary1.energy, np.cos(MCPrimary1.dir.zenith),MCPrimary1.pdg_encoding))
    eventInfo.flux_astro.append(DiffFlux(MCPrimary1.energy, selection))
    
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
def construct_weight_events(licfiles, flux_path, selection, CCNC, norm_list, f_type):
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

    print("Load Pre-calculated nuSQuIDS Flux Files")
    weight_event_list = []
    for norm in norm_list:
        flux = LW.nuSQUIDSAtmFlux(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{flux_path}{norm}_{f_type}_{selection}.hdf'))
        weighter = LW.Weighter(flux, xs, net_generation)
        weight_event_list.append(weighter)

    return weight_event_list

def calculate_weights(i3files, licfiles, liveTime, flux_path, selection, CCNC, norm_list, f_type, splineList):
    
    ## create empty storage object to manage information
    eventInfo = EventInfo()
    atm_flux = InitAtmFlux()
    
    ## weights depend on input flux
    ## either use merged flux or separate
    if f_type == 'all':
        weight_event_list = construct_weight_events(licfiles, flux_path, selection, CCNC, norm_list, f_type)
        for norm in norm_list:
            setattr(eventInfo, f'weight{norm}', [])
        splineList = splineList[0]
    else:
        weight_event_list_atmo  = construct_weight_events(licfiles, flux_path, selection, CCNC, norm_list, f_type='atmo')
        weight_event_list_astro = construct_weight_events(licfiles, flux_path, selection, CCNC, norm_list, f_type='astro')
        for norm in norm_list:
            setattr(eventInfo, f'weight{norm}_atmo',  [])
            setattr(eventInfo, f'weight{norm}_astro', [])
        splineListAtmo  = splineList[0]
        splineListAstro = splineList[1]

    ## determine which reco algo to search for in the i3 file
    if selection == 'track':
        recoInfo = TrackInfo(config.track_reco)
    if selection == 'cascade':
        recoInfo = CascadeInfo(config.cascade_reco)

    ## loop over all i3 - lic file pairs to weight 
    for f, lic in tqdm(zip(i3files, licfiles)):
        dataset, run = get_filename_info(f, lic)
        data_file = dataio.I3File(f, 'r')
        # scan over the frames
        while data_file.more():
            try:
                #frame = data_file.pop_frame()
                #if str(frame.Stop)!='DAQ':
                #    continue
                frame = data_file.pop_physics()

            ## if no physics frames are in the file - skip
            except RuntimeError:
                continue

            ## construct the LeptonWeighter event needed for weighting
            lwEvent = construct_lw_event(frame)

            ## perform the weighting for each event for each normalisation
            for k, norm in enumerate(norm_list):

                ## if fluxes are together
                if f_type == 'all':
                    weight_event = weight_event_list[k]
                    try:
                        ## xsec normalisation applied here!
                        weight = weight_event.weight(lwEvent, norm) * liveTime
                    except RuntimeError:
                        print(f'0 Weight?')
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
                    try:
                        ## xsec normalisation applied here!
                        weight_atmo = weight_event_atmo.weight(lwEvent, norm) * liveTime
                    except RuntimeError:
                        print(f'0 Atmo Weight?')
                    if np.isfinite(weight_atmo) == False:
                        from IPython import embed
                        embed()
                    try:
                        ## xsec normalisation applied here!
                        weight_astro = weight_event_astro.weight(lwEvent, norm) * liveTime
                    except RuntimeError:
                        print(f'0 Astro Weight?')
                    if np.isfinite(weight_astro) == False:
                        from IPython import embed
                        embed()

                    ## add the weight to the eventInfo container
                    _l_atmo = getattr(eventInfo,  f'weight{norm}_atmo')
                    _l_atmo.append(weight_atmo)

                    _l_astro = getattr(eventInfo, f'weight{norm}_astro')
                    _l_astro.append(weight_astro)

            ## collect other information from the event
            eventInfo, recoInfo = extract_event_info(eventInfo, recoInfo, frame, lwEvent, dataset, run, selection, atm_flux)
    
    ##open the nuSQuIDS propagation files - hold them in memory, then fit
    if f_type == 'all':
        eventInfo = propWeightFit(eventInfo, splineList, norm_list)
    else:
        eventInfo = propWeightFit(eventInfo, splineListAtmo,  norm_list, sType='atmo')
        eventInfo = propWeightFit(eventInfo, splineListAstro, norm_list, sType='astro')
    
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
def check_nums(fileList1, fileList2):
    ##force them to be same length - usually it's fine!
    _fileList2 = fileList2[:len(fileList1)]
    datasetList = [''] * len(fileList1)
    runList = [''] * len(fileList1)
    i = 0
    
    for file1, file2 in zip(fileList1, _fileList2):
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
            print(f'Run numbers for {f1} and {f2} are not matching! - trying to correct')
            ##try to find correct number - set search tolerance to 5
            ##if there are lots of mis-ordered files, this becomes expensive
            for iprime in range(5):
                _f2 = os.path.basename(fileList2[i+iprime])
                _f2 = _f2.split(".")
                if runNumber1 == _f2[2]:
                    print(f'Found match: {f1} and {_f2}')
                    _fileList2[i] = fileList2[i+iprime]
                    break
                if iprime == 4:
                    raise IOError(f'Run numbers for {f1} and {f2} are not matching!')
        
        datasetList[i] = dataset1
        runList[i] = runNumber1
        i += 1        

    return fileList1, _fileList2, datasetList, runList

def get_files(i3file_dir, licfile_dir, num_files, selection, CCNC='CC'):
    if not os.path.isdir(i3file_dir):
        raise IOError(f'{file_dir} is not a valid directory!')
    if not os.path.isdir(licfile_dir):
        raise IOError(f'{file_dir} is not a valid directory!')
    if i3file_dir[-1] == '/':
        i3file_dir = i3file_dir[:-1]
    if licfile_dir[-1] == '/':
        licfile_dir = licfile_dir[:-1]
    ##track and cascade folder structure are different - catch both cases
    if licfile_dir.split('/')[-2] == '21408' or licfile_dir.split('/')[-1] == '21408':
        if selection == 'cascade':
            i3files  = sorted(glob(f'{i3file_dir}/*_All_GR_cascade*.i3.bz2'))
        if selection == 'track':
            i3files  = sorted(glob(f'{i3file_dir}/*_All_GR*.i3.zst'))
        _licfiles = sorted(glob(f'{licfile_dir}/All_GR*.lic'))
    elif selection == 'cascade':
        i3files  = sorted(glob(f'{i3file_dir}/*_{CCNC}_cascade.*.i3.bz2'))
        _licfiles = sorted(glob(f'{licfile_dir}/*_{CCNC}.*.lic'))
    else:
        i3files  = sorted(glob(f'{i3file_dir}/*_{CCNC}.*.i3.zst'))
        _licfiles = sorted(glob(f'{licfile_dir}/*_{CCNC}.*.lic'))
    if len(i3files) != len(_licfiles):
        print("------")
        print(f'i3files: {len(i3files)} and licfiles: {len(_licfiles)} have different lengths!')
        print("------")

    i3files, licfiles, datasetList, runList = check_nums(i3files, _licfiles)  

    if len(i3files) == 0:
        raise IOError(f'i3 files not found in path!')
    if len(licfiles) == 0:
        raise IOError(f'lic files not found in path!')
    if len(i3files) != len(licfiles):
        raise IOError(f'Mismatch in files! {len(i3files)} vs {len(licfiles)}')
    return i3files, licfiles, datasetList, runList

def valid_dir(dataset=None, selection='track', do_all=False):
    valid_datasets = [21395, 21396, 21397, 21398, 21399, 21400, 21401, 21402, 21403, 21408]
    if dataset == None and do_all == False:
        raise NotImplementedError(f'Use specific dataset ({valid_datasets}) or set do_all = True!')

    if do_all == False:
        if int(dataset) not in valid_datasets:
            raise ValueError(f'Dataset number {dataset} not valid! Use {valid_datasets}!')
        i3_file_dir, lic_file_dir = build_path(dataset, selection)
        i3_file_dirs = [i3_file_dir]
        lic_file_dirs = [lic_file_dir]
    if do_all == True:
        i3_file_dirs = [''] * len(valid_datasets)
        lic_file_dirs = [''] * len(valid_datasets)        
        for i, dataset in enumerate(valid_datasets):
            i3, lic = build_path(dataset, selection)
            i3_file_dirs[i] = i3
            lic_file_dirs[i] = lic

    return i3_file_dirs, lic_file_dirs

def build_path(dataset, selection='track'):
    lic_file_base_path = '/data/sim/IceCube/2019/generated/snowstorm/lepton_injector/'
    
    if selection == 'track':
        i3_file_base_path = '/data/sim/IceCube/2019/filtered/finallevel/snowstorm/northern_tracks/'
    if selection == 'cascade':
        i3_file_base_path = '/data/user/chill/l5_cascade_lepton_injector/'
    
    i3_dpath = os.path.join(i3_file_base_path, f'{dataset}')
    extension_path = '0000000-0000999/'

    if selection == 'track':
        i3_file_path = os.path.join(i3_dpath, extension_path)
    if selection == 'cascade':
        i3_file_path = i3_dpath

    lic_dpath = os.path.join(lic_file_base_path, f'{dataset}')
    lic_file_path = os.path.join(lic_dpath, extension_path)
    return i3_file_path, lic_file_path


def process_files(i3file_dir, licfile_dir, flux_path, selection, num_files, liveTime, w_dir, norm_list, f_type):
    
    if f_type == 'all':
        splineList, norm_list = initPropFiles(flux_path, norm_list, f_type='all', selection=selection)
        splineList = [splineList]
    else:
        splineListAtmo,  norm_list = initPropFiles(flux_path, norm_list, f_type='atmo',  selection=selection)
        print(f'Atmo: {norm_list}')
        splineListAstro, norm_list = initPropFiles(flux_path, norm_list, f_type='astro', selection=selection)
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

    valid_datasets = [21395, 21396, 21397, 21398, 21399, 21400, 21401, 21402, 21403, 21408]
    if int(dataset_num) not in valid_datasets:
        raise IOError(f'Could not correctly determine dataset number based on path {dataset_num}!')

    for CCNC in ['CC', 'NC']:
        if CCNC == 'NC' and dataset_num == '21408':
            continue
        if CCNC == 'CC' and dataset_num == '21408':
            CCNC = 'GR'
        print(f'Dataset: {dataset_num}, CC/NC:{CCNC}')
        i3files, licfiles, datasetList, runList = get_files(i3file_dir, licfile_dir, num_files, selection, CCNC)
        eventInfo, recoInfo = calculate_weights(i3files, licfiles, liveTime, flux_path, selection, CCNC, norm_list, f_type, splineList)
        ccncList = [CCNC] * len(eventInfo.nu_energy) ##just pick anything
        liveTimeL = [liveTime] * len(eventInfo.nu_energy)
        selectionList = [selection] * len(eventInfo.nu_energy)

        ##unpack eventInfoList - items to save in the dataframe
        data = eventInfo.__dict__
        data.update({'Selection': selectionList, 'IntType': ccncList, 'LiveTime': liveTimeL})
        data.update(recoInfo.__dict__)
        df = pd.DataFrame(data=data)
        df.to_hdf(os.path.join(w_dir, f'weight_df_{datasetList[0]}_{selection}_{CCNC}.hdf'), key='df', mode='w')
        print(f'Created: weight_df_{datasetList[0]}_{selection}_{CCNC}.hdf')
        print(len(df.index.values))

def analysis_wrapper(dataset, selection, do_all, flux_path, num_files, f_type):
    pi = np.pi
    proton_mass = 0.93827231 #GeV
    liveTime = 3.1536e7 #365 days in seconds
    w_dir = '/data/user/chill/icetray_LWCompatible/weights'
    norm_list = [0.2, 0.9, 0.95, 0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015, 1.05, 1.1, 5.0]

    ##if dataset is None and do_all is True - grabs all files for 1 selection
    i3file_dirs, licfile_dirs = valid_dir(dataset, selection, do_all)
    
    ##if do_all is False, size of dirs list is 1
    if do_all == False:
        process_files(i3file_dirs[0], licfile_dirs[0], flux_path, selection, num_files, liveTime, w_dir, norm_list, f_type)
        return

    ##start multi-threading here
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        ##loop over all datasets, CCNC
        for i3file_dir, licfile_dir in zip(i3file_dirs, licfile_dirs):
            futures.append(executor.submit(process_files, i3file_dir, licfile_dir, flux_path, selection, num_files, liveTime, w_dir, norm_list, f_type))
    results = wait(futures)
    for result in results.done:
        print(result.result())

@click.command()
@click.option('--dataset', '-d', default=None)
@click.option('--selection', '-s', default='track', type=click.Choice(['track', 'cascade']))
@click.option('--do_all', is_flag=True)
@click.option('--flux_path', '-f', default='../nuSQuIDS_propagation_files/nuSQuIDS_flux_cache_')
@click.option('--num_files', '-n', default=-1)
@click.option('--f_type', '-f', default='separate', type=click.Choice(['separate', 'all']))
def main(dataset, selection, do_all, flux_path, num_files, f_type):
    analysis_wrapper(dataset, selection, do_all, flux_path, num_files, f_type)

if __name__ == "__main__":
    main()

##end
