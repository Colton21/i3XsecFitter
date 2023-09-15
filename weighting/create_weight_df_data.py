# used for parsing inputs
from optparse import OptionParser
# used for verifying files exist
import os, sys
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
from weighting.event_info import DataEventInfo
from weighting.event_info import CascadeInfo, TrackInfo
from helper.get_data_live_time import wrapper as get_live_time
from helper.get_data_live_time import get_files ##looking for data type files, not generic
from configs.config import config

def calc_live_time(df):
    years = sorted(df.year.unique())
    if len(years) == 1:
        return [df.live_time.values[0]], years
    elif len(years) != 1:

        lt_list = [0] * len(years)
        for i, _y in enumerate(years):
            _df = df[df.year == _y]
            lt_list[i] = _df.live_time.values[0]
        return lt_list, years

def extract_event_info(eventInfo, recoInfo, frame, year, liveTime, selection, yearly_info):
    EventHeader = frame['I3EventHeader']
    run = EventHeader.run_id
    sub_run = EventHeader.sub_run_id
    event_id = EventHeader.event_id
    sub_event_id = EventHeader.sub_event_id
    
    ##make sure it's a float
    liveTime = float(liveTime)
    
    yearList = config.good_run_year_list
    if yearly_info != None:
        found_year = False
        for _y in yearly_info:
            if _y[0] == int(year):
                liveTime = _y[1]
                found_year = True
                break
        if found_year == False:
            raise ValueError(f'Could not find info for {year} in {yearly_info}!')

    ##add the year the event is from
    eventInfo.year.append(int(year))

    eventInfo.run.append(run)
    eventInfo.sub_run.append(sub_run)
    eventInfo.event.append(event_id)
    eventInfo.sub_event.append(sub_event_id)
    eventInfo.Selection.append(selection)
    eventInfo.live_time.append(liveTime)

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

def info_wrapper(i3file_list, data_year, liveTime, selection, yearly_info=None):
    eventInfo = DataEventInfo()
    if selection == 'track':
        recoInfo = TrackInfo(config.track_reco)
    if selection == 'cascade':
        recoInfo = CascadeInfo(config.cascade_reco)
    for i, i3file in enumerate(tqdm(i3file_list, desc='i3 Files')):
        data_file = dataio.I3File(i3file, 'r')
        _year = data_year[i]
        # scan over the frames
        while data_file.more():
            try:
                frame = data_file.pop_physics()
            ##if no physics frames are in the file - skip
            except RuntimeError:
                continue
            eventInfo, recoInfo = extract_event_info(eventInfo, recoInfo, frame, 
                                            _year, liveTime, selection, yearly_info)
    return eventInfo, recoInfo

def process_files(i3file_list, data_year, selection, liveTime, w_dir, yearly_info=None):

    if liveTime == -1 and yearly_info != None:
        eventInfo, recoInfo = info_wrapper(i3file_list, data_year, 
                                           liveTime, selection, yearly_info)
    if liveTime != -1:
        eventInfo, recoInfo = info_wrapper(i3file_list, data_year, liveTime, selection)

    data = eventInfo.__dict__
    data.update(recoInfo.__dict__)
    df = pd.DataFrame(data=data)
    df.to_hdf(os.path.join(w_dir, f'weight_df_data_{selection}.hdf'), key='df', mode='w')
    print(f'Created: weight_df_data_{selection}.hdf')
    print(len(df.index.values))

def analysis_wrapper(selection, test=False, legacy_livetime=False):
    pi = np.pi
    proton_mass = 0.93827231 #GeV
    w_dir = config.weights_dir

    path_track = config.path_track    
    path_cascade = config.path_cascade
    ##old implementation
    if legacy_livetime == True:
        liveTime = get_live_time(selection, path_track, path_cascade,
                                 year_breakdown=False)
    ##yearly_info list of tuples per year
    ##year, livetime, run_start, run_end
    elif legacy_livetime == False:
        if selection == 'cascade':
            yearly_info = get_live_time(selection, path_track, path_cascade, 
                                        year_breakdown=True, alt_list=True)
            print(yearly_info)
        elif selection == 'track':
            yearly_info = get_live_time(selection, path_track, path_cascade, 
                                        year_breakdown=True)
    else:
        raise ValueError(f'{legacy_livetime} must be true or false')

    if selection == 'track':
        files, years = get_files(path_track, selection)
    elif selection == 'cascade':
        files, years = get_files(path_cascade, selection)
    else:
        raise ValueError(f'{selection} must be track or cascade')   

    if legacy_livetime == True:
        if test == True:
            process_files([files[0]], selection, liveTime, w_dir)
        else:    
            process_files(files, years, selection, liveTime, w_dir)
    else:
        if test == True:
            process_files([files[0]], selection, -1, w_dir, yearly_info)
        else:    
            process_files(files, years, selection, -1, w_dir, yearly_info)

@click.command()
@click.option('--selection', '-s', default='track', type=click.Choice(['track', 'cascade']))
@click.option('--test', '-t', is_flag=True)
@click.option('--legacy_livetime', '-lt', is_flag=True)
def main(selection, test, legacy_livetime):
    analysis_wrapper(selection, test, legacy_livetime=legacy_livetime)
    print("Done")

if __name__ == "__main__":
    main()

##end
