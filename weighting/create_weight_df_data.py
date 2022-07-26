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
from event_info import DataEventInfo
from event_info import CascadeInfo, TrackInfo
from get_data_live_time import wrapper as live_time
from get_data_live_time import get_files ##looking for data type files, not generic
from configs.config import config

def extract_event_info(eventInfo, recoInfo, frame, liveTime, selection):
    EventHeader = frame['I3EventHeader']
    run = EventHeader.run_id
    sub_run = EventHeader.sub_run_id
    event_id = EventHeader.event_id
    sub_event_id = EventHeader.sub_event_id
    
    ##make sure it's a float
    liveTime = float(liveTime)

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

def info_wrapper(i3file_list, liveTime, selection):
    eventInfo = DataEventInfo()
    if selection == 'track':
        recoInfo = TrackInfo(config.track_reco)
    if selection == 'cascade':
        recoInfo = CascadeInfo(config.cascade_reco)
    for i3file in tqdm(i3file_list):
        data_file = dataio.I3File(i3file, 'r')
        # scan over the frames
        while data_file.more():
            try:
                frame = data_file.pop_physics()
            ##if no physics frames are in the file - skip
            except RuntimeError:
                continue
            eventInfo, recoInfo = extract_event_info(eventInfo, recoInfo, frame, liveTime, selection)
    return eventInfo, recoInfo

def process_files(i3file_list, selection, liveTime, w_dir):

    eventInfo, recoInfo = info_wrapper(i3file_list, liveTime, selection)
    data = eventInfo.__dict__
    data.update(recoInfo.__dict__)
    df = pd.DataFrame(data=data)
    df.to_hdf(os.path.join(w_dir, f'weight_df_data_{selection}.hdf'), key='df', mode='w')
    print(f'Created: weight_df_data_{selection}.hdf')
    print(len(df.index.values))

def analysis_wrapper(selection, test=False):
    pi = np.pi
    proton_mass = 0.93827231 #GeV
    w_dir = '/data/user/chill/icetray_LWCompatible/weights'

    path_track = config.path_track    
    path_cascade = config.path_cascade
    liveTime = live_time(selection, path_track, path_cascade)

    if selection == 'track':
        files = get_files(path_track, selection)
        if len(files) == 0:
            raise IOError(f'No files found in {path_track}!')
    if selection == 'cascade':
        files = get_files(path_cascade, selection)
        if len(files) == 0:
            raise IOError(f'No files found in {path_cascade}!')

    if test == True:
        process_files([files[0]], selection, liveTime, w_dir)
        return
        
    process_files(files, selection, liveTime, w_dir)

@click.command()
@click.option('--selection', '-s', default='track', type=click.Choice(['track', 'cascade']))
@click.option('--test', '-t', is_flag=True)
def main(selection, test):
    analysis_wrapper(selection, test)

if __name__ == "__main__":
    main()

##end
