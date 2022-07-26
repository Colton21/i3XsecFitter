import sys, os
import click
from glob import glob
import numpy as np
from tqdm import tqdm

#####
from I3Tray import I3Tray
from icecube import icetray, dataio, dataclasses
#####

from configs.config import config

def live_time_calc(run_list):
    run_info_file = '/data/exp/IceCube/2013/filtered/level2pass2/IC86_2013_GoodRunInfo.txt' 
    ##sort again in-case function is called externally
    run_list = sorted(run_list)

    f_run_list = []
    f_live_time_list = []
    f_string_list = []
    f_missing_files_list = []

    with open(run_info_file, 'r') as f:
        run_info = f.read()
        ##first 2 lines are header
        info = run_info.split('\n')[2:]
        for i in info:
            splt = i.split(' ')
            vals = [s for s in splt if s != '']
            run_num = vals[0]
            live_time = vals[3] ##[sec]
            num_i3_strings = vals[4]
            try:
                if vals[8].lower() == 'validated':
                    missing_files = True
                missing_files = False
            except:
                missing_files = False

            f_run_list.append(int(run_num))
            f_live_time_list.append(float(live_time))
            f_string_list.append(int(num_i3_strings))
            f_missing_files_list.append(missing_files)

    ##now check which are which
    l, ind1, ind2 = np.intersect1d(run_list, f_run_list, return_indices=True) 
    f_live_time_list = np.array(f_live_time_list)
    used_live_time = f_live_time_list[ind2]
    total_live_time = np.sum(used_live_time)
    return total_live_time

def check_run_numbers(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError(f'Lenghts of lists do not agree!: {len(list_a)} {len(list_b)}')
    list_a = sorted(list_a)
    list_b = sorted(list_b)
    for a, b in zip(list_a, list_b):
        if a != b:
            raise ValueError(f'{a} and {b} do not agree!')

def get_files(path, selection):
    if selection == 'cascade':
        #files = glob(f'{path}/*.i3.zst.i3.bz2')
        #files = glob(f'{path}/final_*_hansbdt/*.i3.zst.i3.bz2')
        files = glob(f'{path}/final_cascade/*.i3.zst.i3.bz2')
        #files = np.append(files, glob(f'{path}/final_hybrid/*.i3.zst.i3.bz2'))
        #files = np.append(files, glob(f'{path}/final_muon/*.i3.zst.i3.bz2'))
    if selection == 'track':
        files = glob(f'{path}/*0.i3.zst')

    return files

def get_run_numbers(path, selection):
    files = get_files(path, selection)
    print(f"Opening {len(files)} files")
    run_number_list = [0] * len(files)
    for i, f in enumerate(tqdm(files)):
        data_file = dataio.I3File(f, 'r')
        run_number = None
        # scan over the frames
        while data_file.more():
            try:
                frame = data_file.pop_frame()
            ##if no frames are in the file - skip
            except RuntimeError:
                continue
            try:
                header = frame['I3EventHeader']
                run_number = header.run_id
                run_number_list[i] = run_number
                ##once we get the RunID for 1 frame
                ##we have it for the full file
                break
            except KeyError:
                continue
        if run_number == None:
            ##file is empty due to cuts I guess - use name
            bname = os.path.basename(f)
            name = bname.split('.')[0]
            str_num = name.split('_')[4]
            num = str_num[5:]
            run_number_list[i] = int(num)
            #print("Could not find RunID for this file - what's wrong?")
            #print(f)

    return run_number_list

def wrapper(selection, path_track, path_cascade):
    print(f"Collecting info for selection: {selection}")
    if selection == 'track':
        path = path_track
        run_number_list = get_run_numbers(path, selection)
    elif selection == 'cascade':
        path = path_cascade
        run_number_list = get_run_numbers(path, selection)
    elif selection == 'all':
        path = [path_track, path_cascade]
        run_number_list = []
        for p, sel in zip(path, ['track', 'cascade']):
            run_number_list.append(get_run_numbers(p, sel))
        #check_run_numbers(run_number_list[0], run_number_list[1])
    else:
        raise NotImplementedError(f'Selection option {selection} not valid!')
            
    if selection == 'track' or selection == 'cascade':
        live_time = live_time_calc(run_number_list)
        print(f'Live time {live_time} seconds')
        return live_time   
    if selection == 'all':
        live_time_t = live_time_calc(run_number_list[0])
        live_time_c = live_time_calc(run_number_list[1])
        print(f'Track live time {live_time_t} seconds')
        print(f'Cascade live time {live_time_c} seconds')
        return live_time_t, live_time_c

 
@click.command()
@click.option('--selection', '-s', default='all')
def main(selection):
    path_track = config.path_track
    path_cascade = config.path_cascade
    wrapper(selection, path_track, path_cascade)

if __name__ == "__main__":
    main()

##end
