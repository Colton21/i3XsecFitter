import sys, os
import click
from glob import glob
import numpy as np
from tqdm import tqdm

#####
from I3Tray import I3Tray
from icecube import icetray, dataio, dataclasses
#####

sys.path.append('/data/user/chill/icetray_LWCompatible/i3XsecFitter/')
from configs.config import config
    
def get_standard_file(year):
    base = f'/data/exp/IceCube/{year}/filtered'
    print(f'Grabbing standard good run list from {base}')
    if int(year) == 2017:
        run_info_file = f'{base}/level2/IC86_{year}_GoodRunInfo_Versioned.txt' 
    elif int(year) == 2012:
        run_info_file = f'{base}/level2pass2/IC86_{year}_GoodRunInfo.txt' 
    elif int(year) < 2017:
        #run_info_file = f'{base}/level2pass2/IC86_{year}_GoodRunInfo.txt' 
        run_info_file = f'{base}/level2/IC86_{year}_GoodRunInfo_Versioned.txt' 
    else:
        #run_info_file = f'{base}/level2/IC86_{year}_GoodRunInfo.txt' 
        run_info_file = f'{base}/level2/IC86_{year}_GoodRunInfo_Versioned.txt' 
    
    if not os.path.exists(run_info_file):
        raise IOError(f'File {run_info_file} does not exist!!')
    else:
        return run_info_file

def get_alt_file(year):
    base = '/data/user/zzhang1/pass2_GlobalFit/code/submission/goodrunlist/finalpass2/'
    print(f'Grabbing alt good run list from {base}')
    if int(year) == 2010:
        num_str = 79
    elif int(year) >= 2011:
        num_str = 86
    gr_file = os.path.join(base, f'IC{num_str}_{year}_GoodRunInfo_l3_final.txt')
    if not os.path.exists(gr_file):
        raise FileNotFoundError(f'file {gr_file} not found!')
    else:
        return gr_file

def get_good_runs(year, alt_list=False):
    if alt_list == False:
        run_info_file = get_standard_file(year)
    if alt_list == True:
        run_info_file = get_alt_file(year)

    f_run_list = []
    f_live_time_list = []
    f_string_list = []
    f_missing_files_list = []

    with open(run_info_file, 'r') as f:
        run_info = f.read()
        ##first 2 lines are header
        info = run_info.split('\n')[2:]
        
        ##go line by line
        for i in info:
            if i == ' ' or i == '':
                continue
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
    
    return f_run_list, f_live_time_list, f_string_list, f_missing_files_list

def live_time_calc(run_list, selection, just_time=True, year_breakdown=False, alt_list=False):
    ##burn sample years and good run years a bit different (calendar vs season)
    yearList = config.good_run_year_list
    print(f'Calculate the livetime for {selection} over Seasons {yearList}')

    run_list = sorted(run_list)
    print(f'First event is {np.min(run_list)}, Last event is {np.max(run_list)}')

    f_run_list = []
    f_live_time_list = []
    f_string_list = []
    f_missing_files_list = []
    ##year, live time, first run, last run
    yearly_info = [(0, 0, 0, 0)] * len(yearList)
    for i, year in enumerate(yearList):
        _run_list, _live_time_list, \
            _string_list, _missing_files_list = get_good_runs(year, alt_list)
        f_run_list.extend(_run_list)
        f_live_time_list.extend(_live_time_list)
        f_string_list.extend(_string_list)
        f_missing_files_list.extend(_missing_files_list)
        
        _run_list = np.array(_run_list)
        _live_time_list = np.array(_live_time_list)
        ##only events in range and ending in 0
        _mask = (_run_list%10 == 0) & (_run_list >= run_list[0]) & (_run_list <= run_list[-1])
        print(f'Season {year}: {np.sum(_live_time_list[_mask])}s')
        if np.sum(_live_time_list[_mask]) != 0:
            yearly_info[i] = (year, np.sum(_live_time_list[_mask]), _run_list[0], _run_list[-1])
        else:
            yearly_info[i] = (year, 0, 0, 0)

    ##sort again in-case function is called externally
    f_run_list = sorted(f_run_list)
    f_run_list = np.array(f_run_list)
    f_live_time_list = np.array(f_live_time_list)

    mask_10 = (f_run_list%10 == 0) & (f_run_list >= run_list[0]) & (f_run_list <= run_list[-1])
    skimmed_run_list = f_run_list[mask_10]
    print(f'All Runs ending in 0 in my range: {np.sum(f_live_time_list[mask_10])}')

    ##now check which are which
    l, ind1, ind2 = np.intersect1d(run_list, f_run_list, return_indices=True) 
    used_live_time = f_live_time_list[ind2]
    total_live_time = np.sum(used_live_time)

    if len(l) != len(run_list):
        raise ValueError(f'All runs should be in the good run list! ({len(l)}, {len(run_list)}')
    print(f'Number of runs in good run list: {len(l)}')

    if year_breakdown == True:
        print('Returning the live time per year')
        return yearly_info
    elif just_time == True:
        print(f'Returning only the live time of {total_live_time}')
        return total_live_time
    else:
        print('Also grabbing the list of "good" events')
        return total_live_time, run_list[ind2]

def check_run_numbers(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError(f'Lenghts of lists do not agree!: {len(list_a)} {len(list_b)}')
    list_a = sorted(list_a)
    list_b = sorted(list_b)
    for a, b in zip(list_a, list_b):
        if a != b:
            raise ValueError(f'{a} and {b} do not agree!')

def get_files(path, selection):
    files  = []
    fyears = []
    if selection == 'track':
        yearList = config.track_year_list
    elif selection == 'cascade':
        yearList = config.cascade_year_list

    for year in yearList:
        if selection == 'cascade':
            #files = glob(f'{path}/*.i3.zst.i3.bz2')
            #files = glob(f'{path}/final_*_hansbdt/*.i3.zst.i3.bz2')
            #files = np.append(files, glob(f'{path}/final_hybrid/*.i3.zst.i3.bz2'))
            #files = np.append(files, glob(f'{path}/final_muon/*.i3.zst.i3.bz2'))
            fpath = f'{path}/IC86_{year}/burn/final_cascade/Finallevel_IC86_*0.i3.zst'
        
        if selection == 'track':
            fpath = f'{path}/IC{year}/*0.i3.zst'
       
        _files = glob(fpath) 
        _fyear = [year] * len(_files) 
        if len(_files) == 0:
            raise IOError(f'No files found in {fpath}!')
        
        files.extend(_files)
        fyears.extend(_fyear)

    return files, fyears

def get_run_numbers(path, selection):
    files, years = get_files(path, selection)
    print(f"Opening {len(files)} files")
    run_number_list = [0] * len(files)
    for i, f in enumerate(tqdm(files, desc=f'Reading {selection} i3 files')):
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

    return run_number_list

def wrapper(selection, path_track, path_cascade, year_breakdown=False, alt_list=False):
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
    else:
        raise NotImplementedError(f'Selection option {selection} not valid!')
            
    if selection == 'track' or selection == 'cascade':
        live_time = live_time_calc(run_number_list, selection, year_breakdown=year_breakdown,
                                   alt_list=alt_list)
        print(f'Live time {live_time} seconds')
        return live_time
    if selection == 'all':
        live_time_t = live_time_calc(run_number_list[0], selection='track', 
                                     year_breakdown=year_breakdown, alt_list=alt_list)
        live_time_c = live_time_calc(run_number_list[1], selection='cascade',
                                     year_breakdown=year_breakdown, alt_list=alt_list)
        print(f'Track live time {live_time_t} seconds')
        print(f'Cascade live time {live_time_c} seconds')
        return live_time_t, live_time_c

 
@click.command()
@click.option('--selection', '-s', default='all')
@click.option('--year_breakdown', '-yb', is_flag=True)
@click.option('--alt', '-a', is_flag=True)
def main(selection, year_breakdown, alt):
    path_track = config.path_track
    path_cascade = config.path_cascade
    wrapper(selection, path_track, path_cascade, year_breakdown=year_breakdown, alt_list=alt)

if __name__ == "__main__":
    main()

##end
