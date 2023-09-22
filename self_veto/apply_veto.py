import os, sys
import numpy as np

from icecube import photospline
from icecube import MuonGun

def pdg_to_name(pdg):
    if pdg not in [12, -12, 14, -14]:
        raise ValueError(f'pdg: {pdg} not valid, must be(-) 12/14/16')
    name = ''
    if pdg == 12 or pdg == -12:
        name = 'nu_e'
    elif pdg == 14 or pdg == -14:
        name = 'nu_mu'
    elif pdg == 16 or pdg == -16:
        return 'nu_tau'
    
    if pdg < 0:
        name = name + 'bar'

    return name

##default dir are splines from Zelong
def get_veto_splines(source_dir='/data/user/chill/icetray_LWCompatible/veto_pass_rate',
                     pdg=None):
    if pdg == None:
        raise ValueError('must supply valid pdg to eval self veto')
    name = pdg_to_name(pdg)

    if name == 'nu_tau':
        return [1, 1]

    splines = []
    for season in ['December', 'June']:
        fname = f'pr_HillasGaisser2012_H3a_CORSIKA_SouthPole_{season}_SIBYLL2.3c_conv_{name}.fits'
        fpath = os.path.join(source_dir, fname)
        spline = photospline.I3SplineTable(fpath)
        splines.append(spline)

    return splines    

##also event by event
def get_depths(surface, i3pos, i3dir):
    d = surface.intersection(i3pos, i3dir)
    getDepth = i3pos + d.first * i3dir
    impactDepth = MuonGun.depth(getDepth.z)*1.e3
    return impactDepth


##only applies to cascades
##for tracks, set the weight to 1
##passing fraction is set to 1 for nutau
def passing_rate(eventInfo):

    print('Calculating the self-veto passing fractions')

    base = '/cvmfs/icecube.opensciencegrid.org/data/GCD/'
    gcdfile = os.path.join(base, 'GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz')    
    surface = MuonGun.ExtrudedPolygon.from_file(gcdfile)

    ##open the photspline tables
    s_pdg_list = [12, -12, 14, -14]
    spline_list = []
    for pdg in s_pdg_list:        
        splines = get_veto_splines(pdg=pdg)
        spline_list.append(splines)

    energys   = eventInfo.nu_energy
    czens     = np.cos(eventInfo.nu_zenith)
    pdgs      = eventInfo.pdg
    i3posList = eventInfo.i3pos
    i3dirList = eventInfo.i3dir

    pf_list = [1] * len(pdgs)
    i = 0
    ##loop over all events
    for e, z, pdg, i3pos, i3dir in zip(energys, czens, pdgs, i3posList, i3dirList):
        ##only need down-going events
        if abs(pdg) == 16 or z < 0.05:
            i += 1
            continue
        d = get_depths(surface, i3pos, i3dir)
        if pdg == 12:
            _i = 0
        elif pdg == -12:
            _i = 1
        elif pdg == 14:
            _i = 2
        elif pdg == -14:
            _i = 3
        splines = spline_list[_i]
        pf = 0
        ##eval both December and June, then average
        for s in splines:
            pf += s.eval([np.log10(e), z, d])
        pf = pf/2
        if pf < -0.1 or pf > 1.1:
            raise ValueError(f'Passing fracting out of bounds: {pf} for {e}, {z}, {d}, {pdg}')
        if pf > 1 and pf <= 1.1:
            pf = 1
        if pf < 0 and pf > -0.1:
            pf = 0

        pf_list[i] = pf
        i += 1

    print('Done calculating passing fractions')

    return pf_list

def apply_veto(eventInfo, selection):
    if selection == 'track':
        pf_list = [1] * len(eventInfo.nu_energy)
    else:
        pf_list = passing_rate(eventInfo)

    eventInfo.veto_pf = pf_list

    return eventInfo

##end
