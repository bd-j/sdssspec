import os, sys
import numpy as np
from astropy.io import fits as pyfits

ddir='/Users/bjohnson/Projects/sdss_spec/hs_sdssspec/examples/sdss/'
objnums = {'NGC3937': '2515-54180-0377',
            'NGC5227': '0528-52022-0137',
            'SDSS115744': '095-53474-0355',
            'UGC08248': '2618-54506-0310',
            }

def getdata(objname, deredshift=True, ddir=ddir, **extras):
    
    fn = os.path.join(ddir, 'spec-{}.fits'.format(objname)) 
    with pyfits.open(fn) as hdus:
        spec = np.array(hdus[1].data)
        info = np.array(hdus[2].data)
        line = np.array(hdus[3].data)

    obs = {}
    obs['wavelength'] = 10**spec['loglam']
    obs['spectrum'] = spec['flux'] * 1e-17
    obs['unc'] = spec['ivar']**(-0.5) * 1e-17
    obs['sky'] = spec['sky'] * 1e-17
    
    
    obs['z_sdss'] = info['Z']
    gline = line['LINEZ'] > 0.0
    obs['sigma_lines'] = [np.median(line['LINESIGMA'][gline])]
    obs['z_lines'] = [np.median(line['LINEZ'][gline])]

    obs['vdisp_sdss'] = info['VDISP']

    if deredshift:
        obs['deredshifted'] = True
        obs['wavelength'] /= (1 + obs['z_sdss'])
    else:
        obs['deredshifted'] = False
        
    return obs


def eline_mask(wave, maskfile='sl_mask1.lis', **extras):
    isline = np.zeros(len(wave), dtype=bool)
    with open(maskfile, 'r') as f:
        lines = f.readlines()[1:]
    for l in lines:
        lo, hi = [float(w) for w in l.split()[:2]]
        #print(lo, hi)
        isline = isline | ((wave > lo) & (wave < hi))

    return ~isline
