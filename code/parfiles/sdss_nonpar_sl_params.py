import numpy as np
import fsps
from prospect.models import priors, sedmodel
from prospect.sources import StepSFHBasis
tophat = priors.tophat
from sedpy.observate import load_filters, getSED

# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'../results/sdss_test',
              # Fitter parameters
              'nwalkers':128,
              'nburn':[32, 32, 64, 64, 128, 128, 256], 'niter':512,
              'do_powell': False,
              'ftol':0.5e-5, 'maxfev':5000,
              'initial_disp':0.1,
              # Obs data parameters
              'objname':'NGC3937',
              'mask_elines': True,
              'maskfile': '/Users/bjohnson/Projects/sdss_spec/sl_mask1.lis',
              'deredshift': True,
              'wlow':3750.,
              'whigh':7000.,
              # Data Manipulation
              'logify_spectrum': False,
              'normalize_spectrum': True,
              'norm_band_name':'sdss_r0',
              'rescale':True,
              # Model initialization
              'tau': 1.0,
              'total_mass_init': 5e6,
              # SPS parameters
              'agelims': [0., 8.0, 8.3, 8.6, 8.9, 9.2, 9.5, 9.8, 10.14],
              #'agelims': [0., 8.0, 8.5, 9.0, 9.5, 10.14],
              'zcontinuous': 2,
              }

# --------------
# OBS
# --------------

def load_obs(objname='', mask_elines=False, wlow=0, whigh=np.inf,
             norm_band_name='sdss_r0', **kwargs):
    """Load a data file and choose a particular object.
    """
    import sdata
    obs = sdata.getdata(sdata.objnums[objname], **kwargs)

    # Restrict wavelength range and mask
    good = (obs['wavelength'] > wlow) & (obs['wavelength'] < whigh)
    obs['mask'] = good
    if mask_elines:
        obs['mask'] = obs['mask'] & sdata.eline_mask(obs['wavelength'], **kwargs)
    # Add a fake normalization point
    obs['filters'] = load_filters([norm_band_name])
    mags = getSED(obs['wavelength'], obs['spectrum'], filterlist=obs['filters'])
    obs['maggies'] = np.atleast_1d(10**(-0.4 * mags))
    obs['maggies_unc'] = obs['maggies'] / 100  # Hack

    # Add unessential bonus info
    obs['objname'] = objname

    return obs


def expsfh(agelims, tau=3.0, power=1, **extras):
    """Calculate the mass in a set of step functions that is equivalent to an
    exponential SFH.  That is, \int_amin^amax \, dt \, e^(-t/\tau) where
    amin,amax are the age limits of the bins making up the step function.
    """
    from scipy.special import gamma, gammainc
    tage = 10**np.max(agelims) / 1e9
    t = tage - 10**np.array(agelims)/1e9
    nb = len(t)
    mformed = np.zeros(nb-1)
    #t = np.insert(t, 0, tage)
    for i in range(nb-1):
        t1, t2 = t[i+1], t[i]
        normalized_times = (np.array([t1, t2, tage])[:, None]) / tau
        mass = gammainc(power, normalized_times)
        intsfr = (mass[1,...] - mass[0,...]) / mass[2,...]
        mformed[i] = intsfr
    return mformed * 1e3

# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False,
             interp_type='logarithmic', **extras):
    sps = StepSFHBasis(zcontinuous=zcontinuous, interp_type='logarithmic',
                       compute_vega_mags=compute_vega_mags)
    return sps


# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    from prospect.likelihood import NoiseModel, kernels
    pjitter = kernels.Uncorrelated(['phot_jitter'])
    phot_noise = NoiseModel(metric_name='filternames',
                            kernels = [pjitter],
                            weight_by=['maggies_unc'])
    mjitter = kernels.Uncorrelated(['unc_factor'])
    spec_noise = NoiseModel(metric_name='wavelength',
                            kernels = [mjitter],
                            weight_by=['unc'])
    #return None, phot_noise
    return None, None


# --------------
# MODEL_PARAMS
# --------------

model_params = []


# --- SFH --------
model_params.append({'name': 'mass', 'N': 1,
                     'isfree': True,
                     'init': 2e6,
                     'units': r'M$_\odot$',
                     'prior_function': priors.tophat,
                     'prior_args': {'mini':1e4, 'maxi': 1e9}})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [[0.0, 8.0], [8.0, 8.7], ],
                        'units': 'log(yr)',
                        })

model_params.append({'name': 'lumdist', 'N':1,
                         'isfree': False,
                         'init': 1.0,
                         'units': 'Mpc',
                         })

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.3,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': priors.tophat,
                        'prior_args': {'mini':-2, 'maxi':0.1}})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': None})

model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

# --- DUST ---------
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 2, # 2=Calzetti
                        'units': 'index'})

model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.2,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.5,
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.5}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': -0.7,
                        'init_disp': 0.3,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-2.5, 'maxi':-1.3}})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.3,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-2.0, 'maxi':-0.7}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': 'index'})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': False,
                        'init': 0.5,
                        'init_disp': 0.25,
                        'units': 'index',
                        'prior_function': tophat,
                        'prior_args': {'mini': 0.5, 'maxi': 2.5}
                        })

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'init_disp': 0.5,
                        'units': 'index',
                        'prior_function': tophat,
                        'prior_args': {'mini': 0.11, 'maxi': 10.0}
                        })

# --- Stellar Pops ------------
model_params.append({'name': 'tpagb_norm_type', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': 'index'})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'agb_dust', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'index'})

# --- Nebular Emission ------
def gasz_dep(logzsol=0.0, **extras):
    return logzsol

model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'depends_on': gasz_dep,
                        'prior_function':tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'init_disp': 0.5,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})


# ------- WAVELENGTH SCALE ------
model_params.append({'name': 'zred', 'N':1,
                        'isfree': True,
                        'init': 1e-6,
                        'init_disp': 1e-5,
                        'units': None,
                        'prior_function': priors.tophat,
                        'prior_args': {'mini':-3e-4, 'maxi':3e-4}})

model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': True})

model_params.append({'name': 'sigma_smooth', 'N': 1,
                        'isfree': True,
                        'init': 100.0,
                        'init_disp': 50.0,
                        'units': r'$km/s$',
                        'prior_function': priors.tophat,
                        'prior_args': {'mini':1.0, 'maxi':200.0}})
                        #'prior_function': priors.lognormal,
                        #'prior_args': {'log_mean':np.log(2.2)+0.05**2, 'sigma':0.05}})

model_params.append({'name': 'smoothtype', 'N': 1,
                        'isfree': False,
                        'init': 'vel'})

model_params.append({'name': 'fftsmooth', 'N': 1,
                        'isfree': False,
                        'init': True,})

model_params.append({'name': 'min_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 3500.0,
                        'units': r'$\AA$'})

model_params.append({'name': 'max_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 7800.0,
                        'units': r'$\AA$'})

# --- Calibration ---------
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'init_disp': 0.5,
                        'units': '$\sigma_{mags}$',
                        'prior_function': priors.logarithmic,
                        'prior_args': {'mini':0.2, 'maxi':3.0}})


def load_model(agelims=[], mock_params=['logzsol', 'dust2'],
               total_mass_init=3e10, fit_rv=False,
               objname=None, deredshift=False, **kwargs):

    # set up the age bins and initial masses
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1
    mass_init =  expsfh(agelims, **kwargs)
    mass_init *= total_mass_init / mass_init.sum()

    # adjust model parameters
    pnames = [p['name'] for p in model_params]
    mind = pnames.index('mass')
    model_params[mind]['N'] = ncomp
    model_params[mind]['init'] = mass_init
    model_params[mind]['init_disp'] = mass_init * 0.3
    model_params[mind]['prior_args'] = {'maxi':mass_init.max()*1e6,
                                                   'mini':mass_init.min() * 1e-6}
    model_params[pnames.index('agebins')]['N'] = ncomp
    model_params[pnames.index('agebins')]['init'] = agebins.T

    if objname is not None:
        import sdata
        obs = sdata.getdata(sdata.objnums[objname])
        if not deredshift:
            model_params[pnames.index('zred')]['init'] = obs['z_sdss']
            model_params[pnames.index('zred')]['init_disp'] = obs['z_sdss']*0.1
            model_params[pnames.index('zred')]['prior_args']= {'mini':obs['z_sdss']-0.005,
                                                               'maxi':obs['z_sdss'] + 0.005}
        try:
            model_params[pnames.index('sigma_smooth')]['init'] = obs['vdisp_sdss']
        except:
            print('Not setting initial velcity dispersion')

    for k in mock_params:
        try:
            model_params[pnames.index('k')]['init'] = kwargs[k]
        except:
            pass

    # Alter the free and fixed model params based on command line arguments
    if fit_rv:
        if not fit_tesc:
            if model_params[pnames.index('dust_type')]['init'] == 0:
                model_params[pnames.index('dust_index')]['isfree'] = True
            else:
                model_params[pnames.index('mwr')]['isfree'] = True
    
    # initialize the model
    model = sedmodel.SedModel(model_params)
    return model

