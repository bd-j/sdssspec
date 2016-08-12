import sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import gridspec
from matplotlib import rcParams

from prospect.utils import plotting
import prospect.io.read_results as bread
from prospect.sources import StepSFHBasis

from jpost import joint_post

def setup(resfile='', **extras):
    res, pr, mod = bread.results_from(resfile, dangerous=True)
    obs = res['obs']
    sps = StepSFHBasis(**res['run_params'])
    return res, obs, mod, sps


def get_vectors(thetas, obs, mod, sps, **extras):
    spec, phot, cal, sed, jitter = [],[],[],[],[]
    jind = np.array(mod.theta_labels()) == 'unc_factor'
    for theta in thetas:
        s, p, x = mod.mean_model(theta, obs=obs, sps=sps)
        if True in jind:
            jitter.append(theta[jind][0])
        else:
            jitter.append(1.0)
        spec.append(s)
        phot.append(p)
        cal.append(mod._speccal)
        sed.append(mod._spec)
    return spec, phot, cal, sed, jitter


def sedfig(ax, specs, phots, jitter, obs,
           resid=False, peraa=True, masked=False,
           slabel='Model spectra', plabel='Model photometry',
           speccolor='maroon', photcolor='orange', **extras):

    w =  obs['wavelength']
    pw = np.array([filt.wave_effective for filt in obs['filters']])
    f, pf = obs['spectrum'], obs['maggies'],
    u, pu = obs['unc'], obs['maggies_unc']

    norm = 1.0 / obs['normalization_guess']
    # to convert from f_lambda cgs/AA to lambda*f_lambda cgs
    sconv = w * norm
    # to convert from maggies to nu * f_nu cgs
    pconv = 3631e-23 * 2.998e18 / pw
    ylabel = r'$\lambda F_{{\lambda}}$ (intrinsic, cgs)'
    if peraa:
        sconv /= w
        pconv /= pw
        ylabel = r'$F_{{\lambda}}$ (intrinsic, cgs)'
    if resid:
        ylabel = r'$\chi$'
    
    for s, p, j in zip(specs, phots, jitter):
        if resid:
            y = (s - f) / (u * j)
            sm = obs['mask']
            z = (p - pf) / pu
            pm = obs['phot_mask']
        else:
            y = s * sconv
            z = p * pconv
            if masked:
                wm = w[obs['mask']]
                sm = (w <= wm.max()) & (w >= wm.min())
                pm = obs['phot_mask']
            else:
                sm = pm = slice(None)
        
        ax.plot(w[sm], y[sm], linewidth=0.5, label=slabel,
                 color=speccolor, alpha=0.4)
        ax.plot(pw[pm], z[pm] , markersize=8.0, linestyle='', label=plabel,
                 marker='o', alpha=0.5, mec=photcolor, color=photcolor)
        slabel = plabel = None

    ax.set_ylabel(ylabel)
    return ax, [pw, pconv, pm]
    

def calfig(ax, cal, obs, calcolor='maroon', **extras):
    w =  obs['wavelength']
    sm = obs['mask']
    clabel = 'Modeled calibration'
    for c in cal:
        ax.plot(w[sm], c[sm], color=calcolor, alpha=0.5, label=clabel)
        clabel = None
    ax.axhline(1.0, linestyle=':')
    ax.set_ylabel(r'Calibration ($\frac{F_{obs}}{F_{model}} \times Const.$)')
    return ax, None

def sfr_plot(res, mod, ax=None, color='black',
             showpars=None, **kwargs):
    tlabel = np.array(res['theta_labels'])
    thetas, _ = plotting.hist_samples(res, mod.theta_labels(), **kwargs)
    sfhs = masses_to_sfh(thetas, mod)
    sfr, mformed = sfhs[1][:, 0], sfhs[2] / 1e8
    xbins, ybins, sigma = plotting.compute_sigma_level(mformed, sfr, **kwargs)
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], colors=color)
    return ax


def sfh_plot(res, mod, nsample=20, thetas=None,
             ax=None, color='blue', label=None, **kwargs):

    if thetas is None:
        samp = np.random.uniform(size=nsample)
        thetas = plotting.posterior_samples(res, samp, **kwargs)
    sfhs = masses_to_sfh(thetas, mod)
    sfrs = sfhs[1]
    ages = sfhs[0]
    tages = 10**(np.append(ages[:, 0], ages[-1,1]) - 9)
    for sfr in sfrs:
        ax.step(tages, np.append(sfr, 0.0), alpha=0.3, where='post',
                label=label, color=color)
        label = None
    return ax

def sfh_fillplot(res, mod, ax=None, color='blue', label=None, **kwargs):
    slabel = label
    ages = mod.params['agebins']
    dt = np.squeeze(np.diff(10**ages, axis=-1))

    pdict = plotting.get_percentiles(res, **kwargs)
    start, stop = mod.theta_index['mass']
    ptiles = np.array([pdict[p] for p in mod.theta_labels()[start:stop]])
    sfrs = ptiles / dt[:, None]

    for sfr, a in zip(sfrs, ages):
        sig = plotting.fill_between(a, 2 * [sfr[0]], 2 * [sfr[2]], ax=ax,
                                    color=color, alpha=0.5, label=slabel)
        ax.plot(a, 2 * [sfr[1]], color=color)
        slabel=None
    return ax
        
def masses_to_sfh(thetas, mod):
    start, stop = mod.theta_index['mass']
    
    ages = mod.params['agebins']
    dt = np.squeeze(np.diff(10**ages, axis=-1))
    
    masses = np.atleast_2d(thetas)[:, start:stop]
    sfrs = masses / dt
    mformed = masses.sum(axis=-1)
    mwa = (masses * ages.mean(axis=-1)).sum() / masses.sum()

    return ages, sfrs, mformed


def get_axes(npar, figsize=(10,8), **extras):
    fig = pl.figure(figsize=figsize)
    # Set up left hand side
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(left=0.08, right=0.48, hspace=0.1)

    sax = pl.subplot(gs1[0,0])
    rax = pl.subplot(gs1[1,0])

    # Set up right hand side
    gs2 = gridspec.GridSpec(npar, npar)
    gs2.update(left=0.58, right=0.98, wspace=0.05, hspace=0.01, top=0.8)
    haxes = np.array([pl.subplot(gs2[i, j]) for i in range(npar) for j in range(npar)])
    
    return sax, rax, haxes.reshape(npar, npar), fig


if __name__ == "__main__":

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    #rcParams['axes.grid'] = False
    colors = rcParams['axes.color_cycle']
    showpars=['logzsol', 'dust2', 'zred', 'sigma_smooth']
    
    literature = {}#{'tage': 10.0, 'logzsol': -0.4, 'dust2': 3.753/1.086}
    kwargs = {'resfile': '../../results/sdss_test_1468019817_mcmc',
              'labelfontsize': 12,
              'titlefontsize': 10,
              'tickparams': {'labelsize': 8, 'length': 2, 'width': 1},
              'start': 0.5,
              #'thin': 20,
              'showbest': False,
              'ticklabelrotate': ['logzsol', 'sigma_smooth', 'dust2', 'zred', 'gas_logu'],
              #'parlims': np.array([[9,12], [-0.45, -0.18], [2.35, 3.15]]),
              'truths': literature,
              'truthlabel': None,
              'nsample': 10,
              'figsize': (10.5, 6.6) #(14, 7.5)
              }

    if len(sys.argv) > 1:
        kwargs['resfile'] = sys.argv[1]

    # read data and get parameter samples, spec vectors
    res, obs, mod, sps = setup(**kwargs)
    mod.params['max_wave_smooth'] = 7500.0
    mod.params['min_wave_smooth'] = 3750.0
    srand = np.random.uniform(0, 1.0, kwargs['nsample']) 
    thetas = plotting.posterior_samples(res, srand, **kwargs)
    blob = get_vectors(thetas, obs, mod, sps)
    spec, phot, cal, sed, jitter = blob
    if 'gas_logu' in mod.free_params:
        showpars.append('gas_logu')
    kwargs['truths'] = {'zred': obs['z_sdss'], 'sigma_smooth': obs['vdisp_sdss']}
    if obs.get('deredshifted', False):
        kwargs['truths']['zred'] = 0.0
    
    # Set up axes
    sfhax, rax, taxes, fig = get_axes(len(showpars), **kwargs)
    sfrax = taxes[0,-1]
    lfig, (lsax, lrax) = pl.subplots(2, 1, sharex=True, figsize=(13, 5.5))

    # Plot observations
    #sax.plot(obs['wavelength'], obs['spectrum']*obs['rescale'],
    #        color='black', label='Observed spectrum')
    lsax.plot(obs['wavelength'], obs['spectrum']*obs['rescale'],
              color='black', label='Observed spectrum')

    # Plot samples
    #sax, pdat = sedfig(sax, sed, phot, jitter, obs, masked=True)
    lsax, pdat = sedfig(lsax, sed, phot, jitter, obs, masked=True)
    rax, rdat = sedfig(rax, spec, phot, jitter, obs, resid=True,
                       slabel='Spectra', plabel='Photometry')
    lrax, rdat = sedfig(lrax, spec, phot, jitter, obs, resid=True,
                       slabel='Spectra', plabel='Photometry')
    taxes = joint_post(res, taxes, thiscolor='maroon', showpars=showpars, **kwargs)

    # Plot more observations
    pw, pconv, pmask = pdat 
    lsax.errorbar(pw[pmask], obs['maggies'][pmask] * pconv[pmask],
                  yerr=obs['maggies_unc'][pmask] * pconv[pmask],
                  marker='o', markersize=5.0, ecolor='black',
                  color=colors[0], alpha=1.0, mec='black', mew=2,
                  linestyle='', label='Observed photometry')

    # Plot SFH
    sfhax = sfh_fillplot(res, mod, ax=sfhax,
                         color='maroon', label='16th-84h', **kwargs)

    # Axis setup
    [ax.set_ylim(1e-16, 6e-15) for ax in [lsax]]
    [ax.set_yscale('log') for ax in [lsax]]
    
    [ax.axhline(0.0, linestyle=':', color='k')  for ax in [rax, lrax]]
    [ax.set_ylim(-7.9, 7.9)  for ax in [rax, lrax]]
    [ax.set_ylabel(r'$\chi \, (\frac{model-data}{\sigma})$') for ax in [rax, lrax]]
    [ax.text(0.75, 0.8, '$Residuals$', transform=ax.transAxes)  for ax in [rax, lrax]]
    [ax.set_xlabel('$\lambda$')  for ax in [rax, lrax]]
    
    [ax.set_xlim(3600, 7100) for ax in [rax, lsax, lrax]]
    [ax.set_xticklabels('') for ax in [lsax]]

    
    [a.set_visible(False) for i,a in enumerate(taxes.flat) if not a.has_data()]
    [ax.legend(loc=0, prop={'size':10}) for ax in [lsax, sfhax]]
    #[ax.legend(loc=0, prop={'size':8}) for ax in taxes.flat]

    sfhax.set_yscale('log')
    sfhax.set_label('log Age')
    sfhax.set_ylabel('SFR $(M_\odot/yr)$')
    sfhax.set_xlim(6, 10.2)
        
    [f.suptitle(obs['objname']) for f in [fig, lfig]]
    
    pl.show()
