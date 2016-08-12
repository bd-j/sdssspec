import numpy as np
import matplotlib.pyplot as pl
from prospect.utils import plotting


pnmap = {'mass': r'log M$_*$',
         'logzsol': r'$log \, Z/Z_\odot$',
         'tau': r'$log \, \tau_{SF} (Gyr)$',
         'tage': 'Age (Gyr)',
         'dust1': r'$\tau_{V, BC}$',
         'dust2': r'$A_V$',
         'duste_umin': r'$U_{min}$',
         'duste_gamma': r'$\gamma_{dust}$',
         'duste_qpah': r'$Q_{PAH}$',
         'dust_index': r'$\Gamma_{dust}$',
         'mwr': r'$R_v$',
         'uvb': r'$f_{bump}$',
         'sigma_smooth': r'$\sigma_* (km/s)$',
         'zred': r'$z$',
         'gas_logu': r'$log U$',
         }

pmap = {'Z': lambda x: x - np.log10(0.0134),
        'logt': lambda x: 10**x,
        #'zred': lambda x:  x*100,
        'mass': lambda x: np.log10(x),
        'dust2': lambda x: 1.086*x,
        #'tage': lambda x: np.log10(x),
        'tau': lambda x: np.log10(x)}


def eye(x):
    return x


def transform(x, n):
    return pmap.get(n, eye)(x)


def joint_post(res, paxes, showpars=[],
         parlims=None, showbest=True, truths=None,
         thiscolor='maroon', truthcolor='black',
         tickparams={'labelsize': 7, 'length': 2, 'width': 1}, ticklabelrotate=[],
         labelfontsize=12, titlefontsize=9, **kwargs):
    """
    """
   
    npar = len(showpars)
    assert len(showpars) == paxes.shape[0]
    #if parlims is None:
    #    parlims = np.array(npar * [[None, None]])

    for i, p1 in enumerate(showpars):

        theta_names, theta_best = plotting.get_best(res, **kwargs)
        
        # ---- Plot diagonal -----
        dax = paxes[i,i]
        trace, p = plotting.hist_samples(res, [p1], **kwargs)
        x = transform(trace, p[0])
        # Plot the histogram
        hh = dax.hist(x, bins=30, normed=True, alpha=0.5, label='pPDF',
                      histtype='stepfilled', color=thiscolor)
        # Plot truths + best
        if truths is None:
            try:
                truth = plotting.get_truths(res)[p[0]]
                truth = transform(truth, p[0])
                dax.axvline(truth, linestyle='--', label='Truth',
                            color=truthcolor)
            except:
                truth = None
        else:
            try:
                truth = truths[p[0]]
                dax.axvline(truth, linestyle='--', label='"Truth"',
                            color=truthcolor)
            except(KeyError):
                pass
        if showbest:
            best = transform(theta_best[theta_names.index(p[0])], p[0])
            dax.axvline(best, linestyle=':', label='Max. a post.', linewidth=2,
                        color=thiscolor)
        
        # Label with pctiles.
        pct = np.percentile(trace[:,0], [16, 50, 84])
        pct = transform(pct, p[0])
        if abs(np.log10(abs(pct[1]))) > 2:
            # The number of sig figs should be based on the pct differences.  duh.
            fmt =  r'{3}$={0:.2e}^{{+{1:.2e}}}_{{-{2:.2e}}}$'
        else:
            fmt = r'{3}$={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'
        ti = fmt.format(pct[1], pct[2]-pct[1], pct[1]-pct[0], pnmap.get(p1, p1))
        if titlefontsize is not None:
            dax.set_title(ti, fontdict={'fontsize':titlefontsize})

        # Axis label foo
        dax.tick_params(axis='both', which='major', **tickparams)
        dax.set_yticklabels('')
        if i == (npar-1):
            dax.set_xlabel(pnmap.get(p2, p2), fontsize=labelfontsize)
            if p2 in ticklabelrotate:
                pl.setp(dax.xaxis.get_majorticklabels(), rotation=-55,
                        horizontalalignment='center')
        else:
            dax.set_xticklabels('')

        # --- Off-diagonal axis ----
        for j, p2 in enumerate(showpars[(i+1):]):
            k = j+i+1
            ax = paxes[k, i]
            # plot the joint PDFs
            pdf = plotting.joint_pdf(res, p2, p1, pmap=pmap, **kwargs)
            ax.contour(pdf[0], pdf[1], pdf[2], levels = [0.683, 0.955], colors=thiscolor)
            # Axis label foo
            if i == 0:
                ax.set_ylabel(pnmap.get(p2, p2), fontsize=labelfontsize)
            else:
                ax.set_yticklabels('')
            if k == (npar-1):
                print(p1)
                ax.set_xlabel(pnmap.get(p1,p1), fontsize=labelfontsize)
                dax.set_xlim(ax.get_xlim())
                if p1 in ticklabelrotate:
                    pl.setp(ax.xaxis.get_majorticklabels(), rotation=-55,
                            horizontalalignment='center')

            else:
                ax.set_xticklabels('')

            # Axis range foo
            xcur = ax.get_xlim()
            ycur = ax.get_ylim()
            #print(p1, p2, parlims, parlims[i])
            if parlims is not None:
                #xlims = min([parlims[i,0], xcur[0]]), max([parlims[i,1], xcur[1]])
                #ylims = min([parlims[j+i+1,0], ycur[0]]), max([parlims[j+i+1,1], ycur[1]])
                xlims = parlims[i,:]
                ylims = parlims[j+i+1, :]
            else:
                xlims, ylims = xcur, ycur
            print(i, xlims, ylims)
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            dax.set_xlim(*xlims)
            dax.tick_params(axis='both', which='major', **tickparams)
            ax.tick_params(axis='both', which='major', **tickparams)

            # Plot truths (if they exist)
            if truths is None:
                try:
                    truth = get_truths(res)
                    truthxy = [transform(truth[k], k) for k in [p1, p2]]
                    truthxy = np.squeeze(np.copy(truthxy))
                    ax.plot(truthxy[0], truthxy[1], marker='o', color=truthcolor, label='Truth')
                except:
                    pass
            else:
                pass
                #truthxy = truths[p1], truths[p2]
                #ax.plot(truthxy[0], truthxy[1], marker='o', color=truthcolor, label='Truth')

            # Plot bestfits
            if showbest:
                for n, res in enumerate(results):
                    thiscolor = clrs[n]
                    bests = dict(zip(*get_best(res)))
                    best = [transform(bests[k], k) for k in [p1, p2]]
                    best = np.squeeze(np.copy(best))
                    ax.plot(best[0], best[1], 'o', color=thiscolor, label='MAP')

    return paxes
