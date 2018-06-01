import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from lightcurve import *
from astropy.stats import LombScargle as ls
import warnings
warnings.filterwarnings('ignore')

def periodogram(kic=None, lc=None, focus=None, span=None):
    if kic is not None:
        lc = create_from_kic(kic, plsbar=True)

    print('Got LC.')
    lc = lc.remove_nans().remove_outliers()
    print('Clean.')
    dv = lc.activity_proxy()

    plt.style.use('idl_small')

    if span is None:
        span = min(lc.time[-1]-lc.time[0], 200)

    freq = 1./np.linspace(0.05, span/2+0.05, 10001)
    lc.time -= lc.time.min()


    lc.flux = lc.flux / np.median(lc.flux) - 1.0
    lomb = ls(lc.time, lc.flux)
    a = lomb.power(freq)
    print('Calculated periodogram.')
    bp1 = 1/freq[np.argmax(a)]
    fap = lomb.false_alarm_probability(a.max())

    print('Calculated FAP.')
    ml = np.where(lc.time >= span/2)[0][0]
    R = acf(lc.flux, maxlag=ml, plssmooth=True)
    print('Calculated ACF.')

    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.25)
    if kic is not None:
        fig.suptitle('KIC {}'.format(kic), fontsize=18)

    lc.plot(ax[0], 'ko', markersize=3, alpha=0.25)
    ax[0].axhline(dv/2, color='r')
    ax[0].axhline(-dv/2, color='r')
    ax[0].set_ylabel('FLUX', fontsize=14)
    ax[0].tick_params(labelsize=12)
    if focus is None:
        focus = lc.time.mean()
    ax[0].set_xlim(focus-span/2, focus+span/2)

    ax[1].plot(1/freq, a, 'r-')
    ax[1].axvline(bp1, color='k', ls='--', lw=1)
    ax[1].set_ylabel('L-S', fontsize=14)
    ax[1].tick_params(labelsize=12)
    ax[1].set_xlim(0, span/2)
    ax[1].text(.425*span, .9*a.max(), '{0:.2f} d'.format(bp1))
    ax[1].text(.4*span, .8*a.max(), 'FAP={0:.2f}%'.format(100*fap))

    ax[2].plot(lc.time[:ml], R, 'k-')
    ax[2].axhline(0.0, color='gray')
    ax[2].set_ylabel('ACF', fontsize=14)
    ax[2].tick_params(labelsize=12)
    ax[2].set_xlim(0, span/2)
    search = np.where(R < 0)[0][0]
    if search < ml:
        peak = search + np.argmax(R[search:])
        size = np.max(np.abs(R[search:]))
        ax[2].set_ylim(-1.1*size, 1.1*size)
        bp2 = lc.time[peak]
        ax[2].axvline(bp2, color='r', ls='--', lw=1)
        ax[2].text(.425*span, .85*abs(size), '{0:.2f} d'.format(bp2))
    else:
        ax[2].text(.425*span, .8*ax[2].set_ylim()[-1], 'ERROR')
        bp2 = np.nan
    if fap > 0.05:
        bp1 = np.nan
    return bp1, bp2
    
if __name__ == '__main__':
    import argparse
    # TODO: implement command line usage
    
    
