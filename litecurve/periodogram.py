import warnings
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import LombScargle as ls
from scipy.stats import linregress

from .lightcurve import create_from_kic, acf

warnings.filterwarnings('ignore')


def periodogram(kic=None, lc=None, span=None, scale='frequency',
                nperiods=10001, verbose=True):
    """
    Calculates and plots Lomb-Scargle and ACF for a given lightcurve.
    Parameters
    ----------
    kic : int, optional
        KIC/EPIC ID number of star.
        Use when you don't have a LightCurve object ready.
    lc : LightCurve object, optional
        Timeseries in which to perform the periodogram analysis.
    span : list, optional
        Range of periods to search within.
        Minimum defaults to 0.5, maximum defaults to 100 or lc.time.max()/2
    scale : str, optional
        Whether to call LombScargle uniformly on `period` or `frequency`
        Defaults to `frequency`.
    nperiods : int, optional
        Number of samples in LombScargle, defaults to 1e4+1
    verbose : bool, optional
        Whether to print progress, defaults to True

    Returns
    -------
    bp_ls : float
        Best period found with LombScargle
    bp_acf : float
        Best period found with ACF
    # TODO: return period uncertainties
    """
    t1 = perf_counter()
    if lc is None:
        if kic is not None:
            lc = create_from_kic(kic, plsbar=verbose)
        else:
            print('No lightcurve was defined.')
            return
    if verbose:
        print('Got LC.')
    lc = lc.remove_nans().remove_outliers()
    lc_fill, ids = lc.fill()
    if lc.flux.mean() > 0.1:
        lc.flux = lc.flux / np.median(lc.flux)
    if verbose:
        print('Clean.')
    dv = lc.activity_proxy()

    m, b = linregress(lc.time, lc.flux)[:2]
    lc.flux -= m * lc.time + b

    # plt.style.use('idl_small')

    if span is None:
        span = [0.5, min(lc.time.max()/2, 100)]

    assert span[0] < span[1], "bad period search interval ({}).".format(span)
    assert scale in ['period', 'frequency'], "unknown scale {}".format(scale)

    if scale == 'period':
        freq = 1./np.linspace(span[0], span[1], nperiods)
    else:
        freq = np.linspace(1./span[1], 1./span[0], nperiods)

    lomb = ls(lc.time, lc.flux)
    a = lomb.power(freq)
    if verbose:
        print('Calculated periodogram.')

    peaks = np.array([i for i in range(1, len(freq)-1)
                      if a[i-1] < a[i] and a[i+1] < a[i]])
    bp_ls = 1./freq[peaks][a[peaks].argsort()[-3:][::-1]]
    fap = lomb.false_alarm_probability(a.max())
    if verbose:
        print('Calculated FAP.')

    lags = lc_fill.time - lc_fill.time.min()
    ml = np.where(lags >= span[1])[0][0]
    R = acf(lc_fill.flux, maxlag=ml, s=9)
    ti, hi = find_peaks(R, lags[:ml])
    if verbose:
        print('Calculated ACF.')

    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.25, left=0.12)
    if kic is not None:
        fig.suptitle('{0} {1}'.format(["KIC", "EPIC"][kic > 2e8], kic), fontsize=18)

    lc.plot(ax[0], 'ko', markersize=3, alpha=0.25)
    ax[0].axhline(dv/2, color='r')
    ax[0].axhline(-dv/2, color='r')
    ax[0].set_ylabel('FLUX')
    focus = lc.time.mean()
    ax[0].set_xlim(focus-span[1], focus+span[1])

    ax[1].plot(1./freq, a, 'r-')
    for i,bp in enumerate(bp_ls):
        ax[1].axvline(bp, color='k', ls='--', lw=1)
        ax[1].text(.85*span[1], (.9-.2*i)*a.max(), '{0:.2f} d'.format(bp))
    ax[1].set_ylabel('L-S')
    ax[1].set_xlim(span)
    ax[1].set_ylim(0)
    ax[1].text(.825*span[1], .3*a.max(), 'FAP={0:.2f}%'.format(100*fap))

    if fap > 0.05:
        bp_ls = np.nan

    ax[2].plot(lc.time[:ml], R, 'k-')
    ax[2].axhline(0.0, color='gray')
    ax[2].set_ylabel('ACF')
    ax[2].set_xlim(0, span[1])
    bp_acf = lc.time[ti][R[ti].argsort()[-3:][::-1]]
    size = np.max(np.abs(R[50:]))
    ax[2].set_ylim(-1.1*size, 1.1*size)
    for i,bp in enumerate(bp_acf):
        ax[2].axvline(bp, color='r', ls='--', lw=1)
        ax[2].text(.85*span[1], (.85-.2*i)*abs(size), '{0:.2f} d'.format(bp))

    t2 = perf_counter()
    if verbose:
        print('Done in {0:.2f} s'.format(t2-t1))

    return bp_ls, bp_acf, ax


def find_peaks(R, lags):
    peaks = np.array([i for i in range(1, len(lags)-1)
                      if R[i-1] < R[i] and R[i+1] < R[i]])

    dips = np.array([i for i in range(1, len(lags)-1)
                    if R[i-1] > R[i] and R[i+1] > R[i]])

    if lags[dips[0]] > lags[peaks[0]]:
        peaks = peaks[1:]

    '''
    leftdips = np.array([dips[lags[dips] < lags[peak]][-1] for peak in peaks])

    if lags[dips[-1]] > lags[peaks[-1]]:
        rightdips = np.array([dips[lags[dips] > lags[peak]][0] for peak in peaks])
    else:
        rightdips = [dips[lags[dips] > lags[peaks[i]]][0] for i in range(len(peaks)-1)]
        rightdips.append(len(peaks)-1)
        rightdips = np.array(rightdips)

     leftheights = R[leftdips]
     rightheights = R[rightdips]
     peakheights = R[peaks]
     heights = 0.5 * (np.abs(peakheights-leftheights) +
                       np.abs(peakheights-rightheights))
    '''

    # TODO: Calculate actual peak heights
    heights = R[peaks]

    return peaks, heights


if __name__ == '__main__':
    pass
    # TODO: implement command line usage
