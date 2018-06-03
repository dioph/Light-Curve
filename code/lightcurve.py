import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.integrate import simps
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from astropy.io import ascii, fits
from astroquery.mast import Observations
from tqdm import tqdm
from .utils import DisableLogger
from .corrector import SFFCorrector
import copy
import warnings
warnings.filterwarnings('ignore')


class LightCurve(object):
    """
    Implements a simple class for a generic light curve
    
    Attributes
    ----------
    time : array-like
    flux : array-like
    """
    def __init__(self, time, flux):
        self.time = time
        self.flux = flux

    def __getitem__(self, key):
        lc = copy.copy(self)
        lc.time = self.time[key]
        lc.flux = self.flux[key]
        return lc

    def normalize(self):
        """
        Returns a normalized version of the lightcurve
        obtained dividing `flux` by the median flux
        
        Returns
        -------
        lc : LightCurve object 
        """
        lc = copy.copy(self)
        if np.abs(np.nanmedian(lc.flux)) < 1e-9:
            lc.flux = lc.flux + 1
        else:
            lc.flux = lc.flux / np.nanmedian(lc.flux)
        return lc
        
    def remove_nans(self):
        """
        Removes NaN values from flux array
        """
        return self[~np.isnan(self.flux)]
        
    def remove_outliers(self, sigma=5., return_mask=False):
        """
        Removes outliers from lightcurve
        Parameters
        ----------
        sigma : float, optional
            Number of standard deviations to use for clipping limit.
            Default sigma = 5
        return_mask : bool, optional
            Whether to return outlier_mask with the rejected points masked
            Defaults to False
            
        Returns
        -------
        lc : LightCurve object
        outlier_mask : array-like
            Masked array where the points rejected have been masked
        """
        outlier_mask = sigma_clip(data=self.flux, sigma=sigma).mask
        if return_mask:
            return self[~outlier_mask], outlier_mask
        return self[~outlier_mask]
        
    def bins(self, binsize=13, method='mean'):
        """
        """
        available_methods = ['mean', 'median']
        if method not in available_methods:
            raise ValueError("method must be one of: {}".format(available_methods))
        methodf = np.__dict__['nan' + method]
        n_bins = self.time.size // binsize
        lc = copy.copy(self)
        lc.time = np.array([methodf(a) for a in np.array_split(self.time, n_bins)])
        lc.flux = np.array([methodf(a) for a in np.array_split(self.flux, n_bins)])
        if hasattr(lc, 'centroid_col'):
            lc.centrod_col = np.array([methodf(a) for a in np.array_split(self.centroid_col, n_bins)])
        if hasattr(lc, 'centroid_row'):
            lc.centrod_row = np.array([methodf(a) for a in np.array_split(self.centroid_row, n_bins)])
        return lc
    
    def plot(self, ax=None, *args, **kwargs):
        """
        Plots lightcurve in given ax
        
        Returns
        -------
        ax : axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.time, self.flux, *args, **kwargs)
        return ax
        
    def activity_proxy(self, method='dv'):
        """
        Calculates a photometric index for magnetic activity
        
        Parameters
        ----------
        method : string, optional
            The name of the index to be used
            Currently implemented methods:
            --- dv  : (He et al. 2015)
            --- iac : (He et al. 2015)
            Defaults to dv.
        
        Returns
        -------
        act : float
            Photometric activity proxy of lightcurve
        """
        available_methods = ['dv', 'iac']
        dic = {'dv':_dv, 'iac':_iac}
        if method not in available_methods:
            raise ValueError("method must be one of: {}".format(available_methods))
        methodf = dic[method]
        lc = self.normalize()
        lc = lc.remove_nans()
        act = methodf(lc.flux)
        return act
        
    def flatten(self, window_length=101, polyorder=3, **kwargs):
        """
        Removes low-frequency trend using scipy's Savitzky-Golay filter
        
        Returns
        -------
        trend_lc : LightCurve object
            Removed polynomial trend
        flat_lc : LightCurve object
            Detrended lightcurve
        """
        clean_lc = self.remove_nans()
        trend = signal.savgol_filter(x=clean_lc.flux, 
                                    window_length=window_length,
                                    polyorder=polyorder, **kwargs)
        flat_lc = copy.copy(clean_lc)
        trend_lc = copy.copy(clean_lc)
        trend_lc.flux = trend
        flat_lc.flux = clean_lc.flux / trend
        return trend_lc, flat_lc
        
    def fold(self, period, phase=0.0):
        """
        """
        fold_time = ((self.time - phase) / period) % 1
        ids = np.argsort(fold_time)
        return LightCurve(fold_time[ids], self.flux[ids])


class KeplerLightCurve(LightCurve):
    def __init__(self, time, flux, centroid_col, centroid_row):
        super(KeplerLightCurve, self).__init__(time, flux)
        self.centroid_col = centroid_col
        self.centroid_row = centroid_row
        
    def __getitem__(self, key):
        lc = super(KeplerLightCurve, self).__getitem__(key)
        lc.centroid_col = self.centroid_col[key]
        lc.centroid_row = self.centroid_row[key]
        return lc
        
    def correct(self, **kwargs):
        assert np.isnan(self.flux).sum() == 0, "Please remove nans before correcting."
        corrector = SFFCorrector()
        lc_corr = corrector.correct(time=self.time, flux=self.flux, 
            centroid_col=self.centroid_col, centroid_row=self.centroid_row, **kwargs)
        new_lc = copy.copy(self)
        new_lc.time = lc_corr.time
        new_lc.flux = lc_corr.flux
        return new_lc


def create_from_kic(kic, mode='pdc', plsbar=False, quarter=None, campaign=None):
    paths = get_lc_kepler(kic, quarter=quarter, campaign=campaign)
    x, y, ccol, crow = [], [], [], []
    if plsbar:
        bar = tqdm(paths)
    else:
        bar = paths
    for p in bar:
        lc = create_from_file(p, xcol='time', ycol=mode+'sap_flux', mode='kepler')
        lc = lc.normalize()
        x = np.append(x, lc.time)
        y = np.append(y, lc.flux)
        ccol = np.append(ccol, lc.centroid_col)
        crow = np.append(crow, lc.centroid_row)
    return KeplerLightCurve(x, y, ccol, crow)


def create_from_file(filename, xcol='time', ycol='flux', mode='ascii'):
    assert mode in ['ascii', 'fits', 'kepler'], "unknown mode {}".format(mode)
    if mode == 'ascii':
        tbl = ascii.read(filename)
        x = tbl[xcol]
        y = tbl[ycol]
        lc = LightCurve(x, y)
    elif mode == 'fits':
        tbl = fits.open(filename)
        hdu = tbl[1].data
        x = hdu[xcol]
        y = hdu[ycol]
        lc = LightCurve(x, y)
    elif mode == 'kepler':
        tbl = fits.open(filename)
        hdu = tbl[1].data
        x = hdu[xcol]
        y = hdu[ycol]
        ccol = hdu['mom_centr1']
        crow = hdu['mom_centr2']
        lc = KeplerLightCurve(x, y, ccol, crow)
    return lc


def get_lc_kepler(target, quarter=None, campaign=None):
    """
    returns table of LCs from Kepler or K2 for a given target
    """
    
    if 0 < target < 2e8:
        name = 'kplr{:09d}'.format(target)
    elif 2e8 < target < 3e8:
        name = 'ktwo{:09d}'.format(target)
    else:
        #TODO: implement error handling function
        pass
        
    obs = Observations.query_criteria(target_name=name, project=['Kepler', 'K2'])
    products = Observations.get_product_list(obs)
    suffix = 'Lightcurve Long'
    mask = np.array([suffix in fn for fn in products['description']])
    when = campaign if campaign is not None else quarter
    if when is not None:
        mask &= np.array([desc.lower().endswith('q{}'.format(when)) or
                            desc.lower().endswith('c{:02}'.format(when)) or
                            desc.replace('-','').lower().endswith('c{:03d}'.format(when))
                            for desc in products['description']])
    pr = products[mask]
    with DisableLogger():
        dl = Observations.download_products(pr, mrp_only=False)
    return [path[0] for path in list(dl)]


def acf(y, maxlag=None, plssmooth=False):
    """
    Auto-Correlation Function of signal
    Parameters
    ----------
    y : array-like
        Signal
    maxlag : int, optional
        Maximum lag to compute ACF. Defaults to len(y)
    plssmooth : bool, optional
        Whether to smooth ACF using a gaussian kernel.
        
    Returns
    -------
    R : array-like
        First ``maxlag'' samples of ACF(y)
    """
    N = len(y)
    if maxlag is None:
        maxlag = N
    m = np.mean(y)
    s = np.square(y-m).sum()
    R = np.zeros(maxlag)
    lags = np.array(list(range(maxlag)), dtype=int)
    for h in lags:
        a = y[h:]
        b = y[:N-h]
        R[h] = ((a-m)*(b-m)).sum() / s
    if plssmooth:
        R = smooth(R, 7, kernel='gaussian')
    return R


def _dv(y):
    rms = np.sqrt(np.mean(np.square(y-np.median(y))))
    return np.sqrt(8) * rms


def _iac(y):
    N = len(y)
    ml = N//2 + 1
    R = acf(y, maxlag=ml)
    ic = 2/N * simps(R, np.arange(0, ml, 1))
    return ic


def plot_mcmc(samples, labels=None, priors=None, ptrue=None, nbins=30):
    """
    Plots a Giant Triangle Confusogram
    Parameters
    ----------
    samples : 2-D array, shape (N, ndim)
        Samples from ndim variables to be plotted in the GTC
    labels : list of strings, optional
        List of names for each variable (size ndim)
    priors : list of callables, optional
        List of prior functions for the variables distributions (size ndim)
    ptrue : float, optional     #TODO: change into generic list of floats
    nbins : int, optional
        Number of bins to be used in 1D and 2D histograms. Defaults to 30
    """
    p = map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    p = list(p)
    ndim = samples.shape[-1]
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.style.use('dfm_small')
    grid = plt.GridSpec(ndim, ndim, wspace=0.0, hspace=0.0)
    handles = []
    
    ### PLOT 1D
    for i in range(ndim):
        ax = fig.add_subplot(grid[i,i])
        H, edges = np.histogram(samples[:, i], bins=nbins, normed=True)
        centers = (edges[1:]+edges[:-1])/2
        data = ndimage.gaussian_filter1d((centers, H), sigma=1.0)
        data[1] /= data[1].sum()
        l1, = ax.plot(data[0], data[1], 'b-', lw=1, label='posterior')
        if priors is not None:
            pr = priors[i](centers)
            pr /= pr.sum()
            l2, = ax.plot(centers, pr, 'k-', lw=1, label='prior')
        l3 = ax.axvline(p[i][0], color='k', ls='--', label='median')
        mask = np.logical_and(centers-p[i][0] <= p[i][2],
                                p[i][0]-centers <= p[i][1])
        ax.fill_between(centers[mask], np.zeros(mask.sum()), data[1][mask],
                         color='b', alpha=0.3)
        if i < ndim-1:
            ax.set_xticks([])
        else:
            ax.tick_params(rotation=45)
            if ptrue is not None:
                l4 = ax.axvline(ptrue, color='gray', lw=1.5, label='true')
        ax.set_yticks([])
        ax.set_ylim(0)
        if labels is not None:
            ax.set_title('{0} = {1:.2f}$^{{+{2:.2f}}}_{{-{3:.2f}}}$'.format(
                            labels[i], p[i][0], p[i][2], p[i][1]))
    
    handles.append(l1) 
    handles.append(l2)
    try:
        handles.append(l3)
    except:
        pass
    try:
        handles.append(l4)      
    except:
        pass
          
    ### PLOT 2D
    nbins_flat = np.linspace(0, nbins**2, nbins**2)
    for i in range(ndim):
        for j in range(i):
            ax = fig.add_subplot(grid[i,j])
            H, xi, yi = np.histogram2d(samples[:, j], samples[:, i], bins=nbins)
            extents = [xi[0], xi[-1], yi[0], yi[-1]]
            H /= H.sum()
            H_order = np.sort(H.flat)
            H_cumul = np.cumsum(H_order)
            tmp = np.interp([.0455, .3173, 1.0], H_cumul, nbins_flat)
            chainlevels = np.interp(tmp, nbins_flat, H_order)
            data = ndimage.gaussian_filter(H.T, sigma=1.0)
            xbins = (xi[1:]+xi[:-1])/2
            ybins = (yi[1:]+yi[:-1])/2
            ax.contourf(xbins, ybins, data, levels=chainlevels, colors=['#1f77b4','#52aae7','#85ddff'], alpha=0.3)
            ax.contour(data, chainlevels, extent=extents, colors='b')
            if i < ndim-1:
                ax.set_xticks([])
            else:
                ax.tick_params(rotation=45)
                if ptrue is not None:
                    ax.axhline(ptrue, color='gray', lw=1.5)
            if j > 0:
                ax.set_yticks([])
            else:
                ax.tick_params(rotation=45)
    fig.legend(handles=handles)


def make_gauss(m, s):
    """
    Basic 1-D Gaussian implementation
    Parameters
    ----------
    m : float, array-like
        Mean (location parameter)
    s : float, array-like
        Standard deviation (scale parameter)
    
    Returns
    -------
    gauss : callable
        A gaussian function on a variable x with given parameters
    """
    def gauss(x):
        return 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*((x-m)/s)**2)
    return gauss


def gauss(m, s, x):
    return 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*((x-m)/s)**2)


def smooth(y, scale, kernel='boxcar', **kwargs):
    """
    Smooths a signal with a kernel. Wraps astropy.convolution.convolve
    
    Parameters
    ----------
    y : array-like
        Raw signal
    scale : int
        Scale parameter of filter (e.g. width, stddev)
    kernel : str, optional
        Kernel used to convolve signal. Default: ``boxcar''

    Returns
    -------
    s : array-like
        Filtered signal
    """
    if kernel == 'boxcar':
        s = convolve(y, Box1DKernel(scale, **kwargs))
    elif kernel == 'gaussian':
        s = convolve(y, Gaussian1DKernel(scale, **kwargs))
    return s


def filt(y, lo, hi, fs, order=5):
    """
    Filters a signal with a 5th order butterworth passband digital filter
    
    Parameters
    ----------
    y : array-like
        Signal to be filtered
    lo : float
        Lower critical frequency with -3dB
    hi : float
        Higher critical frequency with -3dB
    fs : float
        Sampling frequency (fs > hi > lo)
    
    Returns
    -------
    yf : array-like
        Filtered signal
    """
    nyq = 0.5 * fs
    lo /= nyq
    hi /= nyq
    b, a = signal.butter(order, [lo, hi], btype='band')
    yf = signal.lfilter(b, a, y)
    return yf          


