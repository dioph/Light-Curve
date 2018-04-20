import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.stats import sigma_clip
import copy

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

    def normalize(self):
        """
        Returns a normalized version of the lightcurve
        obtained dividing `flux` by the median flux
        
        Returns
        -------
        lc : LightCurve object 
        """
        lc = copy.copy(self)
        lc.flux = lc.flux / np.nanmedian(lc.flux)
        return lc
        
    def remove_nans(self):
        """
        """
        lc = copy.copy(self)
        nanmask = np.isnan(lc.flux)
        lc.time = self.time[~nanmask]
        lc.flux = self.flux[~nanmask]
        return lc
        
    def remove_outliers(self, sigma=5., return_mask=False):
        """
        """
        lc = copy.copy(self)
        outlier_mask = sigma_clip(data=lc.flux, sigma=sigma).mask
        lc.time = self.time[~outlier_mask]
        lc.flux = self.flux[~outlier_mask]
        if return_mask:
            return lc, outlier_mask
        return lc
        
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
        return lc
    
    def plot(self, ax=None):
        """
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.time, self.flux, 'kx')
        return ax
        
    def activity_proxy(self, method='dv'):
        """
        """
        lc = self.normalize()
        lc = lc.remove_nans()
        rms = np.sqrt(np.mean(np.square(lc.flux)))
        return np.sqrt(8) * rms
        
    def flatten(self, window_length=101, polyorder=3, **kwargs):
        """
        Removes low-frequency trend using scipy's Savitzky-Golay filter
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
        
if __name__ == '__main__':
    import kplr
    kic = 10619192
    client = kplr.API()
    star = client.star(kic)
    lcs = star.get_light_curves(short_cadence=False)
    x, y = [], []
    for lc in lcs:
        with lc.open() as F:
            hdu = F[1].data
            x = np.append(x, hdu['time'])
            yn = hdu['pdcsap_flux'] / np.nanmedian(hdu['pdcsap_flux'])
            y = np.append(y, yn)
    lc = LightCurve(x, y)
    fig, ax = plt.subplots(3, 1)
    lc.plot(ax=ax[0])
    clean = lc.remove_nans()
    clean = clean.remove_outliers()
    clean = clean.bins()
    trend, flat = clean.flatten()
    flat.plot(ax=ax[1])
    trend.plot(ax=ax[2])
    plt.show()
    
