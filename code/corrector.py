import numpy as np
from scipy import linalg, interpolate
from astropy.stats import sigma_clip
from lightcurve import LightCurve


class SFFCorrector(object):
    def __init__(self):
        pass
        
    def correct(self, time, flux, centroid_col, centroid_row, polyorder=5,
                niters=3, bins=15, windows=1, sigma_1=3., sigma_2=5.):
        timecopy = time
        time = np.array_split(time, windows)
        flux = np.array_split(flux, windows)
        centroid_col = np.array_split(centroid_col, windows)
        centroid_row = np.array_split(centroid_row, windows)
        flux_hat = np.array([])
        
        for i in range(windows):
            rot_col, rot_row = self.rotate_centroids(centroid_col[i], centroid_row[i])
            mask = sigma_clip(data=rot_col, sigma=sigma_2).mask
            coeffs = np.polyfit(rot_row[~mask], rot_col[~mask], polyorder)
            poly = np.poly1d(coeffs)
            self.polyprime = poly.deriv()
            
            x = np.linspace(rot_row[~mask].min(), rot_row[~mask].max(), 10000)
            s = np.array([self.arclength(x1=xp, x=x) for xp in rot_row])
            self.trend = np.ones(len(time[i]))
            for n in range(niters):
                bspline = self.fit_bspline(time[i], flux[i])
                iter_trend = bspline(time[i] - time[i][0])
                normflux = flux[i] / iter_trend
                self.trend *= iter_trend
                interp = self.bin_and_interpolate(s, normflux, bins, sigma=sigma_1)
                corrected_flux = normflux / interp(s)
                flux[i] = corrected_flux
            flux_hat = np.append(flux_hat, flux[i])
            
        return LightCurve(timecopy, flux_hat)
        
    def rotate_centroids(self, centroid_col, centroid_row):
        centroids = np.array([centroid_col, centroid_row])
        _, eig_vecs = linalg.eigh(np.cov(centroids))
        return np.dot(eig_vecs, centroids)
        
    def arclength(self, x1, x):
        mask = x < x1
        return np.trapz(y=np.sqrt(1 + self.polyprime(x[mask]) ** 2), x=x[mask])
    
    def fit_bspline(self, time, flux, s=0):
        t2 = time - time[0]
        #knots = np.arange(0, time[-1], 1.5)
        idx = (np.arange(1, len(t2)-1, (len(t2)-2)/50)).astype(int)
        knots = t2[idx]
        t, c, k = interpolate.splrep(t2, flux, t=knots, s=s, task=-1)
        return interpolate.BSpline(t, c, k)
        
    def bin_and_interpolate(self, s, normflux, bins, sigma):
        idx = np.argsort(s)
        s_srtd = s[idx]
        normflux_srtd = normflux[idx]

        mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask
        normflux_srtd = normflux_srtd[~mask]
        s_srtd = s_srtd[~mask]

        knots = np.array([np.min(s_srtd)]
                         + [np.median(split) for split in np.array_split(s_srtd, bins)]
                         + [np.max(s_srtd)])
        bin_means = np.array([normflux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(normflux_srtd, bins)]
                             + [normflux_srtd[-1]])
        return interpolate.interp1d(knots, bin_means, bounds_error=False,
                                    fill_value='extrapolate')


class CBVCorrector(object):
    def __init__(self):
        pass

    def correct(self, cbvs, method='powell'):
        pass
