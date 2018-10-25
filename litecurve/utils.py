import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)


class HiddenPrints():
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


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
    p = map(lambda v: (v[1], v[1] - v[0], v[2] - v[1]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    p = list(p)
    ndim = samples.shape[-1]
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.style.use('dfm_small')
    grid = plt.GridSpec(ndim, ndim, wspace=0.0, hspace=0.0)
    handles = []

    ### PLOT 1D
    for i in range(ndim):
        ax = fig.add_subplot(grid[i, i])
        H, edges = np.histogram(samples[:, i], bins=nbins, normed=True)
        centers = (edges[1:] + edges[:-1]) / 2
        data = ndimage.gaussian_filter1d((centers, H), sigma=1.0)
        data[1] /= data[1].sum()
        l1, = ax.plot(data[0], data[1], 'b-', lw=1, label='posterior')
        if priors is not None:
            pr = priors[i](centers)
            pr /= pr.sum()
            l2, = ax.plot(centers, pr, 'k-', lw=1, label='prior')
        l3 = ax.axvline(p[i][0], color='k', ls='--', label='median')
        mask = np.logical_and(centers - p[i][0] <= p[i][2],
                              p[i][0] - centers <= p[i][1])
        ax.fill_between(centers[mask], np.zeros(mask.sum()), data[1][mask],
                        color='b', alpha=0.3)
        if i < ndim - 1:
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
    nbins_flat = np.linspace(0, nbins ** 2, nbins ** 2)
    for i in range(ndim):
        for j in range(i):
            ax = fig.add_subplot(grid[i, j])
            H, xi, yi = np.histogram2d(samples[:, j], samples[:, i], bins=nbins)
            extents = [xi[0], xi[-1], yi[0], yi[-1]]
            H /= H.sum()
            H_order = np.sort(H.flat)
            H_cumul = np.cumsum(H_order)
            tmp = np.interp([.0455, .3173, 1.0], H_cumul, nbins_flat)
            chainlevels = np.interp(tmp, nbins_flat, H_order)
            data = ndimage.gaussian_filter(H.T, sigma=1.0)
            xbins = (xi[1:] + xi[:-1]) / 2
            ybins = (yi[1:] + yi[:-1]) / 2
            ax.contourf(xbins, ybins, data, levels=chainlevels, colors=['#1f77b4', '#52aae7', '#85ddff'], alpha=0.3)
            ax.contour(data, chainlevels, extent=extents, colors='b')
            if i < ndim - 1:
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