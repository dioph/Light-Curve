#from __future__ import print_function
import warnings, numpy as np, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from os import *
from scipy import fftpack
import lc_io
#import pandas as pd
#from pandas import Series, DataFrame, Panel
#import tkinter as tk    # GUI

class LightCurve(object):
    def __init__(self, tbl, N=0, T=0, xcol='time', ycol='sap_flux'):
        self.tbl = tbl
        self.N = N
        self.T = T
        self.xcol = xcol
        self.ycol = ycol

    def fourier(self):
        xf = np.linspace(0.0, 1/(2.0*self.T), self.N/2)
        yf = fftpack.fft(self.tbl[self.ycol])
        return xf, yf

def init():
    plt.style.use(astropy_mpl_style)
    return getcwd()  # current working directory

# I/O

cwd = init()
# KIC 007447200 light curve
hdulist = lc_io.read('http://archive.stsci.edu/pub/kepler/lightcurves/0074/007447200/kplr007447200-2009166043257_llc.fits')
lc = LightCurve(tbl=lc_io.extract(hdulist[1]))

# interpolate data to a uniform sampling [CDA]

'''
1- points number reduction
2- discontinuities supression
3- corrected linear tendency
'''

# fourier transform [DCDFT]
'''
xf, yf = lc.fourier()
plt.plot(xf, 2/lc.N * np.abs(yf[:lc.N/2]))

plt.figure()
'''
lc_io.plot(lc.tbl[lc.xcol], lc.tbl[lc.ycol], xlabel='Time ['+lc.tbl.columns[lc.xcol].unit+']', ylabel='Flux ['+lc.tbl.columns[lc.ycol].unit+']')

plt.show()

# get rid of white noise [CLEANEST]

# try using wavelet (?)

# parametrize transit
