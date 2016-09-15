from __future__ import print_function
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
    def __init__(self, T=0, N=0, hdulist=None, tbl=None, xcol=None, ycol=None):
        self.T = T
        self.N = N
        self.hdulist = hdulist
        self.tbl = tbl
        self.xcol = xcol
        self.ycol = ycol

    def fourier(self):
        xf = np.linspace(0.0, 1/(2.0*self.T), self.N/2)
        yf = fftpack.fft(self.ycol)
        return xf, yf

def init():
    plt.style.use(astropy_mpl_style)
    return getcwd()  # current working directory

# I/O
cwd = init()

lc_io.clock('Program initialized')

# KIC 007447200 light curve
lc = LightCurve()
arq = input('Please insert file name: ')
lc.hdulist = lc_io.read(arq)
lc.tbl = lc_io.extract(lc.hdulist[1])
lc.T = lc_io.median_cadence(lc.hdulist)[3]
lc.N = len(lc.tbl.field(0))
lc.xcol = lc_io.timecol(lc.tbl)
lc.ycol = lc_io.fluxcol(lc.tbl)

lc_io.clock('Finished creating LightCurve() object')
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

lc_io.plot(lc.xcol, \
           lc.ycol, \
           xlabel='Time ['+lc.tbl.columns['time'].unit+']', \
           ylabel='Flux [e-/s]')

lc_io.clock('Finished plotting')
plt.show()

# get rid of white noise [CLEANEST]

# try using wavelet (?)

# parametrize transit
