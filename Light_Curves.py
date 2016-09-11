from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from os import *
import pandas as pd
from pandas import Series, DataFrame, Panel
from scipy import fftpack
import tkinter as tk    # GUI

class LightCurve(object):
    def __init__(self, tbl=Table(), N=0, T=0, xcol='', ycol=''):
        self.tbl = tbl
        self.N = N
        self.T = T
        self.xcol = xcol
        self.ycol = ycol
    def read(self, arq, cwd='C:\\', form='ipac'):
        self.tbl = Table.read(cwd+arq, format=form) 
        self.N = len(self.tbl)
        self.xcol, self.ycol = self.tbl.colnames[1:]
        self.T = self.tbl[self.xcol][1] - self.tbl[self.xcol][0]
    def plot(self, xlabel='x', ylabel='y'):
        plt.plot(self.tbl[self.xcol], self.tbl[self.ycol])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    def fourier(self):
        xf = np.linspace(0.0, 1/(2.0*self.T), self.N/2)
        yf = fftpack.fft(self.tbl[self.ycol])
        return xf, yf

def init():
    plt.style.use(astropy_mpl_style)
    return getcwd()  # current working directory


cwd = init()
lc = LightCurve() 
lc.read('\plot.tbl', cwd)  # KIC 7447200 light curve

# interpolate data to a uniform sampling

'''
1- points number reduction
2- discontinuities supression
3- corrected linear tendency
'''

# fourier transform [DCDFT]

xf, yf = fourier(lc)
plt.plot(xf, 2/lc.N * np.abs(yf[:lc.N/2]))

plt.figure()
lc.plot(xlabel='Time', ylabel='Flux')

plt.show()

# get rid of white noise [CLEANEST]

# try using wavelet (?)

# parametrize transit
