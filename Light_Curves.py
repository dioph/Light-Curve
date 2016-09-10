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

plt.style.use(astropy_mpl_style)
cwd = getcwd()

t = Table.read(cwd+"\plot.tbl", format='ipac')
N = len(t)
T = t['TIME'][1] - t['TIME'][0]
# interpolate data to a uniform sampling
yf = fftpack.fft(t['LC_INIT'])
xf = np.linspace(0.0, 1/(2.0*T), N/2)
fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N/2]))
plt.show()

plt.plot(t['TIME'], t['LC_INIT'])
plt.xlabel('Time')
plt.ylabel('Flux')
plt.show()
