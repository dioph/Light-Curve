import numpy as np

####
## Fourier transform:
##
## x:   time values
## y:   time-dependent values
## f1:  frequence to start range
## f2:  frequence to stop range
## df:  frequence steps
####
def fourier(x, y, f1, f2, df):
    real = []
    imag = []
    xf = []
    yf = []
    count = 0
    n = len(x)
    for f in np.arange(f1, f2, df):
        real.append(0.0)
        imag.append(0.0)
        for i in range(n):
            real[-1] += y[i] * np.cos(2.0 * np.pi * f * x[i])
            imag[-1] += y[i] * np.sin(2.0 * np.pi * f * x[i])
        xf.append(f)
        if n > 0:
            yf.append((real[-1]**2 + imag[-1]**2) / n**2)
        else:
            yf.append(np.nan)
        count += 1
    xf = np.array(xf, dtype='float32')
    yf = np.array(yf, dtype='float32')
    return xf, yf

