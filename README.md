# litecurve
General python tools for astronomical photometric timeseries processing.

Check out [```lightkurve```](https://github.com/KeplerGO/lightkurve) 
and [```pyke```](https://github.com/KeplerGO/pyke) packages from NASA's KeplerGO team.

### Usage
#### Installation
```
pip install --index-url https://test.pypi.org/simple/ litecurve
```
#### Basic:
```python
from litecurve import create_from_kic

kic = 9654627
lc = create_from_kic(kic)
lc.plot()   # equivalent to plt.plot(lc.time, lc.flux)
```
#### Lightcurve object functions:
```python
from litecurve import create_from_kic, acf
import numpy as np
import matplotlib.pyplot as plt

kic = 9654627
lc = create_from_kic(kic, quarter=4)
lc = lc.remove_nans().remove_outliers(sigma=3.)
lc_fill, ids = lc.fill()
lags = lc_fill.time - lc_fill.time.min()
maxlag = np.where(lags >= 70)[0][0]
R = acf(lc_fill.flux, maxlag=maxlag)

fig, ax = plt.subplots(2, 1)
lc.plot(ax[0], 'ko', markersize=3)
ax[0].set_xlabel('TIME')
ax[0].set_ylabel('FLUX')
ax[1].plot(lags[:maxlag], R, 'k-')
ax[1].set_xlabel('LAGS')
ax[1].set_ylabel('ACF')
```