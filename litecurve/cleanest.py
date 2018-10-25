import numpy as np
from astropy.stats import LombScargle as ls
from matplotlib import pyplot as plt


def cleanest(x, y, n=1):
    '''
    removes sequentially the n first frequencies from y(x) lightcurve
    returns cleanest lightcurve y(x) and its periodogram a2
    '''
    p = np.linspace(0.5, 50.5, 1001)
    f = 1. / p
    for i in range(n):
        #y /= max(y)
        a1 = ls(x, y).power(f)
        bf = f[np.argmax(a1)]
        print('REMOVENDO {0:.2f}'.format(1/bf))
        yf = ls(x, y).model(x, bf)
        d = y - yf + 1
        a2 = ls(x, d).power(f)
        plt.subplot(n+1, 2, 2*i+1)
        plt.plot(x, y, 'r')
        plt.plot(x, yf, 'k')

        plt.subplot(n+1, 2, 2*i+2)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a1, 'b')
        
        y = d
    else:
        plt.subplot(n+1, 2, 2*n+1)
        plt.plot(x, y, 'r')
        plt.subplot(n+1, 2, 2*n+2)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a2, 'b')
        plt.show()
    return y, a2


def cleany(x, y, n=1):
    '''
    removes the nth best frequency from y(x) lightcurve
    returns new d(x) lightcurve and its periodogram a2
    '''
    p = np.linspace(0.5, 50.5, 1001)
    f = 1. / p
    a = ls(x, y).power(f)
    y2 = y
    for i in range(n):
        a1 = ls(x, y).power(f)
        bf = f[np.argmax(a1)]
        print('REMOVENDO {0:.2f}'.format(1/bf))
        yf = ls(x, y).model(x, bf)
        d = y - yf + 1
        a2 = ls(x, d).power(f)
        y = d    
    else:
        plt.subplot(221)
        plt.plot(x, y2, 'r')
        plt.plot(x, yf, 'k')
        plt.subplot(222)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a, 'b')
        d = y2 - yf + 1
        a2 = ls(x, d).power(f)
        plt.subplot(223)
        plt.plot(x, d, 'r')
        plt.subplot(224)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a2, 'b')
        plt.show()
    return d, a2


def join(x, y, n=1):
    '''
    joins n first frequencies in y(x) lightcurve
    returns joined y2(x) lightcurve and array bp of length n with best periods
    '''
    p = np.linspace(0.5, 50.5, 1001)
    f = 1. / p
    a = ls(x, y).power(f)
    plt.subplot(221)
    plt.plot(x, y, 'r')
    y2 = np.ones(len(y))
    bp = []
    for i in range(n):
        a1 = ls(x, y).power(f)
        bf = f[np.argmax(a1)]
        print('REMOVENDO {0:.2f}'.format(1/bf))
        bp.append(1/bf)
        yf = ls(x, y).model(x, bf)
        plt.plot(x, yf, 'k')
        d = y - yf + 1
        a2 = ls(x, d).power(f)
        y = d
        y2 += yf - 1
    else:
        plt.subplot(222)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a, 'b')
        a2 = ls(x, y2).power(f)
        plt.subplot(223)
        plt.plot(x, y2, 'r')
        plt.subplot(224)
        plt.ylim([0.0, 1.0])
        plt.plot(p, a2, 'b')
        plt.show()
    return y2, bp
