from astropy.io import fits
import matplotlib.pyplot as plt

def read(arq, cwd=''):
    try:
        hdulist = fits.open(cwd+arq)
    except:
        hdulist = tbl = None
        print('Error reading FITS file')
    return hdulist

def extract(hdu):
    try:
        tbl = hdu.data
    except:
        tbl = None
        print('Error extracting table from'+arq)
    return tbl

def plot(x, y, xlabel='x', ylabel='y'):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def HDUnum(hdulist):
    valid = True
    n = 0
    while valid:
        try:
            hdulist[n].header[0]
            n += 1
        except:
            valid = False
    return n
'''
def filterNaN(hdulist, ycol):
'''