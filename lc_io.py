from astropy.io import fits
import matplotlib.pyplot as plt, numpy as np, time

# prints error messages
def error(msg):
    print('ERROR - '+msg)

# open fits file and returns hdulist
def read(arq, mode=None, cwd=''):
    try:
        hdulist = fits.open(cwd+arq, mode)
    except:
        hdulist = None
        error('Error reading FITS file: '+cwd+arq)
    return hdulist

#closes fits file
def close(hdulist):
    try:
        hdulist.close()
    except:
        error('Cannot close HDU list')

# opens ascii file
def ascii(arq, mode):
    try:
        file = open(arq, mode)
    except:
        error('Error opening file '+arq)
    return file

# extracts data table from hdu
def extract(hdu):
    try:
        tbl = hdu.data
    except:
        tbl = None
        error('Error extracting table from '+arq)
    return tbl

# returns column from data table
def getcol(tbl, colname):
    try:
        col = tbl.field(colname)
    except:
        error('Could not find column '+colname)
        col = None
    return col

# returns 'TIME' column from data table
def timecol(tbl):
    try:
        col = tbl.field('time')
    except:
        error('Could not find time column')
        col = None
    return col

# returns 'SAP_FLUX' or 'AP_RAW_FLUX' column from data table
def fluxcol(tbl):
    try:
        col = tbl.field('sap_flux')
    except:
        try:
            col = tbl.field('ap_raw_flux')
        except:
            error('Could not find SAP Flux column')
            col = None
    return col

# returns the number of HDUs in a hdulist
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

# writes a fits file from hdu to arq
def writefits(hdu, arq):
    try:
        hdu.writeto(arq)
    except:
        error('Cannot create FITS file: '+arq)

# gets time diff info from hdulist
def median_cadence(hdulist):
    try:
        times = hdulist[1].data.field('barytime')
    except:
        times = getcol(hdulist[1].data, 'time')
    dt = []
    for i in range(1, len(times)):
        if np.isfinite(times[i]) and np.isfinite(times[i-1]):
            dt.append(times[i]-times[i-1])
    dt = np.array(dt, dtype='float32')
    cd = np.median(dt) # * 86400.0
    return times[0], times[-1], len(times), cd

# inserts a keyword to a HDU
def newrow(name, value, comment, hdu):
    try:
        hdu.header.update(name, value, comment)
    except:
        error('Cannot create '+name+' keyword')

# filters out NaN values from timecol and fluxcol
def filternan(hdulist, ycol):
    try:
        isclean = hdulist[1].header['NANCLEAN']
    except:
        new_size = 0
        for i in range(len(hdulist[1].columns.names)):
            if 'time' in hdulist[1].columns.names[i].lower():
                xcol = hdulist[1].columns.names[i]
        try:
            hdulist[1].data.field(ycol)
        except:
            error('Cannot find '+ycol+' column')
        else:
            try:
                for i in range(len(hdulist[1].data.field(0))):
                    if str(hdulist[1].data.field(xcol)[i]) != '-inf' and str(hdulist[1].data.field(ycol)[i]) != '-inf':
                        hdulist[1].data[new_size] = hdulist[1].data[i]
                        new_size += 1
                hdulist[1].data = hdulist[1].data[:new_size]
                comment = 'NaN cadences removed from data'
                newrow('NANCLEAN', True, comment, hdulist[1])
            except:
                error('Failed to filter NaN cadences')
    return hdulist

# write message with current time
def clock(msg=''):
    print(msg + ': ' + time.asctime(time.localtime()))

# read time keywords from hdulist
def time_info(hdulist):
    tstart = 0.0
    tstop = 0.0
    cd = 0.0
### BJDREF
    try:
        bjdrefi = hdulist[1].header['BJDREFI']
    except:
        bjdrefi = 0.0
    try:
        bjdreff = hdulist[1].header['BJDREFF']
    except:
        bjdreff = 0.0
    bjdref = bjdrefi + bjdreff
### TSTART
    try:
        tstart = hdulist[1].header['TSTART']
    except:
        try:
            tstart = hdulist[1].header['STARTBJD'] + 2.4e6
        except:
            try:
                tstart = hdulist[0].header['LC_START'] + 2400000.5
            except:
                error('Cannot find TSTART')
    tstart += bjdref
### TSTOP
    try:
        tstop = hdulist[1].header['TSTOP']
    except:
        try:
            tstop = hdulist[1].header['ENDBJD'] + 2.4e6
        except:
            try:
                tstop = instr[0].header['LC_END'] + 2400000.5
            except:
                error('Cannot find TSTOP')
    tstop += bjdref
### OBSMODE
    cd = 1.0
    try:
        mode = hdulist[0].header['OBSMODE']
    except:
        try:
            mode = hdulist[1].header['DATATYPE']
        except:
            error('Cannot find OBSMODE')
    else:
        if 'short' in mode:
            cd = 54.1782
        elif 'long' in mode:
            cd = 1625.35

    return tstart, tstop, bjdref, cd

# plots two arrays
def plot(x, y, xlabel='x', ylabel='y'):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
