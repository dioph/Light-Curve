from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt, matplotlib.ticker as ticker, numpy as np, time, re, os

# prints error messages
def error(msg):
    print('ERROR - '+msg)
    os.system("pause")

# write message with current time
def clock(msg=''):
    print(msg + ': ' + time.asctime(time.localtime()))

# open FITS file and returns hdulist
def open(arq, mode=None):
    try:
        hdulist = fits.open(arq, mode)
    except:
        hdulist = None
        error('Error reading FITS file: '+arq)
    return hdulist

# closes FITS file
def close(hdulist):
    try:
        hdulist.close()
    except:
        error('Cannot close HDU list')

# extracts data table from HDU
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
        error('Could not find '+colname.upper()+' column')
        col = None
    return col

# returns 'TIME' column from data table
def timecol(tbl):
    try:
        col = tbl.field('time')
    except:
        error('Could not find TIME column')
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
            error('Could not find any SAP Flux column')
            col = None
    return col

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
    cd = np.median(dt) * 86400.0
    return times[0], times[-1], len(times), cd

# inserts a keyword to a HDU
def newrow(name, value, comment, hdu):
    try:
        hdu.header.update(name, value, comment)
    except:
        error('Cannot create '+name.upper()+' keyword')

# returns value from a keyword of a HDU
def getrow(hdu, name):
    try:
        val = hdu.header[name]
    except:
        error('Cannot read '+name.upper()+'keyword')
        val = None
    return val

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
            error('Cannot find '+ycol.upper()+' column')
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
    if 'short' in mode:
        cd = 54.1782
    elif 'long' in mode:
        cd = 1625.35

    return tstart, tstop, bjdref, cd

# get World Coordinate System keywords
def getwcs(hdu):
    pix1 = 0.0
    pix2 = 0.0
    val1 = 0.0
    val2 = 0.0
    delta1 = 0.0
    delta2 = 0.0
    try:
        pix1 = getrow(hdu, 'CRPIX1P')
    except:
        error('Cannot read keyword CRPIX1P')
    try:
        pix2 = getrow(hdu, 'CRPIX2P')
    except:
        error('Cannot read keyword CRPIX2P')
    try:
        val1 = getrow(hdu, 'CRVAL1P')
    except:
        error('Cannot read keyword CRVAL1P')
    try:
        val2 = getrow(hdu, 'CRVAL2P')
    except:
        error('Cannot read keyword CRVAL2P')
    try:
        delta1 = getrow(hdu, 'CDELT1P')
    except:
        error('Cannot read keyword CDELT1P')
    try:
        delta2 = getrow(hdu, 'CDELT2P')
    except:
        error('Cannot read keyword CDELT2P')
    return pix1, pix2, val1, val2, delta1, delta2

# calculate coordinate from WCS
def wcs(i, pix, val, delta):
    return val + (float(i + 1) - pix) * delta

# reads target pixel data file
def readTPF(arq, name):
    tpf = open(arq, 'readonly')

    try:
        id = str(tpf[0].header['KEPLERID'])
    except:
        error('No KEPLERID keyword in ' + arq)
    try:
        ch = str(tpf[0].header['CHANNEL'])
    except:
        error('No CHANNEL keyword in ' + arq)
    try:
        sky = str(tpf[0].header['SKYGROUP'])
    except:
        error('No SKYGROUP keyword in ' + arq)
    try:
        mod = str(tpf[0].header['MODULE'])
    except:
        error('No MODULE keyword in ' + arq)
    try:
        out = str(tpf[0].header['OUTPUT'])
    except:
        error('No OUTPUT keyword in ' + arq)
    try:
        qt = str(tpf[0].header['QUARTER'])
    except:
        error('No QUARTER keyword in ' + arq)
    try:
        s = str(tpf[0].header['SEASON'])
    except:
        s = '0'
    try:
        ra = str(tpf[0].header['RA_OBJ'])
    except:
        error('No RA_OBJ keyword in ' + arq)
    try:
        dec = str(tpf[0].header['DEC_OBJ'])
    except:
        error('No DEC_OBJ keyword in ' + arq)
    try:
        mag = str(float(tpf[0].header['KEPMAG']))
    except:
        mag = ''
    try:
        dim = tpf['TARGETTABLES'].header['TDIM5']
        xdim = int(dim.strip().strip('(').strip(')').split(',')[0])
        ydim = int(dim.strip().strip('(').strip(')').split(',')[1])
    except:
        error('Cannot read TDIM5 keyword in ' + arq)
    try:
        col = tpf['TARGETTABLES'].header['1CRV5P']
    except:
        error('Cannot read 1CRV5P keyword in ' + arq)
    try:
        row = tpf['TARGETTABLES'].header['2CRV5P']
    except:
        error('Cannot read 2CRV5P keyword in ' + arq)
    try:
        data = tpf['TARGETTABLES'].data.field(name)[:]
    except:
        error('Cannot read '+name+' column in '+arq)
        data = None
    close(tpf)
    if len(np.shape(data)) == 3:
        data = np.reshape(data, (np.shape(data)[0], np.shape(data)[1]*np.shape(data)[2]))
    return id, ch, sky, mod, out, qt, s, ra, dec, mag, xdim, ydim, col, row, data

# read target pixel mask data
def read_mask(arq):
    tpf = open(arq, 'readonly')
    try:
        img = tpf['APERTURE'].data
    except:
        error('Cannot read mask definition in '+arq)
    try:
        size1 = tpf['APERTURE'].header['NAXIS1']
    except:
        error('Cannot read NAXIS1 keyword from '+arq)
    try:
        size2 = tpf['APERTURE'].header['NAXIS2']
    except:
        error('Cannot read NAXIS2 keyword from '+arq)
    pix1, pix2, val1, val2, delta1, delta2 = getwcs(tpf['APERTURE'])
    coord1 = np.zeros((size1, size2))
    coord2 = np.zeros((size1, size2))
    for j in range(size2):
        for i in range(size1):
            coord1[i,j] = wcs(i, pix1, val1, delta1)
            coord2[i,j] = wcs(j, pix2, val2, delta2)
    close(tpf)
    return img, coord1, coord2

# bitmap decoding
def bit_in_bitmap(bitmap, bit):
    for i in range(10,-1,-1):
        if bitmap - 2**i >= 0:
            bitmap -= 2**i
            if 2**i == bit:
                return True
    return False

####
## customizable plot:
##
## arq:     FITS file
## yname:   name of the column with flux data
## quality: True if user wants to ignore cadences where the data quality is questionable, False otherwise
## ylabel:  label of y axis; default='e$^-$ s$^{-1}$'
## fill:    color of data filling within the plot
####            
def plot(arq, yname, quality=True, ylabel='e$^-$ s$^{-1}$', fill='#ffff00'):
    plt.style.use(astropy_mpl_style)
### get input columns
    hdulist = open(arq, mode='readonly')
    tstart, tstop, bjdref, cd = time_info(hdulist)
    tbl = extract(hdulist[1])
    xcol = timecol(tbl)
    xcol += bjdref
    ycol = getcol(tbl, yname)
    qlty = getcol(tbl, 'SAP_QUALITY')
    close(hdulist)
### remove bad data
    array = np.array([xcol, ycol, qlty], dtype='float64')
    array = np.rot90(array, 3)
    array = array[~np.isnan(array).any(1)]
    array = array[~np.isinf(array).any(1)]
    if quality:
        array = array[array[:,0] == 0.0]
    timedata = np.array(array[:,2], dtype='float64')
    fluxdata = np.array(array[:,1], dtype='float32')
    if len(timedata) == 0:
        error('Plotting arrays are full of NaN')
### clean up
    timeshift = float(int(tstart/100) * 100.0)
    timedata -= timeshift
    xlabel = 'BJD $-$ %d' % timeshift

    exp = 0
    try:
        exp = len(str(int(np.nanmax(fluxdata)))) - 1
    except:
        exp = 0
    fluxdata /= 10**exp
    if 'e$^-$ s$^{-1}$' in ylabel or 'default' in ylabel:
        if exp == 0:
            ylabel = 'e$^-$ s$^{-1}$'
        else:
            ylabel = '10$^{%d}$ e$^-$ s$^{-1}$' % exp
    else:
        ylabel = re.sub('_', '-', ylabel)
### limits
    xmin = float(np.nanmin(timedata))
    xmax = float(np.nanmax(timedata))
    ymin = float(np.nanmin(fluxdata))
    ymax = float(np.nanmax(fluxdata))
    xr = xmax - xmin
    yr = ymax - ymin
    timedata = np.insert(timedata, [0], [timedata[0]])
    timedata = np.append(timedata, timedata[-1])
    fluxdata = np.insert(fluxdata, [0], -10000.0)
    fluxdata = np.append(fluxdata, -10000.0)
### plot
    plt.figure(figsize=[16,8])
    plt.ticklabel_format(useOffset=False)
    subtime = np.array([], dtype='float64')
    subflux = np.array([], dtype='float32')
    dt = 0
    delta = 2.0 * cd / 86400
    for i in range(1, len(fluxdata)-1):
        dt = timedata[i] - timedata[i-1]
        if dt < delta:
            subtime = np.append(subtime, timedata[i])
            subflux = np.append(subflux, fluxdata[i])
        else:
            plt.plot(subtime, subflux, 'b-')
            subtime = np.array([], dtype='float64')
            subflux = np.array([], dtype='float32')
    plt.plot(subtime, subflux, 'b-')
    plt.fill(timedata, fluxdata, fc=fill, alpha=0.2)
            
    if ymin-yr*0.01 <= 0.0:
        plt.axis([xmin-xr*0.01, xmax+xr*0.01, 1.0e-10, ymax+yr*0.01])
    else:
        plt.axis([xmin-xr*0.01, xmax+xr*0.01, ymin-yr*0.01, ymax+yr*0.01])

    plt.xlabel(xlabel)
    try:
        plt.ylabel(ylabel)
    except:
        ylabel = '10$^{%d}$ e$^-$ s$^{-1}$' % exp
        plt.ylabel(ylabel)

# plot archived photometric time-series for individual target pixels
def plotpixel(arq):
### open TPF
    id, ch, sky, mod, out, qt, s, ra, dec, mag, xdim, ydim, col, row, timedata = readTPF(arq, 'TIME')
    id, ch, sky, mod, out, qt, s, ra, dec, mag, xdim, ydim, col, row, qlty = readTPF(arq, 'QUALITY')
    id, ch, sky, mod, out, qt, s, ra, dec, mag, xdim, ydim, col, row, fluxpix = readTPF(arq, 'FLUX')
### read mask definition from TPF
    img, coord1, coord2 = read_mask(arq)
### print target data
    print('KepID: '+id)
    print('RA (J2000): '+ra)
    print('Dec (J2000): '+dec)
    print('KepMag: '+mag)
    print('SkyGroup: '+sky)
    print('Season: '+str(s))
    print('Channel: '+ch)
    print('Module: '+mod)
    print('Output: '+out)
### remove bad rows
    size = 0
    num_rows = len(fluxpix)
    for i in range(num_rows):
        if qlty[i] == 0 and np.isfinite(timedata[i]) and np.isfinite(fluxpix[i,ydim*xdim/2]):
            size += 1
    time_array = np.empty((size))
    corr_array = np.empty((size))
    cdno_array = np.empty((size))
    qlty_array = np.empty((size))
    fpix_array = np.empty((ydim, xdim, size))
    ferr_array = np.empty((ydim, xdim, size))
### build output light curves
    for i in range(ydim):
        for j in range(xdim):
            size = 0
            for k in range(num_rows):
                if qlty[k] == 0 and np.isfinite(timedata[i]) and np.isfinite(fluxpix[i,ydim*xdim/2]):
                    time_array[size] = timedata[k]
                    qlty_array[size] = qlty[k]
                    fpix_array[i,j,size] = fluxpix[k,i*xdim+j]
                    size += 1
### plot pixel array
    plt.style.use(astropy_mpl_style)

    fmin = 1.0e33
    fmax = -1.0e33
    plt.figure(figsize=[12,12])
    dx = 0.93 / xdim
    dy = 0.94 / ydim
    ax = plt.axes([0.06, 0.05, 0.93, 0.94])
    plt.ticklabel_format(useOffset=False)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.axis([np.min(coord1)-0.5, np.max(coord1)+0.5, np.min(coord2)-0.5, np.max(coord2)+0.5])
    plt.xlabel('time')
    plt.ylabel('flux')
    for i in range(ydim):
        for j in range(xdim):
            tmin = np.amin(time_array)
            tmax = np.amax(time_array)
            fmin = np.amin(fpix_array[i,j,:])
            fmax = np.amax(fpix_array[i,j,:])
            xmin = tmin - (tmax-tmin)/40
            xmax = tmax + (tmax-tmin)/40
            ymin = fmin - (fmax-fmin)/20
            ymax = fmax + (fmax-fmin)/20
            if bit_in_bitmap(img[i,j], 2):
                plt.axes([0.06+float(j)*dx, 0.05+i*dy, dx, dy], axisbg='lightslategray')
            elif img[i,j] == 0:
                plt.axes([0.06+float(j)*dx, 0.05+i*dy, dx, dy], axisbg='black')
            else:
                plt.axes([0.06+float(j)*dx, 0.05+i*dy, dx, dy])
            plt.setp(plt.gca(), xticklabels=[], yticklabels=[])

            ptime = time_array
            ptime = np.insert(ptime, [0], ptime[0])
            ptime = np.append(ptime, ptime[-1])
            pflux = fpix_array[i,j,:]
            pflux = np.insert(pflux, [0], -1000.0)
            pflux = np.append(pflux, -1000.0)
            plt.plot(time_array, fpix_array[i,j,:], 'b-', linewidth=0.5)
            if not bit_in_bitmap(img[i,j], 2):
                plt.fill(ptime, pflux, fc='lightslategray', linewidth=0.0, alpha=1.0)
            plt.fill(ptime, pflux, fc='#fff380', linewidth=0.0, alpha=1.0)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
