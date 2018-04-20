from astropy.io import fits
import numpy as np
from numpy.linalg import inv as inverse
import pyqtgraph as pg
from pyqtgraph import QtGui
import lc


# retorna CBVs do cbvdata de acordo com cbv_list
def get_vectors(cbvdata, cbv_list, caddata):
    vectors = np.zeros((len(cbv_list), len(caddata)))
    for i in range(len(cbv_list)):
        j = int(cbv_list[i])
        dat = cbvdata.field('VECTOR_%s' % j)[np.isnan(cbvdata.field('cadenceno')) == False]
        vectors[i] = dat[np.in1d(cbvdata.field('cadenceno'), caddata)]
    return vectors


# retorna a soma do CBV segundo os coeffs
def cbvsum(vectors, coeffs):
    soma = 0.0
    for i in range(len(coeffs)):
        soma += coeffs[i] * vectors[i]
    return soma


# retorna novo array de fluxo baseado nos coeffs
def adequar(flux, vectors, coeffs):
    newflux = np.copy(flux)
    for i in range(len(coeffs)):
        newflux += coeffs[i] * vectors[i]
    return newflux


# minimos quadrados
def llsq(vectors, flux):
    A = np.matrix(vectors).transpose()
    y = np.matrix(flux).transpose()
    At = A.transpose()
    coeffs = inverse(At * A) * At * y
    coeffs = np.array(0.0 - coeffs)
    return coeffs


# salva novo array de fluxo em novo_arq na coluna CBVSAP_FLUX
def savefit(hdulist, novo_arq, fluxdata, soma, version):
    if version == 1:
        unit = 'e-/cadence'
        fluxdata *= 1625.35
    elif version == 2:
        unit = 'e-/s'
    col1 = fits.Column(name='CBVSAP_MODL', format='E13.7', unit=unit, array=soma)
    col2 = fits.Column(name='CBVSAP_FLUX', format='E13.7', unit=unit, array=fluxdata)
    cols = hdulist[1].columns + col1 + col2
    hdulist[1] = fits.BinTableHDU.from_columns(cols, header=hdulist[1].header)
    hdulist.writeto(novo_arq, clobber=True)


# remove erros sistematicos utilizando CBVs e salva um novo FITS
def tendencia(win, arq, novo_arq, cbv_list, mask=None, data=None, xdim=None, ydim=None):
    ### ler arquivo de entrada
    hdulist = lc.abrir(arq, 'readonly')
    cad = 1625.35
    try:
        __ = str(hdulist[0].header['filever'])
        version = 2
    except KeyError:
        version = 1
    tbl = hdulist[1].data
    mod = str(hdulist[0].header['module'])
    out = str(hdulist[0].header['output'])
    if version == 1:
        if str(hdulist[1].header['datatype']) == 'long cadence':
            quarter = str(hdulist[1].header['quarter'])
            caddata = tbl.field('cadence_number')
            if data is None:
                timedata = tbl.field('barytime')
                fluxdata = tbl.field('ap_raw_flux') / cad
            else:
                inicio, fim, bjdref, cad = data[2]
                timedata = data[0]
                fluxdata = data[1]
        elif str(hdulist[1].header['datatype']) == 'short cadence':
            lc.error('Tendencia nao implementada para cadencias curtas')
    elif version == 2:
        if str(hdulist[0].header['obsmode']) == 'long cadence':
            quarter = str(hdulist[0].header['quarter'])
            caddata = tbl.field('cadenceno')
            if data is None:
                timedata = tbl.field('time')
                fluxdata = tbl.field('sap_flux')
            else:
                inicio, fim, bjdref, cad = data[2]
                timedata = data[0]
                fluxdata = data[1]
        elif str(hdulist[0].header['obsmode']) == 'short cadence':
            lc.error('Tendencia nao implementada para cadencias curtas')
    ### arquivo com CBVs para o quarter correto
    cbvfile = 'quarter' + quarter + '.fits'
    separator = cbv_list[1]
    ### remover infinitos e colunas de fluxo zero
    if data is not None:
        timeshift = float(int(inicio / 100) * 100.0)
        timedata += timeshift - bjdref
        good_data = np.empty((len(caddata)), dtype=bool)
        time_array = tbl.field('time')
        for i in range(len(time_array)):
            for j in range(len(timedata)):
                if abs(time_array[i] - timedata[j]) <= 0.0001:
                    good_data[i] = True
                    break
            else:
                good_data[i] = False
    else:
        good_data = np.logical_and(np.logical_and(np.isfinite(timedata), np.isfinite(fluxdata)), fluxdata != 0.0)
    caddata = caddata[good_data]
    if data is None:
        timedata = timedata[good_data]
        fluxdata = fluxdata[good_data]
    ### lista de CBVs para utilizar
    cbv_list = np.fromstring(cbv_list, dtype='int', sep=separator)
    cbvdata = fits.open(cbvfile)
    cbvdata = cbvdata['MODOUT_%s_%s' % (mod, out)].data
    vectors = get_vectors(cbvdata, cbv_list, caddata)
    ### minimos quadrados
    medflux = np.median(fluxdata)
    flux_array = (fluxdata / medflux) - 1
    fluxmask = flux_array
    if mask is not None:
        maskdata = np.atleast_2d(np.genfromtxt(mask, delimiter=','))
        mask = np.zeros(len(timedata)) == 0.
        for intervalo in maskdata:
            if version == 1:
                ini = intervalo[0] - 2400000.0
                fim = intervalo[1] - 2400000.0
            elif version == 2:
                ini = intervalo[0] - 2454833.0
                fim = intervalo[1] - 2454833.0
            newdata = np.logical_xor(timedata < ini, timedata > fim)
            mask = np.logical_and(mask, newdata)
        fluxmask = flux_array[mask]
    coeffs = llsq(vectors, fluxmask)
    flux_array = medflux * (adequar(flux_array, vectors, coeffs) + 1)
    soma = cbvsum(vectors, coeffs)
    medflux = np.median(flux_array)
    soma_normal = medflux * (1 - soma)
    ### plotar resultado e salvar FITS
    if data is not None:
        pw1, pw2 = plotfit(win, timedata, fluxdata, flux_array, soma_normal, cad, version, row=ydim+2, col=xdim+2,
                           rowspan=int((ydim+2-(ydim%2))/2), colspan=xdim+2)
    else:
        pw1, pw2 = plotfit(win, timedata, fluxdata, flux_array, soma_normal, cad, version)
    if data is None:
        savefit(hdulist, novo_arq, flux_array, soma, version)
    lc.fechar(hdulist)
    return pw1, pw2, flux_array


# plota aproximacao dos CBVs e fluxo apos a remocao destes
def plotfit(win, timedata, fluxdata1, fluxdata2, soma, cad, version, row=0, col=0, rowspan=1, colspan=1):
    if version == 1:
        timeshift = float(int(timedata[0] / 100) * 100.0)
        timedata -= timeshift
        xlabel = 'BJD - %d' % (timeshift + 2400000.0)
    elif version == 2:
        timeshift = float(int((timedata[0] + 54833.0) / 100) * 100.0)
        timedata += 54833.0 - timeshift
        xlabel = 'BJD - %d' % (timeshift + 2400000.0)

    pw1 = win.addPlot(row=row, col=col, rowspan=rowspan, colspan=colspan)
    sub = np.array([], dtype='int32')
    deltamax = 2.0 * cad / 86400
    for i in range(1, len(fluxdata1)):
        dt = timedata[i] - timedata[i - 1]
        if dt < deltamax:
            sub = np.append(sub, 1)
        else:
            sub = np.append(sub, 0)
    sub = np.append(sub, 1)
    m = np.median(fluxdata1)
    pw1.plot(x=timedata, y=fluxdata1/m - 1, pen='b', connect=sub)
    pw1.plot(x=timedata, y=soma/m - 1, pen='r', connect=sub)
    pw1.showAxis('bottom', False)

    pw2 = win.addPlot(row=row + rowspan, col=col, rowspan=rowspan, colspan=colspan)
    sub = np.array([], dtype='int32')
    for i in range(1, len(fluxdata2)):
        dt = timedata[i] - timedata[i - 1]
        if dt < deltamax:
            sub = np.append(sub, 1)
        else:
            sub = np.append(sub, 0)
    sub = np.append(sub, 1)
    m = np.median(fluxdata2)
    rms1 = np.sqrt(2 * np.mean(np.square(fluxdata2/m - 1)))
    rms2 = -np.sqrt(2 * np.mean(np.square(fluxdata2/m - 1)))
    pw2.setLabel('bottom', 'tempo', units=xlabel)
    pw2.plot(x=timedata, y=fluxdata2/m - 1, pen='b', connect=sub)
    line = pg.InfiniteLine(pos=rms1, angle=0, pen='r')
    pw2.addItem(line)
    line = pg.InfiniteLine(pos=rms2, angle=0, pen='r')
    pw2.addItem(line)
    print('activity =', rms1-rms2)
    return pw1, pw2


if __name__ == '__main__':
    import argparse

    app = QtGui.QApplication([])
    font = QtGui.QFont('Century Gothic', 18)
    app.setFont(font)
    win = pg.GraphicsWindow()
    parser = argparse.ArgumentParser(description='remove erros sistematicos utilizando CBVs e salva um novo FITS')
    parser.add_argument('arq', help='nome do arquivo FITS de origem (LC)', type=str)
    parser.add_argument('novo_arq', help='nome do arquivo FITS de saida', type=str)
    parser.add_argument('--vectors', dest='cbv_list', help='lista de CBVs para utilizar', type=str)
    args = parser.parse_args()
    tendencia(win, args.arq, args.novo_arq, args.cbv_list)
    app.exec_()
