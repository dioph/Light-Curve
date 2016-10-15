from astropy.io import fits
import numpy as np, pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from numpy.linalg import inv as inverse
import lc

# retorna CBVs do cbvdata de acordo com cbv_list
def get_vectors(cbvdata, cbv_list, caddata):
    vectors = np.zeros((len(cbv_list),len(caddata)))
    for i in range(len(cbv_list)):
        j = int(cbv_list[i])
        dat = cbvdata.field('VECTOR_%s' % j)[np.isnan(cbvdata.field('cadenceno'))==False]
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
    coeffs = np.array(0.0-coeffs)
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
    hdulist[1] = fits.new_table(cols, header=hdulist[1].header)
    hdulist.writeto(novo_arq)

# remove erros sistematicos utilizando CBVs e salva um novo FITS
class tendencia(pg.GraphicsWindow):
    def __init__(self, arq, novo_arq, cbv_list, mask=None):
        super(tendencia, self).__init__()
        ### ler arquivo de entrada
        hdulist = lc.abrir(arq, 'readonly')
        cad = 1625.35
        try:
            test = str(hdulist[0].header['filever'])
            version = 2
        except KeyError:
            version = 1
        tbl = lc.extrair(hdulist[1])
        mod = str(hdulist[0].header['module'])
        out = str(hdulist[0].header['output'])
        if version == 1:
            if str(hdulist[1].header['datatype']) == 'long cadence':
                quarter = str(hdulist[1].header['quarter'])
                caddata = tbl.field('cadence_number')
                timedata = tbl.field('barytime')
                fluxdata = tbl.field('ap_raw_flux') / cad
            elif str(hdulist[1].header['datatype']) == 'short cadence':
                lc.error('Tendencia nao implementada para cadencias curtas')
        elif version == 2:
            if str(hdulist[0].header['obsmode']) == 'long cadence':
                quarter = str(hdulist[0].header['quarter'])
                caddata = tbl.field('cadenceno')
                timedata = tbl.field('time')
                fluxdata = tbl.field('sap_flux')
            elif str(hdulist[0].header['obsmode']) == 'short cadence':
                lc.error('Tendencia nao implementada para cadencias curtas')
        ### arquivo com CBVs para o quarter correto
        cbvfile = 'quarter'+quarter+'.fits'
        separator = cbv_list[1]
        ### remover infinitos e colunas de fluxo zero
        good_data = np.logical_and(np.logical_and(np.isfinite(timedata), np.isfinite(fluxdata)), fluxdata != 0.0)
        caddata = caddata[good_data]
        timedata = timedata[good_data]
        fluxdata = fluxdata[good_data]
        ### lista de CBVs para utilizar
        cbv_list = np.fromstring(cbv_list, dtype='int', sep=separator)
        cbvdata = fits.open(cbvfile)
        cbvdata = cbvdata['MODOUT_%s_%s' % (mod, out)].data
        vectors = get_vectors(cbvdata, cbv_list, caddata)
        ### minimos quadrados
        medflux = np.median(fluxdata)
        flux_array = (fluxdata/medflux)-1
        coeffs = llsq(vectors, flux_array)
        flux_array = medflux * (adequar(flux_array, vectors, coeffs)+1)
        soma = cbvsum(vectors, coeffs)
        medflux = np.median(flux_array+1)
        soma_normal = medflux * (1 - soma)
        ### plotar resultado e salvar FITS
        self.plotfit(timedata, fluxdata, flux_array, soma_normal, cad, version)
        savefit(hdulist, novo_arq, flux_array, soma, version)
        lc.fechar(hdulist)

    # plota aproximacao dos CBVs e fluxo apos a remocao destes
    def plotfit(self, timedata, fluxdata1, fluxdata2, soma, cad, version):
        
        self.setWindowTitle('fit')
        self.resize(1500,800)
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('leftButtonPan', False)

        if version == 1:
            timeshift = float(int(timedata[0]/100) * 100.0)
            timedata -= timeshift
            xlabel = 'BJD - %d' % (timeshift + 2400000.0)
        elif version == 2:
            timeshift = float(int((timedata[0]+54833.0)/100) * 100.0)
            timedata += 54833.0 - timeshift
            xlabel = 'BJD - %d' % (timeshift + 2400000.0)

        try:
            exp = len(str(int(np.nanmax(fluxdata1))))-1
        except:
            exp = 0
        fluxdata1 /= 10**exp
        soma /= 10**exp
        ylabel1 = '10<sup>%d</sup> e<sup>-</sup> s<sup>-1</sup>' % exp
        try:
            exp = len(str(int(fluxdata2.max())))-1
        except:
            exp = 0
        fluxdata2 /= 10**exp
        ylabel2 = '10<sup>%d</sup> e<sup>-</sup> s<sup>-1</sup>' % exp

        pw = self.addPlot()
        sub = np.array([], dtype='int32')
        deltamax = 2.0 * cad / 86400
        for i in range(1, len(fluxdata1)):
            dt = timedata[i] - timedata[i-1]
            if dt < deltamax:
                sub = np.append(sub, 1)
            else:
                sub = np.append(sub, 0)
        sub = np.append(sub, 1)
        pw.plot(x=timedata, y=fluxdata1, pen='b', connect=sub)
        pw.plot(x=timedata, y=soma, pen='r', connect=sub)
        pw.showAxis('bottom', False)
        pw.setLabel('left', 'flux', units=ylabel1)

        self.nextRow()

        pw = self.addPlot()
        sub = np.array([], dtype='int32')
        for i in range(1, len(fluxdata2)):
            dt = timedata[i] - timedata[i-1]
            if dt < deltamax:
                sub = np.append(sub, 1)
            else:
                sub = np.append(sub, 0)
        sub = np.append(sub, 1)
        pw.plot(x=timedata, y=fluxdata2, pen='b', connect=sub)
        pw.setLabel('bottom', 'time', units=xlabel)
        pw.setLabel('left', 'flux', units=ylabel2)
