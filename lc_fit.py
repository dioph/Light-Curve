from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import inv as inverse
import lc_io

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
    print(fluxdata[1], soma[1])
    col1 = fits.Column(name='CBVSAP_MODL', format='E13.7', unit=unit, array=soma)
    col2 = fits.Column(name='CBVSAP_FLUX', format='E13.7', unit=unit, array=fluxdata)
    cols = hdulist[1].columns + col1 + col2
    hdulist[1] = fits.new_table(cols, header=hdulist[1].header)
    hdulist.writeto(novo_arq)

# plota aproximacao dos CBVs e fluxo apos a remocao destes
def plotfit(timedata, fluxdata1, fluxdata2, soma, cad, version):
    plt.style.use(astropy_mpl_style)
    plt.figure(figsize=[15,8])
    plt.clf()
    if version == 1:
        timeshift = float(int(timedata[0]/100) * 100.0)
        timedata -= timeshift
        xlabel = 'BJD $-$ %d' % (timeshift + 2400000.0)
    elif version == 2:
        timeshift = float(int((timedata[0]+54833.0)/100) * 100.0)
        timedata += 54833.0 - timeshift
        xlabel = 'BJD $-$ %d' % (timeshift + 2400000.0)
    try:
        exp = len(str(int(np.nanmax(fluxdata1))))-1
    except:
        exp = 0
    fluxdata1 /= 10**exp
    soma /= 10**exp
    ylabel1 = '10$^%d$ e$^-$ s$^{-1}$' % exp
    try:
        exp = len(str(int(fluxdata2.max())))-1
    except:
        exp = 0
    fluxdata2 /= 10**exp
    ylabel2 = '10$^%d$ e$^-$ s$^{-1}$' % exp

    xmin = min(timedata)
    xmax = max(timedata)
    ymin1 = min(min(fluxdata1), min(soma))
    ymax1 = max(max(fluxdata1), max(soma))
    ymin2 = min(fluxdata2)
    ymax2 = max(fluxdata2)
    dx = xmax - xmin
    dy1 = (ymax1 - ymin1) * 0.01
    dy2 = (ymax2 - ymin2) * 0.01
    timedata2 = timedata
    ax1 = plt.subplot(211)

    subtime = np.array([], dtype='float64')
    subflux = np.array([], dtype='float32')
    subsoma = np.array([], dtype='float32')
    deltamax = 2.0 * cad / 86400.0
    for i in range(1, len(fluxdata1)-1):
        dt = timedata[i] - timedata[i-1]
        if dt < deltamax:
            subtime = np.append(subtime, timedata[i])
            subflux = np.append(subflux, fluxdata1[i])
            subsoma = np.append(subsoma, soma[i])
        else:
            plt.plot(subtime, subflux, 'b-', linewidth=1.0)
            plt.plot(subtime, subsoma, 'r-', linewidth=2.0)
            subtime = np.array([], dtype='float64')
            subflux = np.array([], dtype='float32')
            subsoma = np.array([], dtype='float32')
    plt.plot(subtime, subflux, 'b-', linewidth=1.0)
    plt.plot(subtime, subsoma, 'r-', linewidth=2.0)

    timedata = np.insert(timedata, [0], timedata[0])
    timedata = np.append(timedata, timedata[-1])
    fluxdata1 = np.insert(fluxdata1, [0], 0.0)
    fluxdata1 = np.append(fluxdata1, 0.0)
    plt.fill(timedata, fluxdata1, fc='#fffacd')
    plt.xlim(xmin-dx, xmax+dx)
    if ymin1 <= dy1:
        plt.ylim(1.0e-10, ymax1+dy1)
    else:
        plt.ylim(ymin1-dy1, ymax1+dy1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel1)
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)

    subtime = np.array([], dtype='float64')
    subflux = np.array([], dtype='float32')
    for i in range(1, len(fluxdata2)-1):
        dt = timedata2[i] - timedata2[i-1]
        if dt < deltamax:
            subtime = np.append(subtime, timedata2[i])
            subflux = np.append(subflux, fluxdata2[i])
        else:
            plt.plot(subtime, subflux, 'b-', linewidth=1.0)
            subtime = np.array([], dtype='float64')
            subflux = np.array([], dtype='float32')
    plt.plot(subtime, subflux, 'b-', linewidth=1.0)

    fluxdata2 = np.insert(fluxdata2, [0], 0.0)
    fluxdata2 = np.append(fluxdata2, 0.0)
    plt.fill(timedata, fluxdata2, fc='#fffacd')
    if ymin2 <= dy2:
        plt.ylim(1.0e-10, ymax2+dy2)
    else:
        plt.ylim(ymin2-dy2, ymax2+dy2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel2)
    plt.grid()

    plt.subplots_adjust(0.1, 0.1, 0.94, 0.94, 0.0, 0.0)
    plt.ticklabel_format(useOffset=False)
    plt.show()

# remove erros sistematicos utilizando CBVs e salva um novo FITS
def tendencia(arq, novo_arq, cbv_list):
### ler arquivo de entrada
    hdulist = lc_io.abrir(arq, 'readonly')
    cad = 1625.35
    try:
        test = str(hdulist[0].header['filever'])
        version = 2
    except KeyError:
        version = 1
    tbl = lc_io.extrair(hdulist[1])
    mod = str(hdulist[0].header['module'])
    out = str(hdulist[0].header['output'])
    if version == 1:
        if str(hdulist[1].header['datatype']) == 'long cadence':
            quarter = str(hdulist[1].header['quarter'])
            caddata = tbl.field('cadence_number')
            timedata = tbl.field('barytime')
            fluxdata = tbl.field('ap_raw_flux') / cad
        elif str(hdulist[1].header['datatype']) == 'short cadence':
            lc_io.error('Tendencia nao implementada para cadencias curtas')
    elif version == 2:
        if str(hdulist[0].header['obsmode']) == 'long cadence':
            quarter = str(hdulist[0].header['quarter'])
            caddata = tbl.field('cadenceno')
            timedata = tbl.field('time')
            fluxdata = tbl.field('sap_flux')
        elif str(hdulist[0].header['obsmode']) == 'short cadence':
            lc_io.error('Tendencia nao implementada para cadencias curtas')
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
    plotfit(timedata, fluxdata, flux_array, soma_normal, cad, version)
    savefit(hdulist, novo_arq, flux_array, soma, version)
    lc_io.fechar(hdulist)
