from astropy.io import fits
import numpy as np
import scipy.optimize as sci
import pyqtgraph as pg
from pyqtgraph import QtGui
import tpf, func


# imprime mensagens de erro
def error(msg):
    print('ERROR - ' + msg)


# abre arquivo FITS e retorna a HDUlist
def abrir(arq, modo=None):
    try:
        hdulist = fits.open(arq, modo)
    except:
        try:
            hdulist = arq.open()
        except:
            hdulist = None
            error('Erro lendo FITS: ' + str(arq))
    return hdulist


# fecha arquivo FITS
def fechar(hdulist):
    try:
        hdulist.close()
    except:
        error('Nao foi possivel fechar FITS')

# retorna coluna de uma tabela de dados
def getcol(tbl, coluna):
    try:
        col = tbl.field(coluna)
    except:
        error('Nao foi possivel encontrar a coluna ' + coluna.upper())
        col = None
    return col


# retorna coluna 'TIME' de uma tabela de dados
def timecol(tbl):
    try:
        col = tbl.field('TIME')
    except:
        error('Nao foi possivel encontrar a coluna TIME')
        col = None
    return col


# retorna coluna 'SAP_FLUX' ou 'AP_RAW_FLUX' de uma tabela de dados
def fluxcol(tbl):
    try:
        col = tbl.field('SAP_FLUX')
    except:
        try:
            col = tbl.field('AP_RAW_FLUX')
        except:
            error('Nao foi possivel encontrar a coluna SAP_FLUX')
            col = None
    return col


# insere uma palavra-chave na HDU
def newrow(nome, valor, comentario, hdu):
    try:
        hdu.header[nome.upper()] = (valor, comentario)
    except:
        error('Nao foi possivel criar ' + nome.upper())


# retorna valor de uma palavra-chave de uma HDU
def getrow(hdu, nome):
    try:
        val = hdu.header[nome]
    except:
        error('Nao foi possivel ler ' + nome.upper())
        val = None
    return val


# le palavras-chave de tempo da HDUlist
def time_info(hdulist):
    inicio = 0.0
    fim = 0.0
    cad = 0.0
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
        inicio = hdulist[1].header['TSTART']
    except:
        try:
            inicio = hdulist[1].header['STARTBJD'] + 2.4e6
        except:
            try:
                inicio = hdulist[0].header['LC_START'] + 2400000.5
            except:
                error('Nao foi possivel encontrar TSTART')
    inicio += bjdref
    ### TSTOP
    try:
        fim = hdulist[1].header['TSTOP']
    except:
        try:
            fim = hdulist[1].header['ENDBJD'] + 2.4e6
        except:
            try:
                fim = hdulist[0].header['LC_END'] + 2400000.5
            except:
                error('Nao foi possivel encontrar TSTOP')
    fim += bjdref
    ### OBSMODE
    cad = 1.0
    try:
        modo = hdulist[0].header['OBSMODE']
    except:
        try:
            modo = hdulist[1].header['DATATYPE']
        except:
            modo = ''
            error('Nao foi possivel encontrar OBSMODE')
    if 'short' in modo:
        cad = 54.178
    elif 'long' in modo:
        cad = 1625.35

    return inicio, fim, bjdref, cad


# plot customizavel
def plot(win, row=0, col=0, rowspan=1, colspan=1, info=None, xcol=None, ycol=None, qcol=None, arq=None,
         yname='sap_flux', quality=True, ylabel='e<sup>-</sup> s<sup>-1</sup>', plslr=False, plsline=False):
    """"
    plot customizavel:

    arq:     arquivo FITS
    yname:   nome da coluna com dados de fluxo
    quality: True se usuario quiser ignorar cadencias onde a qualidade dos dados é questionavel, False caso contrario
    ylabel:  label do eixo y; default = 'e<sup>-</sup> s<sup>-1</sup>'
    """
    ### ler colunas de entrada
    if arq is not None:
        hdulist = abrir(arq, 'readonly')
        inicio, fim, bjdref, cad = time_info(hdulist)
        tbl = hdulist[1].data
        xcol = timecol(tbl)
        xcol += bjdref
        ycol = getcol(tbl, yname)
        qcol = getcol(tbl, 'SAP_QUALITY')
        fechar(hdulist)
    else:
        inicio, fim, bjdref, cad = info
        xcol += bjdref
    ### remover lixo dos dados
    array = np.array(list(zip(xcol, ycol, qcol)))
    array = array[~np.isnan(array).any(1)]
    array = array[~np.isinf(array).any(1)]
    if quality:
        array = array[array[:, 2] == 0.0]
    timedata = np.array(array[:, 0], dtype='float64')
    fluxdata = np.array(array[:, 1], dtype='float32')
    if len(timedata) == 0:
        error('Arrays estao cheios de lixo')
    ### organizar os eixos
    timeshift = float(int(inicio / 100) * 100.0)
    timedata -= timeshift
    xlabel = 'BJD - %d' % timeshift

    try:
        exp = len(str(int(np.nanmax(fluxdata)))) - 1
    except:
        exp = 0
    fluxdata /= (10 ** exp)
    if 'e<sup>-</sup> s<sup>-1</sup>' in ylabel or 'default' in ylabel:
        if exp == 0:
            ylabel = 'e<sup>-</sup> s<sup>-1</sup>'
        else:
            ylabel = '10<sup>%d</sup> e<sup>-</sup> s<sup>-1</sup>' % exp
    ### plot
    diffs = np.array([], dtype=[('x', float), ('y', int)])
    size = 0.0
    pw = win.addPlot(title=yname.upper(), row=row, col=col, rowspan=rowspan, colspan=colspan)
    sub = np.array([], dtype='int32')
    deltamax = 2.0 * cad / 86400
    for i in range(1, len(fluxdata)):
        dt = timedata[i] - timedata[i - 1]
        if dt < deltamax:
            sub = np.append(sub, 1)
            newlistx = np.append(diffs['x'], fluxdata[i - 1] - fluxdata[i])
            newlisty = np.append(diffs['y'], i)
            diffs = np.array([(newlistx[j], newlisty[j]) for j in range(newlistx.size)],
                             dtype=[('x', float), ('y', int)])
            size += 0.01
        else:
            sub = np.append(sub, 0)
    sub = np.append(sub, 1)
    size = int(size) + 1
    diffs.sort(order='x')
    threshold = 3 * diffs[-size]['x']
    diffs = diffs[diffs['x'] > threshold]
    print(threshold, diffs)
    p = pw.plot(x=timedata, y=fluxdata, pen='b', connect=sub)
    for i in diffs['y']:
        if fluxdata[i - 1] > np.nanmedian(fluxdata):
            pw.plot(x=timedata[i - 1:i + 1], y=fluxdata[i - 1:i + 1], pen='y')
    pw.setLabel('bottom', 'tempo', units=xlabel)
    pw.setLabel('left', 'fluxo', units=ylabel)
    lr = pg.LinearRegionItem([timedata[0], timedata[-1]])
    lr.setZValue(-10)
    if plsline:
        rms1 = np.median(fluxdata)+np.sqrt(2*np.mean(np.square(fluxdata-np.median(fluxdata))))
        rms2 = np.median(fluxdata)-np.sqrt(2*np.mean(np.square(fluxdata-np.median(fluxdata))))
        line = pg.InfiniteLine(pos=rms1, angle=0, pen='r')
        pw.addItem(line)
        line = pg.InfiniteLine(pos=rms2, angle=0, pen='r')
        pw.addItem(line)
    if plslr:
        pw.addItem(lr)
    return pw, p, lr, timedata, fluxdata


# cria nova curva de luz aplicando mascara e salva no FITS de saida
def new_curve(arq, mask, novo_arq):
    ### ler arquivo de entrada
    hdulist = abrir(arq, 'readonly')
    inicio, fim, bjdref, cad = time_info(hdulist)
    cards0 = hdulist[0].header.cards
    cards1 = hdulist[1].header.cards
    cards2 = hdulist[2].header.cards
    tbl = hdulist[1].data
    maskdata = np.copy(hdulist[2].data)
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, timedata = tpf.lerTPF(hdulist, 'TIME')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, tcorrdata = tpf.lerTPF(hdulist, 'TIMECORR')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, caddata = tpf.lerTPF(hdulist, 'CADENCENO')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, rawdata = tpf.lerTPF(hdulist, 'RAW_CNTS')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxdata = tpf.lerTPF(hdulist, 'FLUX')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, ferrdata = tpf.lerTPF(hdulist, 'FLUX_ERR')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, fbkgdata = tpf.lerTPF(hdulist, 'FLUX_BKG')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, bkgerrdata = tpf.lerTPF(hdulist, 'FLUX_BKG_ERR')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, qdata = tpf.lerTPF(arq, 'QUALITY')
    timedata = np.array(timedata, dtype='float64')
    tcorrdata = np.array(tcorrdata, dtype='float32')
    caddata = np.array(caddata, dtype='int')
    qdata = np.array(qdata, dtype='int')
    try:
        pos_corr1 = np.array(tbl.field('POS_CORR1'), dtype='float64')
    except:
        pos_corr1 = np.empty(len(timedata))
        pos_corr1[:] = np.nan
    try:
        pos_corr2 = np.array(tbl.field('POS_CORR2'), dtype='float64')
    except:
        pos_corr2 = np.empty(len(timedata))
        pos_corr2[:] = np.nan
    psf_centr1 = np.empty(len(timedata));
    psf_centr1[:] = np.nan
    psf_centr1_err = np.empty(len(timedata));
    psf_centr1_err[:] = np.nan
    psf_centr2 = np.empty(len(timedata));
    psf_centr2[:] = np.nan
    psf_centr2_err = np.empty(len(timedata));
    psf_centr2_err[:] = np.nan
    mom_centr1_err = np.empty(len(timedata));
    mom_centr1_err[:] = np.nan
    mom_centr2_err = np.empty(len(timedata));
    mom_centr2_err[:] = np.nan
    ### ler definicao de mascara
    maskx = np.array([], dtype='int')
    masky = np.array([], dtype='int')
    if mask.lower() != 'all':
        lines = open(mask, 'r')
        for line in lines:
            line = line.split('|')
            if len(line) == 5:
                y0 = int(line[2])
                x0 = int(line[3])
                line = line[4].split(';')
                for items in line:
                    try:
                        masky = np.append(masky, y0 + int(items.split(',')[0]))
                        maskx = np.append(maskx, x0 + int(items.split(',')[1]))
                    except:
                        continue
        lines.close()
        if len(maskx) == 0 or len(masky) == 0:
            error(mask + ' nao possui pixels')
    ### definir novo bitmap para subimagem
    pix1, pix2, val1, val2, delta1, delta2 = tpf.getwcs(hdulist[2])
    if mask.lower() != 'all':
        aperx = np.array([], dtype='int')
        apery = np.array([], dtype='int')
        aperq = np.array([], dtype='int')
        for i in range(maskdata.shape[0]):
            for j in range(maskdata.shape[1]):
                aperx = np.append(aperx, tpf.wcs(j, pix1, val1, delta1))
                apery = np.append(apery, tpf.wcs(i, pix2, val2, delta2))
                if maskdata[i, j] == 0:
                    aperq = np.append(aperq, 0)
                else:
                    aperq = np.append(aperq, 1)
                    maskdata[i, j] = 1
                    for k in range(len(maskx)):
                        if aperx[-1] == maskx[k] and apery[-1] == masky[k]:
                            aperq[-1] = 3
                            maskdata[i, j] = 3
    if mask.lower() == 'all':
        aperx = np.array([], dtype='int')
        apery = np.array([], dtype='int')
        aperq = np.array([], dtype='int')
        for i in range(maskdata.shape[0]):
            for j in range(maskdata.shape[1]):
                aperx = np.append(aperx, tpf.wcs(j, pix1, val1, delta1))
                apery = np.append(apery, tpf.wcs(i, pix2, val2, delta2))
                if maskdata[i, j] == 0:
                    aperq = np.append(aperq, 0)
                else:
                    aperq = np.append(aperq, 3)
                    maskdata[i, j] = 3
    if len(aperq) == 0:
        error('Nenhum pixel valido definido')
    ### construir nova tabela de fluxo
    num_aper = (aperq == 3).sum()
    num_time = len(timedata)
    sap_flux = np.array([], dtype='float32')
    sap_flux_err = np.array([], dtype='float32')
    sap_bkg = np.array([], dtype='float32')
    sap_bkg_err = np.array([], dtype='float32')
    raw_flux = np.array([], dtype='float32')
    for i in range(num_time):
        prov1 = np.array([], dtype='float64')
        prov2 = np.array([], dtype='float64')
        prov3 = np.array([], dtype='float64')
        prov4 = np.array([], dtype='float64')
        prov5 = np.array([], dtype='float64')
        for j in range(len(aperq)):
            if aperq[j] == 3:
                prov1 = np.append(prov1, fluxdata[i, j])
                prov2 = np.append(prov2, ferrdata[i, j])
                prov3 = np.append(prov3, fbkgdata[i, j])
                prov4 = np.append(prov4, bkgerrdata[i, j])
                prov5 = np.append(prov5, rawdata[i, j])
        sap_flux = np.append(sap_flux, prov1.sum())
        sap_flux_err = np.append(sap_flux_err, func.sumerr(prov2))
        sap_bkg = np.append(sap_bkg, prov3.sum())
        sap_bkg_err = np.append(sap_bkg_err, func.sumerr(prov4))
        raw_flux = np.append(raw_flux, prov5.sum())
    ### construir nova tabela de momento
    mom_centr1 = np.zeros(shape=(num_time))
    mom_centr2 = np.zeros(shape=(num_time))
    mom_centr1_err = np.zeros(shape=(num_time))
    mom_centr2_err = np.zeros(shape=(num_time))
    for i in range(num_time):
        xf = np.zeros(shape=(num_aper))
        yf = np.zeros(shape=(num_aper))
        f = np.zeros(shape=(num_aper))
        xfe = np.zeros(shape=(num_aper))
        yfe = np.zeros(shape=(num_aper))
        fe = np.zeros(shape=(num_aper))
        k = 0
        for j in range(len(aperq)):
            if aperq[j] == 3:
                xf[k] = aperx[j] * fluxdata[i, j]
                xfe[k] = aperx[j] * ferrdata[i, j]
                yf[k] = apery[j] * fluxdata[i, j]
                yfe[k] = apery[j] * ferrdata[i, j]
                f[k] = fluxdata[i, j]
                fe[k] = ferrdata[i, j]
                k += 1
        xfsum = xf.sum()
        yfsum = yf.sum()
        fsum = f.sum()
        xfsume = np.sqrt(np.square(xfe).sum() / num_aper)
        yfsume = np.sqrt(np.square(yfe).sum() / num_aper)
        fsume = np.sqrt(np.square(fe).sum() / num_aper)
        mom_centr1[i] = xfsum / fsum
        mom_centr2[i] = yfsum / fsum
        mom_centr1_err[i] = np.sqrt((xfsume / xfsum) ** 2 + (fsume / fsum) ** 2)
        mom_centr2_err[i] = np.sqrt((yfsume / yfsum) ** 2 + (fsume / fsum) ** 2)
    mom_centr1_err *= mom_centr1
    mom_centr2_err *= mom_centr2
    ### construir nova tabela de PSF
    psf_centr1 = np.zeros(shape=(num_time))
    psf_centr2 = np.zeros(shape=(num_time))
    psf_centr1_err = np.zeros(shape=(num_time))
    psf_centr2_err = np.zeros(shape=(num_time))
    modx = np.zeros(shape=(num_aper))
    mody = np.zeros(shape=(num_aper))
    k = 0
    for j in range(len(aperq)):
        if aperq[j] == 3:
            modx[k] = aperx[j]
            mody[k] = apery[j]
            k += 1
    for i in range(num_time):
        modf = np.zeros(shape=(num_aper))
        k = 0
        tentativa = [mom_centr1[i], mom_centr2[i], np.nanmax(fluxdata[i:]), 1.0, 1.0, 0.0, 0.0]
        for j in range(len(aperq)):
            if aperq[j] == 3:
                modf[k] = fluxdata[i, j]
                k += 1
        args = (modx, mody, modf)
        resp = sci.leastsq(func.gaussiana, tentativa, args=args, xtol=1.0e-8, ftol=1.0e-4, full_output=True)
        s_sq = (resp[2]['fvec'] ** 2).sum() / (num_time - len(tentativa))
        psf_centr1[i] = resp[0][0]
        psf_centr2[i] = resp[0][1]
        try:
            psf_centr1_err[i] = np.sqrt(np.diag(resp[1] * s_sq))[0]
        except:
            psf_centr1_err[i] = np.nan
        try:
            psf_centr2_err[i] = np.sqrt(np.diag(resp[1] * s_sq))[1]
        except:
            psf_centr2_err[i] = np.nan
    ### construir extensao primaria de saida
    hdu0 = fits.PrimaryHDU()
    for i in range(len(cards0)):
        if cards0[i].keyword not in hdu0.header.keys():
            newrow(cards0[i].keyword, cards0[i].value, cards0[i].comment, hdu0)
        else:
            hdu0.header.cards[cards0[i].keyword].comment = cards0[i].comment
    saida = fits.HDUList(hdu0)
    ### construir extensao de curva de luz de saida
    col1 = fits.Column(name='TIME', format='D', unit='BJD - 2454833', array=timedata)
    col2 = fits.Column(name='TIMECORR', format='E', unit='d', array=tcorrdata)
    col3 = fits.Column(name='CADENCENO', format='J', array=caddata)
    col4 = fits.Column(name='SAP_FLUX', format='E', array=sap_flux)
    col5 = fits.Column(name='SAP_FLUX_ERR', format='E', array=sap_flux_err)
    col6 = fits.Column(name='SAP_BKG', format='E', array=sap_bkg)
    col7 = fits.Column(name='SAP_BKG_ERR', format='E', array=sap_bkg_err)
    col8 = fits.Column(name='PDCSAP_FLUX', format='E', array=sap_flux)
    col9 = fits.Column(name='PDCSAP_FLUX_ERR', format='E', array=sap_flux_err)
    col10 = fits.Column(name='SAP_QUALITY', format='J', array=qdata)
    col11 = fits.Column(name='PSF_CENTR1', format='E', unit='pixel', array=psf_centr1)
    col12 = fits.Column(name='PSF_CENTR1_ERR', format='E', unit='pixel', array=psf_centr1_err)
    col13 = fits.Column(name='PSF_CENTR2', format='E', unit='pixel', array=psf_centr2)
    col14 = fits.Column(name='PSF_CENTR2_ERR', format='E', unit='pixel', array=psf_centr2_err)
    col15 = fits.Column(name='MOM_CENTR1', format='E', unit='pixel', array=mom_centr1)
    col16 = fits.Column(name='MOM_CENTR1_ERR', format='E', unit='pixel', array=mom_centr1_err)
    col17 = fits.Column(name='MOM_CENTR2', format='E', unit='pixel', array=mom_centr2)
    col18 = fits.Column(name='MOM_CENTR2_ERR', format='E', unit='pixel', array=mom_centr2_err)
    col19 = fits.Column(name='POS_CORR1', format='E', unit='pixel', array=pos_corr1)
    col20 = fits.Column(name='POS_CORR2', format='E', unit='pixel', array=pos_corr2)
    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, \
                         col11, col12, col13, col14, col15, col16, col17, col18, col19, col20])
    hdu1 = fits.BinTableHDU.from_columns(cols)
    newrow('TTYPE1', 'TIME', 'column title: data time stamps', hdu1)
    newrow('TFORM1', 'D', 'data type: float64', hdu1)
    newrow('TUNIT1', 'BJD - 2454833', 'column units: barycenter corrected JD', hdu1)
    newrow('TDISP1', 'D12.7', 'column display format', hdu1)
    newrow('TTYPE2', 'TIMECORR', 'column title: barycentric-timeslice correction', hdu1)
    newrow('TFORM2', 'E', 'data type: float32', hdu1)
    newrow('TUNIT2', 'd', 'column units: days', hdu1)
    newrow('TTYPE3', 'CADENCENO', 'column title: unique cadence number', hdu1)
    newrow('TFORM3', 'J', 'column format: signed integer32', hdu1)
    newrow('TTYPE4', 'SAP_FLUX', 'column title: aperture photometry flux', hdu1)
    newrow('TFORM4', 'E', 'column format: float32', hdu1)
    newrow('TUNIT4', 'e-/s', 'column units: electrons per second', hdu1)
    newrow('TTYPE5', 'SAP_FLUX_ERR', 'column title: aperture phot. flux error', hdu1)
    newrow('TFORM5', 'E', 'column format: float32', hdu1)
    newrow('TUNIT5', 'e-/s', 'column units: electrons per second (1-sigma)', hdu1)
    newrow('TTYPE6', 'SAP_BKG', 'column title: aperture phot. background flux', hdu1)
    newrow('TFORM6', 'E', 'column format: float32', hdu1)
    newrow('TUNIT6', 'e-/s', 'column units: electrons per second', hdu1)
    newrow('TTYPE7', 'SAP_BKG_ERR', 'column title: ap. phot. background flux error', hdu1)
    newrow('TFORM7', 'E', 'column format: float32', hdu1)
    newrow('TUNIT7', 'e-/s', 'column units: electrons per second (1-sigma)', hdu1)
    newrow('TTYPE8', 'PDCSAP_FLUX', 'column title: PDC photometry flux', hdu1)
    newrow('TFORM8', 'E', 'column format: float32', hdu1)
    newrow('TUNIT8', 'e-/s', 'column units: electrons per second', hdu1)
    newrow('TTYPE9', 'PDCSAP_FLUX_ERR', 'column title: PDC flux error', hdu1)
    newrow('TFORM9', 'E', 'column format: float32', hdu1)
    newrow('TUNIT9', 'e-/s', 'column units: electrons per second (1-sigma)', hdu1)
    newrow('TTYPE10', 'SAP_QUALITY', 'column title: aperture photometry quality flag', hdu1)
    newrow('TFORM10', 'J', 'column format: signed integer32', hdu1)
    newrow('TTYPE11', 'PSF_CENTR1', 'column title: PSF fitted column centroid', hdu1)
    newrow('TFORM11', 'E', 'column format: float32', hdu1)
    newrow('TUNIT11', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE12', 'PSF_CENTR1_ERR', 'column title: PSF fitted column error', hdu1)
    newrow('TFORM12', 'E', 'column format: float32', hdu1)
    newrow('TUNIT12', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE13', 'PSF_CENTR2', 'column title: PSF fitted row centroid', hdu1)
    newrow('TFORM13', 'E', 'column format: float32', hdu1)
    newrow('TUNIT13', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE14', 'PSF_CENTR2_ERR', 'column title: PSF fitted row error', hdu1)
    newrow('TFORM14', 'E', 'column format: float32', hdu1)
    newrow('TUNIT14', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE15', 'MOM_CENTR1', 'column title: moment-derived column centroid', hdu1)
    newrow('TFORM15', 'E', 'column format: float32', hdu1)
    newrow('TUNIT15', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE16', 'MOM_CENTR1_ERR', 'column title: moment-derived column error', hdu1)
    newrow('TFORM16', 'E', 'column format: float32', hdu1)
    newrow('TUNIT16', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE17', 'MOM_CENTR2', 'column title: moment-derived row centroid', hdu1)
    newrow('TFORM17', 'E', 'column format: float32', hdu1)
    newrow('TUNIT17', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE18', 'MOM_CENTR2_ERR', 'column title: moment-derived row error', hdu1)
    newrow('TFORM18', 'E', 'column format: float32', hdu1)
    newrow('TUNIT18', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE19', 'POS_CORR1', 'column title: col correction for vel. abbern', hdu1)
    newrow('TFORM19', 'E', 'column format: float32', hdu1)
    newrow('TUNIT19', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE20', 'POS_CORR2', 'column title: row correction for vel. abbern', hdu1)
    newrow('TFORM20', 'E', 'column format: float32', hdu1)
    newrow('TUNIT20', 'pixel', 'column units: pixel', hdu1)
    newrow('TTYPE21', 'RAW_FLUX', 'column title: raw aperture photometry flux', hdu1)
    newrow('TFORM21', 'E', 'column format: float32', hdu1)
    newrow('TUNIT21', 'e-/s', 'column units: electrons per second', hdu1)
    newrow('EXTNAME', 'LIGHTCURVE', 'name of extension', hdu1)
    for i in range(len(cards1)):
        if cards1[i].keyword not in hdu1.header.keys() and cards1[i].keyword[:4] not in ['TTYP', 'TFOR', 'TUNI', 'TDIS',
                                                                                         'TDIM', 'WCAX', '1CTY',
                                                                                         '2CTY', '1CRP', '2CRP', '1CRV',
                                                                                         '2CRV', '1CUN', '2CUN',
                                                                                         '1CDE', '2CDE', '1CDL', '2CDL',
                                                                                         '11PC', '12PC', '21PC',
                                                                                         '22PC']:
            newrow(cards1[i].keyword, cards1[i].value, cards1[i].comment, hdu1)
    saida.append(hdu1)
    ### construir extensao de mascara de saida
    hdu2 = fits.ImageHDU(maskdata)
    for i in range(len(cards2)):
        if cards2[i].keyword not in hdu2.header.keys():
            newrow(cards2[i].keyword, cards2[i].value, cards2[i].comment, hdu2)
        else:
            hdu2.header.cards[cards2[i].keyword].comment = cards2[i].comment
    saida.append(hdu2)
    ### escrever arquivo de saida
    saida.writeto(novo_arq, clobber=True)
    fechar(hdulist)
    return novo_arq


def realtime_newcurve(arq, maskx, masky):
    ### ler arquivo de entrada
    hdulist = abrir(arq, 'readonly')
    inicio, fim, bjdref, cad = time_info(hdulist)
    maskdata = np.copy(hdulist[2].data)
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, timedata = tpf.lerTPF(hdulist, 'TIME')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxdata = tpf.lerTPF(hdulist, 'FLUX')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, qdata = tpf.lerTPF(hdulist, 'QUALITY')
    timedata = np.array(timedata, dtype='float64')
    qdata = np.array(qdata, dtype='int')
    ### ler definicao de mascara
    y0 = int(row)
    x0 = int(col)
    for i in range(len(maskx)):
        masky[i] += y0
        maskx[i] += x0
    if len(maskx) == 0 or len(masky) == 0:
        error('a mask nao possui pixels')
    ### definir novo bitmap para subimagem
    pix1, pix2, val1, val2, delta1, delta2 = tpf.getwcs(hdulist[2])

    aperx = np.array([], dtype='int')
    apery = np.array([], dtype='int')
    aperq = np.array([], dtype='int')
    for i in range(maskdata.shape[0]):
        for j in range(maskdata.shape[1]):
            aperx = np.append(aperx, tpf.wcs(j, pix1, val1, delta1))
            apery = np.append(apery, tpf.wcs(i, pix2, val2, delta2))
            if maskdata[i, j] == 0:
                aperq = np.append(aperq, 0)
            else:
                aperq = np.append(aperq, 1)
                maskdata[i, j] = 1
                for k in range(len(maskx)):
                    if aperx[-1] == maskx[k] and apery[-1] == masky[k]:
                        aperq[-1] = 3
                        maskdata[i, j] = 3
    if len(aperq) == 0:
        error('Nenhum pixel valido definido')
    ### construir nova tabela de fluxo
    num_time = len(timedata)
    sap_flux = np.array([], dtype='float32')
    for i in range(num_time):
        prov = np.array([], dtype='float64')
        for j in range(len(aperq)):
            if aperq[j] == 3:
                prov = np.append(prov, fluxdata[i, j])
        sap_flux = np.append(sap_flux, prov.sum())
    for i in range(len(maskx)):
        masky[i] -= y0
        maskx[i] -= x0
    return timedata, sap_flux, qdata, (inicio, fim, bjdref, cad)


# junta varias curvas de luz em um unico FITS
def stitch(arqs, novo_arq):
    lct = []
    bjd = []
    hdu_out = abrir(arqs[0])
    nrows1 = hdu_out[1].data.shape[0]
    head0 = hdu_out[0].header
    head1 = hdu_out[1].header
    nfiles = 0
    for arq in arqs:
        hdu_in = abrir(arq)
        if nfiles > 0:
            nrows2 = hdu_in[1].data.shape[0]
            nrows = nrows1 + nrows2
            tbl = fits.new_table(hdu_out[1].columns, nrows=nrows)
            for name in hdu_out[1].columns.names:
                try:
                    tbl.data.field(name)[nrows1:] = hdu_in[1].data.field(name)
                except:
                    pass
            hdu_out[1] = tbl
            hdu_out[0].header = head0
            hdu_out[1].header = head1
            nrows1 = nrows
            version = 1.0
        ini = getcol(hdu_in[1], 'lc_start')
        fim = getcol(hdu_in[1], 'lc_end')
        try:
            startbjd = hdu_in[1].header['startbjd']
        except:
            startbjd = getcol(hdu_in[1], 'tstart')
            version = 2.0
        try:
            endbjd = hdu_in[1].header['endbjd']
        except:
            endbjd = getcol(hdu_in[1], 'tstop')
            version = 2.0
        lct.append(ini)
        lct.append(fim)
        bjd.append(startbjd)
        bjd.append(endbjd)
        fechar(hdu_in)
        nfiles += 1
    ini = min(lct)
    fim = max(lct)
    startbjd = min(bjd)
    endbjd = max(bjd)
    hdu_out.header.update('lc_start', ini)
    hdu_out.header.update('lc_end', fim)
    if version == 1.0:
        hdu_out.header.update('startbjd', startbjd)
        hdu_out.header.update('endbjd', endbjd)
    if version == 2.0:
        hdu_out.header.update('tstart', startbjd)
        hdu_out.header.update('tstop', endbjd)
    hdu_out.writeto(novo_arq)
    fechar(hdu_out)


if __name__ == "__main__":
    import argparse

    app = QtGui.QApplication([])
    font = QtGui.QFont('Century Gothic', 18)
    app.setFont(font)
    win = pg.GraphicsWindow()
    parser = argparse.ArgumentParser(description='ferramentas para arquivos de curvas de luz .FITS')
    parser.add_argument('arq', help='nome do arquivo FITS de origem (LC)', type=str)
    args = parser.parse_args()
    plot(win, arq=args.arq)
    app.exec_()
