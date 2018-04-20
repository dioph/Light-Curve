import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtGui
import lc


# imprime mensagens de erro
def error(msg):
    print('ERROR - ' + msg)


# le palavras-chave WCS (World Coordinate System)
def getwcs(hdu):
    try:
        pix1 = lc.getrow(hdu, 'CRPIX1P')
    except:
        pix1 = 0.0
    try:
        pix2 = lc.getrow(hdu, 'CRPIX2P')
    except:
        pix2 = 0.0
    try:
        val1 = lc.getrow(hdu, 'CRVAL1P')
    except:
        val1 = 0.0
    try:
        val2 = lc.getrow(hdu, 'CRVAL2P')
    except:
        val2 = 0.0
    try:
        delta1 = lc.getrow(hdu, 'CDELT1P')
    except:
        delta1 = 0.0
    try:
        delta2 = lc.getrow(hdu, 'CDELT2P')
    except:
        delta2 = 0.0
    return pix1, pix2, val1, val2, delta1, delta2


# calcula coordenadas a partir de WCS
def wcs(i, pix, val, delta):
    return val + (float(i + 1) - pix) * delta


# le TPF (Target Pixel File)
def lerTPF(tpf, nome):
    try:
        id = str(tpf[0].header['KEPLERID'])
    except:
        id = ''
    try:
        qt = str(tpf[0].header['QUARTER'])
    except:
        qt = ''
    try:
        season = str(tpf[0].header['SEASON'])
    except:
        season = ''
    try:
        ra = str(tpf[0].header['RA_OBJ'])
    except:
        ra = ''
    try:
        dec = str(tpf[0].header['DEC_OBJ'])
    except:
        dec = ''
    try:
        mag = str(float(tpf[0].header['KEPMAG']))
    except:
        mag = ''
    try:
        dim = tpf[1].header['TDIM5']
        xdim = int(dim.strip('(').strip(')').split(',')[0])
        ydim = int(dim.strip('(').strip(')').split(',')[1])
    except:
        error('Nao foi possivel ler TDIM5')
        xdim = 0
        ydim = 0
    try:
        col = tpf[1].header['1CRV5P']
    except:
        error('Nao foi possivel ler 1CRV5P')
        col = 0
    try:
        row = tpf[1].header['2CRV5P']
    except:
        error('Nao foi possivel ler 2CRV5P')
        row = 0
    try:
        data = tpf[1].data.field(nome)[:]
    except:
        error('Nao foi possivel ler ' + nome.upper())
        data = None
    if len(np.shape(data)) == 3:
        data = np.reshape(data, (np.shape(data)[0], np.shape(data)[1] * np.shape(data)[2]))
    return id, qt, season, ra, dec, mag, xdim, ydim, col, row, data


# le mask de pixels do TPF
def ler_mask(tpf):
    img = tpf['APERTURE'].data
    try:
        size1 = tpf['APERTURE'].header['NAXIS1']
    except:
        error('Nao foi possivel ler NAXIS1')
    try:
        size2 = tpf['APERTURE'].header['NAXIS2']
    except:
        error('Nao foi possivel ler NAXIS2')
    pix1, pix2, val1, val2, delta1, delta2 = getwcs(tpf['APERTURE'])
    coord1 = np.zeros((size1, size2))
    coord2 = np.zeros((size1, size2))
    for j in range(size2):
        for i in range(size1):
            coord1[i, j] = wcs(i, pix1, val1, delta1)
            coord2[i, j] = wcs(j, pix2, val2, delta2)
    lc.fechar(tpf)
    return img, coord1, coord2


# plota fluxo vs tempo para pixels individuais do alvo
def plotpixel(win, arq):
    ### abrir TPF
    tpf = lc.abrir(arq, 'readonly')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, timedata = lerTPF(tpf, 'TIME')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, qlty = lerTPF(tpf, 'QUALITY')
    id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxpix = lerTPF(tpf, 'FLUX')
    ### ler mask do TPF
    img, coord1, coord2 = ler_mask(tpf)
    lc.fechar(tpf)
    ### remover linhas de lixo
    new_size = 0
    num_rows = len(fluxpix)
    for i in range(num_rows):
        if qlty[i] == 0 and np.isfinite(timedata[i]) and np.isfinite(fluxpix[i, int(ydim * xdim / 2)]):
            new_size += 1
    time_array = np.empty((new_size))
    qlty_array = np.empty((new_size))
    fpix_array = np.empty((ydim, xdim, new_size))
    ### construir curvas de luz
    for i in range(ydim):
        for j in range(xdim):
            new_size = 0
            for k in range(num_rows):
                if qlty[k] == 0 and np.isfinite(timedata[i]) and np.isfinite(fluxpix[i, int(ydim * xdim / 2)]):
                    time_array[new_size] = timedata[k]
                    qlty_array[new_size] = qlty[k]
                    fpix_array[i, j, new_size] = fluxpix[k, i * xdim + j]
                    new_size += 1
    ### plotar array de pixels
    p = [[pg.PlotDataItem() for j in range(xdim)] for i in range(ydim)]
    pw = [[pg.PlotItem() for j in range(xdim)] for i in range(ydim)]
    win.addLabel('fluxo arbitrario', angle=-90, rowspan=ydim)
    for i in range(ydim - 1, -1, -1):
        win.addLabel(str(int(coord2[0, i])), angle=-90, rowspan=1)
        for j in range(xdim):
            pw[i][j] = win.addPlot()
            if img[i, j] >= 2:
                p[i][j] = pw[i][j].plot(x=time_array, y=fpix_array[i, j, :], pen='g', background='g', clickable=True)
            elif img[i, j] != 0:
                p[i][j] = pw[i][j].plot(x=time_array, y=fpix_array[i, j, :], pen='b', background='w', clickable=True)
            pw[i][j].showAxis('left', False)
            pw[i][j].showAxis('bottom', False)
        win.nextRow()
    for j in range(xdim):
        win.addLabel(str(int(coord1[j, 0])), angle=0, col=j + 2, colspan=1)
    win.nextRow()
    win.addLabel('tempo', angle=0, col=2, colspan=xdim)
    return xdim, ydim, p


"""
# limites de intensidade de array 1D em uma dada escala ['linear'|'log'|'sqrt']
def rescale(imagem, escala):
	array = []
	prov = np.array(np.sort(imagem), dtype='float32')
	for i in range(len(prov)):
		if 'nan' not in str(prov[i]).lower():
			array.append(prov[i])
	array = np.array(array, dtype='float32')
	n = max(int(float(len(array)) / 10 + 0.5), 2)
	zmin = np.median(array[:n])
	zmax = np.median(array[-n:])
	if escala == 'log':
		if zmin < 0.0:
			zmin = 100.0
		imagem = np.log10(imagem)
		zmin = np.log10(zmin)
		zmax = np.log10(zmax)
	elif escala == 'sqrt':
		if zmin < 0.0:
			zmin = 100.0
		imagem = np.sqrt(imagem)
		zmin = np.sqrt(zmin)
		zmax = np.sqrt(zmax)
	return imagem, zmin, zmax

# define novas aberturas otimas para o alvo
def plotmask(arq, linha=2177, escala='linear', imin=False, imax=False, colormap='bone'):
	global id, qt, season, ra, dec, mag, xdim, ydim, col, row
	global cmap, mask, zmin, zmax, img, xmin, xmax, ymin, ymax
	zmin = imin
	zmax = imax
	cmap = colormap
	mask = []

	tpf = lc.abrir(arq, 'readonly')
	size = tpf[1].header['NAXIS2']
	
	id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxpix = lerTPF(tpf, 'FLUX')
	lc.fechar(tpf)
	
	img = fluxpix[linha]
	#img = fluxpix
	ymin = np.copy(row)
	ymax = ymin + ydim
	xmin = np.copy(col)
	xmax = xmin + xdim

	img, imin, imax = rescale(img, escala)
	if zmin and zmax:
		if escala == 'log':
			zmin = np.log10(zmin)
			zmax = np.log10(zmax)
		elif escala == 'sqrt':
			zmin = np.sqrt(zmin)
			zmax = np.sqrt(zmax)
	else:
		zmin = np.copy(imin)
		zmax = np.copy(imax)

	ymin = float(ymin) - 0.5
	ymax = float(ymax) - 0.5
	xmin = float(xmin) - 0.5
	xmax = float(xmax) - 0.5

	plt.style.use(astropy_mpl_style)
	plt.figure(figsize=[10,7])
	plotimage()

# plot interativo da imagem atual
def plotimage():
	global cid1, cid2, cid3
	### caixa de texto
	#plt.ion()
	'''
	'''app = QtGui.QApplication([])
	pg.setConfigOptions(imageAxisOrder='row-major')
	win = QtGui.QMainWindow()
	win.resize(800,800)
	imv = pg.ImageView()
	#win.setCentralWidget(imv)
	win.show()
	win.setWindowTitle('plotmask')
	print(img.shape)
	new_img = np.empty((img.shape[0], ydim, xdim))
	for w in range(img.shape[0]):
		k = 0
		for i in range(ydim):
			for j in range(xdim):
				new_img[w,(ydim-1)-i,j] = img[w,k]
				k += 1
	imv.setImage(new_img)
	font = QtGui.QFont('SansSerif', 12)
	win.'''
	'''
	plt.clf()
	plt.axes([0.73, 0.09, 0.25, 0.32])
	plt.text(0.1, 0.8, 'ID: '+id, fontsize=12)
	plt.text(0.1, 0.7, 'RA (J2000): '+ra, fontsize=12)
	plt.text(0.1, 0.6, 'Dec (J2000): '+dec, fontsize=12)
	plt.text(0.1, 0.5, 'Magnitude: '+mag, fontsize=12)
	plt.text(0.1, 0.4, 'Season: '+season, fontsize=12)
	plt.text(0.1, 0.3, 'Quarter: '+qt, fontsize=12)
	plt.text(0.1, 0.2, 'Coluna: '+str(col), fontsize=12)
	plt.text(0.1, 0.1, 'Linha: '+str(row), fontsize=12)
	plt.setp(plt.gca(), xticklabels=[], xticks=[], yticklabels=[], yticks=[])
	plt.xlim(0.0, 1.0)
	plt.ylim(0.05, 0.9)
	### botao LIMPAR
	plt.axes([0.73, 0.86, 0.25, 0.11])
	plt.text(0.5, 0.5, 'LIMPAR', fontsize=24, weight='heavy', verticalalignment='center', horizontalalignment='center')
	plt.setp(plt.gca(), xticklabels=[], xticks=[], yticklabels=[], yticks=[])
	plt.fill([0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0], '#ffffee')
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	cid1 = plt.connect('button_press_event', botao1)
	### botao SALVAR
	plt.axes([0.73, 0.62, 0.25, 0.11])
	plt.text(0.5, 0.5, 'SALVAR', fontsize=24, weight='heavy', verticalalignment='center', horizontalalignment='center')
	plt.setp(plt.gca(), xticklabels=[], xticks=[], yticklabels=[], yticks=[])
	plt.fill([0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0], '#ffffee')
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	cid2 = plt.connect('button_press_event', botao2)
	### plotando imagem
	plt.axes([0.07, 0.09, 0.63, 0.88])
	plt.ticklabel_format(useOffset=False)
	plt.subplots_adjust(0.06, 0.1, 0.93, 0.88)
	new_img = np.empty((ydim, xdim))
	k = 0
	for i in range(ydim):
		for j in range(xdim):
			new_img[i,j] = img[k]
			k += 1
	plt.imshow(new_img, aspect='auto', interpolation='nearest', origin='lower', extent=(xmin,xmax,ymin,ymax), cmap=cmap, vmin=zmin, vmax=zmax)
	plt.gca().set_autoscale_on(False)
	plt.xlabel('Coluna')
	plt.ylabel('Linha')
	### escolher pixels
	for pixel in mask:
		m = int(pixel.split(',')[0])
		n = int(pixel.split(',')[1])
		x = [m-0.5, m+0.5, m+0.5, m-0.5, m-0.5]
		y = [n-0.5, n-0.5, n+0.5, n+0.5, n-0.5]
		plt.fill(x, y, 'g', alpha=0.5, ec='g')
	cid3 = plt.connect('button_press_event', botao3)
	plt.show()

# evento do botao LIMPAR
def botao1(event):
	global mask, cid1, cid2, cid3
	if event.inaxes:
		if event.button == 1:
			if 601 < event.x < 801 and 492 < event.y < 522:
				plt.disconnect(cid1)
				plt.disconnect(cid2)
				plt.disconnect(cid3)
				mask = []
				plt.clf()
				plotimage()

# evento do botao SALVAR
def botao2(event):
	if event.inaxes:
		if event.button == 1:
			if 601 < event.x < 801 and 354 < event.y < 415:
				txt = 'NEW|'
				txt += str(id)+'|'
				txt += str(int(row))+'|'+str(int(col))+'|'
				for coord in sorted(set(mask)):
					txt += str(int(coord.split(',')[1])-row) + ','
					txt += str(int(coord.split(',')[0])-col) + ';'
				saida = input('Nome da mask de saida .txt: ')
				arq = open(saida, 'a')
				arq.write(txt[:-1]+'\n')
				arq.close()

# evento de escolher pixels
def botao3(event):
	global mask, cid3
	if event.inaxes:
		if event.button == 1:
			if 75 < event.x < 580 and 53 < event.y < 550:
				m = int(event.xdata + 0.5)
				n = int(event.ydata + 0.5)
				txt = str(m)+','+str(n)
				if txt in mask:
					prov = []
					for pixel in mask:
						if pixel != txt:
							prov.append(pixel)
					mask = prov
				else:
					mask.append(txt)
				plotimage()

# junta varios TPFs em um unico FITS
def stitch(arqs, novo_arq):
	pass
"""

if __name__ == "__main__":
    import argparse

    app = QtGui.QApplication([])
    font = QtGui.QFont('Century Gothic', 18)
    app.setFont(font)
    win = pg.GraphicsWindow()
    parser = argparse.ArgumentParser(description='plota fluxo vs tempo para pixels individuais do alvo')
    parser.add_argument('arq', help='nome do arquivo FITS de origem (TPF)', type=str)
    args = parser.parse_args()
    plotpixel(win, args.arq)
    app.exec_()
