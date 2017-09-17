import numpy as np
from astropy.stats import LombScargle
import pyqtgraph as pg
from pyqtgraph import QtGui
import kplr
import lc, tpf, fit
import cleanest

class start(QtGui.QWidget):
    def __init__(self):
        super(start, self).__init__()
        self.init_gui()
        self.con()
        self.client = kplr.API()
        self.arq = ''
        self.win = np.array([], dtype=pg.GraphicsWindow)
        self.k = -1
        self.maskx = []
        self.masky = []

    def init_gui(self):
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)

        self.title = QtGui.QLabel('<b>ANALISE DE CURVAS DE LUZ</b>', alignment=0x0004)
        self.bt_n = QtGui.QPushButton('usar um novo arquivo')
        self.kic = QtGui.QPushButton('buscar online')
        self.txt1 = QtGui.QLineEdit()
        self.txt2 = QtGui.QLineEdit()
        self.sub = QtGui.QPushButton('submit')

        self.layout.addWidget(self.title, 0, 0, 1, 2)
        self.layout.addWidget(self.bt_n, 1, 0)
        self.layout.addWidget(self.kic, 1, 1)
        self.layout.addWidget(self.txt1, 2, 0)
        self.layout.addWidget(self.txt2, 2, 1)
        self.layout.addWidget(self.sub, 2, 2)
        self.setGeometry(10, 10, 1000, 600)
        self.txt1.hide()
        self.txt2.hide()
        self.sub.hide()
        self.show()

    def con(self):
        self.bt_n.clicked.connect(self.click_n)
        self.kic.clicked.connect(self.click_kic)

    def click_n(self):
        self.k += 1
        self.win = np.append(self.win, pg.GraphicsWindow())
        self.win[self.k].hide()
        self.fd = QtGui.QFileDialog()
        self.arq = self.fd.getOpenFileName()[0]
        self.main()

    def click_kic(self):
        self.txt1.show()
        self.txt1.setText('KIC da estrela')
        self.txt2.show()
        self.txt2.setText('Quarter (1-6)')
        self.sub.show()
        self.sub.disconnect()
        self.sub.clicked.connect(self.input_kic)

    def input_kic(self):
        self.k += 1
        self.win = np.append(self.win, pg.GraphicsWindow())
        self.kic_n = int(self.txt1.text())
        q = int(self.txt2.text())
        self.star = self.client.star(self.kic_n)
        print('Teff =', self.star.kic_teff)
        self.tpfs = self.star.get_target_pixel_files(short_cadence=False)
        # self.arq = self.tpfs[q - 1]
        for tpf in self.tpfs:
            with tpf.open() as f:
                if (q == f[0].header['quarter']):
                    self.arq = tpf
                    break
        self.main()

    def f_tpf(self):
        self.tbl = self.hdulist[2].data
        id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxpix = tpf.lerTPF(self.hdulist, 'FLUX')
        self.name = str(id) + str(qt)
        print(id, qt, season, ra, dec, mag, xdim, ydim, col, row)
        self.xdim, self.ydim, self.matrix = tpf.plotpixel(self.win[self.k], self.arq)
        for i in range(self.ydim):
            for j in range(self.xdim):
                if self.tbl[i][j] == 3:
                    self.maskx.append(j)
                    self.masky.append(i)

        for i in range(self.ydim - 1, -1, -1):
            for j in range(self.xdim):
                self.matrix[i][j].curve.setClickable(True)
                self.matrix[i][j].sigClicked.connect(self.make_updateMask(i, j))

        self.timedata, self.fluxdata, self.qdata, self.info = lc.realtime_newcurve(self.arq, self.maskx, self.masky)
        lc.fechar(self.hdulist)

        self.pw, self.graph, self.lr, self.timedata, self.fluxdata = lc.plot(self.win[self.k], row=0, col=self.xdim+2,
                                                                             rowspan=self.ydim+2,
                                                                             colspan=self.xdim+2,
                                                                             info=self.info, xcol=self.timedata,
                                                                             ycol=self.fluxdata, qcol=self.qdata)
        self.lr.sigRegionChanged.connect(self.updateLS)
        self.updateLS()

        for i in range(self.ydim + 2):
            self.pw2 = self.win[self.k].addPlot(row=self.ydim + 2 + i, col=2 * self.xdim + 4)
            self.pw2.showAxis('left', False)
            self.pw2.showAxis('bottom', False)
        for j in range(self.xdim + 2):
            self.pw2 = self.win[self.k].addPlot(row=2 * self.ydim + 4, col=self.xdim + 2 + j)
            self.pw2.showAxis('left', False)
            self.pw2.showAxis('bottom', False)

        self.pf1, self.pf2, self.flux_array = fit.tendencia(self.win[self.k], self.arq, 'cotrend' + self.name,
                                                            '1 2 3 4 5 6',
                                                            data=[self.timedata, self.fluxdata, self.info],
                                                            xdim=self.xdim, ydim=self.ydim)

    def f_lc(self):
        self.tbl = self.hdulist[1].data
        self.name = ''
        self.xdim, self.ydim = 2, 2
        self.timedata = lc.timecol(self.tbl)
        self.fluxdata = lc.fluxcol(self.tbl)
        self.timedata = self.timedata[~np.isnan(self.fluxdata)]
        self.fluxdata = self.fluxdata[~np.isnan(self.fluxdata)]
        self.fluxdata = self.fluxdata / np.median(self.fluxdata) - 1.0
        self.info = lc.time_info(self.hdulist)
        self.qdata = np.empty(np.shape(self.fluxdata))
        __, __, self.lr, __, __ = lc.plot(self.win[self.k], row=0, col=self.xdim+2, rowspan=self.ydim+2,
                                          colspan=self.xdim + 2, quality=False, info=self.info, plslr=True,
                                          xcol=self.timedata, ycol=self.fluxdata, qcol=self.qdata)
        self.lr.sigRegionChanged.connect(self.updateLS)
        self.updateLS()

        rot, noise, __ = cleanest.cleanest(self.timedata, self.fluxdata, 2)
        lc.plot(self.win[self.k], row=0, col=0, rowspan=self.ydim+2, colspan=self.xdim+2, quality=False,
                     info=self.info, xcol=self.timedata, ycol=rot, qcol=self.qdata)
        lc.plot(self.win[self.k], row=self.ydim+2, col=self.xdim+2, rowspan=self.ydim+2, colspan=self.xdim+2,
                     quality=False, info=self.info, xcol=self.timedata, ycol=noise, qcol=self.qdata)

    def main(self):
        self.hdulist = lc.abrir(self.arq, 'readonly')
        if self.hdulist[1].name != 'TARGETTABLES':
            self.f_lc()
        else:
            self.f_tpf()
        self.win[self.k].show()

    def updatePlot(self):
        self.pw.hide()
        self.graph.hide()
        self.lr.hide()
        self.lr.disconnect()
        self.pw.showLabel('bottom', False)
        self.pw.showLabel('left', False)
        self.pw.showAxis('bottom', False)
        self.pw.showAxis('left', False)
        self.win[self.k].removeItem(self.pw)
        self.pw, self.graph, self.lr, self.timedata, self.fluxdata = lc.plot(self.win[self.k], row=0, col=self.xdim+2,
                                                                             rowspan=self.ydim+2,
                                                                             colspan=self.xdim+2, info=self.info,
                                                                             xcol=self.timedata, ycol=self.fluxdata,
                                                                             qcol=self.qdata)
        self.win[self.k].removeItem(self.pf1)
        self.win[self.k].removeItem(self.pf2)
        self.pf1, self.pf2, self.flux_array = fit.tendencia(self.win[self.k], self.arq, 'cotrend' + self.arq,
                                                            '1 2 3 4 5 6',
                                                            data=[self.timedata, self.fluxdata, self.info],
                                                            xdim=self.xdim, ydim=self.ydim)
        self.lr.sigRegionChanged.connect(self.updateLS)

    def updateLS(self):
        print(self.lr.getRegion()[0], '---', self.lr.getRegion()[1])
        self.xmin = self.lr.getRegion()[0]
        self.xmax = self.lr.getRegion()[1]
        self.ini = 0
        self.end = -1
        while self.ini < len(self.timedata) and self.timedata[self.ini] < self.xmin:
            self.ini += 1
        while self.end > -(len(self.timedata)) and self.timedata[self.end] > self.xmax:
            self.end -= 1
        self.time = self.timedata[self.ini:self.end]
        self.flux = self.fluxdata[self.ini:self.end]
        if len(self.time) < 10:
            self.freq, self.power = (np.linspace(0.5, 50.5, 501), [0 for i in range(501)])
        else:
            self.freq = 1. / np.linspace(.5, 50.5, 501)
            self.power = LombScargle(self.time, self.flux).power(self.freq)

            '''self.best_freq = self.freq[np.argmax(self.power)]
            self.phase = (self.time * self.best_freq) % 1
            self.phase_fit = np.linspace(0, 1)
            self.flux_fit = LombScargle(self.time, self.flux).model(t=self.phase_fit / self.best_freq,
                                                                    frequency=self.best_freq)'''
        try:
            self.p.hide()
            self.win[self.k].removeItem(self.p)
        except:
            pass
        self.p = self.win[self.k].addPlot(title='Lomb Scargle', row=self.ydim+2, col=0,
                                          rowspan=self.ydim+2, colspan=self.xdim+2)
        self.p.setLabel('bottom', 'periodo', units='dias')
        self.p.setLabel('left', 'amplitude')
        self.p.plot(x=1. / self.freq, y=self.power, pen='r')

    def make_updateMask(self, i, j):
        def updateMask(curve):
            print(self.maskx, self.masky, i, j)
            for k in range(len(self.maskx)):
                if i == self.masky[k] and j == self.maskx[k]:
                    curve.setPen('b')
                    provx = []
                    provy = []
                    for k2 in range(len(self.maskx)):
                        if i != self.masky[k2] or j != self.maskx[k2]:
                            provx.append(self.maskx[k2])
                            provy.append(self.masky[k2])
                    self.maskx = provx
                    self.masky = provy
                    break
            else:
                self.maskx.append(j)
                self.masky.append(i)
                curve.setPen('g')
            self.timedata, self.fluxdata, self.qdata, self.info = lc.realtime_newcurve(self.arq, self.maskx, self.masky)
            self.updatePlot()

        return updateMask


if __name__ == '__main__':
    app = QtGui.QApplication([])
    font = QtGui.QFont('noto sans', 18)
    app.setFont(font)
    ex = start()
    app.exec_()
