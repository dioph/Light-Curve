import pyqtgraph as pg, kplr
from astropy.io import fits
from pyqtgraph import QtGui, Qt
import lc, tpf, fit

class start(QtGui.QWidget):
    def __init__(self):
        super(start, self).__init__()
        self.init_gui()
        self.con()
        self.type = ''
        self.client = kplr.API()
        self.arq = ''
        self.kic = None
        
    def init_gui(self):
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
        self.title = QtGui.QLabel('<b>ANALISE DE CURVAS DE LUZ</b>', alignment=0x0004)
        self.bt_n = QtGui.QPushButton('usar um novo arquivo')
        #self.kic = QtGui.QPushButton('buscar online')
        self.bt_p = QtGui.QPushButton('plotar curva de luz do arquivo atual')
        self.bt_x = QtGui.QPushButton('plotar os pixels do alvo atual')
        self.bt_m = QtGui.QPushButton('definir uma nova mascara para o alvo atual')
        self.bt_e = QtGui.QPushButton('extrair nova curva de luz aplicando mascara')
        self.bt_t = QtGui.QPushButton('remover CBVs da curva de luz')
        self.txt = QtGui.QLineEdit()
        self.txt1 = QtGui.QLineEdit()
        self.txt2 = QtGui.QLineEdit()
        self.sub1 = QtGui.QPushButton('submit')
        self.sub2 = QtGui.QPushButton('submit')
        self.arqname = QtGui.QLabel()
        self.cancel = QtGui.QPushButton('cancel')
        self.info = QtGui.QLabel()
        
        self.layout.addWidget(self.title, 0, 0, 1, 2)
        self.layout.addWidget(self.bt_n, 1, 0)
        #self.layout.addWidget(self.kic, 2, 0)
        self.layout.addWidget(self.bt_p, 2, 0)
        self.layout.addWidget(self.bt_x, 2, 0)
        self.layout.addWidget(self.bt_m, 3, 0)
        self.layout.addWidget(self.bt_e, 4, 0)
        self.layout.addWidget(self.bt_t, 3, 0)
        self.layout.addWidget(self.txt, 2, 1)
        self.layout.addWidget(self.txt1, 6, 0)
        self.layout.addWidget(self.txt2, 5, 0)
        self.layout.addWidget(self.sub1, 3, 1)
        self.layout.addWidget(self.sub2, 7, 0)
        self.layout.addWidget(self.arqname, 1, 1)
        self.layout.addWidget(self.cancel, 5, 1)
        self.layout.addWidget(self.info, 2, 1, 3, 1)

        self.bt_p.hide()
        self.bt_x.hide()
        self.bt_m.hide()
        self.bt_e.hide()
        self.bt_t.hide()
        self.txt.hide()
        self.txt1.hide()
        self.txt2.hide()
        self.sub1.hide()
        self.sub2.hide()
        self.cancel.hide()
        self.info.hide()
        
        self.setGeometry(10, 10, 1000, 600)
        self.show()

    def con(self):
        self.bt_n.clicked.connect(self.click_n)
        #self.kic.clicked.connect(self.click_kic)
        self.bt_p.clicked.connect(self.click_p)
        self.bt_x.clicked.connect(self.click_x)
        self.bt_m.clicked.connect(self.click_m)
        self.bt_e.clicked.connect(self.click_e)
        self.bt_t.clicked.connect(self.click_t)
        self.cancel.clicked.connect(self.options)
        
    def click_n(self):
        self.cancel.show()
        if self.type == 'tpf':
            self.txt1.setText('Nome do arquivo .fits')
            self.txt1.show()
            self.sub2.show()
            self.sub2.disconnect()
            self.sub2.clicked.connect(self.input_n)
        else:
            self.txt.setText('Nome do arquivo .fits')
            self.txt.show()
            self.sub1.show()
            self.sub1.disconnect()
            self.sub1.clicked.connect(self.input_n)

    def input_n(self):
        if self.type == 'tpf':
            self.txt1.hide()
            self.arq = self.txt1.text()
        else:
            self.txt.hide()
            self.arq = self.txt.text()
        self.hdulist = fits.open(self.arq)
        print('type: '+self.hdulist[1].name)
        if self.hdulist[1].name == 'LIGHTCURVE':
            self.type = 'lc'
        else:
            self.type = 'tpf'
        self.tbl = self.hdulist[1].data
        self.options()
        print(self.arq+' aberto com sucesso')
    '''
    def click_kic(self):
        if self.type == 'tpf':
            self.txt1.hide()
            self.kic = int(self.txt1.text())
        else:
            self.txt.hide()
            self.kic = int(self.txt.text())
        self.star = self.client.star(self.kic)
        lcs = self.star.get_light_curves(short_cadence=False)
        time, flux, ferr, quality, quarter = [], [], [], [], []
        for lc in lcs:
            with lc.open() as f:
                self.tbl = f[1].data
                time = np.r_[time,self.tbl["time"]]
                flux = np.r_[flux,self.tbl["sap_flux"]]
                ferr = np.r_[ferr,self.tbl["sap_flux_err"]]
                quality = np.r_[quality,self.tbl["sap_quality"]]
                quarter = np.r_[quarter,f[0].header["QUARTER"] + np.zeros(len(self.tbl["time"]))]
    '''   
    def click_p(self):
        self.options()
        self.cancel.show()
        self.txt.setText('Nome da coluna de dados')
        self.txt.show()
        self.sub1.show()
        self.sub1.disconnect()
        self.sub1.clicked.connect(self.input_p)

    def input_p(self):
        self.txt.hide()
        self.col = self.txt.text()
        self.win = lc.plot(self.arq, self.col)
        self.options()

    def click_x(self):
        self.options()
        self.view = tpf.plotpixel(self.arq)

    def click_m(self):
        self.options()
        tpf.plotmask(self.arq)

    def click_e(self):
        self.options()
        self.cancel.show()
        self.txt1.setText('Nome da mascara de pixels .txt')
        self.txt2.setText('Nome do arquivo .fits de saida')
        self.txt1.show()
        self.txt2.show()
        self.sub2.show()
        self.sub2.disconnect()
        self.sub2.clicked.connect(self.input_e)

    def input_e(self):
        self.novo_arq = self.txt2.text()
        self.mask = self.txt1.text()
        lc.new_curve(self.arq, self.mask, self.novo_arq)
        self.options()

    def click_t(self):
        self.options()
        self.cancel.show()
        self.txt1.setText('Lista de vetores CBV')
        self.txt2.setText('Nome do arquivo .fits de saida')
        self.txt1.show()
        self.txt2.show()
        self.sub2.show()
        self.sub2.disconnect()
        self.sub2.clicked.connect(self.input_t)

    def input_t(self):
        self.novo_arq = self.txt2.text()
        self.cbv_list = self.txt1.text()
        self.win = fit.tendencia(self.arq, self.novo_arq, self.cbv_list)
        self.options()
        
    def options(self):
        print('OPTIONS')
        self.cancel.hide()
        self.arqname.setText('Arquivo atual: '+self.arq)
        self.txt.hide()
        self.txt1.hide()
        self.txt2.hide()
        self.sub1.hide()
        self.sub2.hide()
        if self.type == 'lc':
            self.bt_p.show()
            self.bt_x.hide()
            self.bt_m.hide()
            self.bt_e.hide()
            self.bt_t.show()
            self.info.hide()
            
        elif self.type == 'tpf':
            self.bt_p.hide()
            self.bt_x.show()
            self.bt_m.show()
            self.bt_e.show()
            self.bt_t.hide()
            id, qt, season, ra, dec, mag, xdim, ydim, col, row, fluxpix = tpf.lerTPF(self.arq, 'FLUX')
            self.info.setText('ID: '+id+'\nRA (J2000): '+ra+'\nDec (J2000): '+dec+ '\nMagnitude: '+mag+'\nQuarter: '+qt)
            self.info.show()     
            
if __name__ == '__main__':
	app = QtGui.QApplication([])
	ex = start()
	app.exec_()

